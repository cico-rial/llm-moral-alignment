"""
PPO + LoRA fine-tuning loop using trl==0.11
GPU target: RTX 4090 32GB
"""

import torch
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, PPOv2Trainer, PPOv2Config
from trl.core import LengthSampler

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# ──────────────────────────────────────────────
# 1. MODEL REGISTRY  (6 quantised options)
# ──────────────────────────────────────────────
MODEL_OPTIONS = {
    "phi3-mini":      "microsoft/Phi-3-mini-4k-instruct",      # ~3.8B  – very fast
    "gemma2-2b":      "google/gemma-2-2b-it",                  # ~2.6B  – lightweight
    "gemma-3-270m": "google/gemma-3-270m-it",
    "llama3.2-3b":    "meta-llama/Llama-3.2-3B-Instruct",      # ~3.2B  – solid baseline
    "mistral-7b":     "mistralai/Mistral-7B-Instruct-v0.3",    # ~7B    – 4-bit needed
    "qwen2.5-7b":     "Qwen/Qwen2.5-7B-Instruct",             # ~7.6B  – strong instruct
}

# ──────────────────────────────────────────────
# ▶  SELECT YOUR MODEL HERE
# ──────────────────────────────────────────────
SELECTED_MODEL = "gemma-3-270m"

# ──────────────────────────────────────────────
# 2. GAME REGISTRY
# ──────────────────────────────────────────────
@dataclass(frozen=True)
class Game:
    """
    A symmetric 2-action, 2-player game defined entirely by its payoff matrix.
 
    Payoff matrix (row = my action, col = opponent action):
 
                      act1              act2
        act1    (aa_me, aa_op)    (ab_me, ab_op)
        act2    (ba_me, ba_op)    (bb_me, bb_op)
 
    *_me  = points earned by the acting player for that cell.
    *_op  = points earned by the opponent for that cell.
    """
    name:  str    # human-readable label used in logs and checkpoints
    act1:  str    # label for action 1  (e.g. "COOPERATE")
    act2:  str    # label for action 2  (e.g. "DEFECT")
 
    aa_me: int;  aa_op: int   # both play act1
    ab_me: int;  ab_op: int   # I play act1, opponent plays act2
    ba_me: int;  ba_op: int   # I play act2, opponent plays act1
    bb_me: int;  bb_op: int   # both play act2
 
    def lookup_score(self, my_action: str, their_action: str) -> tuple[int, int]:
        """Return (my_points, their_points) for a given action pair."""
        if my_action == self.act1 and their_action == self.act1:
            return self.aa_me, self.aa_op
        if my_action == self.act1 and their_action == self.act2:
            return self.ab_me, self.ab_op
        if my_action == self.act2 and their_action == self.act1:
            return self.ba_me, self.ba_op
        if my_action == self.act2 and their_action == self.act2:
            return self.bb_me, self.bb_op
        return INVALID_ACTION_REWARD
 
    def payoff_table_str(self, opponent_name: str = "opponent") -> str:
        """Human-readable payoff table, ready to embed in a prompt."""
        return (
            f"You play <{self.act1}> and {opponent_name} plays <{self.act1}>"
            f"  ->  you get {self.aa_me},  {opponent_name} gets {self.aa_op}\n"
            f"You play <{self.act1}> and {opponent_name} plays <{self.act2}>"
            f"  ->  you get {self.ab_me},  {opponent_name} gets {self.ab_op}\n"
            f"You play <{self.act2}> and {opponent_name} plays <{self.act1}>"
            f"  ->  you get {self.ba_me},  {opponent_name} gets {self.ba_op}\n"
            f"You play <{self.act2}> and {opponent_name} plays <{self.act2}>"
            f"  ->  you get {self.bb_me},  {opponent_name} gets {self.bb_op}"
        )
    

# ── Built-in game library ────────────────────────────────────────────────────
GAMES: dict[str, Game] = {
 
    # Classic Prisoner's Dilemma  (defection dominates, but cooperation is socially optimal)
    "prisoners_dilemma": Game(
        name="Prisoner's Dilemma",
        act1="COOPERATE", act2="DEFECT",
        aa_me=3, aa_op=3,
        ab_me=0, ab_op=5,
        ba_me=5, ba_op=0,
        bb_me=1, bb_op=1,
    ),
 
    # Stag Hunt  (coordination game – mutual trust dominates)
    "stag_hunt": Game(
        name="Stag Hunt",
        act1="STAG", act2="HARE",
        aa_me=4, aa_op=4,
        ab_me=0, ab_op=3,
        ba_me=3, ba_op=0,
        bb_me=2, bb_op=2,
    ),
 
    # Hawk-Dove / Chicken  (anti-coordination – both hawking is the worst outcome)
    "hawk_dove": Game(
        name="Hawk-Dove",
        act1="DOVE", act2="HAWK",
        aa_me=3, aa_op=3,
        ab_me=1, ab_op=5,
        ba_me=5, ba_op=1,
        bb_me=0, bb_op=0,
    ),
 
    # Battle of the Sexes  (asymmetric coordination – players prefer different equilibria)
    "battle_of_sexes": Game(
        name="Battle of the Sexes",
        act1="OPERA", act2="FOOTBALL",
        aa_me=3, aa_op=2,
        ab_me=0, ab_op=0,
        ba_me=0, ba_op=0,
        bb_me=2, bb_op=3,
    ),
 
    # Pure coordination  (symmetric – just match your partner)
    "coordination": Game(
        name="Coordination",
        act1="LEFT", act2="RIGHT",
        aa_me=2, aa_op=2,
        ab_me=0, ab_op=0,
        ba_me=0, ba_op=0,
        bb_me=2, bb_op=2,
    ),
}
 
# ──────────────────────────────────────────────
# ▶  SELECT YOUR GAME HERE
# ──────────────────────────────────────────────
SELECTED_GAME = "prisoners_dilemma"

# ──────────────────────────────────────────────
# 3. QUANTISATION CONFIG
#    4-bit for 7B models, 8-bit for <4B models
# ──────────────────────────────────────────────
def get_bnb_config(model_key: str) -> BitsAndBytesConfig:
    large_models = {"mistral-7b", "qwen2.5-7b"}
    if model_key in large_models:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=False,
        )
    

# 4. LORA CONFIG
# ──────────────────────────────────────────────
LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    # Target the attention + MLP projections (works for most modern architectures)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)


# ──────────────────────────────────────────────
# 5. PPO CONFIG
# ──────────────────────────────────────────────
PPO_CFG = PPOConfig(
    model_name=MODEL_OPTIONS[SELECTED_MODEL],
    learning_rate=1e-5,
    batch_size=8,               # number of (prompt, response) pairs per PPO step
    mini_batch_size=2,          # gradient-accumulation sub-batches
    gradient_accumulation_steps=4,
    ppo_epochs=4,               # inner optimisation epochs per batch
    max_grad_norm=0.5,
    kl_penalty="kl",            # or "full" for full KL
    init_kl_coef=0.2,
    adap_kl_ctrl=True,
    target_kl=6.0,
    seed=42,
    optimize_cuda_cache=True,
    log_with=None,              # swap to "wandb" or "tensorboard" if desired
)

# Number of environment samples to collect before each PPO step
ROLLOUT_BATCH_SIZE = PPO_CFG.batch_size  # keep in sync


# ──────────────────────────────────────────────
# 6. GENERATION KWARGS
# ──────────────────────────────────────────────
GENERATION_KWARGS = {
    "min_length": -1,
    "top_k": 0,
    "top_p": 0.95,
    "do_sample": True,
    "pad_token_id": None,       # filled in after tokenizer is loaded
    "max_new_tokens": 256,
}


# ──────────────────────────────────────────────
# 7. MODEL + TOKENIZER LOADING
# ──────────────────────────────────────────────
def load_model_and_tokenizer(model_key: str):
    model_id = MODEL_OPTIONS[model_key]
    bnb_config = get_bnb_config(model_key)

    print(f"[load] Loading tokenizer from {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[load] Loading base model with quantisation …")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Inject LoRA adapters
    base_model = get_peft_model(base_model, LORA_CONFIG)
    base_model.print_trainable_parameters()

    # Wrap with value head for PPO
    model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)

    return model, tokenizer


# ──────────────────────────────────────────────
# 8.  CHAT TEMPLATES  (one per model family)
# ──────────────────────────────────────────────
# Each template receives a list of message dicts:
#   [{"role": "system", "content": "…"}, {"role": "user", "content": "…"}]
# and returns the fully-formatted string that is fed to the tokeniser.
#
# The templates deliberately do NOT append the assistant turn-start token
# so the model generates it freely during the PPO rollout.
 
def _apply_llama3_template(messages: list[dict]) -> str:
    """
    Meta LLaMA-3 / LLaMA-3.2 instruct format.
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>\n\n{sys}<|eot_id|>
    <|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>\n\n
    """
    out = "<|begin_of_text|>"
    for msg in messages:
        out += (
            f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n"
            f"{msg['content']}<|eot_id|>"
        )
    out += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return out
 
 
def _apply_mistral_template(messages: list[dict]) -> str:
    """
    Mistral v0.3 instruct format.
    [INST] {optional_sys + user} [/INST]
    Note: Mistral merges system into the first user turn.
    """
    system_text = ""
    turns = []
    for msg in messages:
        if msg["role"] == "system":
            system_text = msg["content"].strip() + "\n\n"
        elif msg["role"] == "user":
            turns.append(f"[INST] {system_text}{msg['content']} [/INST]")
            system_text = ""          # only prepend system to the first user turn
        elif msg["role"] == "assistant":
            turns.append(msg["content"])
    return " ".join(turns)
 
 
def _apply_phi3_template(messages: list[dict]) -> str:
    """
    Microsoft Phi-3 instruct format.
    <|system|>\n{sys}<|end|>\n<|user|>\n{user}<|end|>\n<|assistant|>\n
    """
    out = ""
    for msg in messages:
        role_tag = {"system": "<|system|>", "user": "<|user|>",
                    "assistant": "<|assistant|>"}[msg["role"]]
        out += f"{role_tag}\n{msg['content']}<|end|>\n"
    out += "<|assistant|>\n"
    return out
 
 
def _apply_gemma2_template(messages: list[dict]) -> str:
    """
    Google Gemma-2 instruct format.
    <bos><start_of_turn>user\n{optional_sys + user}<end_of_turn>\n
    <start_of_turn>model\n
    Note: Gemma-2 has no native system role; we prepend it to the user turn.
    """
    system_text = ""
    out = "<bos>"
    for msg in messages:
        if msg["role"] == "system":
            system_text = msg["content"].strip() + "\n\n"
        elif msg["role"] == "user":
            out += f"<start_of_turn>user\n{system_text}{msg['content']}<end_of_turn>\n"
            system_text = ""
        elif msg["role"] == "assistant":
            out += f"<start_of_turn>model\n{msg['content']}<end_of_turn>\n"
    out += "<start_of_turn>model\n"
    return out

def _apply_gemma3_template(messages: list[dict]) -> str:
    """
    Google Gemma-2 instruct format.
    <bos><start_of_turn>user\n{optional_sys + user}<end_of_turn>\n
    <start_of_turn>model\n
    Note: Gemma-2 has no native system role; we prepend it to the user turn.
    """
    system_text = ""
    out = "<bos>"
    for msg in messages:
        if msg["role"] == "system":
            system_text = msg["content"].strip() + "\n\n"
        elif msg["role"] == "user":
            out += f"<start_of_turn>user\n{system_text}{msg['content']}<end_of_turn>\n"
            system_text = ""
        elif msg["role"] == "assistant":
            out += f"<start_of_turn>model\n{msg['content']}<end_of_turn>\n"
    out += "<start_of_turn>model\n"
    return out
 
 
def _apply_qwen25_template(messages: list[dict]) -> str:
    """
    Qwen-2.5 instruct format (ChatML).
    <|im_start|>system\n{sys}<|im_end|>\n
    <|im_start|>user\n{user}<|im_end|>\n
    <|im_start|>assistant\n
    """
    out = ""
    for msg in messages:
        out += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    out += "<|im_start|>assistant\n"
    return out
 
 
# Dispatch table: model key → template function
_TEMPLATE_FN = {
    "llama3.2-3b": _apply_llama3_template,
    "mistral-7b":  _apply_mistral_template,
    "phi3-mini":   _apply_phi3_template,
    "gemma2-2b":   _apply_gemma2_template,
    "gemma-3-270m": _apply_gemma3_template,
    "qwen2.5-7b":  _apply_qwen25_template,
}

# ──────────────────────────────────────────────
# 9. ETHICAL FRAMEWORK REGISTRY
# ──────────────────────────────────────────────

INVALID_ACTION_REWARD: float = -5.0   # applied whenever the LLM outputs neither valid action
DEONTOLOGICAL_PENALTY: float = -5.0   # applied when defecting against a cooperator


@dataclass(frozen=True)
class EthicalFramework:
    """
    Encapsulates one ethical reward scheme.

    Fields
    ------
    name        : str  – key used in FRAMEWORKS registry and logs.
    description : str  – injected into the system prompt so the model knows
                         which objective it is being trained on.
    """
    name:        str
    description: str

    def compute_reward(
        self,
        player_action:        str | None,
        opponent_action:      str | None,
        game:                 "Game",
        last_opponent_action: str | None = None,
    ) -> float:
        """
        Compute the scalar training reward.

        Parameters
        ----------
        player_action        : str | None  Parsed LLM action (None = invalid output).
        opponent_action      : str | None  Opponent's action this round.
        game                 : Game        Active game (provides payoff lookup).
        last_opponent_action : str | None  Opponent's action in the *previous* round
                                           (required by the deontological framework).

        Returns
        -------
        float – reward signal fed to PPO.
        """
        # ── Invalid-action guard (applies to every framework) ─────────────────
        if player_action is None or opponent_action is None:
            return INVALID_ACTION_REWARD

        my_score, their_score = game.lookup_score(player_action, opponent_action)

        if self.name == "selfish":
            # Only the LLM's own payoff matters.
            return float(my_score)

        elif self.name == "utilitarian":
            # Social welfare: maximise the sum of both players' payoffs.
            return float(my_score + their_score)

        elif self.name == "deontological":
            # Fixed penalty for defecting against an opponent who cooperated
            # last round; 0 in every other case (no reward for outcomes, only
            # a penalty for violating the fairness rule).
            betrayal = (
                player_action == game.act2   # LLM defected
                and last_opponent_action == game.act1   # opponent cooperated last round
            )
            return DEONTOLOGICAL_PENALTY if betrayal else 0.0

        else:
            raise ValueError(
                f"Unknown ethical framework '{self.name}'. "
                f"Available: {list(FRAMEWORKS.keys())}"
            )


FRAMEWORKS: dict[str, EthicalFramework] = {

    "selfish": EthicalFramework(
        name="selfish",
        description=(
            "Your sole objective is to maximise YOUR OWN points. "
            "You do not care about the opponent's score."
        ),
    ),

    "utilitarian": EthicalFramework(
        name="utilitarian",
        description=(
            "Your objective is to maximise the TOTAL points earned by both players combined. "
            "A good outcome for your opponent is also good for you."
        ),
    ),

    "deontological": EthicalFramework(
        name="deontological",
        description=(
            "You follow a strict fairness rule: you must NEVER defect against an opponent "
            "who cooperated in the previous round. Violating this rule is always wrong, "
            "regardless of the points involved."
        ),
    ),
}

# ──────────────────────────────────────────────
# ▶  SELECT YOUR ETHICAL FRAMEWORK HERE
# ──────────────────────────────────────────────
SELECTED_FRAMEWORK = "selfish"


# ──────────────────────────────────────────────
# 10. PROMPT BUILDER
# ──────────────────────────────────────────────
def _build_raw_messages(
    game: Game,
    framework: "EthicalFramework | None" = None,
    opponent_name: str = "A",
    last_player_action: str | None = None,
    last_opponent_action: str | None = None,
) -> list[dict]:
    """
    Build the message list for one round of an iterated game.
 
    Parameters
    ----------
    game                 : Game       Fully specifies the payoff matrix and action labels.
    opponent_name        : str        Short name used to refer to the opponent in the prompt.
    last_player_action   : str | None Action the player took last round (None = first round).
    last_opponent_action : str | None Action the opponent took last round (None = first round).
 
    Returns
    -------
    list[dict]  -- ready to be passed to get_prompt() / any chat template.
    """
 
    # Resolve framework (fallback to SELECTED_FRAMEWORK global)
    if framework is None:
        framework = FRAMEWORKS[SELECTED_FRAMEWORK]

    # System message: stable rules + persona
    system_content = f"""\
    You are a rational agent playing an iterated two-player game against an opponent named {opponent_name}.

    ACTIONS
    You must choose exactly one of two actions each round:
    <{game.act1}> or <{game.act2}>

    PAYOFF TABLE (your points / opponent's points)
    {game.payoff_table_str(opponent_name)}

    OBJECTIVE
    {framework.description}

    OUTPUT FORMAT
    Reply with exactly one token: <{game.act1}> or <{game.act2}>
    Do not explain your reasoning. Do not add any other text."""
 
    # User message: dynamic episode state + action request
    if last_player_action is None and last_opponent_action is None:
        history_line = "This is the first round. You have no prior history with this opponent."
    elif last_player_action is None and last_opponent_action is not None:
        my_score = game.lookup_score(last_player_action, last_opponent_action)
        history_line = (
            f"Last round: you played an invalid action and scored {my_score} point(s)"
        )
    else:
        my_score, their_score = game.lookup_score(last_player_action, last_opponent_action)
        history_line = (
            f"Last round: you played <{last_player_action}> and {opponent_name} played "
            f"<{last_opponent_action}>. "
            f"You scored {my_score} point(s) and {opponent_name} scored {their_score} point(s)."
        )
 
    user_content = f"""\
    {history_line}
 
    What action do you choose this round?
    Your answer: """
 
    return [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": user_content},
    ]
 
def get_prompt(
    model_key: str = SELECTED_MODEL,
    game: Game | None = None,
    framework: EthicalFramework | None = None,
    opponent_name: str = "A",
    last_player_action: str | None = None,
    last_opponent_action: str | None = None,
) -> str:
    """
    Build a chat-formatted prompt string for the selected model and game.
 
    Parameters
    ----------
    model_key            : str        One of the keys in MODEL_OPTIONS.
    game                 : Game       Game definition (defaults to GAMES[SELECTED_GAME]).
    opponent_name        : str        Name used to refer to the opponent.
    last_player_action   : str | None Last round's player action (None = first round).
    last_opponent_action : str | None Last round's opponent action (None = first round).
 
    Returns
    -------
    str  --  Fully formatted prompt, ready to be tokenised and fed to the model.
    """
    if model_key not in _TEMPLATE_FN:
        raise ValueError(
            f"No chat template registered for '{model_key}'. "
            f"Available: {list(_TEMPLATE_FN.keys())}"
        )
    if game is None:
        game = GAMES[SELECTED_GAME]
 
    messages = _build_raw_messages(
        game=game,
        framework=framework,
        opponent_name=opponent_name,
        last_player_action=last_player_action,
        last_opponent_action=last_opponent_action,
    )
    return _TEMPLATE_FN[model_key](messages)


# ──────────────────────────────────────────────
# 11. Get Player's and Opponent's actions.
# ──────────────────────────────────────────────
def get_opponent_action(game: Game, last_player_action: str, opponent_name):
    AVAILABLE_OPPONENTS = ["AC", "AD", "TFT", "RAND"]

    if opponent_name == "AC": # always cooperate
        return game.act1
    
    elif opponent_name == "AD": # always defect
        return game.act2
    
    elif opponent_name == "TFT":
        if last_player_action is None: # if action is invalid or is the first turn
            return game.act1
        return last_player_action
    
    elif opponent_name == "RAND":
        actions = [game.act1, game.act2]
        random_action = round(torch.rand(1,).item())
        return actions[random_action]
    
    else:
        raise ValueError(
            f"{opponent_name} is not a valid opponent."
            f"Select from: {AVAILABLE_OPPONENTS}")
        
        
def process_player_response(response_text: str, game: Game):
    """Process the player's response to extract the action."""
    if game.act1 in response_text and game.act2 in response_text:
        return None
    
    elif game.act1 in response_text:
        return game.act1
    
    elif game.act2 in response_text:
        return game.act2
    
    else:
        return None
    
def log_game(prompt: str ,response: str, actions: list | None = None, log: str | None = None):
    if log == "full":
        print(f"Prompt:\n{prompt}")
        print("")
        print(f"Response:\n{response}")
        print("*"*60)
    elif log == "short" and actions is not None:
        print("")
        print(f"LLM:      {actions[0]}")
        print(f"opponent: {actions[1]}")
        print("")


# ──────────────────────────────────────────────
# 12. ROLLOUT  →  collect N (prompt, response, reward) triples
# ──────────────────────────────────────────────
def collect_rollouts(
    ppo_trainer: PPOTrainer,
    tokenizer,
    n: int,
    model_key: str = SELECTED_MODEL,
    game: Game | None = None,
    framework: EthicalFramework | None = None,
    opponent_name: str = "A",
    last_player_action: str | None = None,
    last_opponent_action: str | None = None,
    log: str | None = None
):
    """
    Generate `n` responses and score them.
 
    Returns
    -------
    queries   : list[torch.Tensor]  – tokenised prompts
    responses : list[torch.Tensor]  – tokenised model outputs
    rewards   : list[torch.Tensor]  – scalar reward tensors
    actions   : list[list]  – list of actions 
    """

    if game is None:
        game = GAMES[SELECTED_GAME]
    if framework is None:
        framework = FRAMEWORKS[SELECTED_FRAMEWORK]

    queries, responses, rewards, actions = [], [], [], []
    
    GENERATION_KWARGS["pad_token_id"] = tokenizer.pad_token_id
 
    for _ in range(n):
        prompt_text = get_prompt(
            model_key=model_key,
            game=game,
            framework=framework,
            opponent_name=opponent_name,
            last_player_action=last_player_action,
            last_opponent_action=last_opponent_action,
        )
 
        # Tokenise prompt
        input_ids = tokenizer.encode(prompt_text, return_tensors="pt").squeeze(0).to(device)
 
        # Generate response via PPOTrainer (handles ref-model tracking)
        response_ids = ppo_trainer.generate(
            input_ids,
            **GENERATION_KWARGS,
        ).squeeze(0).to(device)
 
        # Decode only the newly generated tokens
        response_text = tokenizer.decode(
            response_ids[len(input_ids):], skip_special_tokens=True
        )

        current_player_action = process_player_response(response_text, game) 
        current_opponent_action = get_opponent_action(game, last_player_action, opponent_name)
 
        # Score via the active ethical framework
        r = framework.compute_reward(
            player_action=current_player_action,
            opponent_action=current_opponent_action,
            game=game,
            last_opponent_action=last_opponent_action,  # needed by deontological
        )
        reward_tensor = torch.tensor(r, dtype=torch.float32)
 
        queries.append(input_ids)
        responses.append(response_ids[len(input_ids):])   # response tokens only
        rewards.append(reward_tensor)
        actions.append([current_player_action,current_opponent_action])

        # log for debug
        readable_prompt = tokenizer.decode(tokenizer.encode(prompt_text, return_tensors="pt").squeeze(0), skip_special_tokens=True)
        log_game(
            readable_prompt, 
            response_text, 
            actions=[current_player_action, current_opponent_action], 
            log=log)

        last_opponent_action = current_opponent_action
        last_player_action = current_player_action
 
    return queries, responses, rewards, actions


def train(
    num_steps: int = 200,
    game: Game | None = None,
    framework: EthicalFramework | None = None,
    opponent_name: str = "A",
    log: str | None = None
):
    """
    Main PPO training loop.
 
    Parameters
    ----------
    num_steps       : int       Number of PPO update steps (each step consumes ROLLOUT_BATCH_SIZE samples).
    game            : Game      Game to train on (defaults to GAMES[SELECTED_GAME]).
    opponent_name   : str       Name used for the opponent in prompts.
    """

    if game is None:
        game = GAMES[SELECTED_GAME]
    if framework is None:
        framework = FRAMEWORKS[SELECTED_FRAMEWORK]

    model, tokenizer = load_model_and_tokenizer(SELECTED_MODEL)
 
    ppo_trainer = PPOTrainer(
        config=PPO_CFG,
        model=model,
        ref_model=None,   # trl auto-creates a frozen copy when None
        tokenizer=tokenizer,
    )
 
    print(f"\n{'='*60}")
    print(f"  Model   : {SELECTED_MODEL}  ({MODEL_OPTIONS[SELECTED_MODEL]})")
    print(f"  Game    : {game.name}  [{game.act1} / {game.act2}]")
    print(f"  Ethics  : {framework.name}")
    print(f"  Opponent: {opponent_name}")
    print(f"  Steps   : {num_steps}")
    print(f"  Rollouts: {ROLLOUT_BATCH_SIZE} per step")
    print(f"{'='*60}\n")

    # single episode. could insert a for loop for running several episodes!

    last_player_action = None 
    last_opponent_action = None
 
    for step in range(num_steps):
 
        # ── 9a. Collect rollouts ──────────────────────────────────
        queries, responses, rewards, actions = collect_rollouts(
            ppo_trainer, tokenizer,
            n=ROLLOUT_BATCH_SIZE,
            model_key=SELECTED_MODEL,
            game=game,
            framework=framework,
            opponent_name=opponent_name,
            log=log,
            last_player_action=last_player_action,
            last_opponent_action=last_opponent_action,
        )

        print(rewards)

        # ── 9b. last action retrieval ─────────────────────────────
        last_player_action = actions[-1][0] 
        last_opponent_action = actions[-1][1]
 
        # ── 9c. PPO update step ───────────────────────────────────
        stats = ppo_trainer.step(queries, responses, rewards)
 
        # ── 9d. Logging ───────────────────────────────────────────
        mean_reward = sum(r.item() for r in rewards) / len(rewards)
        print(
            f"[step {step+1:>4}/{num_steps}]  "
            f"mean_reward={mean_reward:+.4f}  "
            f"kl={stats.get('objective/kl', float('nan')):.4f}  "
            f"policy_loss={stats.get('ppo/loss/policy', float('nan')):.4f}"
        )
 
        ppo_trainer.log_stats(stats, {"query": queries, "response": responses}, rewards)
 
        # ── 9d. Periodic checkpoint (game name embedded) ───────────
        if (step + 1) % 50 == 0:
            ckpt_path = (
                f"./checkpoints/{SELECTED_MODEL}"
                f"_{game.name.replace(' ', '_')}"
                f"_{framework.name}"
                f"_step{step+1}"
            )
            ppo_trainer.save_pretrained(ckpt_path, safe_serialization=False)
            print(f"  ✔ Checkpoint saved → {ckpt_path}")
 
    # ── 10. Final save ────────────────────────────────────────────
    final_path = f"./checkpoints/{SELECTED_MODEL}_final"
    ppo_trainer.save_pretrained(ckpt_path, safe_serialization=False)
    print(f"\n✅ Training complete. Model saved to {final_path}")


if __name__ == "__main__":
    train(
        num_steps=200,
        game=GAMES[SELECTED_GAME],
        framework=FRAMEWORKS[SELECTED_FRAMEWORK],
        opponent_name="TFT",
        log="full"
    )