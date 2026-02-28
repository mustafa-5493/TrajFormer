# config.py
# TrajFormer — Decision Transformer for Hopper-v5 Control
#
# Decision Transformer formulation:
#   Input:  (R, s, a, R, s, a, ...) sequences
#           R = reward-to-go (how much reward from this point forward)
#           s = state (11-dim for Hopper)
#           a = action (3-dim for Hopper)
#   Output: predicted action at each timestep
#
# conditioning on reward-to-go rather than past rewards
# allows the model to be "told" how well to perform at inference time.
# Set R=high at inference → model tries to perform well.
# This is the core insight of Decision Transformer (Chen et al. 2021).

import torch
from pathlib import Path

# =============================================================================
# PROJECT ROOT
# =============================================================================

ROOT = Path(__file__).parent.resolve()

# =============================================================================
# PATHS
# =============================================================================

DATA_DIR        = ROOT / "data" / "trajectories"
CHECKPOINT_DIR  = ROOT / "outputs" / "checkpoints"
PLOTS_DIR       = ROOT / "outputs" / "plots"
BENCHMARK_DIR   = ROOT / "outputs" / "benchmarks"
EXPORTS_DIR     = ROOT / "outputs" / "exports"

for _dir in [DATA_DIR, CHECKPOINT_DIR, PLOTS_DIR, BENCHMARK_DIR, EXPORTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# =============================================================================
# DEVICE
# =============================================================================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()

# =============================================================================
# ENVIRONMENT
# =============================================================================

ENV_NAME        = "Hopper-v5"
STATE_DIM       = 11        # Hopper observation space
ACTION_DIM      = 3         # Hopper action space
MAX_EPISODE_STEPS = 1000    # Max steps per episode
ACTION_RANGE    = 1.0       # Actions clipped to [-1, 1]

# =============================================================================
# DATA COLLECTION
# =============================================================================

# Number of expert trajectories to collect
# Each trajectory = one full episode (up to MAX_EPISODE_STEPS steps)
# 500 episodes × ~300 avg steps = ~150,000 transitions
# Enough to train on, small enough to fit in RAM
N_COLLECT_EPISODES  = 500

# Controller type for data collection
# "pid"    : simple PID-like controller, fast, deterministic
# "random" : random actions (baseline, not recommended)
# "sine"   : sinusoidal joint commands (effective for hopper)
CONTROLLER_TYPE = "sine"

# Save raw trajectories as individual .npz files
# One file per episode: states, actions, rewards, dones
SAVE_INDIVIDUAL = True

# =============================================================================
# DECISION TRANSFORMER — SEQUENCE
# =============================================================================

# Context length K: number of timesteps the transformer sees at once
# Decision Transformer paper uses K=20 for Hopper
# Each timestep = (R, s, a) triple → K timesteps = 3K tokens
CONTEXT_LENGTH  = 20        # K in the paper
N_TOKENS        = CONTEXT_LENGTH * 3   # R, s, a per timestep = 60 tokens

# Reward-to-go target at inference time
# Set high to elicit good behavior from the model
# Hopper expert typically achieves ~3000-3500 total reward per episode
EVAL_TARGET_RETURN = 3000.0

# Reward scale — normalize reward-to-go to reasonable range
# Divides all reward-to-go values by this before feeding to model
REWARD_SCALE    = 1000.0

# =============================================================================
# DECISION TRANSFORMER — ARCHITECTURE
# =============================================================================

# These fit comfortably in 4GB VRAM
# Comparable to DT paper's Hopper config (which used similar dims)
D_MODEL         = 128       # Token embedding dimension
N_HEADS         = 8         # Attention heads (D_MODEL must be divisible by N_HEADS)
N_LAYERS        = 6         # Transformer decoder layers
D_FF            = 512       # Feed-forward hidden dimension (4 × D_MODEL)
DROPOUT         = 0.1

# Embedding dimensions for each modality
# All projected to D_MODEL before entering transformer
STATE_EMBED_DIM  = D_MODEL
ACTION_EMBED_DIM = D_MODEL
RETURN_EMBED_DIM = D_MODEL

# =============================================================================
# TRAINING
# =============================================================================

BATCH_SIZE      = 64
NUM_EPOCHS      = 50
LEARNING_RATE   = 1e-4
WEIGHT_DECAY    = 1e-4
GRAD_CLIP       = 1.0

# Learning rate warmup
# Linearly increase LR for first WARMUP_STEPS steps
WARMUP_STEPS    = 500

# AMP
USE_AMP         = True      # Disabled automatically on MPS

# Train/val split
TRAIN_RATIO     = 0.9
VAL_RATIO       = 0.1

# Early stopping
PATIENCE        = 10

# Reproducibility
SEED            = 42

# =============================================================================
# SANITY CHECK THRESHOLDS
# =============================================================================

# These catch silent data bugs BEFORE training starts.
# Training aborts if these are violated.

# Predicted action std across a batch — if below this, model is collapsing
MIN_PRED_STD    = 0.01

# Dataset action std — if below this, dataset is degenerate
MIN_DATA_ACTION_STD = 0.05

# Maximum allowed initial loss — if above this, something is wrong
MAX_INITIAL_LOSS = 10.0

# =============================================================================
# BENCHMARK
# =============================================================================

BENCHMARK_WARMUP_RUNS   = 20
BENCHMARK_TIMED_RUNS    = 200

# =============================================================================
# EXPORT
# =============================================================================

ONNX_OPSET          = 14
ONNX_FILENAME       = "trajformer.onnx"
TORCHSCRIPT_FILENAME = "trajformer.pt"

# =============================================================================
# LOGGING
# =============================================================================

LOG_INTERVAL    = 10        # Print stats every N batches
SAVE_BEST_ONLY  = True