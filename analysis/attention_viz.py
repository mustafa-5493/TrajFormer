# analysis/attention_viz.py
# Visualize attention weights from TrajFormer.
#
# Decision Transformer attention visualization reveals WHAT the model
# attends to when predicting actions — which past states, rewards,
# and actions influence each prediction.
#
# Token sequence structure (K=20 context):
#   [R0, s0, a0, R1, s1, a1, ..., R19, s19, a19]
#    0   1   2   3   4   5  ...   57   58   59
#
# Interesting patterns to look for:
#   - Do state tokens attend strongly to recent reward-to-go tokens?
#     (model checking "how am I doing?")
#   - Do action tokens attend to nearby state tokens?
#     (model checking "what's my current state?")
#   - Does attention become more focused in later layers?
#     (hierarchical feature extraction)
#
# Output:
#   outputs/plots/attention_layer_{N}.png  — per-layer heatmap
#   outputs/plots/attention_summary.png   — all layers side by side
#   outputs/plots/attention_by_token_type.png — R vs s vs a patterns

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import gymnasium as gym
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from typing import List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    DEVICE,
    CONTEXT_LENGTH,
    STATE_DIM,
    ACTION_DIM,
    EVAL_TARGET_RETURN,
    REWARD_SCALE,
    MAX_EPISODE_STEPS,
    CHECKPOINT_DIR,
    BENCHMARK_DIR,
    PLOTS_DIR,
    N_LAYERS,
    N_HEADS,
)
from src.model import build_model, load_checkpoint
from src.evaluate import evaluate_episode, EVAL_ENV


# Token type labels for axis ticks
def get_token_labels(K: int) -> List[str]:
    """
    Generate token labels for a K-timestep context.
    Returns list of 3K labels: ['R0','s0','a0','R1','s1','a1',...]
    """
    labels = []
    for t in range(K):
        labels.extend([f"R{t}", f"s{t}", f"a{t}"])
    return labels


def get_token_colors(K: int) -> List[str]:
    """
    Color code by token type:
        R (return-to-go) : red
        s (state)        : blue
        a (action)       : green
    """
    colors = []
    for _ in range(K):
        colors.extend(["#dc2626", "#2563eb", "#16a34a"])
    return colors


def collect_attention_weights(
    model:       torch.nn.Module,
    env:         gym.Env,
    state_mean:  np.ndarray,
    state_std:   np.ndarray,
) -> List[torch.Tensor]:
    """
    Run one episode and collect attention weights at the
    final timestep (full context window filled).

    Returns list of attention weight tensors, one per layer.
    Each tensor: (1, N_HEADS, N_TOKENS, N_TOKENS)
    """
    from collections import deque

    K = CONTEXT_LENGTH
    states_buf  = deque(maxlen=K)
    actions_buf = deque(maxlen=K)
    rtg_buf     = deque(maxlen=K)
    ts_buf      = deque(maxlen=K)

    obs, _ = env.reset()
    obs    = obs.astype(np.float32)
    obs_norm = (obs - state_mean) / (state_std + 1e-8)

    states_buf.append(obs_norm)
    actions_buf.append(np.zeros(ACTION_DIM, dtype=np.float32))
    rtg_buf.append(EVAL_TARGET_RETURN / REWARD_SCALE)
    ts_buf.append(0)

    target_weights = None
    step = 0

    for step in range(K + 10):  # run until context is full
        pad_len = K - len(states_buf)

        states_ctx  = np.array(
            [np.zeros(STATE_DIM)] * pad_len + list(states_buf),
            dtype=np.float32
        )
        actions_ctx = np.array(
            [np.zeros(ACTION_DIM)] * pad_len + list(actions_buf),
            dtype=np.float32
        )
        rtg_ctx = np.array(
            [rtg_buf[0]] * pad_len + list(rtg_buf),
            dtype=np.float32
        )
        ts_ctx = np.array(
            [0] * pad_len + list(ts_buf),
            dtype=np.int64
        )

        states_t  = torch.tensor(states_ctx).unsqueeze(0).to(DEVICE)
        actions_t = torch.tensor(actions_ctx).unsqueeze(0).to(DEVICE)
        rtg_t     = torch.tensor(rtg_ctx[:, None]).unsqueeze(0).to(DEVICE)
        ts_t      = torch.tensor(ts_ctx).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            _ = model(rtg_t, states_t, actions_t, ts_t)

        # Capture weights when context is full
        if len(states_buf) == K and target_weights is None:
            target_weights = model.get_attention_weights()
            # target_weights: list of (1, H, T, T) per layer
            break

        # Step environment
        action = model.predict_action(rtg_t, states_t, actions_t, ts_t)
        action = action.squeeze(0).cpu().numpy()
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        next_obs, reward, term, trunc, _ = env.step(action)
        if term or trunc:
            break

        next_obs_norm = (
            next_obs.astype(np.float32) - state_mean
        ) / (state_std + 1e-8)
        new_rtg = max(rtg_buf[-1] - reward / REWARD_SCALE, 0.0)

        states_buf.append(next_obs_norm)
        actions_buf.append(action)
        rtg_buf.append(new_rtg)
        ts_buf.append(min(step + 1, MAX_EPISODE_STEPS - 1))

    return target_weights


def plot_single_layer(
    weights:  torch.Tensor,   # (1, H, T, T)
    layer_idx: int,
    K:        int,
    save_path: Path,
) -> None:
    """
    Plot attention heatmap for one layer.
    Shows mean attention across all heads.
    """
    # Average across heads and batch: (T, T)
    attn = weights.squeeze(0).mean(dim=0).cpu().numpy()
    T    = attn.shape[0]

    labels = get_token_labels(K)
    # Only show every 3rd label to avoid crowding
    tick_labels = [l if i % 6 == 0 else "" for i, l in enumerate(labels)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        f"Layer {layer_idx + 1} Attention Weights — TrajFormer",
        fontsize=13, fontweight="bold"
    )

    # Left: full attention matrix (mean across heads)
    ax = axes[0]
    im = ax.imshow(attn, cmap="Blues", aspect="auto", vmin=0)
    ax.set_xticks(range(0, T, 3))
    ax.set_xticklabels(
        [labels[i] for i in range(0, T, 3)],
        rotation=90, fontsize=7
    )
    ax.set_yticks(range(0, T, 3))
    ax.set_yticklabels(
        [labels[i] for i in range(0, T, 3)],
        fontsize=7
    )
    ax.set_xlabel("Key tokens (attended to)")
    ax.set_ylabel("Query tokens (attending from)")
    ax.set_title("Mean attention across heads")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Add token type color bands
    colors = get_token_colors(K)
    for i, c in enumerate(colors):
        ax.axvline(i - 0.5, color=c, alpha=0.1, linewidth=0.5)

    # Right: per-head attention (grid of H heads)
    H     = weights.shape[1]
    n_col = 4
    n_row = (H + n_col - 1) // n_col

    ax = axes[1]
    ax.axis("off")

    # Create inset axes for each head
    head_weights = weights.squeeze(0).cpu().numpy()  # (H, T, T)

    # Show first 4 heads as subplots
    show_heads = min(H, 4)
    for h in range(show_heads):
        inset = ax.inset_axes([
            (h % 2) * 0.5,
            1.0 - ((h // 2) + 1) * 0.5,
            0.48, 0.48
        ])
        inset.imshow(
            head_weights[h], cmap="Reds",
            aspect="auto", vmin=0
        )
        inset.set_title(f"Head {h+1}", fontsize=8)
        inset.set_xticks([])
        inset.set_yticks([])

    axes[1].set_title("Per-head attention (first 4 heads)")

    # Legend
    patches = [
        mpatches.Patch(color="#dc2626", label="R (return-to-go)"),
        mpatches.Patch(color="#2563eb", label="s (state)"),
        mpatches.Patch(color="#16a34a", label="a (action)"),
    ]
    fig.legend(handles=patches, loc="lower center",
               ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.name}")


def plot_attention_summary(
    all_weights: List[torch.Tensor],
    K:           int,
    save_path:   Path,
) -> None:
    """
    All layers side by side — shows how attention evolves
    through the network depth.
    """
    n_layers = len(all_weights)
    fig, axes = plt.subplots(
        2, (n_layers + 1) // 2,
        figsize=(4 * ((n_layers + 1) // 2), 8)
    )
    axes = axes.flatten()

    fig.suptitle(
        "Attention Weights Across All Layers — TrajFormer",
        fontsize=13, fontweight="bold"
    )

    for layer_idx, weights in enumerate(all_weights):
        attn = weights.squeeze(0).mean(dim=0).cpu().numpy()
        ax   = axes[layer_idx]

        im = ax.imshow(attn, cmap="Blues", aspect="auto", vmin=0)
        ax.set_title(f"Layer {layer_idx + 1}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

        # Mark token type boundaries
        T = attn.shape[0]
        for t in range(0, T, 3):
            ax.axvline(t - 0.5, color="white", alpha=0.3, linewidth=0.5)
            ax.axhline(t - 0.5, color="white", alpha=0.3, linewidth=0.5)

    # Hide unused axes
    for i in range(n_layers, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.name}")


def plot_token_type_attention(
    all_weights: List[torch.Tensor],
    K:           int,
    save_path:   Path,
) -> None:
    """
    Aggregate attention by token type (R, s, a).

    For each layer: how much does each token TYPE attend
    to each other token TYPE on average?

    This reveals the learned information flow:
        - Do state tokens attend to reward tokens?
        - Do action tokens attend to state tokens?
    """
    n_layers = len(all_weights)
    T        = 3 * K
    type_labels = ["R (return)", "s (state)", "a (action)"]

    # Aggregate attention by token type
    # For each layer: (3, 3) matrix of mean attention between types
    type_attn = np.zeros((n_layers, 3, 3))

    for layer_idx, weights in enumerate(all_weights):
        attn = weights.squeeze(0).mean(dim=0).cpu().numpy()  # (T, T)

        for q_type in range(3):   # query type (attending from)
            for k_type in range(3):   # key type (attending to)
                # Extract sub-matrix for this type pair
                q_indices = list(range(q_type, T, 3))
                k_indices = list(range(k_type, T, 3))
                sub = attn[np.ix_(q_indices, k_indices)]
                type_attn[layer_idx, q_type, k_type] = sub.mean()

    # Plot
    fig, axes = plt.subplots(
        2, (n_layers + 1) // 2,
        figsize=(4 * ((n_layers + 1) // 2), 7)
    )
    axes = axes.flatten()

    fig.suptitle(
        "Token Type Attention Patterns per Layer\n"
        "(Row=Query type, Col=Key type)",
        fontsize=12, fontweight="bold"
    )

    for layer_idx in range(n_layers):
        ax  = axes[layer_idx]
        mat = type_attn[layer_idx]

        im = ax.imshow(mat, cmap="YlOrRd", aspect="auto",
                       vmin=0, vmax=mat.max())
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(["R", "s", "a"], fontsize=9)
        ax.set_yticklabels(["R", "s", "a"], fontsize=9)
        ax.set_title(f"Layer {layer_idx + 1}", fontsize=10)

        # Annotate values
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{mat[i,j]:.3f}",
                        ha="center", va="center",
                        fontsize=8, color="black")

    for i in range(n_layers, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.name}")


def run_attention_viz() -> None:
    """Full attention visualization pipeline."""

    # Load normalization stats
    stats_path = BENCHMARK_DIR / "normalization_stats.json"
    with open(str(stats_path)) as f:
        stats = json.load(f)
    state_mean = np.array(stats["state_mean"], dtype=np.float32)
    state_std  = np.array(stats["state_std"],  dtype=np.float32)

    # Load model
    model = build_model(DEVICE)
    ckpt  = CHECKPOINT_DIR / "best_model.pth"
    load_checkpoint(ckpt, model, optimizer=None, device=DEVICE)
    model.eval()

    print("\nCollecting attention weights from environment rollout...")
    env = gym.make(EVAL_ENV, max_episode_steps=MAX_EPISODE_STEPS)

    all_weights = collect_attention_weights(
        model, env, state_mean, state_std
    )
    env.close()

    if all_weights is None:
        print("Failed to collect attention weights — episode too short.")
        return

    print(f"Collected attention from {len(all_weights)} layers")
    print(f"Generating visualizations...\n")

    K = CONTEXT_LENGTH

    # Per-layer heatmaps
    for layer_idx, weights in enumerate(all_weights):
        save_path = PLOTS_DIR / f"attention_layer_{layer_idx+1:02d}.png"
        plot_single_layer(weights, layer_idx, K, save_path)

    # Summary: all layers side by side
    plot_attention_summary(
        all_weights, K,
        PLOTS_DIR / "attention_summary.png"
    )

    # Token type patterns
    plot_token_type_attention(
        all_weights, K,
        PLOTS_DIR / "attention_token_types.png"
    )

    print(f"\nAll attention plots saved to: {PLOTS_DIR}")
    print("Files:")
    for f in sorted(PLOTS_DIR.glob("attention_*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    run_attention_viz()