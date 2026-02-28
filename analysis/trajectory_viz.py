# analysis/trajectory_viz.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import gymnasium as gym
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from collections import deque
from typing import List, Dict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    DEVICE,
    DATA_DIR,
    BENCHMARK_DIR,
    PLOTS_DIR,
    CONTEXT_LENGTH,
    STATE_DIM,
    ACTION_DIM,
    EVAL_TARGET_RETURN,
    REWARD_SCALE,
    MAX_EPISODE_STEPS,
    CHECKPOINT_DIR,
)
from src.model import build_model, load_checkpoint

EVAL_ENV    = "Hopper-v4"
JOINT_NAMES = ["Hip torque", "Knee torque", "Ankle torque"]


def load_model_and_stats():
    stats_path = BENCHMARK_DIR / "normalization_stats.json"
    with open(str(stats_path)) as f:
        stats = json.load(f)
    state_mean = np.array(stats["state_mean"], dtype=np.float32)
    state_std  = np.array(stats["state_std"],  dtype=np.float32)

    model = build_model(DEVICE)
    load_checkpoint(
        CHECKPOINT_DIR / "best_model.pth",
        model, optimizer=None, device=DEVICE
    )
    model.eval()
    return model, state_mean, state_std


def _make_tensors(states_buf, actions_buf, rtg_buf, ts_buf, K):
    """Build padded context tensors with correct dtypes."""
    pad = K - len(states_buf)

    s_t = torch.tensor(
        np.array(
            [np.zeros(STATE_DIM, dtype=np.float32)] * pad + list(states_buf),
            dtype=np.float32
        ),
        device=DEVICE, dtype=torch.float32
    ).unsqueeze(0)

    a_t = torch.tensor(
        np.array(
            [np.zeros(ACTION_DIM, dtype=np.float32)] * pad + list(actions_buf),
            dtype=np.float32
        ),
        device=DEVICE, dtype=torch.float32
    ).unsqueeze(0)

    r_t = torch.tensor(
        np.array(
            [[rtg_buf[0]] * pad + list(rtg_buf)],
            dtype=np.float32
        ),
        device=DEVICE, dtype=torch.float32
    ).unsqueeze(-1)

    ts_t = torch.tensor(
        np.array(
            [[0] * pad + list(ts_buf)],
            dtype=np.int64
        ),
        device=DEVICE, dtype=torch.long
    )

    return s_t, a_t, r_t, ts_t


# =============================================================================
# PLOT 1: ACTION PREDICTION ACCURACY
# =============================================================================

def plot_action_prediction(
    model:      torch.nn.Module,
    state_mean: np.ndarray,
    state_std:  np.ndarray,
    save_path:  Path,
    n_steps:    int = 100,
) -> None:
    files = sorted(DATA_DIR.glob("episode_*.npz"))
    ep    = np.load(str(files[0]))

    expert_states  = ep["states"][:n_steps].astype(np.float32)
    expert_actions = ep["actions"][:n_steps].astype(np.float32)
    expert_rtg     = ep["returns_to_go"][:n_steps].astype(np.float32)

    K = CONTEXT_LENGTH
    predicted_actions = []

    states_buf  = deque(maxlen=K)
    actions_buf = deque(maxlen=K)
    rtg_buf     = deque(maxlen=K)
    ts_buf      = deque(maxlen=K)

    for t in range(n_steps):
        obs_norm = (
            expert_states[t] - state_mean
        ) / (state_std + 1e-8)

        states_buf.append(obs_norm.astype(np.float32))
        actions_buf.append(
            expert_actions[t - 1].astype(np.float32) if t > 0
            else np.zeros(ACTION_DIM, dtype=np.float32)
        )
        rtg_buf.append(float(expert_rtg[t] / REWARD_SCALE))
        ts_buf.append(t)

        s_t, a_t, r_t, ts_t = _make_tensors(
            states_buf, actions_buf, rtg_buf, ts_buf, K
        )

        with torch.no_grad():
            pred = model.predict_action(r_t, s_t, a_t, ts_t)
        predicted_actions.append(pred.squeeze(0).cpu().numpy())

    predicted = np.array(predicted_actions)
    actual    = expert_actions
    timesteps = np.arange(n_steps)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(
        "TrajFormer: Predicted vs Expert Actions",
        fontsize=13, fontweight="bold"
    )

    colors = ["#2563eb", "#dc2626", "#16a34a"]

    for dim in range(ACTION_DIM):
        ax = axes[dim]
        ax.plot(
            timesteps, actual[:, dim],
            label="Expert (SAC)", color=colors[dim],
            linewidth=1.5, alpha=0.9
        )
        ax.plot(
            timesteps, predicted[:, dim],
            label="TrajFormer", color=colors[dim],
            linewidth=1.5, linestyle="--", alpha=0.7
        )
        ax.set_ylabel(JOINT_NAMES[dim], fontsize=9)
        ax.set_ylim(-1.1, 1.1)
        ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
        ax.grid(True, alpha=0.3)
        if dim == 0:
            ax.legend(loc="upper right", fontsize=9)

        mse = np.mean((predicted[:, dim] - actual[:, dim]) ** 2)
        ax.set_title(f"MSE = {mse:.4f}", fontsize=8, loc="left")

    axes[-1].set_xlabel("Timestep")
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.name}")


# =============================================================================
# PLOT 2: RTG SENSITIVITY
# =============================================================================

def plot_rtg_sensitivity(
    model:      torch.nn.Module,
    state_mean: np.ndarray,
    state_std:  np.ndarray,
    save_path:  Path,
) -> None:
    env = gym.make(EVAL_ENV, max_episode_steps=MAX_EPISODE_STEPS)

    target_returns = [500, 1000, 1500, 2000, 2500, 3000]
    colors         = plt.cm.RdYlGn(
        np.linspace(0.2, 0.9, len(target_returns))
    )

    results = []

    for target_rtg in target_returns:
        K = CONTEXT_LENGTH
        states_buf  = deque(maxlen=K)
        actions_buf = deque(maxlen=K)
        rtg_buf     = deque(maxlen=K)
        ts_buf      = deque(maxlen=K)

        obs, _ = env.reset(seed=42)
        obs_norm = (obs.astype(np.float32) - state_mean) / (state_std + 1e-8)

        states_buf.append(obs_norm)
        actions_buf.append(np.zeros(ACTION_DIM, dtype=np.float32))
        rtg_buf.append(float(target_rtg / REWARD_SCALE))
        ts_buf.append(0)

        episode_rewards = []
        action_history  = []
        step = 0

        while step < 200:
            s_t, a_t, r_t, ts_t = _make_tensors(
                states_buf, actions_buf, rtg_buf, ts_buf, K
            )

            with torch.no_grad():
                action = model.predict_action(r_t, s_t, a_t, ts_t)
            action = action.squeeze(0).cpu().numpy()
            action = np.clip(action, -1.0, 1.0).astype(np.float32)

            next_obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc

            episode_rewards.append(reward)
            action_history.append(action)

            next_obs_norm = (
                next_obs.astype(np.float32) - state_mean
            ) / (state_std + 1e-8)
            new_rtg = max(rtg_buf[-1] - reward / REWARD_SCALE, 0.0)

            states_buf.append(next_obs_norm)
            actions_buf.append(action)
            rtg_buf.append(new_rtg)
            ts_buf.append(min(step + 1, MAX_EPISODE_STEPS - 1))

            step += 1
            if done:
                break

        results.append({
            "target":   target_rtg,
            "achieved": sum(episode_rewards),
            "steps":    step,
            "actions":  np.array(action_history),
            "rewards":  np.cumsum(episode_rewards),
        })

    env.close()

    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(2, 2, figure=fig)
    fig.suptitle(
        "TrajFormer: Return-to-Go Conditioning Sensitivity",
        fontsize=13, fontweight="bold"
    )

    ax1 = fig.add_subplot(gs[0, 0])
    for i, r in enumerate(results):
        ax1.plot(
            r["rewards"], label=f"RTG={r['target']}",
            color=colors[i], linewidth=1.5
        )
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Cumulative reward")
    ax1.set_title("Cumulative reward by target RTG")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    targets  = [r["target"]   for r in results]
    achieved = [r["achieved"] for r in results]
    ax2.plot(targets, targets, "--", color="gray",
             linewidth=1, label="Perfect conditioning", alpha=0.6)
    ax2.scatter(targets, achieved,
                color=[colors[i] for i in range(len(results))], s=80, zorder=5)
    ax2.plot(targets, achieved, color="steelblue", linewidth=1.5)
    ax2.set_xlabel("Target RTG")
    ax2.set_ylabel("Achieved return")
    ax2.set_title("Target vs achieved return")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 0])
    steps = [r["steps"] for r in results]
    ax3.bar(range(len(results)), steps,
            color=colors, edgecolor="white", linewidth=0.5)
    ax3.set_xticks(range(len(results)))
    ax3.set_xticklabels([str(r["target"]) for r in results])
    ax3.set_xlabel("Target RTG")
    ax3.set_ylabel("Episode length (steps)")
    ax3.set_title("Episode length by target RTG")
    ax3.grid(True, alpha=0.3, axis="y")

    ax4 = fig.add_subplot(gs[1, 1])
    action_mags = [np.abs(r["actions"]).mean() for r in results]
    ax4.plot(targets, action_mags,
             color="steelblue", linewidth=2, marker="o", markersize=6)
    ax4.set_xlabel("Target RTG")
    ax4.set_ylabel("Mean |action|")
    ax4.set_title("Action magnitude vs target RTG")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.name}")

    print(f"\nRTG sensitivity summary:")
    print(f"  {'Target':>8}  {'Achieved':>10}  {'Steps':>7}")
    for r in results:
        print(f"  {r['target']:>8}  {r['achieved']:>10.1f}  {r['steps']:>7}")


def run_trajectory_viz() -> None:
    model, state_mean, state_std = load_model_and_stats()
    print("\nGenerating trajectory visualizations...")

    plot_action_prediction(
        model, state_mean, state_std,
        PLOTS_DIR / "action_prediction.png",
        n_steps=100,
    )

    plot_rtg_sensitivity(
        model, state_mean, state_std,
        PLOTS_DIR / "rtg_sensitivity.png",
    )

    print(f"\nAll trajectory plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    run_trajectory_viz()