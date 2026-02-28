# data/collect.py
# Collect expert trajectories from a pretrained SAC agent.
#
# I tried a hand-designed sinusoidal controller first — Hopper
# fell over after ~24 steps every episode. Not useful for training.
# Switched to sb3/sac-Hopper-v3 from HuggingFace, which runs for
# 400-1000 steps per episode at 1500+ mean return.
#
# Data collected on Hopper-v4 and kept on v4 throughout — training,
# evaluation, everything. Mixing versions changes the observation
# and reward scaling in ways that are annoying to track down.
#
# Output: one .npz per episode under data/trajectories/.
# Each file has states, actions, rewards, returns_to_go, dones.
# RTG is precomputed here so the dataset loader doesn't have to.
#
# Usage:
#   python data/collect.py
#   python data/collect.py --n_episodes 200

import numpy as np
import gymnasium as gym
import argparse
import json
from pathlib import Path
from tqdm import tqdm
from typing import Dict
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    STATE_DIM,
    ACTION_DIM,
    MAX_EPISODE_STEPS,
    N_COLLECT_EPISODES,
    DATA_DIR,
    BENCHMARK_DIR,
    MIN_DATA_ACTION_STD,
    SEED,
)

# Use v4 for HuggingFace pretrained model compatibility
COLLECT_ENV = "Hopper-v4"


# =============================================================================
# LOAD PRETRAINED AGENT
# =============================================================================

def load_expert_agent():
    """
    Download and load pretrained SAC agent for Hopper from HuggingFace.
    Cached locally after first download.
    """
    from huggingface_sb3 import load_from_hub
    from stable_baselines3 import SAC

    print("Loading pretrained SAC agent from HuggingFace...")
    print("(Downloads ~5MB on first run, cached after)")

    checkpoint = load_from_hub(
        repo_id  = "sb3/sac-Hopper-v3",
        filename = "sac-Hopper-v3.zip",
    )

    # Load with custom env to match v4
    env = gym.make(COLLECT_ENV, max_episode_steps=MAX_EPISODE_STEPS)
    model = SAC.load(checkpoint, env=env)

    print("✓ Expert agent loaded")
    return model


# =============================================================================
# COLLECTION
# =============================================================================

def compute_returns_to_go(rewards: np.ndarray) -> np.ndarray:
    """
    Compute reward-to-go for each timestep.
    RTG[t] = sum(rewards[t:])
    """
    T   = len(rewards)
    rtg = np.zeros(T, dtype=np.float32)
    running = 0.0
    for t in reversed(range(T)):
        running += rewards[t]
        rtg[t]   = running
    return rtg


def collect_episode(
    env:   gym.Env,
    model,
    deterministic: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Run one episode with the pretrained agent.
    deterministic=False adds stochasticity → more diverse trajectories.
    """
    obs, _ = env.reset()

    states  = []
    actions = []
    rewards = []
    dones   = []

    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        action     = np.clip(action, -1.0, 1.0).astype(np.float32)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        states.append(obs.astype(np.float32))
        actions.append(action)
        rewards.append(float(reward))
        dones.append(done)

        obs = next_obs
        if done:
            break

    return {
        "states":  np.array(states,  dtype=np.float32),
        "actions": np.array(actions, dtype=np.float32),
        "rewards": np.array(rewards, dtype=np.float32),
        "dones":   np.array(dones,   dtype=bool),
    }


def collect_dataset(
    n_episodes:   int  = N_COLLECT_EPISODES,
    deterministic: bool = False,
    seed:         int  = SEED,
) -> None:
    """
    Collect n_episodes expert trajectories and save to DATA_DIR.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Skip if already collected
    existing = list(DATA_DIR.glob("episode_*.npz"))
    if len(existing) >= n_episodes:
        print(f"Found {len(existing)} existing episodes — skipping collection.")
        _print_dataset_stats()
        return

    # Load agent
    model = load_expert_agent()

    env = gym.make(COLLECT_ENV, max_episode_steps=MAX_EPISODE_STEPS)
    env.reset(seed=seed)

    print(f"\nCollecting {n_episodes} episodes...")

    episode_rewards = []
    episode_lengths = []
    all_actions     = []

    for ep in tqdm(range(n_episodes), desc="Collecting"):
        traj = collect_episode(env, model, deterministic=deterministic)
        rtg  = compute_returns_to_go(traj["rewards"])

        out_path = DATA_DIR / f"episode_{ep:06d}.npz"
        np.savez(
            str(out_path),
            states        = traj["states"],
            actions       = traj["actions"],
            rewards       = traj["rewards"],
            returns_to_go = rtg,
            dones         = traj["dones"],
        )

        episode_rewards.append(traj["rewards"].sum())
        episode_lengths.append(len(traj["rewards"]))
        all_actions.append(traj["actions"])

    env.close()

    # ==========================================================================
    # SANITY CHECKS
    # ==========================================================================
    all_actions_np = np.concatenate(all_actions, axis=0)
    action_std     = all_actions_np.std(axis=0)
    action_mean    = all_actions_np.mean(axis=0)

    print(f"\n{'='*50}")
    print("DATASET SANITY CHECKS")
    print(f"{'='*50}")
    print(f"Episodes collected  : {n_episodes}")
    print(f"Total transitions   : {sum(episode_lengths):,}")
    print(f"Mean episode length : {np.mean(episode_lengths):.1f} steps")
    print(f"Mean episode reward : {np.mean(episode_rewards):.2f}")
    print(f"Reward std          : {np.std(episode_rewards):.2f}")
    print(f"Min episode reward  : {np.min(episode_rewards):.2f}")
    print(f"Max episode reward  : {np.max(episode_rewards):.2f}")
    print(f"\nAction statistics (per dimension):")
    for i in range(ACTION_DIM):
        status = "✓" if action_std[i] > MIN_DATA_ACTION_STD else "✗ DEGENERATE"
        print(f"  dim {i}: mean={action_mean[i]:+.3f}  "
              f"std={action_std[i]:.3f}  {status}")

    if (action_std < MIN_DATA_ACTION_STD).any():
        raise RuntimeError(
            f"Dataset failed sanity check: action std too low.\n"
            f"Action std: {action_std}\n"
            f"Delete {DATA_DIR} and re-collect."
        )

    print(f"\n✓ Dataset sanity check PASSED")
    print(f"{'='*50}")

    # Save metadata
    meta = {
        "n_episodes":        n_episodes,
        "total_transitions": int(sum(episode_lengths)),
        "mean_reward":       float(np.mean(episode_rewards)),
        "std_reward":        float(np.std(episode_rewards)),
        "mean_length":       float(np.mean(episode_lengths)),
        "min_reward":        float(np.min(episode_rewards)),
        "max_reward":        float(np.max(episode_rewards)),
        "action_std":        action_std.tolist(),
        "action_mean":       action_mean.tolist(),
        "agent":             "SAC pretrained (sb3/sac-Hopper-v3)",
        "env":               COLLECT_ENV,
        "deterministic":     deterministic,
    }
    meta_path = BENCHMARK_DIR / "dataset_metadata.json"
    with open(str(meta_path), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved: {meta_path}")


def _print_dataset_stats():
    files = sorted(DATA_DIR.glob("episode_*.npz"))
    if not files:
        print("No episodes found.")
        return
    rewards = []
    lengths = []
    for f in files[:100]:
        d = np.load(str(f))
        rewards.append(d["rewards"].sum())
        lengths.append(len(d["rewards"]))
    print(f"Dataset: {len(files)} episodes")
    print(f"Mean reward : {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Mean length : {np.mean(lengths):.1f} steps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_episodes",    type=int,  default=N_COLLECT_EPISODES)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seed",          type=int,  default=SEED)
    args = parser.parse_args()

    # Delete old sinusoidal data first
    import shutil
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
        print(f"Cleared old dataset: {DATA_DIR}")

    collect_dataset(
        n_episodes    = args.n_episodes,
        deterministic = args.deterministic,
        seed          = args.seed,
    )