# data/collect.py
# Mixed-quality trajectory collection for TrajFormer.
#
# Expert tier uses oversampling — collect 400 stochastic episodes,
# keep the top 200 by total return. This guarantees the expert tier
# contains genuinely high-return trajectories rather than relying
# on deterministic inference which generalizes poorly from v3 to v4.

import numpy as np
import gymnasium as gym
import json
import shutil
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    STATE_DIM,
    ACTION_DIM,
    MAX_EPISODE_STEPS,
    DATA_DIR,
    BENCHMARK_DIR,
    MIN_DATA_ACTION_STD,
    SEED,
)

COLLECT_ENV = "Hopper-v4"

# Oversample expert tier — collect 400, keep top 200
EXPERT_COLLECT = 400
EXPERT_KEEP    = 200
MEDIUM_EPISODES = 200
RANDOM_EPISODES = 100


def load_expert_agent():
    from huggingface_sb3 import load_from_hub
    from stable_baselines3 import SAC

    print("Loading pretrained SAC agent...")
    checkpoint = load_from_hub(
        repo_id  = "sb3/sac-Hopper-v3",
        filename = "sac-Hopper-v3.zip",
    )
    env   = gym.make(COLLECT_ENV, max_episode_steps=MAX_EPISODE_STEPS)
    model = SAC.load(checkpoint, env=env)
    print("✓ Agent loaded")
    return model


def compute_returns_to_go(rewards: np.ndarray) -> np.ndarray:
    T   = len(rewards)
    rtg = np.zeros(T, dtype=np.float32)
    running = 0.0
    for t in reversed(range(T)):
        running += rewards[t]
        rtg[t]   = running
    return rtg


def collect_episode_sac(
    env:          gym.Env,
    model,
    deterministic: bool = False,
) -> Dict[str, np.ndarray]:
    obs, _ = env.reset()
    states, actions, rewards, dones = [], [], [], []

    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        action     = np.clip(action, -1.0, 1.0).astype(np.float32)
        next_obs, reward, term, trunc, _ = env.step(action)
        done = term or trunc

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


def collect_episode_random(env: gym.Env) -> Dict[str, np.ndarray]:
    obs, _ = env.reset()
    states, actions, rewards, dones = [], [], [], []

    while True:
        action = env.action_space.sample().astype(np.float32)
        next_obs, reward, term, trunc, _ = env.step(action)
        done = term or trunc

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


def save_episode(
    traj:   Dict[str, np.ndarray],
    path:   Path,
    tier:   str,
) -> None:
    rtg = compute_returns_to_go(traj["rewards"])
    np.savez(
        str(path),
        states        = traj["states"],
        actions       = traj["actions"],
        rewards       = traj["rewards"],
        returns_to_go = rtg,
        dones         = traj["dones"],
        tier          = np.array([tier]),
    )


def collect_dataset(seed: int = SEED) -> None:
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
        print(f"Cleared old dataset: {DATA_DIR}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    env   = gym.make(COLLECT_ENV, max_episode_steps=MAX_EPISODE_STEPS)
    model = load_expert_agent()
    env.reset(seed=seed)

    ep_idx      = 0
    all_rewards = []
    all_lengths = []
    all_actions = []
    tier_stats  = {}

    # ==========================================================================
    # EXPERT TIER — oversample then keep top EXPERT_KEEP by return
    # ==========================================================================
    print(f"\nCollecting {EXPERT_COLLECT} candidate expert episodes "
          f"(keeping top {EXPERT_KEEP})...")

    candidates: List[Dict] = []
    for _ in tqdm(range(EXPERT_COLLECT), desc="expert candidates"):
        traj = collect_episode_sac(env, model, deterministic=False)
        candidates.append(traj)

    # Sort by total return, keep top EXPERT_KEEP
    candidates.sort(key=lambda t: t["rewards"].sum(), reverse=True)
    expert_episodes = candidates[:EXPERT_KEEP]

    expert_rewards = [t["rewards"].sum() for t in expert_episodes]
    print(f"  Kept top {EXPERT_KEEP}: "
          f"mean={np.mean(expert_rewards):.1f}  "
          f"min={np.min(expert_rewards):.1f}  "
          f"max={np.max(expert_rewards):.1f}")
    tier_stats["expert"] = {
        "mean": float(np.mean(expert_rewards)),
        "min":  float(np.min(expert_rewards)),
        "max":  float(np.max(expert_rewards)),
    }

    for traj in expert_episodes:
        save_episode(traj, DATA_DIR / f"episode_{ep_idx:06d}.npz", "expert")
        all_rewards.append(traj["rewards"].sum())
        all_lengths.append(len(traj["rewards"]))
        all_actions.append(traj["actions"])
        ep_idx += 1

    # ==========================================================================
    # MEDIUM TIER — stochastic SAC, no filtering
    # ==========================================================================
    print(f"\nCollecting {MEDIUM_EPISODES} medium episodes...")
    medium_rewards = []

    for _ in tqdm(range(MEDIUM_EPISODES), desc="medium"):
        traj = collect_episode_sac(env, model, deterministic=False)
        save_episode(traj, DATA_DIR / f"episode_{ep_idx:06d}.npz", "medium")
        r = traj["rewards"].sum()
        medium_rewards.append(r)
        all_rewards.append(r)
        all_lengths.append(len(traj["rewards"]))
        all_actions.append(traj["actions"])
        ep_idx += 1

    print(f"  medium: mean={np.mean(medium_rewards):.1f}  "
          f"min={np.min(medium_rewards):.1f}  "
          f"max={np.max(medium_rewards):.1f}")
    tier_stats["medium"] = {
        "mean": float(np.mean(medium_rewards)),
        "min":  float(np.min(medium_rewards)),
        "max":  float(np.max(medium_rewards)),
    }

    # ==========================================================================
    # RANDOM TIER
    # ==========================================================================
    print(f"\nCollecting {RANDOM_EPISODES} random episodes...")
    random_rewards = []

    for _ in tqdm(range(RANDOM_EPISODES), desc="random"):
        traj = collect_episode_random(env)
        save_episode(traj, DATA_DIR / f"episode_{ep_idx:06d}.npz", "random")
        r = traj["rewards"].sum()
        random_rewards.append(r)
        all_rewards.append(r)
        all_lengths.append(len(traj["rewards"]))
        all_actions.append(traj["actions"])
        ep_idx += 1

    print(f"  random: mean={np.mean(random_rewards):.1f}  "
          f"min={np.min(random_rewards):.1f}  "
          f"max={np.max(random_rewards):.1f}")
    tier_stats["random"] = {
        "mean": float(np.mean(random_rewards)),
        "min":  float(np.min(random_rewards)),
        "max":  float(np.max(random_rewards)),
    }

    env.close()

    # ==========================================================================
    # SANITY CHECKS
    # ==========================================================================
    all_actions_np = np.concatenate(all_actions, axis=0)
    action_std     = all_actions_np.std(axis=0)
    action_mean    = all_actions_np.mean(axis=0)
    all_rewards_np = np.array(all_rewards)

    print(f"\n{'='*55}")
    print("DATASET SANITY CHECKS")
    print(f"{'='*55}")
    print(f"Total episodes      : {ep_idx}")
    print(f"Total transitions   : {sum(all_lengths):,}")
    print(f"Mean episode length : {np.mean(all_lengths):.1f}")
    print(f"\nReturn distribution:")

    bins = [0, 100, 500, 1000, 1500, 2000, 2500, 3000, 9999]
    for i in range(len(bins) - 1):
        count = ((all_rewards_np >= bins[i]) &
                 (all_rewards_np < bins[i+1])).sum()
        bar   = "█" * (count // 5)
        print(f"  {bins[i]:>5}-{bins[i+1]:<5}: {count:>3} episodes  {bar}")

    print(f"\nAction statistics:")
    for i in range(ACTION_DIM):
        status = "✓" if action_std[i] > MIN_DATA_ACTION_STD else "✗ DEGENERATE"
        print(f"  dim {i}: mean={action_mean[i]:+.3f}  "
              f"std={action_std[i]:.3f}  {status}")

    if (action_std < MIN_DATA_ACTION_STD).any():
        raise RuntimeError(
            f"Dataset failed sanity check: action std too low.\n"
            f"Action std: {action_std}"
        )

    print(f"\n✓ Dataset sanity check PASSED")
    print(f"{'='*55}")

    meta = {
        "total_episodes":    ep_idx,
        "total_transitions": int(sum(all_lengths)),
        "mean_reward":       float(np.mean(all_rewards)),
        "min_reward":        float(np.min(all_rewards)),
        "max_reward":        float(np.max(all_rewards)),
        "tier_stats":        tier_stats,
        "action_std":        action_std.tolist(),
        "env":               COLLECT_ENV,
        "expert_oversample": EXPERT_COLLECT,
        "expert_kept":       EXPERT_KEEP,
    }
    meta_path = BENCHMARK_DIR / "dataset_metadata.json"
    with open(str(meta_path), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved: {meta_path}")


if __name__ == "__main__":
    collect_dataset()