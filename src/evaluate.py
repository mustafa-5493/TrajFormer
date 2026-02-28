# src/evaluate.py
# Environment rollout evaluation.
#
# Runs the trained model in Hopper-v4 and records episode returns.
# At each step we maintain a sliding context window of the last K
# timesteps, feed it to the model, execute the predicted action,
# then shift the window forward.
#
# RTG is decremented by the reward received at each step — so the
# model always knows how much return is still expected from here.
#
# I also run a random baseline for the same number of episodes
# so the comparison is apples-to-apples on the same environment seed.
#
# Results saved to outputs/benchmarks/eval_results.json

import torch
import numpy as np
import json
import gymnasium as gym
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional
from collections import deque

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    DEVICE,
    STATE_DIM,
    ACTION_DIM,
    CONTEXT_LENGTH,
    EVAL_TARGET_RETURN,
    REWARD_SCALE,
    MAX_EPISODE_STEPS,
    CHECKPOINT_DIR,
    BENCHMARK_DIR,
)
from src.model import build_model, load_checkpoint


# Use same env as data collection
EVAL_ENV = "Hopper-v4"


def evaluate_episode(
    model:       torch.nn.Module,
    env:         gym.Env,
    state_mean:  np.ndarray,
    state_std:   np.ndarray,
    target_return: float = EVAL_TARGET_RETURN,
    device:      torch.device = DEVICE,
) -> Dict:
    """
    Run one evaluation episode.

    Maintains a sliding context window of the last K timesteps.
    At each step:
        1. Feed context to model
        2. Get predicted action for current timestep
        3. Execute action in environment
        4. Update context

    Returns episode stats.
    """
    model.eval()

    # Initialize context buffers (sliding window of size K)
    K = CONTEXT_LENGTH

    states_buf  = deque(maxlen=K)
    actions_buf = deque(maxlen=K)
    rtg_buf     = deque(maxlen=K)
    ts_buf      = deque(maxlen=K)

    obs, _ = env.reset()
    obs    = obs.astype(np.float32)

    # Normalize first state
    obs_norm = (obs - state_mean) / (state_std + 1e-8)

    # Initialize with first state
    states_buf.append(obs_norm)
    actions_buf.append(np.zeros(ACTION_DIM, dtype=np.float32))
    rtg_buf.append(target_return / REWARD_SCALE)
    ts_buf.append(0)

    total_reward = 0.0
    step         = 0
    rewards      = []

    while True:
        # Pad context to length K if needed
        pad_len = K - len(states_buf)

        states_ctx  = np.array(
            [np.zeros(STATE_DIM)] * pad_len + list(states_buf),
            dtype=np.float32
        )                                                   # (K, 11)
        actions_ctx = np.array(
            [np.zeros(ACTION_DIM)] * pad_len + list(actions_buf),
            dtype=np.float32
        )                                                   # (K, 3)
        rtg_ctx     = np.array(
            [rtg_buf[0]] * pad_len + list(rtg_buf),
            dtype=np.float32
        )                                                   # (K,)
        ts_ctx      = np.array(
            [0] * pad_len + list(ts_buf),
            dtype=np.int64
        )                                                   # (K,)

        # Convert to tensors — add batch dim
        states_t  = torch.tensor(states_ctx,        device=device).unsqueeze(0)
        actions_t = torch.tensor(actions_ctx,       device=device).unsqueeze(0)
        rtg_t     = torch.tensor(rtg_ctx[:, None],  device=device).unsqueeze(0)
        ts_t      = torch.tensor(ts_ctx,            device=device).unsqueeze(0)

        # Get action prediction
        action = model.predict_action(rtg_t, states_t, actions_t, ts_t)
        action = action.squeeze(0).cpu().numpy()
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # Step environment
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        total_reward += reward
        rewards.append(reward)
        step += 1

        # Update context
        next_obs_norm = (
            next_obs.astype(np.float32) - state_mean
        ) / (state_std + 1e-8)

        # RTG decreases by reward received
        new_rtg = max(rtg_buf[-1] - reward / REWARD_SCALE, 0.0)

        states_buf.append(next_obs_norm)
        actions_buf.append(action)
        rtg_buf.append(new_rtg)
        ts_buf.append(min(step, MAX_EPISODE_STEPS - 1))

        if done:
            break

        obs = next_obs

    return {
        "total_reward": total_reward,
        "n_steps":      step,
        "rewards":      rewards,
    }


def evaluate_random(
    env:         gym.Env,
    n_episodes:  int = 20,
) -> Dict:
    """Baseline: random agent."""
    rewards = []
    lengths = []

    for _ in range(n_episodes):
        env.reset()
        total = 0.0
        steps = 0
        while True:
            action = env.action_space.sample()
            _, r, term, trunc, _ = env.step(action)
            total += r
            steps += 1
            if term or trunc:
                break
        rewards.append(total)
        lengths.append(steps)

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward":  float(np.std(rewards)),
        "mean_steps":  float(np.mean(lengths)),
    }


def run_evaluation(
    n_episodes:    int   = 20,
    target_return: float = EVAL_TARGET_RETURN,
    checkpoint:    Path  = None,
) -> Dict:
    """
    Full evaluation pipeline.
    Compares TrajFormer against random baseline.
    """
    ckpt_path = checkpoint or (CHECKPOINT_DIR / "best_model.pth")

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {ckpt_path}.\n"
            f"Run: python src/train.py"
        )

    # Load normalization stats
    stats_path = BENCHMARK_DIR / "normalization_stats.json"
    with open(str(stats_path)) as f:
        stats = json.load(f)

    state_mean = np.array(stats["state_mean"], dtype=np.float32)
    state_std  = np.array(stats["state_std"],  dtype=np.float32)

    # Load model
    model = build_model(DEVICE)
    load_checkpoint(ckpt_path, model, optimizer=None, device=DEVICE)
    model.eval()

    print(f"\nEvaluating TrajFormer...")
    print(f"  Checkpoint   : {ckpt_path}")
    print(f"  Target return: {target_return}")
    print(f"  Episodes     : {n_episodes}")

    env = gym.make(EVAL_ENV, max_episode_steps=MAX_EPISODE_STEPS)

    # Random baseline
    print(f"\nRunning random baseline ({n_episodes} episodes)...")
    random_stats = evaluate_random(env, n_episodes)
    print(f"  Random agent: {random_stats['mean_reward']:.1f} "
          f"± {random_stats['std_reward']:.1f} reward  "
          f"({random_stats['mean_steps']:.0f} steps)")

    # TrajFormer evaluation
    print(f"\nRunning TrajFormer ({n_episodes} episodes)...")
    model_rewards = []
    model_lengths = []

    for ep in tqdm(range(n_episodes), desc="Evaluating"):
        result = evaluate_episode(
            model, env, state_mean, state_std,
            target_return=target_return,
        )
        model_rewards.append(result["total_reward"])
        model_lengths.append(result["n_steps"])

    env.close()

    model_stats = {
        "mean_reward": float(np.mean(model_rewards)),
        "std_reward":  float(np.std(model_rewards)),
        "min_reward":  float(np.min(model_rewards)),
        "max_reward":  float(np.max(model_rewards)),
        "mean_steps":  float(np.mean(model_lengths)),
        "all_rewards": model_rewards,
    }

    # Results
    improvement = (
        model_stats["mean_reward"] / max(random_stats["mean_reward"], 1)
    )

    print(f"\n{'='*55}")
    print("EVALUATION RESULTS")
    print(f"{'='*55}")
    print(f"Random agent  : {random_stats['mean_reward']:>8.1f} "
          f"± {random_stats['std_reward']:.1f}  "
          f"({random_stats['mean_steps']:.0f} steps)")
    print(f"TrajFormer    : {model_stats['mean_reward']:>8.1f} "
          f"± {model_stats['std_reward']:.1f}  "
          f"({model_stats['mean_steps']:.0f} steps)")
    print(f"Improvement   : {improvement:.1f}x over random")
    print(f"Expert SAC    : ~3000+ (reference)")
    print(f"{'='*55}")

    # Save results
    results = {
        "model":       "TrajFormer",
        "checkpoint":  str(ckpt_path),
        "target_return": target_return,
        "n_episodes":  n_episodes,
        "model_stats": model_stats,
        "random_stats": random_stats,
        "improvement_over_random": improvement,
    }
    out_path = BENCHMARK_DIR / "eval_results.json"
    with open(str(out_path), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {out_path}")

    return results


if __name__ == "__main__":
    run_evaluation(n_episodes=20)