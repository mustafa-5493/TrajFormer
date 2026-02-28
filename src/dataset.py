# src/dataset.py
# Dataset and dataloader for TrajFormer.
#
# Each training sample is a K-step window from one episode.
# Windows never cross episode boundaries — a window that starts
# near the end of an episode just gets a shorter valid range,
# it doesn't bleed into the next one.
#
# RTG is divided by REWARD_SCALE before being fed to the model.
# Without this the raw values (~0-3500) dwarf the state and action
# embeddings and destabilize early training.
#
# State normalization stats are computed from the training split
# only, then applied to val. Stored to disk so inference can use
# the same stats without reloading the full dataset.
#
# The sanity check on the first batch exists because silent data
# bugs — wrong dtypes, collapsed actions, denormalized states —
# are hard to catch from loss curves alone. Better to abort early
# with a clear message than debug a 4-hour training run.

import torch
import numpy as np
import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Tuple, Dict, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    DATA_DIR,
    BENCHMARK_DIR,
    CONTEXT_LENGTH,
    STATE_DIM,
    ACTION_DIM,
    REWARD_SCALE,
    TRAIN_RATIO,
    BATCH_SIZE,
    SEED,
    MIN_DATA_ACTION_STD,
)


class TrajFormerDataset(Dataset):
    """
    Dataset of trajectory windows for Decision Transformer training.

    Each item is a random K-length window from one expert episode.
    Windows are sampled fresh each epoch (online sampling) —
    this creates effective data augmentation from fixed trajectories.

    Args:
        episodes:     List of episode dicts (states, actions, rewards, rtg)
        context_len:  Window size K
        state_mean:   Dataset mean for normalization (computed from train set)
        state_std:    Dataset std for normalization
    """

    def __init__(
        self,
        episodes:    list,
        context_len: int,
        state_mean:  np.ndarray,
        state_std:   np.ndarray,
    ):
        self.episodes    = episodes
        self.context_len = context_len
        self.state_mean  = state_mean
        self.state_std   = state_std

        # Pre-compute valid (episode_idx, start_idx) pairs
        # A window is valid if episode has at least context_len steps
        self.windows = []
        for ep_idx, ep in enumerate(episodes):
            T = len(ep["states"])
            if T < context_len:
                continue
            for start in range(T - context_len + 1):
                self.windows.append((ep_idx, start))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_idx, start = self.windows[idx]
        ep  = self.episodes[ep_idx]
        end = start + self.context_len

        # Extract window
        states  = ep["states"][start:end]           # (K, 11)
        actions = ep["actions"][start:end]           # (K, 3)
        rtg     = ep["returns_to_go"][start:end]     # (K,)

        # Normalize states
        states = (states - self.state_mean) / (self.state_std + 1e-8)

        # Normalize RTG
        rtg = rtg / REWARD_SCALE                     # (K,) → ~[0, 4]

        # Timestep indices (absolute position in episode)
        timesteps = np.arange(start, end, dtype=np.int64)  # (K,)

        return {
            "returns_to_go": torch.tensor(
                rtg[:, None], dtype=torch.float32
            ),                                       # (K, 1)
            "states":        torch.tensor(
                states, dtype=torch.float32
            ),                                       # (K, 11)
            "actions":       torch.tensor(
                actions, dtype=torch.float32
            ),                                       # (K, 3)
            "timesteps":     torch.tensor(
                timesteps, dtype=torch.long
            ),                                       # (K,)
            "target_actions": torch.tensor(
                actions, dtype=torch.float32
            ),                                       # (K, 3) — training target
        }


def load_episodes(data_dir: Path) -> list:
    """
    Load all episode .npz files from data_dir.
    Returns list of episode dicts.
    """
    files = sorted(data_dir.glob("episode_*.npz"))
    if not files:
        raise FileNotFoundError(
            f"No episode files found in {data_dir}.\n"
            f"Run: python data/collect.py"
        )

    episodes = []
    for f in files:
        d = np.load(str(f))
        episodes.append({
            "states":        d["states"].astype(np.float32),
            "actions":       d["actions"].astype(np.float32),
            "rewards":       d["rewards"].astype(np.float32),
            "returns_to_go": d["returns_to_go"].astype(np.float32),
        })

    print(f"Loaded {len(episodes)} episodes from {data_dir}")
    return episodes


def compute_state_stats(episodes: list) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and std of states across all episodes.
    Used for state normalization.
    Saved to disk for reuse at inference time.
    """
    all_states = np.concatenate(
        [ep["states"] for ep in episodes], axis=0
    )
    mean = all_states.mean(axis=0).astype(np.float32)
    std  = all_states.std(axis=0).astype(np.float32)
    return mean, std


def build_dataloaders(
    data_dir:    Path = DATA_DIR,
    context_len: int  = CONTEXT_LENGTH,
    batch_size:  int  = BATCH_SIZE,
    seed:        int  = SEED,
) -> Tuple[DataLoader, DataLoader, np.ndarray, np.ndarray]:
    """
    Build train and validation dataloaders.

    Returns:
        train_loader, val_loader, state_mean, state_std
    """
    print("\nBuilding dataloaders...")

    # Load all episodes
    episodes = load_episodes(data_dir)

    # Split episodes into train/val (episode-level split)
    # NEVER split within an episode — preserves temporal coherence
    rng        = np.random.default_rng(seed)
    n_episodes = len(episodes)
    indices    = rng.permutation(n_episodes)
    n_train    = int(n_episodes * TRAIN_RATIO)

    train_eps = [episodes[i] for i in indices[:n_train]]
    val_eps   = [episodes[i] for i in indices[n_train:]]

    # Compute normalization stats from TRAINING set only
    # Applying train stats to val prevents data leakage
    state_mean, state_std = compute_state_stats(train_eps)

    # Save stats for inference
    stats = {
        "state_mean": state_mean.tolist(),
        "state_std":  state_std.tolist(),
        "reward_scale": REWARD_SCALE,
    }
    stats_path = BENCHMARK_DIR / "normalization_stats.json"
    with open(str(stats_path), "w") as f:
        json.dump(stats, f, indent=2)

    # Build datasets
    train_ds = TrajFormerDataset(
        train_eps, context_len, state_mean, state_std
    )
    val_ds = TrajFormerDataset(
        val_eps, context_len, state_mean, state_std
    )

    # Build loaders
    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = 0,       # 0 for Windows stability
        pin_memory  = True,
        drop_last   = True,    # keeps batch size consistent
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 0,
        pin_memory  = True,
    )

    print(f"Train episodes : {len(train_eps)}")
    print(f"Val episodes   : {len(val_eps)}")
    print(f"Train windows  : {len(train_ds):,}")
    print(f"Val windows    : {len(val_ds):,}")

    return train_loader, val_loader, state_mean, state_std


def sanity_check_batch(loader: DataLoader) -> None:
    """
    Load one batch and verify data integrity.
    Called before training starts.
    Catches pipeline bugs that would cause silent training failures.
    """
    batch = next(iter(loader))

    rtg     = batch["returns_to_go"]
    states  = batch["states"]
    actions = batch["actions"]
    ts      = batch["timesteps"]
    targets = batch["target_actions"]

    print(f"\nBatch sanity check:")
    print(f"  returns_to_go : {tuple(rtg.shape)}  "
          f"range=[{rtg.min():.3f}, {rtg.max():.3f}]")
    print(f"  states        : {tuple(states.shape)}  "
          f"mean={states.mean():.3f}  std={states.std():.3f}")
    print(f"  actions       : {tuple(actions.shape)}  "
          f"std={actions.std(dim=[0,1]).tolist()}")
    print(f"  timesteps     : {tuple(ts.shape)}  "
          f"range=[{ts.min()}, {ts.max()}]")

    # Check action diversity
    action_std = actions.std(dim=[0, 1])
    if (action_std < MIN_DATA_ACTION_STD).any():
        raise RuntimeError(
            f"Batch sanity check FAILED: actions not diverse.\n"
            f"Action std: {action_std}\n"
            f"This indicates a data pipeline bug."
        )

    # Check states are normalized (mean ~0, std ~1)
    if abs(states.mean().item()) > 1.0:
        print(f"  WARNING: State mean={states.mean():.3f} — "
              f"normalization may be off")

    # Check RTG is in reasonable range
    if rtg.max() > 10.0:
        print(f"  WARNING: RTG max={rtg.max():.3f} — "
              f"check REWARD_SCALE in config.py")

    print(f"  ✓ Batch sanity check PASSED")


if __name__ == "__main__":
    train_loader, val_loader, mean, std = build_dataloaders()
    sanity_check_batch(train_loader)
    print("\nDataset ready for training.")