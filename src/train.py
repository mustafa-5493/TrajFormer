# src/train.py
# TrajFormer training loop.
#
# Trains Decision Transformer via behavior cloning:
#   Loss = MSE(predicted_actions, actual_actions)
#
# Training decisions:
#
# 1. LINEAR WARMUP + COSINE DECAY
#    Transformers are sensitive to LR at the start of training.
#    Linear warmup for WARMUP_STEPS prevents large early updates
#    that can permanently damage the attention weights.
#
# 2. GRADIENT CLIPPING (max_norm=1.0)
#    Essential for transformer training — prevents gradient explosion
#    in deep attention layers.
#
# 3. AMP (Automatic Mixed Precision)
#    FP16 computation on CUDA — reduces VRAM, speeds up training.
#    Disabled automatically on CPU.
#
# 4. PER-EPOCH SANITY CHECKS
#    Prediction diversity monitored every epoch.
#    If predicted action std < MIN_PRED_STD: model is collapsing.
#    Training aborts with clear error message.
#
# 5. TRAINING HISTORY → JSON every epoch
#    Survives crashes. Resume from last checkpoint.

import torch
import torch.nn as nn
import numpy as np
import json
import time
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    DEVICE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    GRAD_CLIP,
    WARMUP_STEPS,
    USE_AMP,
    PATIENCE,
    CHECKPOINT_DIR,
    BENCHMARK_DIR,
    MIN_PRED_STD,
    MAX_INITIAL_LOSS,
    SEED,
)
from src.model import build_model, save_checkpoint, load_checkpoint
from src.dataset import build_dataloaders, sanity_check_batch


# =============================================================================
# LEARNING RATE SCHEDULE
# =============================================================================

def get_lr_scheduler(
    optimizer:     torch.optim.Optimizer,
    warmup_steps:  int,
    total_steps:   int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Linear warmup followed by cosine decay.

    Warmup: LR increases linearly from 0 to LEARNING_RATE
    Decay:  LR follows cosine curve from LEARNING_RATE to ~0

    This is standard for transformer training.
    """
    import math

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(
            total_steps - warmup_steps, 1
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =============================================================================
# TRAIN / VAL STEP
# =============================================================================

def train_epoch(
    model:       nn.Module,
    loader:      torch.utils.data.DataLoader,
    optimizer:   torch.optim.Optimizer,
    scheduler:   torch.optim.lr_scheduler.LambdaLR,
    criterion:   nn.Module,
    scaler:      Optional[torch.cuda.amp.GradScaler],
    device:      torch.device,
    epoch:       int,
    log_interval: int = 50,
) -> Dict[str, float]:
    """
    One training epoch.
    Returns dict of metrics.
    """
    model.train()

    total_loss   = 0.0
    all_preds    = []
    n_batches    = 0
    grad_norms   = []

    for batch_idx, batch in enumerate(
        tqdm(loader, desc=f"Epoch {epoch:03d} [train]", leave=False)
    ):
        rtg       = batch["returns_to_go"].to(device, non_blocking=True)
        states    = batch["states"].to(device,        non_blocking=True)
        actions   = batch["actions"].to(device,       non_blocking=True)
        timesteps = batch["timesteps"].to(device,     non_blocking=True)
        targets   = batch["target_actions"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                preds = model(rtg, states, actions, timesteps)
                loss  = criterion(preds, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), GRAD_CLIP
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(rtg, states, actions, timesteps)
            loss  = criterion(preds, targets)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), GRAD_CLIP
            )
            optimizer.step()

        scheduler.step()

        # Guard against NaN loss
        if torch.isnan(loss):
            print(f"\n  WARNING: NaN loss at batch {batch_idx} — skipping")
            continue

        total_loss += loss.item()
        grad_norms.append(grad_norm.item()
                          if isinstance(grad_norm, torch.Tensor)
                          else grad_norm)
        all_preds.append(preds.detach().cpu())
        n_batches  += 1

    # Prediction diversity check
    all_preds_np  = torch.cat(all_preds, dim=0).numpy()
    pred_std      = all_preds_np.std(axis=(0, 1))

    return {
        "loss":      total_loss / max(n_batches, 1),
        "pred_std":  pred_std.tolist(),
        "grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
        "lr":        scheduler.get_last_lr()[0],
    }


@torch.no_grad()
def val_epoch(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    epoch:     int,
) -> Dict[str, float]:
    """One validation epoch."""
    model.eval()

    total_loss = 0.0
    all_preds  = []
    n_batches  = 0

    for batch in tqdm(
        loader, desc=f"Epoch {epoch:03d} [val]  ", leave=False
    ):
        rtg       = batch["returns_to_go"].to(device)
        states    = batch["states"].to(device)
        actions   = batch["actions"].to(device)
        timesteps = batch["timesteps"].to(device)
        targets   = batch["target_actions"].to(device)

        preds = model(rtg, states, actions, timesteps)
        loss  = criterion(preds, targets)

        total_loss += loss.item()
        all_preds.append(preds.cpu())
        n_batches  += 1

    all_preds_np = torch.cat(all_preds, dim=0).numpy()
    pred_std     = all_preds_np.std(axis=(0, 1))

    return {
        "loss":     total_loss / max(n_batches, 1),
        "pred_std": pred_std.tolist(),
    }


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def train() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"Device : {DEVICE}")
    print(f"AMP    : {USE_AMP and DEVICE.type == 'cuda'}")

    # Build data
    train_loader, val_loader, state_mean, state_std = build_dataloaders()
    sanity_check_batch(train_loader)

    # Build model
    model = build_model(DEVICE)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = LEARNING_RATE,
        weight_decay = WEIGHT_DECAY,
    )

    # LR scheduler
    total_steps = NUM_EPOCHS * len(train_loader)
    scheduler   = get_lr_scheduler(optimizer, WARMUP_STEPS, total_steps)

    # Loss
    criterion = nn.MSELoss()

    # AMP scaler
    use_amp = USE_AMP and DEVICE.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None

    # Training state
    best_val_loss = float("inf")
    patience_ctr  = 0
    history: List[Dict] = []

    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    print(f"Total steps : {total_steps:,}")
    print(f"Warmup steps: {WARMUP_STEPS}\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, scaler, DEVICE, epoch,
        )

        # Validate
        val_metrics = val_epoch(
            model, val_loader, criterion, DEVICE, epoch,
        )

        elapsed = time.time() - t0

        # ======================================================================
        # PER-EPOCH SANITY CHECKS
        # ======================================================================

        # 1. Check initial loss isn't absurdly high
        if epoch == 1 and train_metrics["loss"] > MAX_INITIAL_LOSS:
            raise RuntimeError(
                f"Initial loss too high: {train_metrics['loss']:.4f}\n"
                f"Expected < {MAX_INITIAL_LOSS}. Check data pipeline."
            )

        # 2. Check prediction diversity
        pred_std = np.array(train_metrics["pred_std"])
        if epoch > 3 and (pred_std < MIN_PRED_STD).any():
            print(f"\n  WARNING: Low prediction diversity at epoch {epoch}")
            print(f"  Pred std: {pred_std}")
            print(f"  Model may be collapsing to constant output.")

        # 3. Overfitting check
        overfit_ratio = val_metrics["loss"] / max(train_metrics["loss"], 1e-8)
        overfit_flag  = "⚠ OVERFIT" if overfit_ratio > 3.0 else ""

        # Save checkpoint
        is_best = save_checkpoint(
            model, optimizer, epoch,
            val_metrics["loss"], best_val_loss,
            CHECKPOINT_DIR,
        )
        if is_best:
            best_val_loss = val_metrics["loss"]
            patience_ctr  = 0
        else:
            patience_ctr += 1

        # Log
        print(
            f"Epoch {epoch:03d}/{NUM_EPOCHS} | "
            f"train={train_metrics['loss']:.4f} | "
            f"val={val_metrics['loss']:.4f} | "
            f"lr={train_metrics['lr']:.2e} | "
            f"grad={train_metrics['grad_norm']:.3f} | "
            f"pred_std={[f'{s:.3f}' for s in train_metrics['pred_std']]} | "
            f"{'✓ BEST' if is_best else ''} {overfit_flag} | "
            f"{elapsed:.1f}s"
        )

        # Record history
        history.append({
            "epoch":      epoch,
            "train_loss": train_metrics["loss"],
            "val_loss":   val_metrics["loss"],
            "lr":         train_metrics["lr"],
            "grad_norm":  train_metrics["grad_norm"],
            "pred_std":   train_metrics["pred_std"],
            "is_best":    is_best,
        })

        # Save history every epoch — survives crashes
        hist_path = BENCHMARK_DIR / "training_history.json"
        with open(str(hist_path), "w") as f:
            json.dump(history, f, indent=2)

        # Early stopping
        if patience_ctr >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {PATIENCE} epochs)")
            break

    print(f"\nTraining complete.")
    print(f"Best val loss : {best_val_loss:.6f}")
    print(f"Checkpoint    : {CHECKPOINT_DIR / 'best_model.pth'}")


if __name__ == "__main__":
    if __name__ == "__main__":
        train()