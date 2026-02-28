# benchmark/vram_profiler.py
# VRAM usage profiling for TrajFormer.
#
# Measures peak VRAM at each stage:
#   - Model load
#   - Single inference
#   - Batch inference
#   - Training step
#
# Results saved to outputs/benchmarks/vram_results.json
#
# Usage:
#   python benchmark/vram_profiler.py

import torch
import numpy as np
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    DEVICE,
    CONTEXT_LENGTH,
    STATE_DIM,
    ACTION_DIM,
    BENCHMARK_DIR,
    BATCH_SIZE,
    LEARNING_RATE,
)
from src.model import build_model, load_checkpoint


def mb(bytes: int) -> float:
    return bytes / (1024 ** 2)


def reset_peak():
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def peak_mb() -> float:
    if DEVICE.type == "cuda":
        return mb(torch.cuda.max_memory_allocated())
    return 0.0


def current_mb() -> float:
    if DEVICE.type == "cuda":
        return mb(torch.cuda.memory_allocated())
    return 0.0


def make_batch(B: int):
    K = CONTEXT_LENGTH
    return (
        torch.randn(B, K, 1,          device=DEVICE),
        torch.randn(B, K, STATE_DIM,   device=DEVICE),
        torch.randn(B, K, ACTION_DIM,  device=DEVICE),
        torch.arange(K, device=DEVICE).unsqueeze(0).expand(B, -1),
    )


def run_vram_profile() -> None:
    if DEVICE.type != "cuda":
        print("VRAM profiling requires CUDA. Skipping.")
        return

    print(f"\nVRAM profiling on {DEVICE}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    total_vram = mb(torch.cuda.get_device_properties(0).total_memory)
    print(f"Total VRAM: {total_vram:.0f} MB\n")

    results = {"device": torch.cuda.get_device_name(0),
               "total_vram_mb": total_vram}

    # --- Baseline ---
    torch.cuda.empty_cache()
    reset_peak()
    baseline = current_mb()
    results["baseline_mb"] = baseline
    print(f"Baseline (empty)     : {baseline:.1f} MB")

    # --- Model load ---
    torch.cuda.empty_cache()
    reset_peak()
    model = build_model(DEVICE)
    model.eval()
    model_mb = current_mb()
    results["model_load_mb"] = model_mb
    print(f"Model loaded         : {model_mb:.1f} MB  "
          f"(+{model_mb - baseline:.1f} MB)")

    # --- Single inference ---
    reset_peak()
    rtg, s, a, ts = make_batch(B=1)
    with torch.no_grad():
        _ = model(rtg, s, a, ts)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    single_peak = peak_mb()
    results["single_inference_peak_mb"] = single_peak
    print(f"Single inference     : {single_peak:.1f} MB peak  "
          f"(+{single_peak - model_mb:.1f} MB activations)")

    # --- Batch inference ---
    torch.cuda.empty_cache()
    reset_peak()
    rtg, s, a, ts = make_batch(B=BATCH_SIZE)
    with torch.no_grad():
        _ = model(rtg, s, a, ts)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    batch_peak = peak_mb()
    results["batch_inference_peak_mb"] = batch_peak
    results["batch_size"] = BATCH_SIZE
    print(f"Batch inference      : {batch_peak:.1f} MB peak  "
          f"(B={BATCH_SIZE})")

    # --- Training step (forward + backward) ---
    torch.cuda.empty_cache()
    reset_peak()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.MSELoss()

    rtg, s, a, ts = make_batch(B=BATCH_SIZE)
    targets = torch.randn(BATCH_SIZE, CONTEXT_LENGTH, ACTION_DIM, device=DEVICE)

    optimizer.zero_grad()
    preds = model(rtg, s, a, ts)
    loss  = criterion(preds, targets)
    loss.backward()
    optimizer.step()

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    train_peak = peak_mb()
    results["training_step_peak_mb"] = train_peak
    print(f"Training step        : {train_peak:.1f} MB peak  "
          f"(forward + backward, B={BATCH_SIZE})")

    # Summary
    headroom = total_vram - train_peak
    print(f"\nVRAM summary:")
    print(f"  Model weights    : {model_mb:.1f} MB")
    print(f"  Training peak    : {train_peak:.1f} MB "
          f"({100*train_peak/total_vram:.1f}% of {total_vram:.0f} MB)")
    print(f"  Headroom         : {headroom:.1f} MB")
    print(f"  Fits in 4GB      : {'YES' if train_peak < 4096 else 'NO'}")

    # Save
    out = BENCHMARK_DIR / "vram_results.json"
    with open(str(out), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out}")


if __name__ == "__main__":
    run_vram_profile()