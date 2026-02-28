# benchmark/latency_test.py
# Per-stage inference latency profiling for TrajFormer.
#
# Measures wall-clock time for each stage of the inference pipeline:
#   1. Data preparation (CPU)
#   2. Embedding layer
#   3. Transformer decoder
#   4. Action head
#   5. Full forward pass (end-to-end)
#
# Results saved to outputs/benchmarks/latency_results.json
#
# Usage:
#   python benchmark/latency_test.py

import torch
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    DEVICE,
    CONTEXT_LENGTH,
    STATE_DIM,
    ACTION_DIM,
    BENCHMARK_DIR,
    BENCHMARK_WARMUP_RUNS,
    BENCHMARK_TIMED_RUNS,
    BATCH_SIZE,
)
from src.model import build_model, load_checkpoint


def make_dummy_batch(B: int = 1) -> dict:
    K = CONTEXT_LENGTH
    return {
        "returns_to_go": torch.randn(B, K, 1,          device=DEVICE),
        "states":        torch.randn(B, K, STATE_DIM,   device=DEVICE),
        "actions":       torch.randn(B, K, ACTION_DIM,  device=DEVICE),
        "timesteps":     torch.arange(K, device=DEVICE).unsqueeze(0).expand(B, -1),
    }


def timed_run(fn, warmup: int, timed: int) -> Dict[str, float]:
    """
    Run fn() warmup times (discarded), then timed times.
    Returns mean and std latency in milliseconds.
    """
    # Warmup
    for _ in range(warmup):
        fn()
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

    # Timed
    latencies = []
    for _ in range(timed):
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)

    return {
        "mean_ms": float(np.mean(latencies)),
        "std_ms":  float(np.std(latencies)),
        "min_ms":  float(np.min(latencies)),
        "max_ms":  float(np.max(latencies)),
        "p99_ms":  float(np.percentile(latencies, 99)),
    }


def run_latency_benchmark() -> None:
    print(f"\nLatency benchmark on {DEVICE}")
    print(f"Warmup: {BENCHMARK_WARMUP_RUNS} runs")
    print(f"Timed : {BENCHMARK_TIMED_RUNS} runs")
    print(f"Batch : 1 (single inference)\n")

    model = build_model(DEVICE)
    ckpt  = BENCHMARK_DIR.parent.parent / "outputs" / "checkpoints" / "best_model.pth"
    if ckpt.exists():
        load_checkpoint(ckpt, model, optimizer=None, device=DEVICE)
    model.eval()

    batch = make_dummy_batch(B=1)
    rtg   = batch["returns_to_go"]
    s     = batch["states"]
    a     = batch["actions"]
    ts    = batch["timesteps"]

    results = {}

    # --- Stage 1: Embedding ---
    with torch.no_grad():
        def embed_fn():
            _ = model.embedding(rtg, s, a, ts)
        stats = timed_run(embed_fn, BENCHMARK_WARMUP_RUNS, BENCHMARK_TIMED_RUNS)
    results["embedding"] = stats
    print(f"Embedding       : {stats['mean_ms']:.3f} ± {stats['std_ms']:.3f} ms")

    # --- Stage 2: Transformer ---
    with torch.no_grad():
        tokens = model.embedding(rtg, s, a, ts)
        def transformer_fn():
            _ = model.transformer(tokens)
        stats = timed_run(transformer_fn, BENCHMARK_WARMUP_RUNS, BENCHMARK_TIMED_RUNS)
    results["transformer"] = stats
    print(f"Transformer     : {stats['mean_ms']:.3f} ± {stats['std_ms']:.3f} ms")

    # --- Stage 3: Action head ---
    with torch.no_grad():
        tokens  = model.embedding(rtg, s, a, ts)
        hidden  = model.transformer(tokens)
        s_toks  = hidden[:, 1::3, :]
        def action_head_fn():
            _ = model.action_head(s_toks)
        stats = timed_run(action_head_fn, BENCHMARK_WARMUP_RUNS, BENCHMARK_TIMED_RUNS)
    results["action_head"] = stats
    print(f"Action head     : {stats['mean_ms']:.3f} ± {stats['std_ms']:.3f} ms")

    # --- Stage 4: Full forward pass ---
    with torch.no_grad():
        def full_fn():
            _ = model(rtg, s, a, ts)
        stats = timed_run(full_fn, BENCHMARK_WARMUP_RUNS, BENCHMARK_TIMED_RUNS)
    results["full_forward"] = stats
    print(f"Full forward    : {stats['mean_ms']:.3f} ± {stats['std_ms']:.3f} ms")

    # --- Stage 5: Batch throughput ---
    batch_b = make_dummy_batch(B=BATCH_SIZE)
    with torch.no_grad():
        def batch_fn():
            _ = model(
                batch_b["returns_to_go"],
                batch_b["states"],
                batch_b["actions"],
                batch_b["timesteps"],
            )
        stats = timed_run(batch_fn, BENCHMARK_WARMUP_RUNS, BENCHMARK_TIMED_RUNS)
    results["batch_forward"] = stats
    throughput = BATCH_SIZE / (stats["mean_ms"] / 1000)
    results["throughput_samples_per_sec"] = throughput
    print(f"Batch forward   : {stats['mean_ms']:.3f} ± {stats['std_ms']:.3f} ms "
          f"(B={BATCH_SIZE})")
    print(f"Throughput      : {throughput:.0f} samples/sec")

    # Summary
    total = results["full_forward"]["mean_ms"]
    print(f"\nInference latency breakdown:")
    print(f"  Embedding   : {results['embedding']['mean_ms']:.3f} ms  "
          f"({100*results['embedding']['mean_ms']/total:.1f}%)")
    print(f"  Transformer : {results['transformer']['mean_ms']:.3f} ms  "
          f"({100*results['transformer']['mean_ms']/total:.1f}%)")
    print(f"  Action head : {results['action_head']['mean_ms']:.3f} ms  "
          f"({100*results['action_head']['mean_ms']/total:.1f}%)")
    print(f"  Total       : {total:.3f} ms")
    print(f"  Max rate    : {1000/total:.0f} Hz  "
          f"({'real-time capable' if 1000/total > 50 else 'below 50Hz'})")

    # Save
    results["device"]   = str(DEVICE)
    results["n_params"] = sum(p.numel() for p in model.parameters())
    out = BENCHMARK_DIR / "latency_results.json"
    with open(str(out), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out}")


if __name__ == "__main__":
    run_latency_benchmark()