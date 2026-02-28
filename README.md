# TrajFormer

A Decision Transformer trained from scratch on a GTX 1050 (4GB VRAM) to control a simulated hopping robot. Attention, positional encoding, and transformer blocks are written from scratch — no `nn.Transformer`.


---

## What This Is

Decision Transformer (Chen et al. 2021) reformulates robot control as sequence modeling. Instead of learning a value function or a policy gradient, the model learns to predict actions conditioned on a desired return-to-go — essentially, you tell it how well you want it to perform, and it figures out the actions that historically led to that outcome.

The input sequence looks like this:

```
(R₀, s₀, a₀, R₁, s₁, a₁, ..., R₁₉, s₁₉, a₁₉)
```

Where R is reward-to-go, s is the 11-dimensional Hopper state, and a is the 3-dimensional joint torque command. At inference, set R high — the model produces actions that match that target.

This is the same architectural paradigm behind RT-2, Decision Transformer, and most modern robot foundation models. This project is a minimal, fully transparent implementation of that idea.

---

## Results

| Agent | Mean Return | Mean Episode Length |
|---|---|---|
| Random | 14.6 ± 9.3 | 20 steps |
| **TrajFormer** | **1365.3 ± 200.5** | **369 steps** |
| Expert SAC | ~3000+ | ~1000 steps |

Evaluated over 20 rollout episodes in Hopper-v4. The model keeps the robot upright and moving forward for 369 steps on average — compared to 20 steps for a random agent.

---

## Benchmarks

| Metric | Value |
|---|---|
| Model size | 5.1 MB |
| Training peak VRAM | 332.5 MB (8.1% of 4GB) |
| Single inference latency | 4.87 ms |
| Max inference rate | 205 Hz |
| Throughput | 2,593 samples/sec |
| Latency breakdown | Transformer 89.5%, Embedding 9.2%, Head 4.5% |

205 Hz exceeds the control frequency of most real-time robot controllers. The transformer accounts for 89.5% of inference time — the expected bottleneck for attention-based architectures.

---

## What The Attention Learns

The token type attention patterns across all 6 layers show a consistent signal: **action tokens attend most strongly to state tokens**. The a→s attention dominates in every layer, peaking at 0.039 in layer 3.

This is physically correct. To predict joint torques, the model looks at joint positions and velocities — the state. It learned this structure from data, not from any hardcoded inductive bias.

![Token type attention patterns](outputs/plots/attention_token_types.png)

The causal mask holds perfectly across all layers — clean lower-triangular structure, no future token leakage.

![Attention across all layers](outputs/plots/attention_summary.png)

---

## Architecture

```
(R, s, a) sequences
      ↓
TrajFormerEmbedding
  - Linear projections: R→D, s→D, a→D
  - Sinusoidal positional encoding (fixed)
  - Learned timestep embedding (shared across R,s,a at same t)
  - Interleaved token sequence: (R₀, s₀, a₀, R₁, s₁, a₁, ...)
      ↓
TransformerDecoder — 6 layers
  Each block (pre-norm / GPT-style):
    x = x + MultiHeadAttention(LayerNorm(x))   ← causal, 8 heads
    x = x + FFN(LayerNorm(x))                  ← GELU, 4× expansion
      ↓
Action head: state token representations → predicted actions
  Linear → GELU → Linear → Tanh
  Output clipped to [-1, 1] (Hopper action range)
```

| Parameter | Value |
|---|---|
| D_MODEL | 128 |
| N_HEADS | 8 |
| N_LAYERS | 6 |
| D_FF | 512 |
| Context length K | 20 timesteps (60 tokens) |
| Trainable parameters | 1,334,275 |

Everything in `core/` is written from scratch. The attention mechanism, causal mask, multi-head projection, sinusoidal encoding, and transformer blocks do not use `nn.Transformer` or `nn.MultiheadAttention`.

---

## Training

**Dataset:** 500 expert episodes collected using a pretrained SAC agent (sb3/sac-Hopper-v3 from HuggingFace). 209,621 total transitions. Mean episode reward 1534, mean length 419 steps.

**Objective:** MSE between predicted and actual actions (behavior cloning).

**Schedule:** Linear warmup (500 steps) → cosine decay. AdamW, lr=1e-4, weight_decay=1e-4. Gradient clipping at 1.0.

**Hardware:** NVIDIA GTX 1050, 4GB VRAM. AMP enabled. ~4 min/epoch.

```
Epoch 001: train=0.0956  val=0.0579
Epoch 005: train=0.0417  val=0.0422
Epoch 016: train=0.0319  val=0.0392  ← best
Epoch 026: train=0.0283  val=0.0398  ← early stop
```

Training completed in ~1.7 hours. Early stopping triggered after 26 epochs without val improvement.

---

## Project Structure

```
TrajFormer/
├── core/
│   ├── attention.py      # Scaled dot-product + multi-head attention
│   ├── positional.py     # Sinusoidal PE + learned timestep embedding
│   └── transformer.py    # Decoder blocks + full decoder stack
├── src/
│   ├── model.py          # TrajFormer: full Decision Transformer
│   ├── dataset.py        # Trajectory windows + normalization
│   ├── train.py          # Training loop with sanity checks
│   └── evaluate.py       # Environment rollout evaluation
├── data/
│   └── collect.py        # Expert trajectory collection (SAC agent)
├── analysis/
│   ├── attention_viz.py  # Attention heatmaps + token type patterns
│   └── trajectory_viz.py # Predicted vs actual trajectory plots
├── benchmark/
│   ├── latency_test.py   # Per-stage inference latency
│   └── vram_profiler.py  # VRAM usage profiling
└── config.py             # All hyperparameters in one place
```

---

## Reproducing

```bash
# Install
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
pip install gymnasium[mujoco] stable-baselines3 huggingface-sb3 \
    huggingface-hub shimmy numpy matplotlib tqdm scikit-learn

# Collect expert data (~5 min)
python data/collect.py

# Train (~2-4 hours on GTX 1050)
python src/train.py

# Evaluate
python src/evaluate.py

# Visualize attention
python analysis/attention_viz.py
```

---

## Limitations

**RTG conditioning is weak.** The model achieves ~600 reward regardless of target return-to-go value. This is a known failure mode of small Decision Transformers trained on datasets without sufficient return diversity — the model learns good average behavior but doesn't learn to modulate performance based on the conditioning signal. The original DT paper observes the same pattern on medium-quality datasets.

The action prediction MSE (0.037–0.041) shows the model learned to imitate expert motion reasonably well — the behavior cloning objective works. The RTG conditioning does not.

---

## References

- Chen et al. (2021) — [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345)
- Vaswani et al. (2017) — [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Fu et al. (2020) — [D4RL: Datasets for Deep Data-Driven Reinforcement Learning](https://arxiv.org/abs/2004.07219)
- Raffin et al. (2021) — [Stable-Baselines3](https://jmlr.org/papers/v22/20-1364.html)