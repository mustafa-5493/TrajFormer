# src/model.py
# Full model definition for TrajFormer.
#
# Wires together the embedding and transformer from core/ into
# a single forward pass: (R, s, a) sequences → predicted actions.
#
# We predict actions from state token positions (3t+1) because
# by that point in the sequence each state token has already
# attended to the preceding R and s tokens — it carries the
# context we need. Predicting from R or a tokens gave worse
# results in early experiments.
#
# Loss is plain MSE against expert actions (behavior cloning).
# No value function, no policy gradient.

import torch
import torch.nn as nn
import json
from pathlib import Path
from typing import Optional, Tuple, Dict

from core.positional import TrajFormerEmbedding
from core.transformer import TransformerDecoder, count_parameters


class TrajFormer(nn.Module):
    """
    Decision Transformer for continuous robot control.

    Args:
        state_dim:      Observation dimension (11 for Hopper)
        action_dim:     Action dimension (3 for Hopper)
        d_model:        Transformer model dimension
        n_heads:        Number of attention heads
        n_layers:       Number of transformer blocks
        d_ff:           Feed-forward hidden dimension
        context_length: Number of timesteps K in context window
        dropout:        Dropout probability
        max_timesteps:  Maximum episode length
    """

    def __init__(
        self,
        state_dim:      int,
        action_dim:     int,
        d_model:        int,
        n_heads:        int,
        n_layers:       int,
        d_ff:           int,
        context_length: int,
        dropout:        float = 0.1,
        max_timesteps:  int   = 1000,
    ):
        super().__init__()

        self.state_dim      = state_dim
        self.action_dim     = action_dim
        self.d_model        = d_model
        self.context_length = context_length

        # Embedding layer: (R, s, a) → token sequence
        self.embedding = TrajFormerEmbedding(
            state_dim      = state_dim,
            action_dim     = action_dim,
            d_model        = d_model,
            context_length = context_length,
            max_timesteps  = max_timesteps,
            dropout        = dropout,
        )

        # Transformer decoder
        self.transformer = TransformerDecoder(
            n_layers = n_layers,
            d_model  = d_model,
            n_heads  = n_heads,
            d_ff     = d_ff,
            dropout  = dropout,
            causal   = True,
        )

        # Action prediction head
        # Projects state token representations → action predictions
        # tanh output: clips actions to [-1, 1] (Hopper action range)
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, action_dim),
            nn.Tanh(),
        )

        # Initialize weights
        self._init_weights()

        # Validate shapes at construction time
        self._validate_dimensions()

    def _init_weights(self):
        """
        Initialize weights following GPT-2 convention.
        Linear layers: normal(0, 0.02)
        Embedding layers: normal(0, 0.02)
        LayerNorm: weight=1, bias=0

        Proper initialization prevents vanishing/exploding gradients
        at the start of training.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        returns_to_go: torch.Tensor,   # (B, K, 1)
        states:        torch.Tensor,   # (B, K, STATE_DIM)
        actions:       torch.Tensor,   # (B, K, ACTION_DIM)
        timesteps:     torch.Tensor,   # (B, K)
    ) -> torch.Tensor:
        """
        Forward pass.

        Returns:
            predicted_actions: (B, K, ACTION_DIM)
            Action predictions for each timestep in the context.
            Training uses all K predictions.
            Inference uses only the last prediction (most recent timestep).
        """
        # Embed inputs → token sequence
        # (B, K, *) → (B, 3K, D)
        tokens = self.embedding(returns_to_go, states, actions, timesteps)

        # Pass through transformer
        # (B, 3K, D) → (B, 3K, D)
        hidden = self.transformer(tokens)

        # Extract state token positions
        # Tokens are interleaved: (R_0, s_0, a_0, R_1, s_1, a_1, ...)
        # State tokens are at positions 1, 4, 7, ... (index 3t+1)
        # We predict actions FROM state tokens — they have full causal context
        state_tokens = hidden[:, 1::3, :]   # (B, K, D)

        # Predict actions
        # (B, K, D) → (B, K, ACTION_DIM)
        predicted_actions = self.action_head(state_tokens)

        return predicted_actions

    def predict_action(
        self,
        returns_to_go: torch.Tensor,   # (B, K, 1)
        states:        torch.Tensor,   # (B, K, STATE_DIM)
        actions:       torch.Tensor,   # (B, K, ACTION_DIM)
        timesteps:     torch.Tensor,   # (B, K)
    ) -> torch.Tensor:
        """
        Single-step inference — returns only the LAST predicted action.
        Used during environment rollout.

        Returns:
            action: (B, ACTION_DIM)
        """
        with torch.no_grad():
            predicted = self.forward(
                returns_to_go, states, actions, timesteps
            )
            return predicted[:, -1, :]   # (B, ACTION_DIM)

    def get_attention_weights(self):
        """Return attention weights from all layers for visualization."""
        return self.transformer.get_all_attention_weights()

    def _validate_dimensions(self):
        """
        Dry run with dummy inputs.
        Catches shape bugs at construction time — before any training.
        """
        B, K = 2, self.context_length
        device = next(self.parameters()).device

        dummy_returns   = torch.zeros(B, K, 1,              device=device)
        dummy_states    = torch.zeros(B, K, self.state_dim,  device=device)
        dummy_actions   = torch.zeros(B, K, self.action_dim, device=device)
        dummy_timesteps = torch.arange(K, device=device).unsqueeze(0).expand(B, -1)

        with torch.no_grad():
            out = self.forward(
                dummy_returns, dummy_states,
                dummy_actions, dummy_timesteps,
            )

        expected = (B, K, self.action_dim)
        assert out.shape == expected, (
            f"Dimension validation failed: expected {expected}, got {out.shape}"
        )


def build_model(device: torch.device) -> TrajFormer:
    """
    Construct TrajFormer from config and move to device.
    Prints parameter summary.
    """
    from config import (
        STATE_DIM, ACTION_DIM, D_MODEL, N_HEADS,
        N_LAYERS, D_FF, CONTEXT_LENGTH, DROPOUT,
        MAX_EPISODE_STEPS,
    )

    model = TrajFormer(
        state_dim      = STATE_DIM,
        action_dim     = ACTION_DIM,
        d_model        = D_MODEL,
        n_heads        = N_HEADS,
        n_layers       = N_LAYERS,
        d_ff           = D_FF,
        context_length = CONTEXT_LENGTH,
        dropout        = DROPOUT,
        max_timesteps  = MAX_EPISODE_STEPS,
    ).to(device)

    params = count_parameters(model)

    print(f"\nTrajFormer constructed on {device}")
    print(f"  Parameters : {params['trainable']:,} trainable")
    print(f"  D_MODEL    : {D_MODEL}")
    print(f"  N_HEADS    : {N_HEADS}")
    print(f"  N_LAYERS   : {N_LAYERS}")
    print(f"  Context K  : {CONTEXT_LENGTH} timesteps ({CONTEXT_LENGTH*3} tokens)")

    return model


def save_checkpoint(
    model:     TrajFormer,
    optimizer: torch.optim.Optimizer,
    epoch:     int,
    val_loss:  float,
    best_loss: float,
    path:      Path,
) -> bool:
    """
    Save checkpoint. Returns True if this is the best model so far.
    """
    is_best = val_loss < best_loss

    torch.save({
        "epoch":            epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss":         val_loss,
        "best_loss":        best_loss,
    }, str(path / "last_model.pth"))

    if is_best:
        torch.save({
            "epoch":            epoch,
            "model_state_dict": model.state_dict(),
            "val_loss":         val_loss,
        }, str(path / "best_model.pth"))

    return is_best


def load_checkpoint(
    path:      Path,
    model:     TrajFormer,
    optimizer: Optional[torch.optim.Optimizer],
    device:    torch.device,
) -> dict:
    """
    Load checkpoint. Cross-device safe (map_location=cpu → .to(device)).
    """
    ckpt = torch.load(str(path), map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    return ckpt


def save_model_summary(model: TrajFormer, path: Path) -> None:
    """Save model architecture summary to JSON."""
    params = count_parameters(model)
    summary = {
        "architecture":   "Decision Transformer",
        "state_dim":      model.state_dim,
        "action_dim":     model.action_dim,
        "d_model":        model.d_model,
        "context_length": model.context_length,
        "trainable_params": params["trainable"],
        "total_params":   params["total"],
    }
    out = path / "model_summary.json"
    with open(str(out), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Model summary saved: {out}")


if __name__ == "__main__":
    from config import DEVICE, BENCHMARK_DIR
    model = build_model(DEVICE)
    save_model_summary(model, BENCHMARK_DIR)
    print("\nModel construction: PASSED ✓")