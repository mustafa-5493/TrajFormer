# core/transformer.py
# Transformer decoder blocks.
#
# Decoder-only (GPT-style) — no cross-attention, no encoder.
# The full context (R, s, a) lives in a single sequence so there
# is no separate source to attend to.
#
# We use pre-norm (LayerNorm before attention and FFN) rather than
# the post-norm in the original Transformer paper. Post-norm needs
# careful LR warmup to not diverge early; pre-norm is more forgiving
# and worked better in our experiments.
#
# FFN uses GELU. We tried ReLU — GELU converged slightly faster
# and final val loss was marginally lower, so I kept it.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

from core.attention import MultiHeadAttention


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    Applied independently to each token position.
    Two linear layers with GELU activation:
        FFN(x) = W_2 * GELU(W_1 * x + b_1) + b_2

    Why GELU over ReLU?
        GELU (Gaussian Error Linear Unit) is smoother than ReLU,
        empirically performs better in transformer language/control models.
        Used in GPT-2, BERT, and Decision Transformer.

    D → D_FF → D  (expand then contract)
    """

    def __init__(
        self,
        d_model:  int,
        d_ff:     int,
        dropout:  float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Single Transformer decoder block (GPT-style, pre-norm).

    Pre-norm formulation:
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))

    The residual connections allow gradients to flow directly
    through the network — essential for training deep transformers.

    Args:
        d_model:  Model dimension
        n_heads:  Number of attention heads
        d_ff:     Feed-forward hidden dimension
        dropout:  Dropout probability
        causal:   If True, use causal (autoregressive) attention
    """

    def __init__(
        self,
        d_model:  int,
        n_heads:  int,
        d_ff:     int,
        dropout:  float = 0.1,
        causal:   bool  = True,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.attention = MultiHeadAttention(
            d_model = d_model,
            n_heads = n_heads,
            dropout = dropout,
            causal  = causal,
        )
        self.ffn = FeedForward(
            d_model = d_model,
            d_ff    = d_ff,
            dropout = dropout,
        )

    def forward(
        self,
        x:    torch.Tensor,                  # (B, T, D)
        mask: Optional[torch.Tensor] = None, # optional additional mask
    ) -> torch.Tensor:
        """
        Pre-norm transformer block forward pass.

        Returns:
            (B, T, D) — same shape as input
        """
        # Self-attention with residual
        x = x + self.attention(self.norm1(x), mask=mask)

        # Feed-forward with residual
        x = x + self.ffn(self.norm2(x))

        return x

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Expose attention weights for visualization."""
        return self.attention.get_attention_weights()


class TransformerDecoder(nn.Module):
    """
    Stack of N transformer decoder blocks.

    This is the full transformer body of TrajFormer.
    Takes token embeddings as input, outputs contextualized representations.

    Args:
        n_layers: Number of stacked transformer blocks
        d_model:  Model dimension
        n_heads:  Number of attention heads
        d_ff:     Feed-forward hidden dimension
        dropout:  Dropout probability
        causal:   If True, use causal attention (required for DT)
    """

    def __init__(
        self,
        n_layers: int,
        d_model:  int,
        n_heads:  int,
        d_ff:     int,
        dropout:  float = 0.1,
        causal:   bool  = True,
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model = d_model,
                n_heads = n_heads,
                d_ff    = d_ff,
                dropout = dropout,
                causal  = causal,
            )
            for _ in range(n_layers)
        ])

        # Final layer norm — applied after last block
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x:    torch.Tensor,                  # (B, T, D)
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Pass through all transformer blocks.

        Returns:
            (B, T, D) — contextualized token representations
        """
        for block in self.blocks:
            x = block(x, mask=mask)

        return self.norm(x)

    def get_all_attention_weights(self) -> List[torch.Tensor]:
        """
        Return attention weights from all layers.
        Used by analysis/attention_viz.py to visualize
        how attention evolves across layers.

        Returns:
            List of (B, H, T, T) tensors, one per layer.
            None entries for layers that haven't run forward yet.
        """
        return [
            block.get_attention_weights()
            for block in self.blocks
        ]


# =============================================================================
# PARAMETER COUNT UTILITY
# =============================================================================

def count_parameters(model: nn.Module) -> dict:
    """
    Count trainable and total parameters.
    Printed at model construction time.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    return {
        "trainable":  trainable,
        "total":      total,
        "frozen":     total - trainable,
    }


# =============================================================================
# SHAPE VALIDATION
# =============================================================================

def _validate_transformer_shapes():
    from config import D_MODEL, N_HEADS, D_FF, N_LAYERS, N_TOKENS, DROPOUT

    B, T, D = 2, N_TOKENS, D_MODEL

    decoder = TransformerDecoder(
        n_layers = N_LAYERS,
        d_model  = D_MODEL,
        n_heads  = N_HEADS,
        d_ff     = D_FF,
        dropout  = DROPOUT,
        causal   = True,
    )
    decoder.eval()

    x   = torch.randn(B, T, D)
    out = decoder(x)

    assert out.shape == (B, T, D), (
        f"Expected ({B}, {T}, {D}), got {out.shape}"
    )

    # Verify attention weights are cached in all layers
    weights = decoder.get_all_attention_weights()
    assert len(weights) == N_LAYERS
    assert all(w.shape == (B, N_HEADS, T, T) for w in weights)

    # Parameter count
    params = count_parameters(decoder)
    print(f"  TransformerDecoder: ({B}, {T}, {D}) → ({B}, {T}, {D}) ✓")
    print(f"  Layers            : {N_LAYERS} blocks ✓")
    print(f"  Attention weights : {N_LAYERS} × ({B}, {N_HEADS}, {T}, {T}) ✓")
    print(f"  Parameters        : {params['trainable']:,} trainable")


if __name__ == "__main__":
    print("Validating transformer shapes...")
    _validate_transformer_shapes()
    print("All transformer checks passed.")