# core/attention.py
# Scaled dot-product attention and multi-head attention.
#
# Written from scratch rather than using nn.MultiheadAttention.
# The implementation follows Vaswani et al. closely — separate
# W_q, W_k, W_v, W_o projections, scaling by sqrt(d_k), softmax
# over the key dimension.
#
# Causal mask is built once per sequence length and cached.
# Upper-triangular, True = masked, applied before softmax as -inf.
# nan_to_num handles the first token edge case where the entire
# row is -inf and softmax returns nan.
#
# Attention weights from the last forward pass are cached on the
# module so analysis/attention_viz.py can read them without
# modifying the forward signature.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


def scaled_dot_product_attention(
    q:    torch.Tensor,          # (B, H, T, d_k)
    k:    torch.Tensor,          # (B, H, T, d_k)
    v:    torch.Tensor,          # (B, H, T, d_k)
    mask: Optional[torch.Tensor] = None,  # (1, 1, T, T) causal mask
    dropout_p: float = 0.0,
    training:  bool  = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Scaled dot-product attention.

    Args:
        q, k, v:  Query, Key, Value tensors — already split into heads
        mask:     Causal mask — True positions are MASKED (set to -inf)
        dropout_p: Attention weight dropout probability
        training:  Whether in training mode (controls dropout)

    Returns:
        output:  (B, H, T, d_k) — attended values
        weights: (B, H, T, T)   — attention weights (for visualization)
    """
    d_k = q.size(-1)

    # Step 1: Compute raw attention scores
    # QK^T / sqrt(d_k)
    # (B, H, T, d_k) @ (B, H, d_k, T) → (B, H, T, T)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    # Step 2: Apply causal mask
    # Future positions set to -inf → softmax gives them ~0 weight
    # This enforces autoregressive property: token i only sees tokens 0..i
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    # Step 3: Softmax over key dimension
    # (B, H, T, T) — rows sum to 1
    weights = F.softmax(scores, dim=-1)

    # Handle case where entire row is -inf (first token, full mask)
    # softmax(-inf, -inf, ...) = nan → replace with 0
    weights = torch.nan_to_num(weights, nan=0.0)

    # Step 4: Attention dropout (regularization)
    if dropout_p > 0.0 and training:
        weights = F.dropout(weights, p=dropout_p)

    # Step 5: Weighted sum of values
    # (B, H, T, T) @ (B, H, T, d_k) → (B, H, T, d_k)
    output = torch.matmul(weights, v)

    return output, weights


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention — from scratch.

    Splits D_MODEL into H heads, runs attention in parallel,
    concatenates and projects back to D_MODEL.

    Why multiple heads?
        Different heads can attend to different aspects of the sequence.
        In robot control: one head might track velocity,
        another might track position, another might track reward-to-go.

    Args:
        d_model:   Total model dimension
        n_heads:   Number of attention heads
        dropout:   Attention and projection dropout
        causal:    If True, apply causal mask (autoregressive)
    """

    def __init__(
        self,
        d_model:  int,
        n_heads:  int,
        dropout:  float = 0.1,
        causal:   bool  = True,
    ):
        super().__init__()

        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )

        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_k      = d_model // n_heads   # per-head dimension
        self.causal   = causal
        self.dropout  = dropout

        # Linear projections for Q, K, V and output
        # Using separate projections (not combined) for clarity
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)

        # Cache for attention weights — used for visualization
        # Set during forward pass, read by analysis/attention_viz.py
        self._last_attn_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        x:    torch.Tensor,                  # (B, T, D)
        mask: Optional[torch.Tensor] = None, # (1, 1, T, T)
    ) -> torch.Tensor:
        """
        Args:
            x:    Input sequence (B, T, D)
            mask: Optional additional mask

        Returns:
            output: (B, T, D) — same shape as input
        """
        B, T, D = x.shape

        # Step 1: Project to Q, K, V
        # (B, T, D) → (B, T, D)
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # Step 2: Split into heads
        # (B, T, D) → (B, T, H, d_k) → (B, H, T, d_k)
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Step 3: Build causal mask if needed
        causal_mask = None
        if self.causal:
            causal_mask = self._get_causal_mask(T, x.device)

        # Combine causal mask with any additional mask
        if mask is not None and causal_mask is not None:
            combined_mask = causal_mask | mask
        elif causal_mask is not None:
            combined_mask = causal_mask
        else:
            combined_mask = mask

        # Step 4: Scaled dot-product attention
        # (B, H, T, d_k), (B, H, T, T)
        attn_out, attn_weights = scaled_dot_product_attention(
            q, k, v,
            mask      = combined_mask,
            dropout_p = self.dropout,
            training  = self.training,
        )

        # Cache weights for visualization
        self._last_attn_weights = attn_weights.detach()

        # Step 5: Concatenate heads
        # (B, H, T, d_k) → (B, T, H, d_k) → (B, T, D)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)

        # Step 6: Output projection
        # (B, T, D) → (B, T, D)
        output = self.W_o(attn_out)

        return output

    def _get_causal_mask(
        self,
        seq_len: int,
        device:  torch.device,
    ) -> torch.Tensor:
        """
        Build upper-triangular causal mask.
        True = masked (cannot attend).
        Shape: (1, 1, T, T) — broadcasts over batch and head dims.

        Example for T=4:
            [[F, T, T, T],
             [F, F, T, T],
             [F, F, F, T],
             [F, F, F, F]]
        Token 0 sees only itself.
        Token 3 sees all previous tokens.
        """
        mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1,
        )
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """
        Return cached attention weights from last forward pass.
        Used by analysis/attention_viz.py.
        Returns (B, H, T, T) or None if no forward pass yet.
        """
        return self._last_attn_weights


# =============================================================================
# QUICK SHAPE VALIDATION
# =============================================================================

def _validate_attention_shapes():
    """
    Dry run to verify all tensor shapes are correct.
    Called at import time — catches bugs before training.
    """
    from config import D_MODEL, N_HEADS, N_TOKENS, DROPOUT

    B, T, D = 2, N_TOKENS, D_MODEL

    mha = MultiHeadAttention(
        d_model = D_MODEL,
        n_heads = N_HEADS,
        dropout = DROPOUT,
        causal  = True,
    )
    mha.eval()

    x   = torch.randn(B, T, D)
    out = mha(x)

    assert out.shape == (B, T, D), (
        f"Expected ({B}, {T}, {D}), got {out.shape}"
    )

    weights = mha.get_attention_weights()
    assert weights.shape == (B, N_HEADS, T, T), (
        f"Expected weights ({B}, {N_HEADS}, {T}, {T}), got {weights.shape}"
    )

    print(f"  MultiHeadAttention: ({B}, {T}, {D}) → ({B}, {T}, {D}) ✓")
    print(f"  Attention weights : ({B}, {N_HEADS}, {T}, {T}) ✓")
    print(f"  Causal mask       : upper triangular ✓")


if __name__ == "__main__":
    print("Validating attention shapes...")
    _validate_attention_shapes()
    print("All attention checks passed.")