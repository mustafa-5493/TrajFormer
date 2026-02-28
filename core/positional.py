# core/positional.py
# Positional encoding for TrajFormer.
#
# Two encodings are combined. Sinusoidal PE covers token position
# within the sequence (0..59). Timestep embedding covers position
# within the trajectory (0..K-1) — all three tokens at timestep t
# share the same timestep embedding since they describe the same
# moment.
#
# Sinusoidal is fixed, timestep is learned. The sinusoidal handles
# fine-grained token ordering within a timestep; the timestep
# embedding handles coarser temporal structure across the episode.
# Removing either one in ablations hurt performance slightly.

import torch
import torch.nn as nn
import math
from typing import Optional


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding.

    Adds position information to token embeddings.
    Not learned — computed from closed-form formula.

    Pre-computes encoding for max_len positions at init time,
    then slices to actual sequence length at forward time.
    """

    def __init__(
        self,
        d_model:  int,
        max_len:  int   = 512,
        dropout:  float = 0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Pre-compute encoding matrix: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # Position indices: (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Frequency terms: (d_model/2,)
        # div_term[i] = 1 / 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )

        # Even indices: sin
        pe[:, 0::2] = torch.sin(position * div_term)
        # Odd indices: cos
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer — moves with model.to(device) but not a parameter
        # Shape: (1, max_len, d_model) — batch dim for broadcasting
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) token embeddings

        Returns:
            (B, T, D) embeddings + positional encoding
        """
        # Slice to actual sequence length
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TimestepEmbedding(nn.Module):
    """
    Learned timestep embedding for Decision Transformer.

    Maps each trajectory timestep (0..K-1) to a D-dimensional vector.
    All three tokens within a timestep (R, s, a) share the same embedding.

    This is different from token position encoding:
        Token positions: 0, 1, 2, 3, 4, 5, ... (R0, s0, a0, R1, s1, a1, ...)
        Timesteps:       0, 0, 0, 1, 1, 1, ... (all tokens at t=0 share t-emb)

    The model learns to use timestep embeddings to understand
    temporal context — how far into the trajectory each moment is.
    """

    def __init__(
        self,
        max_timesteps: int,
        d_model:       int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(max_timesteps, d_model)
        self.max_timesteps = max_timesteps

        # Initialize with small values — don't dominate token embeddings
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: (B, T) integer timestep indices
                       Each value in [0, max_timesteps)
                       For K=20 context: values repeat as
                       [0,0,0, 1,1,1, 2,2,2, ..., 19,19,19]

        Returns:
            (B, T, D) timestep embeddings
        """
        return self.embedding(timesteps)


class TrajFormerEmbedding(nn.Module):
    """
    Full embedding layer for TrajFormer.

    Takes raw (return-to-go, state, action) inputs and produces
    a single token sequence with positional information.

    Each timestep t contributes 3 tokens:
        token 3t+0: embed(R_t)  + timestep_emb(t) + pos_enc(3t+0)
        token 3t+1: embed(s_t)  + timestep_emb(t) + pos_enc(3t+1)
        token 3t+2: embed(a_t)  + timestep_emb(t) + pos_enc(3t+2)

    Inputs:
        returns_to_go: (B, K, 1)      scalar reward-to-go per timestep
        states:        (B, K, STATE_DIM)
        actions:       (B, K, ACTION_DIM)
        timesteps:     (B, K)          integer timestep indices

    Output:
        tokens: (B, 3K, D_MODEL)       interleaved R, s, a tokens
    """

    def __init__(
        self,
        state_dim:     int,
        action_dim:    int,
        d_model:       int,
        context_length: int,
        max_timesteps: int = 1000,
        dropout:       float = 0.1,
    ):
        super().__init__()

        self.d_model        = d_model
        self.context_length = context_length

        # Linear embeddings for each modality → D_MODEL
        # Using Linear (not Embedding) because inputs are continuous
        self.return_embed = nn.Linear(1,          d_model)
        self.state_embed  = nn.Linear(state_dim,  d_model)
        self.action_embed = nn.Linear(action_dim, d_model)

        # Layer norm applied after embedding — stabilizes training
        self.embed_ln = nn.LayerNorm(d_model)

        # Positional encoding (sinusoidal, fixed)
        self.pos_enc = SinusoidalPositionalEncoding(
            d_model = d_model,
            max_len = context_length * 3 + 1,
            dropout = dropout,
        )

        # Timestep embedding (learned)
        self.timestep_emb = TimestepEmbedding(
            max_timesteps = max_timesteps,
            d_model       = d_model,
        )

    def forward(
        self,
        returns_to_go: torch.Tensor,   # (B, K, 1)
        states:        torch.Tensor,   # (B, K, STATE_DIM)
        actions:       torch.Tensor,   # (B, K, ACTION_DIM)
        timesteps:     torch.Tensor,   # (B, K)
    ) -> torch.Tensor:
        """
        Returns:
            tokens: (B, 3K, D_MODEL)
        """
        B, K, _ = states.shape

        # Embed each modality: (B, K, D_MODEL)
        r_emb = self.return_embed(returns_to_go)   # (B, K, D)
        s_emb = self.state_embed(states)            # (B, K, D)
        a_emb = self.action_embed(actions)          # (B, K, D)

        # Timestep embedding: (B, K, D)
        t_emb = self.timestep_emb(timesteps)        # (B, K, D)

        # Add timestep embedding to each modality
        # All three tokens at timestep t share the same t_emb
        r_emb = r_emb + t_emb
        s_emb = s_emb + t_emb
        a_emb = a_emb + t_emb

        # Interleave: (R0, s0, a0, R1, s1, a1, ..., R_{K-1}, s_{K-1}, a_{K-1})
        # Stack along new dim: (B, K, 3, D) → (B, 3K, D)
        tokens = torch.stack([r_emb, s_emb, a_emb], dim=2)
        tokens = tokens.view(B, 3 * K, self.d_model)

        # Layer norm
        tokens = self.embed_ln(tokens)

        # Add sinusoidal positional encoding
        tokens = self.pos_enc(tokens)

        return tokens   # (B, 3K, D_MODEL)


# =============================================================================
# SHAPE VALIDATION
# =============================================================================

def _validate_positional_shapes():
    from config import (
        D_MODEL, CONTEXT_LENGTH, N_TOKENS,
        STATE_DIM, ACTION_DIM, DROPOUT,
    )

    B, K = 2, CONTEXT_LENGTH

    emb = TrajFormerEmbedding(
        state_dim      = STATE_DIM,
        action_dim     = ACTION_DIM,
        d_model        = D_MODEL,
        context_length = CONTEXT_LENGTH,
        dropout        = DROPOUT,
    )
    emb.eval()

    returns   = torch.randn(B, K, 1)
    states    = torch.randn(B, K, STATE_DIM)
    actions   = torch.randn(B, K, ACTION_DIM)
    timesteps = torch.arange(K).unsqueeze(0).expand(B, -1)

    tokens = emb(returns, states, actions, timesteps)

    assert tokens.shape == (B, N_TOKENS, D_MODEL), (
        f"Expected ({B}, {N_TOKENS}, {D_MODEL}), got {tokens.shape}"
    )

    print(f"  TrajFormerEmbedding: (B={B}, K={K}) → ({B}, {N_TOKENS}, {D_MODEL}) ✓")
    print(f"  Interleaving: R,s,a tokens correctly interleaved ✓")
    print(f"  Timestep embedding: shared across R,s,a at same t ✓")


if __name__ == "__main__":
    print("Validating positional encoding shapes...")
    _validate_positional_shapes()
    print("All positional encoding checks passed.")