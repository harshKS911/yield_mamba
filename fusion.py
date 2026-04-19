"""
fusion.py — Multi-Component Fusion Attention
=============================================
Fuses embeddings from 4 reaction components (reactant, product,
reagent, catalyst) using stacked cross-attention transformer layers,
then projects to a scalar yield prediction.

Architecture:
  ┌─────────────────────────────────────────────┐
  │  Per-component mol_emb (B, hidden) × 4      │
  │              ↓ stack as sequence             │
  │     Component Sequence  (B, 4, hidden)       │
  │              ↓                               │
  │   N × FusionTransformerLayer                 │
  │   (self-attention across components          │
  │    + cross-attention to token sequences)     │
  │              ↓                               │
  │   Global Mean Pool  →  (B, hidden)           │
  │              ↓                               │
  │   YieldHead  →  (B, 1)                       │
  └─────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from config import FusionConfig


# ── Scaled Dot-Product Attention (SDPA, uses Flash Attention if available) ───

class MultiHeadAttention(nn.Module):
    """Standard MHA — PyTorch's F.scaled_dot_product_attention uses
    FlashAttention2 kernels automatically on CUDA with torch >= 2.0."""

    def __init__(self, hidden: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert hidden % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = hidden // num_heads
        self.scale     = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.out    = nn.Linear(hidden, hidden, bias=False)
        self.drop   = nn.Dropout(dropout)

    def forward(self,
                query: torch.Tensor,        # (B, Tq, H)
                key:   torch.Tensor,        # (B, Tk, H)
                value: torch.Tensor,        # (B, Tk, H)
                key_padding_mask: torch.Tensor = None,  # (B, Tk) True=ignore
                ) -> torch.Tensor:          # (B, Tq, H)
        B, Tq, H = query.shape
        _, Tk, _  = key.shape
        nh, hd    = self.num_heads, self.head_dim

        q = self.q_proj(query).view(B, Tq, nh, hd).transpose(1, 2)  # (B,nh,Tq,hd)
        k = self.k_proj(key  ).view(B, Tk, nh, hd).transpose(1, 2)  # (B,nh,Tk,hd)
        v = self.v_proj(value).view(B, Tk, nh, hd).transpose(1, 2)  # (B,nh,Tk,hd)

        # Build attn_mask from key_padding_mask
        attn_mask = None
        if key_padding_mask is not None:
            # (B, 1, 1, Tk) — True means "mask out (set to -inf)"
            attn_mask = (~key_padding_mask).unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.expand(B, nh, Tq, Tk)
            attn_mask = attn_mask.to(dtype=query.dtype) * -1e9

        # Use PyTorch's built-in SDPA (FlashAttention2 if available)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.drop.p if self.training else 0.0,
        )                                                   # (B, nh, Tq, hd)

        out = out.transpose(1, 2).contiguous().view(B, Tq, H)
        return self.out(out)                               # (B, Tq, H)


# ── FusionTransformerLayer ────────────────────────────────────────────────────

class FusionTransformerLayer(nn.Module):
    """
    One fusion layer:
      1. Self-attention across the 4 component embeddings
      2. Cross-attention: each component attends to all token sequences
      3. FFN

    This lets each component 'see' both the other components (self-attn)
    and the full token-level detail of each component (cross-attn).
    """

    def __init__(self, cfg: FusionConfig):
        super().__init__()
        H   = cfg.hidden_size
        nh  = cfg.num_heads
        dp  = cfg.dropout
        ffn = cfg.ffn_dim

        # ── Self-attention over component sequence ────────────────────────────
        self.self_norm = nn.LayerNorm(H)
        self.self_attn = MultiHeadAttention(H, nh, dp)

        # ── Cross-attention: component queries ← token-sequence keys/values ──
        # One cross-attention per component (to its own token sequence)
        self.cross_norm  = nn.LayerNorm(H)
        self.cross_attn  = MultiHeadAttention(H, nh, dp)

        # ── FFN ───────────────────────────────────────────────────────────────
        self.ffn_norm = nn.LayerNorm(H)
        self.ffn      = nn.Sequential(
            nn.Linear(H, ffn),
            nn.GELU(),
            nn.Dropout(dp),
            nn.Linear(ffn, H),
            nn.Dropout(dp),
        )

    def forward(self,
                comp_seq: torch.Tensor,          # (B, 4, H)  component embeddings
                token_seqs: Dict[str, torch.Tensor],   # name → (B, T, H)
                token_masks: Dict[str, torch.Tensor],  # name → (B, T) bool
                ) -> torch.Tensor:               # (B, 4, H)

        B, C, H = comp_seq.shape

        # ── 1. Self-attention over 4 components ──────────────────────────────
        h = self.self_norm(comp_seq)
        h = self.self_attn(h, h, h)
        comp_seq = comp_seq + h

        # ── 2. Cross-attention: each component → its own token sequence ──────
        comp_names = ["reactant", "product", "reagent", "catalyst"]
        updated = []
        for i, name in enumerate(comp_names):
            qi    = self.cross_norm(comp_seq[:, i:i+1, :])   # (B, 1, H)
            tok_k = token_seqs[name]                          # (B, T, H)
            tok_v = tok_k
            mask  = token_masks[name]                         # (B, T)
            attn_out = self.cross_attn(qi, tok_k, tok_v, mask)  # (B, 1, H)
            updated.append(comp_seq[:, i:i+1, :] + attn_out)

        comp_seq = torch.cat(updated, dim=1)                  # (B, 4, H)

        # ── 3. FFN ────────────────────────────────────────────────────────────
        comp_seq = comp_seq + self.ffn(self.ffn_norm(comp_seq))

        return comp_seq


# ── Yield Prediction Head ─────────────────────────────────────────────────────

class YieldHead(nn.Module):
    """MLP that maps a fused representation → scalar yield (0–100)."""

    def __init__(self, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, hidden // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, hidden) → (B, 1) in [0, 100]"""
        return torch.sigmoid(self.net(x)) * 100.0


# ── Full Fusion Module ────────────────────────────────────────────────────────

class ReactionFusionModel(nn.Module):
    """
    Combines 4 per-component Mamba embeddings via stacked
    fusion transformer layers + yield head.

    Input:
        encoder_outputs : dict returned by SharedReactionEncoder.forward()
          Each value has keys: 'mol_emb' (B, H), 'token_emb' (B, T, H)
        token_masks     : dict  name → (B, T) bool mask

    Output:
        yield_pred : (B, 1) — predicted reaction yield in [0, 100]
        fused_emb  : (B, H) — fused reaction embedding
    """

    COMPONENT_NAMES = ["reactant", "product", "reagent", "catalyst"]

    def __init__(self, cfg: FusionConfig):
        super().__init__()
        self.cfg    = cfg

        self.layers = nn.ModuleList(
            [FusionTransformerLayer(cfg) for _ in range(cfg.num_layers)]
        )
        self.pool_norm = nn.LayerNorm(cfg.hidden_size)
        self.head      = YieldHead(cfg.hidden_size, cfg.dropout)

    def forward(self,
                encoder_outputs: dict,
                token_masks: dict,
                ) -> dict:

        B = next(iter(encoder_outputs.values()))["mol_emb"].shape[0]

        # ── Build component sequence (B, 4, H) ───────────────────────────────
        comp_seq = torch.stack(
            [encoder_outputs[n]["mol_emb"] for n in self.COMPONENT_NAMES],
            dim=1
        )                                                   # (B, 4, H)

        # ── Token sequences for cross-attention ──────────────────────────────
        token_seqs = {n: encoder_outputs[n]["token_emb"]
                      for n in self.COMPONENT_NAMES}

        # ── Stacked fusion layers ─────────────────────────────────────────────
        for layer in self.layers:
            comp_seq = layer(comp_seq, token_seqs, token_masks)

        # ── Pooling: mean over 4 components ──────────────────────────────────
        fused = self.pool_norm(comp_seq.mean(dim=1))       # (B, H)

        # ── Yield prediction ──────────────────────────────────────────────────
        yield_pred = self.head(fused)                      # (B, 1)

        return {
            "yield_pred": yield_pred,
            "fused_emb":  fused,
        }
