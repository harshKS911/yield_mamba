"""
model.py — Full Reaction Yield Prediction Model
Wraps SharedReactionEncoder + ReactionFusionModel into one nn.Module.
"""

import torch
import torch.nn as nn

from config import Config
from mamba_encoder import SharedReactionEncoder
from fusion import ReactionFusionModel


class ReactionYieldPredictor(nn.Module):
    """
    End-to-end model:
        [4 × SMILES token sequences]
                ↓ SharedReactionEncoder (shared Mamba weights + role embeddings)
        [4 × (token_emb, mol_emb)]
                ↓ ReactionFusionModel (stacked cross-attention fusion)
        [yield_pred: (B,1), fused_emb: (B,H)]
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.encoder = SharedReactionEncoder(
            cfg.mamba,
            use_grad_checkpoint=cfg.hardware.grad_checkpointing,
        )
        self.fusion  = ReactionFusionModel(cfg.fusion)

    def forward(self, batch: dict) -> dict:
        """
        batch: dict with keys
          {reactant_ids, reactant_mask, product_ids, product_mask,
           reagent_ids, reagent_mask, catalyst_ids, catalyst_mask}

        Returns: {yield_pred (B,1), fused_emb (B,H)}
        """
        # ── Encode all 4 components ───────────────────────────────────────────
        enc_out = self.encoder(batch)                       # per-component dicts

        # ── Build token_masks for cross-attention ─────────────────────────────
        token_masks = {
            name: batch[f"{name}_mask"]
            for name in self.encoder.COMPONENT_NAMES
        }

        # ── Fuse + predict ────────────────────────────────────────────────────
        return self.fusion(enc_out, token_masks)

    # ── Convenience ──────────────────────────────────────────────────────────

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def print_summary(self):
        enc_params   = sum(p.numel() for p in self.encoder.parameters()) / 1e6
        fusion_params= sum(p.numel() for p in self.fusion.parameters())  / 1e6
        total        = self.num_parameters / 1e6
        print(f"┌─────────────────────────────────────┐")
        print(f"│  ReactionYieldPredictor              │")
        print(f"│  Encoder params  : {enc_params:6.2f} M          │")
        print(f"│  Fusion params   : {fusion_params:6.2f} M          │")
        print(f"│  Total params    : {total:6.2f} M          │")
        print(f"└─────────────────────────────────────┘")