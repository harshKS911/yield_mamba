"""
train.py — Main Entry Point
============================
All hyperparameters live in hyperparameters.yml.
Edit that file, then run:

    python train.py
    python train.py --config hyperparameter.yml           # explicit path
    python train.py --config experiments/big_model.yml   # different experiment
"""

import argparse
import os
import random

import numpy as np
import torch

from config import load_config
from tokenizer import SMILESTokenizer
from dataset import build_dataloaders
from model import ReactionYieldPredictor
from trainer import Trainer


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ── CLI (only the config path — everything else is in the YAML) ───────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config", type=str, default="hyperparameter.yml",
        help="Path to YAML config file (default: hyperparameter.yml)"
    )
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Load all hyperparameters from YAML ────────────────────────────────────
    cfg = load_config(args.config)
    set_seed(cfg.training.seed)

    # ── Tokenizer (un-fitted; will be built from dataset) ────────────────────
    tokenizer = SMILESTokenizer(max_len=cfg.data.max_smiles_len)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_dl, val_dl, test_dl, tokenizer = build_dataloaders(cfg, tokenizer)

    # ── Update vocab size in mamba config AFTER tokenizer is built ────────────
    cfg.mamba.vocab_size = tokenizer.vocab_size
    print(f"[Main] Vocab size → {cfg.mamba.vocab_size}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ReactionYieldPredictor(cfg)
    model.print_summary()

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        cfg      = cfg,
        model    = model,
        train_dl = train_dl,
        val_dl   = val_dl,
        test_dl  = test_dl,
    )

    # ── Auto-resume if enabled in YAML ────────────────────────────────────────
    best_path = os.path.join(cfg.checkpoint.save_dir, "model_best.pt")
    if cfg.checkpoint.resume and os.path.exists(best_path):
        print(f"[Main] Resuming from {best_path}")
        trainer.load(best_path)

    trainer.train()


if __name__ == "__main__":
    main()