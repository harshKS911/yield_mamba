"""
dataset.py — Reaction Yield Dataset
Reads ORD-style CSV safely, builds per-component tensors for all 4 roles.
"""

import os
import csv
import pandas as pd
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader, random_split

from tokenizer import SMILESTokenizer
from config import Config

COMPONENT_COLS = {
    "reactant": "Reactant_SMILES_new",
    "product":  "Product_SMILES_new",
    "reagent":  "Reagent_SMILES_new",
    "catalyst": "Catalyst_SMILES_new",
}
YIELD_COL = "Yield_Clipped"


# ── Safe CSV reader — bypasses pandas/numpy C-parser bug ─────────────────────

def _read_csv_safe(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows   = list(reader)
    return pd.DataFrame(rows)


# ── Dataset ───────────────────────────────────────────────────────────────────

class ReactionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: SMILESTokenizer):
        self.df        = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self._validate()

    def _validate(self):
        for name, col in COMPONENT_COLS.items():
            if col not in self.df.columns:
                raise ValueError(f"Missing column '{col}' for component '{name}'")
        if YIELD_COL not in self.df.columns:
            raise ValueError(f"Missing yield column: {YIELD_COL}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row  = self.df.iloc[idx]
        item = {}
        for name, col in COMPONENT_COLS.items():
            smi = str(row[col]) if pd.notna(row[col]) else ""
            ids, mask = self.tokenizer.encode(smi)
            item[f"{name}_ids"]  = ids
            item[f"{name}_mask"] = mask

        yield_val = float(row[YIELD_COL])
        yield_val = max(0.0, min(100.0, yield_val))
        item["yield_label"] = torch.tensor([yield_val], dtype=torch.float32)
        return item


# ── Collate ───────────────────────────────────────────────────────────────────

def collate_fn(batch):
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}


# ── Build DataLoaders ─────────────────────────────────────────────────────────

def build_dataloaders(
    cfg: Config,
    tokenizer: SMILESTokenizer,
) -> Tuple[DataLoader, DataLoader, DataLoader, SMILESTokenizer]:

    # ── Load CSV safely ───────────────────────────────────────────────────────
    df = _read_csv_safe(cfg.data.csv_path)
    print(f"[Dataset] Loaded {len(df)} rows from '{cfg.data.csv_path}'")

    # ── Cast numerics ─────────────────────────────────────────────────────────
    df[YIELD_COL] = pd.to_numeric(df[YIELD_COL], errors="coerce")
    if "Yield" in df.columns:
        df["Yield"] = pd.to_numeric(df["Yield"], errors="coerce")

    # ── Drop missing yields ───────────────────────────────────────────────────
    df = df.dropna(subset=[YIELD_COL]).reset_index(drop=True)
    print(f"[Dataset] After dropping NaN yields: {len(df)} rows")

    # ── Build tokenizer ───────────────────────────────────────────────────────
    if tokenizer.vocab_size <= 5:
        all_smiles = []
        for col in COMPONENT_COLS.values():
            if col in df.columns:
                all_smiles.extend(df[col].dropna().tolist())
        tokenizer = SMILESTokenizer.build_from_smiles(
            all_smiles,
            max_len=cfg.data.max_smiles_len,
            min_freq=cfg.data.min_token_freq,
        )
        os.makedirs(cfg.checkpoint.save_dir, exist_ok=True)
        tok_path = os.path.join(cfg.checkpoint.save_dir, "tokenizer.json")
        tokenizer.save(tok_path)
        print(f"[Dataset] Tokenizer saved → {tok_path}")

    # ── Splits ────────────────────────────────────────────────────────────────
    full_ds = ReactionDataset(df, tokenizer)
    n       = len(full_ds)
    n_test  = max(1, int(n * cfg.data.test_split))
    n_val   = max(1, int(n * cfg.data.val_split))
    n_train = n - n_val - n_test

    gen = torch.Generator().manual_seed(cfg.training.seed)
    train_ds, val_ds, test_ds = random_split(
        full_ds, [n_train, n_val, n_test], generator=gen
    )
    print(f"[Dataset] Splits → train={n_train}  val={n_val}  test={n_test}")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    loader_kwargs = dict(
        batch_size         = cfg.training.batch_size,
        num_workers        = cfg.data.num_workers,
        collate_fn         = collate_fn,
        pin_memory         = (cfg.device == "cuda"),
        persistent_workers = (cfg.data.num_workers > 0),
    )
    train_dl = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_dl   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_dl  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    return train_dl, val_dl, test_dl, tokenizer