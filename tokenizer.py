"""
tokenizer.py — SMILES Tokenizer with learnable vocabulary
Builds vocab from dataset; handles all 4 reaction components uniformly.
"""

import re
import json
import os
from collections import Counter
from typing import List, Optional

import torch
import numpy as np


# ── Regex: atom-level SMILES tokenisation (Schwaller et al.) ─────────────────
_SMILES_PATTERN = re.compile(
    r"(\[[^\]]+\]"       # bracketed atoms  e.g. [NH3+]
    r"|Br|Cl|Si|Se|@@"   # two-char atoms / stereo
    r"|[BCNOPSFIbcnopsfih]"  # single-char atoms (aromatic + aliphatic)
    r"|[=\-\+\#\(\)\/\\@\.\%]"  # bonds and special chars
    r"|\d+"              # ring-closure numbers
    r")"
)

SPECIAL_TOKENS = {
    "[PAD]": 0,
    "[MASK]": 1,
    "[CLS]": 2,
    "[SEP]": 3,
    "[UNK]": 4,
}
NUM_SPECIAL = len(SPECIAL_TOKENS)


class SMILESTokenizer:
    """
    Tokenizer that:
     1. Splits SMILES string into atom/bond tokens via regex.
     2. Maps tokens to integer ids using a learned vocabulary.
     3. Handles multi-component SMILES strings (dot-separated mixtures).
     4. Pads / truncates to fixed max_len.
    """

    def __init__(self,
                 vocab: Optional[dict] = None,
                 max_len: int = 128):
        self.max_len = max_len
        self.vocab   = vocab if vocab is not None else dict(SPECIAL_TOKENS)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    # ── Vocabulary ────────────────────────────────────────────────────────────

    @classmethod
    def build_from_smiles(cls,
                          smiles_list: List[str],
                          max_len: int = 128,
                          min_freq: int = 2) -> "SMILESTokenizer":
        """Build vocabulary from a list of SMILES strings."""
        counter: Counter = Counter()
        tokenizer = cls(max_len=max_len)
        for smi in smiles_list:
            if not isinstance(smi, str) or not smi.strip():
                continue
            tokens = tokenizer.tokenize(smi)
            counter.update(tokens)

        vocab = dict(SPECIAL_TOKENS)
        for tok, freq in counter.most_common():
            if freq >= min_freq and tok not in vocab:
                vocab[tok] = len(vocab)
        tokenizer.vocab = vocab
        tokenizer.inv_vocab = {v: k for k, v in vocab.items()}
        print(f"[Tokenizer] Vocabulary size: {len(vocab)}")
        return tokenizer

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"vocab": self.vocab, "max_len": self.max_len}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SMILESTokenizer":
        with open(path) as f:
            data = json.load(f)
        tok = cls(vocab=data["vocab"], max_len=data["max_len"])
        tok.inv_vocab = {v: k for k, v in tok.vocab.items()}
        return tok

    # ── Tokenisation ──────────────────────────────────────────────────────────

    def tokenize(self, smiles: str) -> List[str]:
        """Split a SMILES string (possibly multi-component) into token list."""
        if not smiles or not isinstance(smiles, str):
            return []
        return _SMILES_PATTERN.findall(smiles.strip())

    def encode(self,
               smiles: str,
               add_cls: bool = True,
               pad: bool = True,
               return_mask: bool = True):
        """
        Encode a SMILES string to a padded tensor.

        Returns:
            ids   : LongTensor  (max_len,)
            mask  : BoolTensor  (max_len,)  True = valid token
        """
        tokens = self.tokenize(smiles)
        unk_id = self.vocab.get("[UNK]", 4)

        # Convert tokens → ids
        ids = [self.vocab.get(t, unk_id) for t in tokens]

        # Prepend [CLS]
        if add_cls:
            cls_id = self.vocab.get("[CLS]", 2)
            ids = [cls_id] + ids

        # Truncate
        ids = ids[:self.max_len]

        # Mask (1 = real, 0 = pad)
        length = len(ids)
        mask   = [True] * length

        # Pad
        if pad:
            pad_id = self.vocab.get("[PAD]", 0)
            ids  += [pad_id]  * (self.max_len - length)
            mask += [False]   * (self.max_len - length)

        return (
            torch.tensor(ids,  dtype=torch.long),
            torch.tensor(mask, dtype=torch.bool),
        )

    def encode_batch(self, smiles_list: List[str]):
        """Encode a list of SMILES into batched tensors."""
        ids_list, mask_list = [], []
        for smi in smiles_list:
            ids, mask = self.encode(smi)
            ids_list.append(ids)
            mask_list.append(mask)
        return torch.stack(ids_list), torch.stack(mask_list)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
