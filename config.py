"""
config.py — Load, validate and expose hyperparameters from hyperparameters.yml
All other modules import from here; YAML is the single source of truth.
"""

import os
import yaml
import torch
from dataclasses import dataclass, field, asdict
from typing import Optional, List


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses (one per YAML section)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    csv_path:       str   = "data.csv"
    val_split:      float = 0.10
    test_split:     float = 0.10
    num_workers:    int   = 4
    max_smiles_len: int   = 128
    min_token_freq: int   = 2


@dataclass
class MambaConfig:
    hidden_size:    int   = 256
    num_layers:     int   = 8
    d_state:        int   = 16
    d_conv:         int   = 4
    expand:         int   = 2
    dt_rank:        str   = "auto"
    dt_min:         float = 0.001
    dt_max:         float = 0.1
    pad_token_id:   int   = 0
    mask_token_id:  int   = 1
    vocab_size:     int   = 512     # set automatically after tokenizer is built


@dataclass
class FusionConfig:
    hidden_size:    int   = 256
    num_heads:      int   = 8
    num_layers:     int   = 3
    ffn_dim:        int   = 512
    dropout:        float = 0.10
    num_components: int   = 4


@dataclass
class TrainingConfig:
    epochs:              int   = 100
    batch_size:          int   = 64
    seed:                int   = 42
    optimizer:           str   = "adamw"
    lr:                  float = 3e-4
    weight_decay:        float = 1e-4
    betas:               List  = field(default_factory=lambda: [0.9, 0.95])
    eps:                 float = 1e-8
    scheduler:           str   = "cosine"
    warmup_steps:        int   = 200
    min_lr_ratio:        float = 0.05
    grad_clip:           float = 1.0
    label_smoothing:     float = 0.0
    loss:                str   = "huber"
    huber_delta:         float = 10.0
    accumulation_steps:  int   = 1
    early_stop_patience: int   = 15
    log_every_n_steps:   int   = 10


@dataclass
class HardwareConfig:
    device:             str  = "auto"
    fp16:               bool = True
    grad_checkpointing: bool = True
    compile_model:      bool = False
    cudnn_benchmark:    bool = True


@dataclass
class CheckpointConfig:
    save_dir:   str  = "checkpoints"
    save_every: int  = 5
    resume:     bool = False


@dataclass
class WandbConfig:
    project:       str           = "reaction-yield-mamba"
    entity:        Optional[str] = None
    run_name:      Optional[str] = None
    tags:          List[str]     = field(default_factory=lambda: ["mamba", "yield"])
    log_gradients: bool          = True
    log_freq:      int           = 100
    offline:       bool          = False


# ─────────────────────────────────────────────────────────────────────────────
# Master Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    data:       DataConfig       = field(default_factory=DataConfig)
    mamba:      MambaConfig      = field(default_factory=MambaConfig)
    fusion:     FusionConfig     = field(default_factory=FusionConfig)
    training:   TrainingConfig   = field(default_factory=TrainingConfig)
    hardware:   HardwareConfig   = field(default_factory=HardwareConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    wandb:      WandbConfig      = field(default_factory=WandbConfig)
    device:     str              = "cpu"   # resolved from hardware.device

    def to_dict(self) -> dict:
        return {
            "data":       asdict(self.data),
            "mamba":      asdict(self.mamba),
            "fusion":     asdict(self.fusion),
            "training":   asdict(self.training),
            "hardware":   asdict(self.hardware),
            "checkpoint": asdict(self.checkpoint),
            "wandb":      asdict(self.wandb),
        }

    def validate(self):
        """Catch common misconfiguration early with clear messages."""

        assert self.mamba.hidden_size == self.fusion.hidden_size, (
            f"mamba.hidden_size ({self.mamba.hidden_size}) must equal "
            f"fusion.hidden_size ({self.fusion.hidden_size})"
        )
        assert self.fusion.ffn_dim >= self.fusion.hidden_size, (
            f"fusion.ffn_dim ({self.fusion.ffn_dim}) should be "
            f">= fusion.hidden_size ({self.fusion.hidden_size})"
        )
        assert self.fusion.hidden_size % self.fusion.num_heads == 0, (
            f"fusion.hidden_size ({self.fusion.hidden_size}) must be divisible "
            f"by fusion.num_heads ({self.fusion.num_heads})"
        )
        assert 0.0 < self.data.val_split < 1.0, \
            "data.val_split must be in (0, 1)"
        assert 0.0 < self.data.test_split < 1.0, \
            "data.test_split must be in (0, 1)"
        assert (self.data.val_split + self.data.test_split) < 1.0, \
            "val_split + test_split must be < 1.0"
        assert self.training.loss in ("huber", "mse", "mae"), \
            f"Unsupported loss: '{self.training.loss}'. Choose: huber | mse | mae"
        assert self.training.optimizer in ("adamw", "adam", "sgd"), \
            f"Unsupported optimizer: '{self.training.optimizer}'"
        assert self.training.scheduler in ("cosine", "linear", "constant"), \
            f"Unsupported scheduler: '{self.training.scheduler}'"
        assert self.training.accumulation_steps >= 1, \
            "accumulation_steps must be >= 1"
        assert self.training.lr > 0, \
            "lr must be positive"

        return self


# ─────────────────────────────────────────────────────────────────────────────
# YAML → Dataclass helper
# ─────────────────────────────────────────────────────────────────────────────

def _fill(dataclass_type, yml_section: dict):
    """
    Fill a dataclass from a YAML dict section.
    - Keys in YAML but not in dataclass are silently ignored.
    - Keys missing from YAML fall back to dataclass defaults.
    """
    instance = dataclass_type()
    valid_fields = {f.name for f in instance.__dataclass_fields__.values()}
    for key, val in yml_section.items():
        if key in valid_fields:
            setattr(instance, key, val)
    return instance


# ─────────────────────────────────────────────────────────────────────────────
# Public loader  ← only function you call from outside
# ─────────────────────────────────────────────────────────────────────────────

def load_config(yaml_path: str = "hyperparameters.yml") -> Config:
    """
    Load hyperparameters.yml and return a fully validated Config object.

    Usage:
        from config import load_config
        cfg = load_config("hyperparameters.yml")
    """
    if not os.path.exists(yaml_path):
        print(f"[Config] WARNING: '{yaml_path}' not found — using all defaults.")
        raw = {}
    else:
        with open(yaml_path, "r") as f:
            raw = yaml.safe_load(f) or {}
        print(f"[Config] Loaded from '{yaml_path}'")

    cfg = Config(
        data       = _fill(DataConfig,       raw.get("data",       {})),
        mamba      = _fill(MambaConfig,      raw.get("mamba",      {})),
        fusion     = _fill(FusionConfig,     raw.get("fusion",     {})),
        training   = _fill(TrainingConfig,   raw.get("training",   {})),
        hardware   = _fill(HardwareConfig,   raw.get("hardware",   {})),
        checkpoint = _fill(CheckpointConfig, raw.get("checkpoint", {})),
        wandb      = _fill(WandbConfig,      raw.get("wandb",      {})),
    )

    # ── Resolve device ────────────────────────────────────────────────────────
    dev = cfg.hardware.device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.device = dev

    # ── Apply global torch settings ───────────────────────────────────────────
    if cfg.hardware.cudnn_benchmark and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    cfg.validate()
    _print_summary(cfg)
    return cfg


def _print_summary(cfg: Config):
    sep = "═" * 52
    print(f"\n{sep}")
    print("  Hyperparameter Summary")
    print(sep)
    print(f"  Data CSV          : {cfg.data.csv_path}")
    print(f"  Device            : {cfg.device}")
    print(f"  AMP FP16          : {cfg.hardware.fp16}")
    print(f"  Grad Checkpointing: {cfg.hardware.grad_checkpointing}")
    print(f"  Mamba hidden      : {cfg.mamba.hidden_size}")
    print(f"  Mamba layers      : {cfg.mamba.num_layers}  (d_state={cfg.mamba.d_state})")
    print(f"  Fusion heads      : {cfg.fusion.num_heads}")
    print(f"  Fusion layers     : {cfg.fusion.num_layers}  (ffn_dim={cfg.fusion.ffn_dim})")
    print(f"  Batch size        : {cfg.training.batch_size}  "
          f"(accum={cfg.training.accumulation_steps}  "
          f"eff={cfg.training.batch_size * cfg.training.accumulation_steps})")
    print(f"  LR                : {cfg.training.lr}  ({cfg.training.scheduler} schedule)")
    print(f"  Warmup steps      : {cfg.training.warmup_steps}")
    print(f"  Epochs            : {cfg.training.epochs}")
    print(f"  Loss              : {cfg.training.loss}  (δ={cfg.training.huber_delta})")
    print(f"  Early stop        : {cfg.training.early_stop_patience} epochs")
    print(f"  WandB project     : {cfg.wandb.project}")
    print(f"  Save dir          : {cfg.checkpoint.save_dir}")
    print(f"{sep}\n")