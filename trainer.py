"""
trainer.py — GPU-Optimised Training Loop
All settings read from Config (populated from hyperparameters.yml).

Features:
  ✓ AMP mixed precision          ✓ Gradient checkpointing
  ✓ Cosine / linear / constant LR schedule with warmup
  ✓ Huber / MSE / MAE loss       ✓ Gradient clipping + accumulation
  ✓ WandB: loss, MAE, RMSE, R²   ✓ Early stopping + checkpointing
"""

import os
import math
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

import wandb

from config import Config
from model import ReactionYieldPredictor


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    mae  = float(np.abs(preds - targets).mean())
    rmse = float(np.sqrt(((preds - targets) ** 2).mean()))
    ss_res = ((targets - preds) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    r2   = float(1 - ss_res / (ss_tot + 1e-8))
    return {"mae": mae, "rmse": rmse, "r2": r2}


# ── LR Schedules ─────────────────────────────────────────────────────────────

def _cosine_lambda(step, warmup, total, min_ratio):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

def _linear_lambda(step, warmup, total, min_ratio):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return max(min_ratio, 1.0 - progress * (1.0 - min_ratio))

def _constant_lambda(step, warmup, *_):
    return step / max(1, warmup) if step < warmup else 1.0


# ── Loss factory ─────────────────────────────────────────────────────────────

def build_loss(cfg: Config) -> nn.Module:
    name = cfg.training.loss
    if name == "huber":
        return nn.HuberLoss(delta=cfg.training.huber_delta)
    elif name == "mse":
        return nn.MSELoss()
    elif name == "mae":
        return nn.L1Loss()
    raise ValueError(f"Unknown loss: {name}")


# ── Optimizer factory ────────────────────────────────────────────────────────

def build_optimizer(cfg: Config, model: nn.Module):
    decay   = [p for n, p in model.named_parameters()
                if p.requires_grad and "norm" not in n and "bias" not in n]
    nodecay = [p for n, p in model.named_parameters()
                if p.requires_grad and ("norm" in n or "bias" in n)]

    param_groups = [
        {"params": decay,   "weight_decay": cfg.training.weight_decay},
        {"params": nodecay, "weight_decay": 0.0},
    ]
    name = cfg.training.optimizer
    betas = tuple(cfg.training.betas)

    if name == "adamw":
        use_fused = cfg.device == "cuda"
        try:
            return torch.optim.AdamW(
                param_groups, lr=cfg.training.lr,
                betas=betas, eps=cfg.training.eps, fused=use_fused
            )
        except TypeError:
            return torch.optim.AdamW(
                param_groups, lr=cfg.training.lr,
                betas=betas, eps=cfg.training.eps
            )
    elif name == "adam":
        return torch.optim.Adam(
            param_groups, lr=cfg.training.lr,
            betas=betas, eps=cfg.training.eps
        )
    elif name == "sgd":
        return torch.optim.SGD(
            param_groups, lr=cfg.training.lr, momentum=0.9
        )
    raise ValueError(f"Unknown optimizer: {name}")


# ── Scheduler factory ────────────────────────────────────────────────────────

def build_scheduler(cfg: Config, optimizer, total_steps: int):
    warmup     = cfg.training.warmup_steps
    min_ratio  = cfg.training.min_lr_ratio
    name       = cfg.training.scheduler

    fn_map = {
        "cosine":   _cosine_lambda,
        "linear":   _linear_lambda,
        "constant": _constant_lambda,
    }
    fn = fn_map[name]
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: fn(s, warmup, total_steps, min_ratio)
    )


# ── Trainer ───────────────────────────────────────────────────────────────────

class Trainer:

    def __init__(self,
                 cfg:      Config,
                 model:    ReactionYieldPredictor,
                 train_dl: DataLoader,
                 val_dl:   DataLoader,
                 test_dl:  Optional[DataLoader] = None):

        self.cfg      = cfg
        self.model    = model
        self.train_dl = train_dl
        self.val_dl   = val_dl
        self.test_dl  = test_dl
        self.device   = torch.device(cfg.device)

        if self.device.type == "cuda":
            # Improves matmul throughput on Ampere+ and removes TF32 warning.
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True

        model.to(self.device)

        # Optional torch.compile
        if cfg.hardware.compile_model and hasattr(torch, "compile"):
            print("[Trainer] Compiling model with torch.compile …")
            try:
                # Avoid aggressive CUDA graph capture, which can conflict with
                # repeated forwards per step / dynamic training loops.
                self.model = torch.compile(model, mode="max-autotune-no-cudagraphs")
            except Exception:
                self.model = torch.compile(model, mode="default")
        else:
            self.model = model

        self.criterion = build_loss(cfg)
        self.optimizer = build_optimizer(cfg, self.model)

        total_steps = (cfg.training.epochs
                       * len(train_dl)
                       // cfg.training.accumulation_steps)
        self.scheduler = build_scheduler(cfg, self.optimizer, total_steps)

        use_amp = cfg.hardware.fp16 and self.device.type == "cuda"
        self.use_amp = use_amp
        self.scaler  = GradScaler("cuda") if use_amp else None

        self.global_step   = 0
        self.best_val_mae  = float("inf")
        self.no_improve    = 0
        os.makedirs(cfg.checkpoint.save_dir, exist_ok=True)

    # ── WandB ─────────────────────────────────────────────────────────────────

    def _init_wandb(self):
        wc = self.cfg.wandb
        if wc.offline:
            os.environ["WANDB_MODE"] = "offline"
        wandb.login()
        wandb.init(
            project = wc.project,
            entity  = wc.entity,
            name    = wc.run_name,
            tags    = wc.tags,
            config  = self.cfg.to_dict(),
        )
        if wc.log_gradients:
            wandb.watch(self.model, log="gradients", log_freq=wc.log_freq)

    # ── One train step ────────────────────────────────────────────────────────

    def _maybe_mark_cudagraph_step_begin(self):
        marker = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
        if callable(marker):
            marker()

    def _train_step(self, batch: dict):
        self._maybe_mark_cudagraph_step_begin()
        targets = batch["yield_label"].to(self.device, non_blocking=True)
        inputs  = {
            k: v.to(self.device, non_blocking=True)
            for k, v in batch.items() if k != "yield_label"
        }

        with autocast("cuda", enabled=self.use_amp):
            out  = self.model(inputs)
            loss = self.criterion(out["yield_pred"], targets)
            loss = loss / self.cfg.training.accumulation_steps

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        preds_np = out["yield_pred"].detach().cpu().numpy()
        tgts_np  = targets.detach().cpu().numpy()
        return loss.item() * self.cfg.training.accumulation_steps, preds_np, tgts_np

    # ── Train epoch ───────────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        total_loss = 0.0
        all_preds, all_tgts = [], []
        t0 = time.time()
        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(self.train_dl):
            loss, preds_np, tgts_np = self._train_step(batch)
            total_loss += loss

            # Reuse predictions from the train forward pass (no second forward).
            all_preds.append(preds_np)
            all_tgts.append(tgts_np)

            # Gradient step
            if (step + 1) % self.cfg.training.accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                if self.cfg.training.grad_clip > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.training.grad_clip
                    )
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

                # Per-step WandB log
                if self.global_step % self.cfg.training.log_every_n_steps == 0:
                    lr  = self.optimizer.param_groups[0]["lr"]
                    mem = (torch.cuda.memory_reserved(self.device) / 1e9
                           if self.device.type == "cuda" else 0.0)
                    wandb.log({
                        "train/step_loss": loss,
                        "train/lr":        lr,
                        "gpu/mem_GB":      mem,
                    }, step=self.global_step)

        metrics = compute_metrics(
            np.concatenate(all_preds).ravel(),
            np.concatenate(all_tgts).ravel()
        )
        metrics["loss"]          = total_loss / len(self.train_dl)
        metrics["epoch_time_s"]  = time.time() - t0
        return metrics

    # ── Eval epoch ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _eval_epoch(self, dl: DataLoader) -> dict:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_tgts = [], []

        for batch in dl:
            self._maybe_mark_cudagraph_step_begin()
            targets = batch["yield_label"].to(self.device, non_blocking=True)
            inputs  = {
                k: v.to(self.device, non_blocking=True)
                for k, v in batch.items() if k != "yield_label"
            }
            with autocast("cuda", enabled=self.use_amp):
                out  = self.model(inputs)
                loss = self.criterion(out["yield_pred"], targets)
            total_loss += loss.item()
            all_preds.append(out["yield_pred"].cpu().numpy())
            all_tgts.append(targets.cpu().numpy())

        metrics = compute_metrics(
            np.concatenate(all_preds).ravel(),
            np.concatenate(all_tgts).ravel()
        )
        metrics["loss"] = total_loss / len(dl)
        return metrics

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def save(self, tag: str):
        path = os.path.join(self.cfg.checkpoint.save_dir, f"model_{tag}.pt")
        torch.save({
            "model_state":   self.model.state_dict(),
            "optim_state":   self.optimizer.state_dict(),
            "global_step":   self.global_step,
            "best_val_mae":  self.best_val_mae,
        }, path)
        print(f"  ✓ Checkpoint saved → {path}")

    def load(self, path: str):
        ck = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ck["model_state"])
        self.optimizer.load_state_dict(ck["optim_state"])
        self.global_step  = ck.get("global_step",  0)
        self.best_val_mae = ck.get("best_val_mae", float("inf"))
        print(f"  ✓ Checkpoint loaded ← {path}")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def train(self):
        self._init_wandb()

        for epoch in range(1, self.cfg.training.epochs + 1):
            train_m = self._train_epoch(epoch)
            val_m   = self._eval_epoch(self.val_dl)

            print(
                f"Epoch {epoch:03d}/{self.cfg.training.epochs} | "
                f"Train loss={train_m['loss']:.4f} MAE={train_m['mae']:.2f} "
                f"R²={train_m['r2']:.3f} | "
                f"Val   loss={val_m['loss']:.4f}  MAE={val_m['mae']:.2f}  "
                f"R²={val_m['r2']:.3f} | "
                f"{train_m['epoch_time_s']:.1f}s"
            )

            wandb.log({
                "train/epoch_loss": train_m["loss"],
                "val/epoch_loss":   val_m["loss"],
                "train/mae":        train_m["mae"],
                "val/mae":          val_m["mae"],
                "train/rmse":       train_m["rmse"],
                "val/rmse":         val_m["rmse"],
                "train/r2":         train_m["r2"],
                "val/r2":           val_m["r2"],
                "train/epoch_time": train_m["epoch_time_s"],
                "epoch":            epoch,
            }, step=self.global_step)

            # Periodic save
            if epoch % self.cfg.checkpoint.save_every == 0:
                self.save(f"epoch{epoch:03d}")

            # Best model + early stopping
            if val_m["mae"] < self.best_val_mae:
                self.best_val_mae = val_m["mae"]
                self.no_improve   = 0
                self.save("best")
                print(f"  ★ New best val MAE = {self.best_val_mae:.3f}")
                wandb.run.summary.update({
                    "best_val_mae": self.best_val_mae,
                    "best_val_r2":  val_m["r2"],
                    "best_epoch":   epoch,
                })
            else:
                self.no_improve += 1
                if self.no_improve >= self.cfg.training.early_stop_patience:
                    print(f"\n  Early stopping at epoch {epoch}.")
                    break

        # Final test
        if self.test_dl is not None:
            best_path = os.path.join(self.cfg.checkpoint.save_dir, "model_best.pt")
            if os.path.exists(best_path):
                self.load(best_path)
            test_m = self._eval_epoch(self.test_dl)
            print(f"\n── Test Results ──────────────────────────────────────")
            print(f"  MAE={test_m['mae']:.3f}  RMSE={test_m['rmse']:.3f}  R²={test_m['r2']:.4f}")
            wandb.run.summary.update({
                "test_mae": test_m["mae"], "test_rmse": test_m["rmse"], "test_r2": test_m["r2"]
            })
            wandb.log({"test/mae": test_m["mae"], "test/rmse": test_m["rmse"],
                       "test/r2": test_m["r2"]}, step=self.global_step)

        wandb.finish()
        print("\n✓ Training complete.")