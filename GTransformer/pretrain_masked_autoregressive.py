# In[]
# 03_pretrain_masked.py
# Masked node feature reconstruction pretraining for Graph Transformer (Torch)
# Dataset format: H_{idx}.npy + Y_{idx}.npz (Ybus sparse)
#
# Objective:
#   Masked node feature reconstruction:
#     - randomly mask a ratio of nodes (mask all features -> set to zero)
#     - GT predicts node features
#     - MSE computed ONLY on masked nodes
#
# No argparse: configure parameters in main.

from __future__ import annotations

import os
import re
import sys
sys.path.append(r"/home/user/Desktop/zyh/self-refl/GTransformer")

import time
import math
import json
import random
from dataclasses import asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from scipy.sparse import load_npz

# -----------------------------
# Import model
# -----------------------------
from gt_torch_model import GTConfig, GTransformer
from gt_torch_model import ybus_to_adjacency


# -----------------------------
# Utilities
# -----------------------------
def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_seconds(sec: float) -> str:
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h:d}h{m:02d}m{s:02d}s"
    return f"{m:d}m{s:02d}s"


def get_gpu_mem_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


class EMA:
    """Exponential moving average for scalars (for stable tqdm postfix)."""
    def __init__(self, alpha: float = 0.03):
        self.alpha = alpha
        self.value = None

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value


# -----------------------------
# Dataset
# -----------------------------
def scan_indices(dataset_dir: str) -> List[int]:
    """
    Scan dataset_dir for H_{idx}.npy with matching Y_{idx}.npz.
    """
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")

    h_pat = re.compile(r"^H_(\d+)\.npy$")
    indices = []
    for fn in os.listdir(dataset_dir):
        m = h_pat.match(fn)
        if m:
            idx = int(m.group(1))
            y_fn = f"Y_{idx}.npz"
            if os.path.exists(os.path.join(dataset_dir, y_fn)):
                indices.append(idx)

    indices.sort()
    if len(indices) == 0:
        raise RuntimeError(
            f"No valid samples found in {dataset_dir}. Expect files like H_0.npy and Y_0.npz."
        )
    return indices


class YantianHYDatasetNodeOnly(Dataset):
    """
    Loads (H.npy, Y.npz) samples.
    Produces:
      H: float32 [N, din]
      A: float32 [N, N] adjacency derived from Y (binary/abs/real/imag)
    """
    def __init__(self, dataset_dir: str, indices: List[int], adj_mode: str = "binary", drop_diag: bool = False):
        self.dataset_dir = dataset_dir
        self.indices = indices
        self.adj_mode = adj_mode
        self.drop_diag = drop_diag

        H0 = np.load(os.path.join(dataset_dir, f"H_{indices[0]}.npy"), mmap_mode="r")
        self.N = int(H0.shape[0])
        self.din = int(H0.shape[1])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Dict:
        idx = self.indices[i]

        h_path = os.path.join(self.dataset_dir, f"H_{idx}.npy")
        y_path = os.path.join(self.dataset_dir, f"Y_{idx}.npz")

        H = np.load(h_path, mmap_mode="r").astype(np.float32)  # [N, din]
        Y = load_npz(y_path)
        A = ybus_to_adjacency(Y, mode=self.adj_mode, drop_diagonal=self.drop_diag)  # [N, N] float32

        return {"idx": idx, "H": H, "A": A}


def collate_fn_node_only(batch: List[Dict]) -> Dict:
    idxs = [b["idx"] for b in batch]
    H = torch.from_numpy(np.stack([b["H"] for b in batch], axis=0))  # [B, N, din]
    A = torch.from_numpy(np.stack([b["A"] for b in batch], axis=0))  # [B, N, N]
    return {"idx": idxs, "H": H, "A": A}


# -----------------------------
# Masking
# -----------------------------
@torch.no_grad()
def make_node_feature_mask(
    B: int,
    N: int,
    din: int,
    node_mask_ratio: float,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    node_mask = (torch.rand(B, N, device=device) < node_mask_ratio)
    feature_mask = (~node_mask).float().unsqueeze(-1).expand(B, N, din)
    return feature_mask, node_mask


# -----------------------------
# LR scheduler (Warmup + Cosine)
# -----------------------------
def build_warmup_cosine_scheduler(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.05):
    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# -----------------------------
# Loss / Eval
# -----------------------------
def compute_node_loss(node_pred: torch.Tensor, H: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """
    node_pred, H: [B, N, din]
    node_mask: [B, N] True=masked node
    """
    B, N, din = H.shape
    mask = node_mask.unsqueeze(-1).expand(B, N, din).float()
    mse = (node_pred - H) ** 2
    num = mask.sum().clamp_min(1.0)
    return (mse * mask).sum() / num


@torch.no_grad()
def run_eval(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    node_mask_ratio: float,
    amp: bool
) -> Dict[str, float]:
    model.eval()
    node_losses = []
    node_ema = EMA(0.2)

    pbar = tqdm(loader, desc="Eval", dynamic_ncols=True, leave=False)
    for batch in pbar:
        H = batch["H"].to(device=device, non_blocking=True)
        A = batch["A"].to(device=device, non_blocking=True)

        if not torch.isfinite(H).all():
            print("[Eval Skip] non-finite H, idx=", batch["idx"])
            continue
        if not torch.isfinite(A).all():
            print("[Eval Skip] non-finite A, idx=", batch["idx"])
            continue

        B, N, din = H.shape
        feature_mask, node_mask = make_node_feature_mask(B, N, din, node_mask_ratio, device)

        with autocast(device_type="cuda", enabled=amp):
            out = model(H, A, feature_mask=feature_mask, return_embeddings=False)
            node_pred = out["node_pred"]
            nloss = compute_node_loss(node_pred, H, node_mask)

        if not torch.isfinite(nloss):
            print("[Eval Skip] non-finite node loss, idx=", batch["idx"])
            continue

        node_losses.append(float(nloss.item()))
        pbar.set_postfix({"node": f"{node_ema.update(node_losses[-1]):.6f}"})

    mean_node = float(np.mean(node_losses)) if node_losses else 0.0
    return {"node_loss": mean_node, "total_loss": mean_node}


def save_checkpoint(
    save_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    epoch: int,
    global_step: int,
    best_val: float,
    cfg: Dict
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "best_val": best_val,
        "cfg": cfg,
    }
    torch.save(ckpt, save_path)


def try_load_checkpoint(
    load_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[GradScaler],
    device: torch.device
) -> Tuple[int, int, float]:
    if not os.path.exists(load_path):
        return 0, 0, float("inf")

    ckpt = torch.load(load_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])

    epoch = int(ckpt.get("epoch", 0))
    global_step = int(ckpt.get("global_step", 0))
    best_val = float(ckpt.get("best_val", float("inf")))
    return epoch, global_step, best_val


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":

    # ====== Paths ======
    WORKPATH = "/home/user/Desktop/zyh/self-refl/GTransformer"
    DATAPATH = "/data2/zyh/yantian752_260105"
    DATASET_DIR = DATAPATH

    EXP_NAME = "pretrain_masked_node_only_v1"
    OUT_DIR = os.path.join(WORKPATH, "runs", EXP_NAME)
    CKPT_LATEST = os.path.join(OUT_DIR, "ckpt_latest.pt")
    CKPT_BEST = os.path.join(OUT_DIR, "ckpt_best.pt")
    LOG_JSONL = os.path.join(OUT_DIR, "train_log.jsonl")
    os.makedirs(OUT_DIR, exist_ok=True)

    # ====== System / speed ======
    SEED = 20260105
    set_all_seeds(SEED)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = True if torch.cuda.is_available() else False

    # ====== Data split ======
    all_indices = scan_indices(DATASET_DIR)
    num_total = len(all_indices)
    #num_total = 220000
    train_ratio = 0.9
    num_train = int(num_total * train_ratio)
    train_indices = all_indices[:num_train]
    val_indices = all_indices[num_train:]

    # ====== Masking hyperparams ======
    NODE_MASK_RATIO = 0.5

    # ====== Training hyperparams ======
    EPOCHS = 100
    BATCH_SIZE = 32 if torch.cuda.is_available() else 2
    NUM_WORKERS = 32
    PIN_MEMORY = True

    LR = 3e-4
    WEIGHT_DECAY = 1e-2
    GRAD_CLIP_NORM = 1.0
    WARMUP_STEPS = 1500

    EVAL_EVERY_EPOCH = 1
    SAVE_EVERY_EPOCH = 1

    # ====== Model hyperparams ======
    ADJ_MODE = "binary"
    DROP_DIAG = False

    D_MODEL = 128
    N_HEADS = 8
    D_FF = 256
    N_LAYERS = 3
    K_MIN = 1
    K_MAX = 5
    DROPOUT = 0.1
    ATTN_DROPOUT = 0.1

    # ====== Build datasets/loaders ======
    train_ds = YantianHYDatasetNodeOnly(DATASET_DIR, train_indices, adj_mode=ADJ_MODE, drop_diag=DROP_DIAG)
    val_ds = YantianHYDatasetNodeOnly(DATASET_DIR, val_indices, adj_mode=ADJ_MODE, drop_diag=DROP_DIAG) if len(val_indices) > 0 else None

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
        collate_fn=collate_fn_node_only,
    )

    val_loader = None
    if val_ds is not None and len(val_ds) > 0:
        val_loader = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=max(1, NUM_WORKERS // 2),
            pin_memory=PIN_MEMORY,
            drop_last=False,
            persistent_workers=(NUM_WORKERS > 0),
            prefetch_factor=2 if NUM_WORKERS > 0 else None,
            collate_fn=collate_fn_node_only,
        )

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * EPOCHS

    # ====== Build model ======
    cfg = GTConfig(
        din=train_ds.din,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_ff=D_FF,
        n_layers=N_LAYERS,
        k_min=K_MIN,
        k_max=K_MAX,
        dropout=DROPOUT,
        attn_dropout=ATTN_DROPOUT,
        adj_mode=ADJ_MODE,
    )
    model = GTransformer(cfg).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = build_warmup_cosine_scheduler(
        optimizer, warmup_steps=WARMUP_STEPS, total_steps=total_steps, min_lr_ratio=0.05
    )
    scaler = GradScaler("cuda", enabled=amp)

    # ====== Resume if exists ======
    start_epoch, global_step, best_val = try_load_checkpoint(
        CKPT_LATEST, model, optimizer, scheduler, scaler, device
    )

    # ====== Log config ======
    run_cfg = {
        "WORKPATH": WORKPATH,
        "DATAPATH": DATAPATH,
        "DATASET_DIR": DATASET_DIR,
        "EXP_NAME": EXP_NAME,
        "seed": SEED,
        "device": str(device),
        "amp": amp,
        "train_size": len(train_ds),
        "val_size": len(val_ds) if val_ds is not None else 0,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "mask": {"node_mask_ratio": NODE_MASK_RATIO},
        "optim": {
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip_norm": GRAD_CLIP_NORM,
            "warmup_steps": WARMUP_STEPS,
            "epochs": EPOCHS,
        },
        "model": asdict(cfg),
        "num_params": count_parameters(model),
        "resume": {
            "start_epoch": start_epoch,
            "global_step": global_step,
            "best_val": best_val,
            "ckpt_latest": CKPT_LATEST,
        }
    }

    with open(os.path.join(OUT_DIR, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_cfg, f, ensure_ascii=False, indent=2)

    print("=" * 90)
    print(f"[Run] {EXP_NAME}")
    print(f"[Path] DATASET_DIR: {DATASET_DIR}")
    print(f"[Data] total={num_total}, train={len(train_ds)}, val={len(val_ds) if val_ds is not None else 0}")
    print(f"[Shape] N={train_ds.N}, din={train_ds.din}")
    print(f"[Device] {device}, amp={amp}, gpus={torch.cuda.device_count()}")
    print(f"[Model] params={run_cfg['num_params']:,}")
    print(f"[Train] epochs={EPOCHS}, steps/epoch={steps_per_epoch}, total_steps={total_steps}")
    print(f"[Resume] start_epoch={start_epoch}, global_step={global_step}, best_val={best_val}")
    print("=" * 90)

    # ====== Training loop ======
    node_ema = EMA(0.03)

    t0 = time.time()
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        epoch_t0 = time.time()
        epoch_node_losses: List[float] = []
        skipped_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", dynamic_ncols=True)
        for batch in pbar:
            H = batch["H"].to(device=device, non_blocking=True)
            A = batch["A"].to(device=device, non_blocking=True)

            if not torch.isfinite(H).all():
                print("[Skip] non-finite H, idx=", batch["idx"])
                skipped_batches += 1
                continue
            if not torch.isfinite(A).all():
                print("[Skip] non-finite A, idx=", batch["idx"])
                skipped_batches += 1
                continue

            B, N, din = H.shape
            feature_mask, node_mask = make_node_feature_mask(B, N, din, NODE_MASK_RATIO, device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=amp):
                out = model(H, A, feature_mask=feature_mask, return_embeddings=False)
                node_pred = out["node_pred"]
                loss = compute_node_loss(node_pred, H, node_mask)

            # if loss itself is non-finite, skip safely (no unscale happened)
            if not torch.isfinite(loss):
                print("[Skip] non-finite loss, idx=", batch["idx"], "loss=", float(loss.detach().cpu()))
                optimizer.zero_grad(set_to_none=True)
                # 这里不需要 scaler.update()（因为还未 unscale/step），但加也无害
                scaler.update()
                skipped_batches += 1
                continue

            # backward (scaled)
            scaler.scale(loss).backward()

            # unscale once per iteration
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

            # IMPORTANT FIX:
            # if grad_norm is non-finite, we MUST call scaler.update() before continue,
            # otherwise next iteration calling unscale_ will crash.
            if not torch.isfinite(grad_norm):
                print("[Skip] non-finite grad_norm, idx=", batch["idx"])
                # optional debug stats
                h_min = float(H.min().detach().cpu())
                h_max = float(H.max().detach().cpu())
                h_std = float(H.std().detach().cpu())
                print(f"  [H stats] min={h_min:.3e}, max={h_max:.3e}, std={h_std:.3e}")

                optimizer.zero_grad(set_to_none=True)
                scaler.update()  # CRITICAL: reset scaler state (and likely reduce scale)
                skipped_batches += 1
                continue

            # optimizer step (AMP)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()

            # only step scheduler if optimizer step was not skipped by scaler
            if scale_after >= scale_before:
                scheduler.step()
                global_step += 1

            # record losses
            loss_now = float(loss.detach().cpu())
            epoch_node_losses.append(loss_now)

            # epoch running mean
            loss_avg = float(np.mean(epoch_node_losses)) if len(epoch_node_losses) > 0 else 0.0

            lr_now = optimizer.param_groups[0]["lr"]
            mem_mb = get_gpu_mem_mb()

            # Print more informative loss numbers in tqdm
            pbar.set_postfix({
                "lr": f"{lr_now:.2e}",
                "loss_now": f"{loss_now:.6f}",
                "loss_avg": f"{loss_avg:.6f}",
                "node_ema": f"{node_ema.update(loss_now):.6f}",
                "gN": f"{float(grad_norm):.4f}",
                "memMB": f"{mem_mb:.0f}",
                "skip": skipped_batches,
                "step": global_step,
            })

        # epoch summary
        epoch_time = time.time() - epoch_t0
        train_node = float(np.mean(epoch_node_losses)) if epoch_node_losses else float("inf")
        train_total = train_node

        # eval
        val_metrics = {"node_loss": 0.0, "total_loss": float("inf")}
        if val_loader is not None and ((epoch + 1) % EVAL_EVERY_EPOCH == 0):
            val_metrics = run_eval(
                model.module if isinstance(model, nn.DataParallel) else model,
                val_loader,
                device=device,
                node_mask_ratio=NODE_MASK_RATIO,
                amp=amp,
            )

        # save latest
        if (epoch + 1) % SAVE_EVERY_EPOCH == 0:
            save_checkpoint(
                CKPT_LATEST, model, optimizer, scheduler, scaler,
                epoch=epoch + 1, global_step=global_step, best_val=best_val,
                cfg=run_cfg
            )

        # save best
        if val_loader is not None and val_metrics["total_loss"] < best_val:
            best_val = val_metrics["total_loss"]
            save_checkpoint(
                CKPT_BEST, model, optimizer, scheduler, scaler,
                epoch=epoch + 1, global_step=global_step, best_val=best_val,
                cfg=run_cfg
            )

        # log line
        log_line = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "epoch": epoch + 1,
            "global_step": global_step,
            "train": {"node": train_node, "total": train_total, "skipped_batches": skipped_batches},
            "val": val_metrics,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "epoch_time_sec": float(epoch_time),
            "gpu_peak_mem_mb": float(get_gpu_mem_mb()),
            "best_val_total": float(best_val),
        }
        with open(LOG_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_line, ensure_ascii=False) + "\n")

        print(
            f"[Epoch {epoch+1:03d}/{EPOCHS}] "
            f"train_total={train_total:.6f} (node={train_node:.6f}, skip={skipped_batches}) | "
            f"val_total={val_metrics['total_loss']:.6f} (node={val_metrics['node_loss']:.6f}) | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | "
            f"epoch_time={format_seconds(epoch_time)} | "
            f"best_val={best_val:.6f}"
        )

    total_time = time.time() - t0
    print("=" * 90)
    print(f"Training finished. Total time: {format_seconds(total_time)}")
    print(f"Latest checkpoint: {CKPT_LATEST}")
    print(f"Best checkpoint:   {CKPT_BEST}")
    print("=" * 90)

# %%
