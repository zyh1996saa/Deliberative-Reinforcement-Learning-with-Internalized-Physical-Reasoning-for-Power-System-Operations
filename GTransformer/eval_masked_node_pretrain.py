# In[]
# 04_eval_masked_node_pretrain.py
# Evaluate masked-node reconstruction performance on validation set.
# Produces metrics + plots:
#   - masked abs error distribution
#   - per-sample max node error distribution
#   - top-K hardest nodes by average masked-node MAE

from __future__ import annotations

import os
import re
import sys
sys.path.append(r"/home/user/Desktop/zyh/self-refl/GTransformer")

import math
import json
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from tqdm import tqdm

import matplotlib.pyplot as plt
from scipy.sparse import load_npz

from gt_torch_model import GTConfig, GTransformer, ybus_to_adjacency


# -----------------------------
# Utilities
# -----------------------------
def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def scan_indices(dataset_dir: str) -> List[int]:
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")

    h_pat = re.compile(r"^H_(\d+)\.npy$")
    indices = []
    for fn in os.listdir(dataset_dir):
        m = h_pat.match(fn)
        if m:
            idx = int(m.group(1))
            if os.path.exists(os.path.join(dataset_dir, f"Y_{idx}.npz")):
                indices.append(idx)

    indices.sort()
    if len(indices) == 0:
        raise RuntimeError(f"No valid samples found in {dataset_dir}.")
    return indices


def load_checkpoint_flexible(model: nn.Module, ckpt_path: str, device: torch.device) -> Dict:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # handle DataParallel "module." prefix mismatch
    model_sd = model.state_dict()
    if any(k.startswith("module.") for k in sd.keys()) and not any(k.startswith("module.") for k in model_sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    elif (not any(k.startswith("module.") for k in sd.keys())) and any(k.startswith("module.") for k in model_sd.keys()):
        sd = {("module." + k): v for k, v in sd.items()}

    missing, unexpected = model.load_state_dict(sd, strict=False)
    info = {
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }
    if isinstance(ckpt, dict):
        info.update({k: ckpt.get(k, None) for k in ["epoch", "global_step", "best_val"]})
    return info


@torch.no_grad()
def make_node_feature_mask(B: int, N: int, din: int, ratio: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      feature_mask: [B,N,din] float, 1 keep / 0 mask
      node_mask: [B,N] bool, True masked
    """
    node_mask = (torch.rand(B, N, device=device) < ratio)
    feature_mask = (~node_mask).float().unsqueeze(-1).expand(B, N, din)
    return feature_mask, node_mask


def safe_percentiles(x: np.ndarray, ps: List[float]) -> Dict[str, float]:
    if x.size == 0:
        return {f"p{int(p)}": float("nan") for p in ps}
    return {f"p{int(p)}": float(np.percentile(x, p)) for p in ps}


# -----------------------------
# Dataset (node-only)
# -----------------------------
class YantianHYDatasetNodeOnly(Dataset):
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
        H = np.load(os.path.join(self.dataset_dir, f"H_{idx}.npy"), mmap_mode="r").astype(np.float32)
        Y = load_npz(os.path.join(self.dataset_dir, f"Y_{idx}.npz"))
        A = ybus_to_adjacency(Y, mode=self.adj_mode, drop_diagonal=self.drop_diag)
        return {"idx": idx, "H": H, "A": A}


def collate_fn_node_only(batch: List[Dict]) -> Dict:
    idxs = [b["idx"] for b in batch]
    H = torch.from_numpy(np.stack([b["H"] for b in batch], axis=0))
    A = torch.from_numpy(np.stack([b["A"] for b in batch], axis=0))
    return {"idx": idxs, "H": H, "A": A}


# -----------------------------
# Evaluation
# -----------------------------
@torch.no_grad()
def evaluate_masked_node(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mask_ratio: float,
    amp: bool,
    mc_masks: int = 1
) -> Dict:
    """
    Returns metrics and arrays for visualization:
      - masked_abs_errors: flattened abs errors on masked positions (all features)
      - per_sample_max_node_mae: per-sample max node MAE (masked nodes only)
      - per_node_avg_mae: [N] average masked-node MAE across val
      - per_node_count: [N] number of times node got masked (for reference)
    """
    model.eval()

    masked_abs_errors_all = []
    per_sample_max_node_mae = []

    # aggregated metrics
    sum_sq = 0.0
    sum_abs = 0.0
    cnt_vals = 0.0

    # per-feature metrics (masked only)
    # accumulate abs and sq per feature
    per_feat_sum_abs = None
    per_feat_sum_sq = None
    per_feat_cnt = 0.0

    # per-node average mae (masked nodes only)
    N = loader.dataset.N
    per_node_sum_mae = np.zeros((N,), dtype=np.float64)
    per_node_cnt = np.zeros((N,), dtype=np.int64)

    pbar = tqdm(loader, desc="Eval(val)", dynamic_ncols=True)
    for batch in pbar:
        H = batch["H"].to(device=device, non_blocking=True)  # [B,N,din]
        A = batch["A"].to(device=device, non_blocking=True)

        if not torch.isfinite(H).all():
            print("[Eval Skip] non-finite H, idx=", batch["idx"])
            continue
        if not torch.isfinite(A).all():
            print("[Eval Skip] non-finite A, idx=", batch["idx"])
            continue

        B, Nn, din = H.shape
        if per_feat_sum_abs is None:
            per_feat_sum_abs = np.zeros((din,), dtype=np.float64)
            per_feat_sum_sq = np.zeros((din,), dtype=np.float64)

        # multiple random masks per batch for more stable estimation
        for _ in range(mc_masks):
            feature_mask, node_mask = make_node_feature_mask(B, Nn, din, mask_ratio, device)

            with autocast(device_type="cuda", enabled=amp):
                out = model(H, A, feature_mask=feature_mask, return_embeddings=False)
                pred = out["node_pred"]  # [B,N,din]

            # abs / sq errors
            diff = pred - H
            abs_err = diff.abs()                 # [B,N,din]
            sq_err = diff.pow(2)                 # [B,N,din]

            # masked positions only
            mask3 = node_mask.unsqueeze(-1).expand(B, Nn, din)   # bool [B,N,din]

            masked_abs = abs_err[mask3]          # [num_masked_vals]
            masked_sq = sq_err[mask3]

            # accumulate scalar metrics
            if masked_abs.numel() > 0:
                sum_abs += float(masked_abs.sum().item())
                sum_sq += float(masked_sq.sum().item())
                cnt_vals += float(masked_abs.numel())

                masked_abs_errors_all.append(masked_abs.detach().float().cpu().numpy())

            # per-feature metrics (masked only)
            # sum over nodes masked, per feature
            # [B,N,din] -> [din] with mask
            if node_mask.any():
                # masked per-feature sums
                # create float mask [B,N,1]
                nm = node_mask.float().unsqueeze(-1)  # [B,N,1]
                feat_abs_sum = (abs_err * nm).sum(dim=(0, 1)).detach().cpu().numpy()  # [din]
                feat_sq_sum = (sq_err * nm).sum(dim=(0, 1)).detach().cpu().numpy()   # [din]
                # counts per feature: masked_nodes * B * 1 per feature
                feat_cnt = float(node_mask.sum().item()) * float(din)

                per_feat_sum_abs += feat_abs_sum
                per_feat_sum_sq += feat_sq_sum
                per_feat_cnt += feat_cnt

            # per-sample max node MAE (masked nodes only)
            # node_mae: [B,N] mean abs over features
            node_mae = abs_err.mean(dim=-1)  # [B,N]
            # set unmasked nodes to -inf so max only considers masked nodes
            node_mae_masked = node_mae.masked_fill(~node_mask, float("-inf"))
            # if some sample has 0 masked nodes (rare), avoid -inf
            max_mae = torch.where(
                node_mask.any(dim=1),
                node_mae_masked.max(dim=1).values,
                torch.zeros((B,), device=device, dtype=node_mae.dtype)
            )
            per_sample_max_node_mae.append(max_mae.detach().float().cpu().numpy())

            # per-node accumulation (masked nodes only)
            # For each node j: accumulate mean over batch where masked
            # node_mae: [B,N], node_mask: [B,N]
            node_mae_cpu = node_mae.detach().cpu().numpy()
            node_mask_cpu = node_mask.detach().cpu().numpy()
            # accumulate sums and counts
            # vectorized: sum over batch for each node
            per_node_sum_mae += (node_mae_cpu * node_mask_cpu).sum(axis=0)
            per_node_cnt += node_mask_cpu.sum(axis=0).astype(np.int64)

        # progress display (running masked MAE)
        if cnt_vals > 0:
            running_mae = sum_abs / cnt_vals
            running_rmse = math.sqrt(sum_sq / cnt_vals)
            pbar.set_postfix({"MAE(masked)": f"{running_mae:.5f}", "RMSE(masked)": f"{running_rmse:.5f}"})

    masked_abs_errors = np.concatenate(masked_abs_errors_all, axis=0) if len(masked_abs_errors_all) > 0 else np.array([], dtype=np.float32)
    per_sample_max_node_mae = np.concatenate(per_sample_max_node_mae, axis=0) if len(per_sample_max_node_mae) > 0 else np.array([], dtype=np.float32)

    metrics = {}
    if cnt_vals > 0:
        metrics["masked_mae"] = float(sum_abs / cnt_vals)
        metrics["masked_mse"] = float(sum_sq / cnt_vals)
        metrics["masked_rmse"] = float(math.sqrt(sum_sq / cnt_vals))
    else:
        metrics["masked_mae"] = float("nan")
        metrics["masked_mse"] = float("nan")
        metrics["masked_rmse"] = float("nan")

    # per-feature
    if per_feat_cnt > 0:
        metrics["per_feature_masked_mae"] = (per_feat_sum_abs / per_feat_cnt).tolist()
        metrics["per_feature_masked_rmse"] = np.sqrt(per_feat_sum_sq / per_feat_cnt).tolist()
    else:
        metrics["per_feature_masked_mae"] = None
        metrics["per_feature_masked_rmse"] = None

    # percentiles
    metrics["masked_abs_error_percentiles"] = safe_percentiles(masked_abs_errors, [50, 90, 95, 99])
    metrics["per_sample_max_node_mae_percentiles"] = safe_percentiles(per_sample_max_node_mae, [50, 90, 95, 99])

    # per-node avg mae (masked only)
    per_node_avg_mae = np.zeros((N,), dtype=np.float32)
    valid = per_node_cnt > 0
    per_node_avg_mae[valid] = (per_node_sum_mae[valid] / per_node_cnt[valid]).astype(np.float32)

    return {
        "metrics": metrics,
        "masked_abs_errors": masked_abs_errors,
        "per_sample_max_node_mae": per_sample_max_node_mae,
        "per_node_avg_mae": per_node_avg_mae,
        "per_node_cnt": per_node_cnt,
    }


def plot_and_save_hist(x: np.ndarray, title: str, xlabel: str, out_path: str, bins: int = 100) -> None:
    plt.figure()
    # 默认颜色即可（不手动指定颜色）
    plt.hist(x, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_and_save_topk_nodes(per_node_avg_mae: np.ndarray, per_node_cnt: np.ndarray, out_path: str, k: int = 30) -> List[Tuple[int, float, int]]:
    # only nodes that were actually masked at least once
    valid = per_node_cnt > 0
    idxs = np.where(valid)[0]
    maes = per_node_avg_mae[valid]

    if idxs.size == 0:
        return []

    order = np.argsort(-maes)  # descending
    topk = order[:min(k, order.size)]
    top_nodes = [(int(idxs[i]), float(maes[i]), int(per_node_cnt[idxs[i]])) for i in topk]

    plt.figure(figsize=(12, 5))
    plt.bar(range(len(top_nodes)), [v[1] for v in top_nodes])
    plt.xticks(range(len(top_nodes)), [str(v[0]) for v in top_nodes], rotation=60, ha="right")
    plt.title(f"Top-{len(top_nodes)} Hardest Nodes (avg masked-node MAE)")
    plt.xlabel("Node index")
    plt.ylabel("Avg MAE")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    return top_nodes


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":

    # ====== Paths ======
    WORKPATH = "/home/user/Desktop/zyh/self-refl/GTransformer"
    DATAPATH = "/data2/zyh/yantian752_260105"
    DATASET_DIR = DATAPATH

    # checkpoint: choose best or latest
    EXP_NAME = "pretrain_masked_node_only_v1"
    OUT_DIR = os.path.join(WORKPATH, "runs", EXP_NAME)
    CKPT_BEST = os.path.join(OUT_DIR, "ckpt_best.pt")
    CKPT_LATEST = os.path.join(OUT_DIR, "ckpt_latest.pt")
    CKPT_PATH = CKPT_BEST if os.path.exists(CKPT_BEST) else CKPT_LATEST

    EVAL_OUT_DIR = os.path.join(OUT_DIR, "eval_val")
    os.makedirs(EVAL_OUT_DIR, exist_ok=True)

    # ====== Eval config ======
    SEED = 20260105
    set_all_seeds(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = True if torch.cuda.is_available() else False

    # Data split (must match training split logic)
    all_indices = scan_indices(DATASET_DIR)
    train_ratio = 0.9
    num_train = int(len(all_indices) * train_ratio)
    val_indices = all_indices[num_train:]

    # Mask settings (must reflect evaluation protocol you want)
    NODE_MASK_RATIO = 0.15
    MC_MASKS = 3  # number of random masks per batch; set 1 for fastest, 3~5 for smoother stats

    # Loader
    BATCH_SIZE = 8 if torch.cuda.is_available() else 2
    NUM_WORKERS = 8
    PIN_MEMORY = True

    ADJ_MODE = "binary"
    DROP_DIAG = False

    val_ds = YantianHYDatasetNodeOnly(DATASET_DIR, val_indices, adj_mode=ADJ_MODE, drop_diag=DROP_DIAG)
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

    # Build model (must match training hyperparams)
    cfg = GTConfig(
        din=val_ds.din,
        d_model=128,
        n_heads=8,
        d_ff=256,
        n_layers=3,
        k_min=1,
        k_max=5,
        dropout=0.1,
        attn_dropout=0.1,
        adj_mode=ADJ_MODE,
    )
    model = GTransformer(cfg).to(device)

    # Load ckpt
    ckpt_info = load_checkpoint_flexible(model, CKPT_PATH, device)
    print("=" * 90)
    print("[Eval] checkpoint:", CKPT_PATH)
    print("[Eval] ckpt_info:", ckpt_info)
    print(f"[Eval] val_size={len(val_ds)}, N={val_ds.N}, din={val_ds.din}, mask_ratio={NODE_MASK_RATIO}, MC_MASKS={MC_MASKS}")
    print("=" * 90)

    # Run eval
    res = evaluate_masked_node(
        model=model,
        loader=val_loader,
        device=device,
        mask_ratio=NODE_MASK_RATIO,
        amp=amp,
        mc_masks=MC_MASKS
    )

    metrics = res["metrics"]
    masked_abs_errors = res["masked_abs_errors"]
    per_sample_max_node_mae = res["per_sample_max_node_mae"]
    per_node_avg_mae = res["per_node_avg_mae"]
    per_node_cnt = res["per_node_cnt"]

    # Save metrics JSON
    metrics_path = os.path.join(EVAL_OUT_DIR, "val_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)# 04_eval_masked.py
# Evaluate masked node feature reconstruction performance for GTransformer (Torch)
# Dataset format: H_{idx}.npy + Y_{idx}.npz (Ybus sparse)
#
# Key features:
#   - Evaluate on masked nodes only (default, consistent with pretraining objective)
#   - Per-feature-dimension metrics + per-feature-group metrics
#   - Optional de-normalization (standard/minmax) via normalization.npz / normalization.json
#   - Visualization: error histograms, scatter (pred vs true), saved to OUT_DIR
#
# No argparse: configure parameters in main.

from __future__ import annotations

import os
import re
import sys
import json
import math
import random
import csv
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from tqdm import tqdm
import matplotlib.pyplot as plt

from scipy.sparse import load_npz

# -----------------------------
# Import model
# -----------------------------
sys.path.append(r"/home/user/Desktop/zyh/self-refl/GTransformer")
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


def scan_indices(dataset_dir: str) -> List[int]:
    """Scan dataset_dir for H_{idx}.npy with matching Y_{idx}.npz."""
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


def safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """If keys start with 'module.', strip it for non-DataParallel model loading."""
    if not state_dict:
        return state_dict
    has_module = any(k.startswith("module.") for k in state_dict.keys())
    if not has_module:
        return state_dict
    new_sd = {}
    for k, v in state_dict.items():
        new_sd[k[len("module."):]] = v if k.startswith("module.") else v
    return new_sd


def pearson_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation (safe)"""
    if x.size == 0:
        return float("nan")
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x_mean = x.mean()
    y_mean = y.mean()
    xv = x - x_mean
    yv = y - y_mean
    denom = math.sqrt(float((xv * xv).sum()) * float((yv * yv).sum()))
    if denom <= 0:
        return float("nan")
    return float((xv * yv).sum() / denom)


# -----------------------------
# Dataset
# -----------------------------
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
    """
    node_mask: True means this node is masked (all features zeroed)
    feature_mask: 1 for keep, 0 for masked (broadcasted on feature dim)
    """
    node_mask = (torch.rand(B, N, device=device) < node_mask_ratio)
    feature_mask = (~node_mask).float().unsqueeze(-1).expand(B, N, din)
    return feature_mask, node_mask


# -----------------------------
# De-normalization (optional)
# -----------------------------
class Denormalizer:
    """
    Support:
      - standard: x_raw = x * std + mean
      - minmax : x_raw = x * (max - min) + min
    Expected files:
      1) normalization.npz with keys:
         - type: stored as a numpy scalar string or omitted (assume standard if mean/std present)
         - mean, std  (shape [din])
         - or min, max (shape [din])
      2) normalization.json with fields:
         {"type": "standard", "mean": [...], "std": [...]}
         {"type": "minmax",  "min":  [...], "max": [...]}
    """
    def __init__(self, mode: str = "identity", mean=None, std=None, vmin=None, vmax=None):
        self.mode = mode
        self.mean = mean
        self.std = std
        self.vmin = vmin
        self.vmax = vmax

    def enabled(self) -> bool:
        return self.mode != "identity"

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "identity":
            return x
        if self.mode == "standard":
            return x * self.std + self.mean
        if self.mode == "minmax":
            return x * (self.vmax - self.vmin) + self.vmin
        raise ValueError(f"Unknown denorm mode: {self.mode}")

    @staticmethod
    def try_load(path_npz: str, path_json: str, din: int, device: torch.device) -> "Denormalizer":
        # Prefer explicit path if exists
        if os.path.exists(path_npz):
            obj = np.load(path_npz, allow_pickle=True)
            keys = set(obj.files)

            mode = None
            if "type" in keys:
                t = obj["type"]
                if isinstance(t, np.ndarray) and t.shape == ():
                    mode = str(t.item())
                else:
                    mode = str(t)

            if ("mean" in keys and "std" in keys):
                mean = torch.tensor(obj["mean"], dtype=torch.float32, device=device).view(1, 1, din)
                std = torch.tensor(obj["std"], dtype=torch.float32, device=device).view(1, 1, din)
                return Denormalizer(mode=(mode or "standard"), mean=mean, std=std)

            if ("min" in keys and "max" in keys):
                vmin = torch.tensor(obj["min"], dtype=torch.float32, device=device).view(1, 1, din)
                vmax = torch.tensor(obj["max"], dtype=torch.float32, device=device).view(1, 1, din)
                return Denormalizer(mode=(mode or "minmax"), vmin=vmin, vmax=vmax)

        if os.path.exists(path_json):
            with open(path_json, "r", encoding="utf-8") as f:
                j = json.load(f)
            mode = str(j.get("type", "identity")).lower()
            if mode == "standard":
                mean = torch.tensor(j["mean"], dtype=torch.float32, device=device).view(1, 1, din)
                std = torch.tensor(j["std"], dtype=torch.float32, device=device).view(1, 1, din)
                return Denormalizer(mode="standard", mean=mean, std=std)
            if mode == "minmax":
                vmin = torch.tensor(j["min"], dtype=torch.float32, device=device).view(1, 1, din)
                vmax = torch.tensor(j["max"], dtype=torch.float32, device=device).view(1, 1, din)
                return Denormalizer(mode="minmax", vmin=vmin, vmax=vmax)

        return Denormalizer(mode="identity")


# -----------------------------
# Metrics accumulator
# -----------------------------
class RunningMetrics:
    """
    Streaming metrics per-dimension:
      - sum_abs_err, sum_sq_err
      - sum_y, sum_y2, sum_pred, sum_pred2, sum_pred_y (for R2/corr; corr uses sampled arrays)
      - sum_abs_rel_err (MAPE-like, denominator clamp)
      - count
    """
    def __init__(self, din: int):
        self.din = din
        self.sum_abs = np.zeros(din, dtype=np.float64)
        self.sum_sq = np.zeros(din, dtype=np.float64)
        self.sum_abs_rel = np.zeros(din, dtype=np.float64)

        self.sum_y = np.zeros(din, dtype=np.float64)
        self.sum_y2 = np.zeros(din, dtype=np.float64)
        self.sum_pred = np.zeros(din, dtype=np.float64)
        self.sum_pred2 = np.zeros(din, dtype=np.float64)

        self.count = np.zeros(din, dtype=np.int64)

    def update(self, err: np.ndarray, y: np.ndarray, pred: np.ndarray, eps: float = 1e-6) -> None:
        """
        err, y, pred: shape [M, din] (M masked nodes aggregated in this batch)
        """
        if err.size == 0:
            return
        abs_err = np.abs(err).astype(np.float64)
        sq_err = (err.astype(np.float64) ** 2)

        denom = np.maximum(np.abs(y).astype(np.float64), eps)
        abs_rel = abs_err / denom

        self.sum_abs += abs_err.sum(axis=0)
        self.sum_sq += sq_err.sum(axis=0)
        self.sum_abs_rel += abs_rel.sum(axis=0)

        y64 = y.astype(np.float64)
        p64 = pred.astype(np.float64)
        self.sum_y += y64.sum(axis=0)
        self.sum_y2 += (y64 ** 2).sum(axis=0)
        self.sum_pred += p64.sum(axis=0)
        self.sum_pred2 += (p64 ** 2).sum(axis=0)

        self.count += err.shape[0]

    def finalize(self) -> Dict[str, np.ndarray]:
        c = np.maximum(self.count.astype(np.float64), 1.0)

        mae = self.sum_abs / c
        rmse = np.sqrt(self.sum_sq / c)
        mape = self.sum_abs_rel / c  # mean(|err|/max(|y|,eps))

        # R^2: 1 - SSE / SST
        # SST = sum((y - mean_y)^2) = sum(y^2) - n*mean_y^2
        mean_y = self.sum_y / c
        sst = self.sum_y2 - c * (mean_y ** 2)
        sse = self.sum_sq
        # If sst is ~0 (constant target), R2 undefined; set nan
        r2 = np.full(self.din, np.nan, dtype=np.float64)
        valid = sst > 1e-12
        r2[valid] = 1.0 - (sse[valid] / sst[valid])

        # NRMSE: RMSE / (p95 - p5) is better but needs distribution.
        # Here use RMSE / (|mean_y|+eps) as a simple scale proxy.
        nrmse = rmse / (np.abs(mean_y) + 1e-6)

        return {
            "count": self.count.copy(),
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "r2": r2,
            "nrmse": nrmse,
        }


# -----------------------------
# Visualization helpers
# -----------------------------
def plot_histogram(errors: np.ndarray, title: str, out_path: str, bins: int = 120) -> None:
    """
    errors: 1D array
    """
    if errors.size == 0:
        return
    plt.figure()
    plt.hist(errors, bins=bins)
    plt.title(title)
    plt.xlabel("Error")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_scatter(y: np.ndarray, pred: np.ndarray, title: str, out_path: str, max_points: int = 20000) -> None:
    if y.size == 0:
        return
    if y.size > max_points:
        idx = np.random.choice(y.size, size=max_points, replace=False)
        y = y[idx]
        pred = pred[idx]
    plt.figure()
    plt.scatter(y, pred, s=2, alpha=0.3)
    plt.title(title)
    plt.xlabel("True")
    plt.ylabel("Pred")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def write_csv(path: str, header: List[str], rows: List[List]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# -----------------------------
# Main eval
# -----------------------------
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    node_mask_ratio: float,
    denorm: Denormalizer,
    feature_names: List[str],
    feature_groups: List[Dict],
    eval_on: str = "masked",   # "masked" | "all"
    n_mask_repeats: int = 1,
    max_store_errors: int = 2_000_000,
    eps_rel: float = 1e-6,
) -> Dict:
    """
    Returns dict with:
      - metrics_dim: per-dim metrics arrays
      - metrics_group: per-group summarized metrics
      - stored_errors_by_group: sampled errors for hist/scatter
    """
    din = len(feature_names)
    run = RunningMetrics(din)

    # store distributions (limited)
    stored = {}
    for g in feature_groups:
        stored[g["name"]] = {
            "err": [],       # raw error
            "abs_err": [],   # abs error
            "rel_err": [],   # abs(err)/max(|y|,eps)
            "y": [],         # true
            "pred": [],      # pred
        }

    model.eval()
    pbar = tqdm(loader, desc="Eval", dynamic_ncols=True)

    for _rep in range(n_mask_repeats):
        for batch in pbar:
            H = batch["H"].to(device=device, non_blocking=True)  # [B,N,din]
            A = batch["A"].to(device=device, non_blocking=True)  # [B,N,N]
            if not torch.isfinite(H).all() or not torch.isfinite(A).all():
                continue

            B, N, _ = H.shape
            feature_mask, node_mask = make_node_feature_mask(B, N, din, node_mask_ratio, device)

            with autocast(device_type="cuda", enabled=amp):
                out = model(H, A, feature_mask=feature_mask, return_embeddings=False)
                node_pred = out["node_pred"]  # [B,N,din]

            if not torch.isfinite(node_pred).all():
                continue

            # De-normalize both target and prediction if needed
            H_raw = denorm.inverse(H)
            pred_raw = denorm.inverse(node_pred)

            if eval_on == "masked":
                sel = node_mask  # [B,N]
            elif eval_on == "all":
                sel = torch.ones_like(node_mask, dtype=torch.bool, device=device)
            else:
                raise ValueError(f"Unknown eval_on: {eval_on}")

            # Flatten masked nodes: take all selected nodes across batch
            # y_sel, p_sel: [M, din]
            y_sel = H_raw[sel]         # torch: [M, din]
            p_sel = pred_raw[sel]      # torch: [M, din]
            if y_sel.numel() == 0:
                continue

            err = (p_sel - y_sel)

            # move to cpu numpy for streaming metrics
            y_np = y_sel.detach().cpu().numpy()
            p_np = p_sel.detach().cpu().numpy()
            e_np = err.detach().cpu().numpy()

            run.update(e_np, y_np, p_np, eps=eps_rel)

            # store distribution (per group)
            # NOTE: store limited number of points to avoid OOM
            for g in feature_groups:
                name = g["name"]
                dims = g["dims"]  # list[int]
                # build 1D arrays for distribution: flatten across dims
                e_g = e_np[:, dims].reshape(-1)
                y_g = y_np[:, dims].reshape(-1)
                p_g = p_np[:, dims].reshape(-1)

                remaining = max_store_errors - len(stored[name]["err"])
                if remaining <= 0:
                    continue

                # sample if too many
                if e_g.size > remaining:
                    idx = np.random.choice(e_g.size, size=remaining, replace=False)
                    e_g = e_g[idx]
                    y_g = y_g[idx]
                    p_g = p_g[idx]

                abs_e = np.abs(e_g)
                rel_e = abs_e / np.maximum(np.abs(y_g), eps_rel)

                stored[name]["err"].extend(e_g.tolist())
                stored[name]["abs_err"].extend(abs_e.tolist())
                stored[name]["rel_err"].extend(rel_e.tolist())
                stored[name]["y"].extend(y_g.tolist())
                stored[name]["pred"].extend(p_g.tolist())

    metrics_dim = run.finalize()

    # group summary: aggregate per-dim metrics by mean over dims (weighted by counts)
    metrics_group = {}
    for g in feature_groups:
        dims = np.array(g["dims"], dtype=np.int64)
        # use count-weighted average
        cnt = metrics_dim["count"][dims].astype(np.float64)
        w = cnt / np.maximum(cnt.sum(), 1.0)

        def wavg(arr):
            a = arr[dims].astype(np.float64)
            # if nan exists (e.g., r2), weighted nanmean
            mask = np.isfinite(a)
            if mask.sum() == 0:
                return float("nan")
            ww = w.copy()
            ww[~mask] = 0.0
            s = ww.sum()
            if s <= 0:
                return float("nan")
            ww = ww / s
            return float((a * ww).sum())

        metrics_group[g["name"]] = {
            "dims": g["dims"],
            "count": int(metrics_dim["count"][dims].sum()),
            "mae": wavg(metrics_dim["mae"]),
            "rmse": wavg(metrics_dim["rmse"]),
            "mape": wavg(metrics_dim["mape"]),
            "r2": wavg(metrics_dim["r2"]),
            "nrmse": wavg(metrics_dim["nrmse"]),
        }

    # convert stored to numpy arrays for later plotting/corr
    stored_np = {}
    for name, d in stored.items():
        stored_np[name] = {k: np.asarray(v, dtype=np.float64) for k, v in d.items()}

    # compute Pearson r per group based on stored sample (true vs pred)
    for name in metrics_group.keys():
        y = stored_np[name]["y"]
        p = stored_np[name]["pred"]
        metrics_group[name]["pearson_r"] = pearson_corrcoef(y, p)

    return {
        "metrics_dim": metrics_dim,
        "metrics_group": metrics_group,
        "stored_errors_by_group": stored_np,
    }


if __name__ == "__main__":

    # =========================
    # 1) Paths / I/O
    # =========================
    WORKPATH = "/home/user/Desktop/zyh/self-refl/GTransformer"
    DATAPATH = "/data2/zyh/yantian752_260105"
    DATASET_DIR = DATAPATH

    # Choose which checkpoint to evaluate
    EXP_NAME = "pretrain_masked_node_only_v1"
    OUT_DIR_TRAIN = os.path.join(WORKPATH, "runs", EXP_NAME)
    CKPT_PATH = os.path.join(OUT_DIR_TRAIN, "ckpt_best.pt")  # or ckpt_latest.pt

    # Eval outputs
    EVAL_NAME = "eval_masked_node_only_v1"
    OUT_DIR = os.path.join(OUT_DIR_TRAIN, EVAL_NAME)
    safe_makedirs(OUT_DIR)

    # Optional normalization files (for inverse transform)
    # If you have saved normalization stats, put them here:
    NORM_NPZ = os.path.join(DATASET_DIR, "normalization.npz")
    NORM_JSON = os.path.join(DATASET_DIR, "normalization.json")

    # =========================
    # 2) System
    # =========================
    SEED = 20260106
    set_all_seeds(SEED)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = True if torch.cuda.is_available() else False

    # =========================
    # 3) Data split (match your training split style)
    # =========================
    all_indices = scan_indices(DATASET_DIR)

    # If you intentionally limited total samples in training (e.g., 220000), keep consistent:
    # num_total = min(220000, len(all_indices))
    num_total = len(all_indices)

    train_ratio = 0.9
    num_train = int(num_total * train_ratio)

    train_indices = all_indices[:num_train]
    val_indices = all_indices[num_train:num_total]

    # Choose which split to evaluate: "val" | "train" | "all"
    EVAL_SPLIT = "val"

    if EVAL_SPLIT == "train":
        eval_indices = train_indices
    elif EVAL_SPLIT == "val":
        eval_indices = val_indices
    elif EVAL_SPLIT == "all":
        eval_indices = all_indices[:num_total]
    else:
        raise ValueError("EVAL_SPLIT must be 'train'|'val'|'all'.")

    # Optionally evaluate only a subset for quick check (None means full)
    EVAL_MAX_SAMPLES = None  # e.g., 5000
    if EVAL_MAX_SAMPLES is not None:
        eval_indices = eval_indices[:int(EVAL_MAX_SAMPLES)]

    # DataLoader params
    BATCH_SIZE = 32 if torch.cuda.is_available() else 2
    NUM_WORKERS = 16
    PIN_MEMORY = True

    # =========================
    # 4) Masking / Eval mode
    # =========================
    NODE_MASK_RATIO = 0.5
    EVAL_ON = "masked"  # "masked" | "all"
    N_MASK_REPEATS = 2  # repeat eval with different random masks, then aggregate (streaming)

    # Relative error epsilon
    EPS_REL = 1e-6

    # Store errors for plotting (cap to avoid OOM)
    MAX_STORE_ERRORS = 1_500_000

    # =========================
    # 5) Model config (load from ckpt if possible)
    # =========================
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    ckpt = torch.load(CKPT_PATH, map_location=device)
    if "cfg" in ckpt and isinstance(ckpt["cfg"], dict) and ("model" in ckpt["cfg"]):
        cfg_dict = ckpt["cfg"]["model"]
        cfg = GTConfig(**cfg_dict)
    else:
        raise RuntimeError(
            "Checkpoint does not contain cfg['model']. "
            "Please ensure your training checkpoint saved run_cfg with model config."
        )

    # Build a dataset once to know din
    ADJ_MODE = cfg.adj_mode if hasattr(cfg, "adj_mode") else "binary"
    DROP_DIAG = False

    eval_ds = YantianHYDatasetNodeOnly(DATASET_DIR, eval_indices, adj_mode=ADJ_MODE, drop_diag=DROP_DIAG)
    din = eval_ds.din

    # =========================
    # 6) Feature names & feature groups (请在这里按你的“特征类别”显式配置)
    # =========================

    FEATURE_NAMES = [f"f{i}" for i in range(din)]
    FEATURE_GROUPS = [{"name": f"f{i}", "dims": [i]} for i in range(din)]



    # =========================
    # 8) Build model & load weights
    # =========================
    model = GTransformer(cfg).to(device)

    # load state dict robustly (handle DataParallel prefix)
    state_dict = ckpt["model"]
    # If current model is not DataParallel, strip "module."
    state_dict_stripped = strip_module_prefix(state_dict)
    missing, unexpected = model.load_state_dict(state_dict_stripped, strict=False)
    if missing or unexpected:
        print("[Warn] load_state_dict non-strict:")
        print("  missing:", missing[:20], "..." if len(missing) > 20 else "")
        print("  unexpected:", unexpected[:20], "..." if len(unexpected) > 20 else "")

    model.eval()

    # =========================
    # 9) DataLoader
    # =========================
    eval_loader = DataLoader(
        eval_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
        collate_fn=collate_fn_node_only,
    )

    print("=" * 90)
    print(f"[Eval] CKPT: {CKPT_PATH}")
    print(f"[Eval] Split={EVAL_SPLIT}, samples={len(eval_ds)}, B={BATCH_SIZE}, workers={NUM_WORKERS}")
    print(f"[Eval] din={din}, node_mask_ratio={NODE_MASK_RATIO}, eval_on={EVAL_ON}, repeats={N_MASK_REPEATS}")
    print(f"[Out ] {OUT_DIR}")
    print("=" * 90)

    # =========================
    # 10) Run evaluation
    # =========================
    with torch.no_grad():
        res = evaluate(
            model=model,
            loader=eval_loader,
            device=device,
            amp=amp,
            node_mask_ratio=NODE_MASK_RATIO,
            denorm=denorm,
            feature_names=FEATURE_NAMES,
            feature_groups=FEATURE_GROUPS,
            eval_on=EVAL_ON,
            n_mask_repeats=N_MASK_REPEATS,
            max_store_errors=MAX_STORE_ERRORS,
            eps_rel=EPS_REL,
        )

    metrics_dim = res["metrics_dim"]
    metrics_group = res["metrics_group"]
    stored = res["stored_errors_by_group"]

    # =========================
    # 11) Save metrics (dim-level)
    # =========================
    dim_rows = []
    for i, name in enumerate(FEATURE_NAMES):
        dim_rows.append([
            name,
            int(metrics_dim["count"][i]),
            float(metrics_dim["mae"][i]),
            float(metrics_dim["rmse"][i]),
            float(metrics_dim["nrmse"][i]),
            float(metrics_dim["mape"][i]),
            float(metrics_dim["r2"][i]) if np.isfinite(metrics_dim["r2"][i]) else "",
        ])

    write_csv(
        os.path.join(OUT_DIR, "metrics_by_dim.csv"),
        header=["feature", "count", "mae", "rmse", "nrmse", "mape", "r2"],
        rows=dim_rows,
    )

    # group-level
    group_rows = []
    for gname, md in metrics_group.items():
        group_rows.append([
            gname,
            int(md["count"]),
            float(md["mae"]),
            float(md["rmse"]),
            float(md["nrmse"]),
            float(md["mape"]),
            float(md["r2"]) if np.isfinite(md["r2"]) else "",
            float(md.get("pearson_r", float("nan"))) if np.isfinite(md.get("pearson_r", float("nan"))) else "",
            str(md["dims"]),
        ])

    write_csv(
        os.path.join(OUT_DIR, "metrics_by_group.csv"),
        header=["group", "count", "mae", "rmse", "nrmse", "mape", "r2", "pearson_r", "dims"],
        rows=group_rows,
    )

    # Save a JSON summary
    summary = {
        "ckpt": CKPT_PATH,
        "split": EVAL_SPLIT,
        "num_samples": len(eval_ds),
        "node_mask_ratio": NODE_MASK_RATIO,
        "eval_on": EVAL_ON,
        "mask_repeats": N_MASK_REPEATS,
        "denorm": {"enabled": denorm.enabled(), "mode": denorm.mode},
        "group_metrics": metrics_group,
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # =========================
    # 12) Visualization
    # =========================
    # For each group:
    #   - raw error histogram
    #   - absolute error histogram
    #   - relative error histogram
    #   - scatter: true vs pred
    for g in FEATURE_GROUPS:
        name = g["name"]
        e = stored[name]["err"]
        ae = stored[name]["abs_err"]
        re = stored[name]["rel_err"]
        y = stored[name]["y"]
        p = stored[name]["pred"]

        plot_histogram(e,  title=f"[{name}] Error (pred-true)", out_path=os.path.join(OUT_DIR, f"hist_err_{name}.png"))
        plot_histogram(ae, title=f"[{name}] Abs Error |pred-true|", out_path=os.path.join(OUT_DIR, f"hist_abs_err_{name}.png"))
        plot_histogram(re, title=f"[{name}] Rel Error |err|/|true|", out_path=os.path.join(OUT_DIR, f"hist_rel_err_{name}.png"))

        plot_scatter(y, p, title=f"[{name}] Pred vs True (sample)", out_path=os.path.join(OUT_DIR, f"scatter_{name}.png"))

    # Optional: overall absolute error histogram (all groups merged)
    all_abs = np.concatenate([stored[g["name"]]["abs_err"] for g in FEATURE_GROUPS if stored[g["name"]]["abs_err"].size > 0], axis=0) \
        if len(FEATURE_GROUPS) > 0 else np.array([], dtype=np.float64)
    if all_abs.size > 0:
        plot_histogram(all_abs, title="[ALL] Abs Error |pred-true|", out_path=os.path.join(OUT_DIR, "hist_abs_err_ALL.png"))

    # =========================
    # 13) Print key results
    # =========================
    print("=" * 90)
    print("[完成] 评估结束，关键指标（按组）如下：")
    # sort by rmse descending
    sorted_groups = sorted(metrics_group.items(), key=lambda kv: (kv[1]["rmse"] if kv[1]["rmse"] is not None else 0.0), reverse=True)
    for gname, md in sorted_groups[: min(20, len(sorted_groups))]:
        print(
            f"  {gname:<20s} "
            f"count={md['count']:<10d} "
            f"MAE={md['mae']:.6g}  RMSE={md['rmse']:.6g}  "
            f"NRMSE={md['nrmse']:.6g}  MAPE={md['mape']:.6g}  "
            f"R2={md['r2'] if md['r2']==md['r2'] else 'nan'}  "
            f"r={md.get('pearson_r', float('nan')):.6g}"
        )

    print("-" * 90)
    print("已输出：")
    print(f"  - {os.path.join(OUT_DIR, 'metrics_by_dim.csv')}")
    print(f"  - {os.path.join(OUT_DIR, 'metrics_by_group.csv')}")
    print(f"  - {os.path.join(OUT_DIR, 'summary.json')}")
    print(f"  - 各组误差分布与散点图：{OUT_DIR}")
    print("=" * 90)


    # Print key metrics
    print("[Masked Metrics]")
    print("  masked_mae :", metrics["masked_mae"])
    print("  masked_rmse:", metrics["masked_rmse"])
    print("  masked_mse :", metrics["masked_mse"])
    print("  masked_abs_error_percentiles:", metrics["masked_abs_error_percentiles"])
    print("  per_sample_max_node_mae_percentiles:", metrics["per_sample_max_node_mae_percentiles"])
    if metrics["per_feature_masked_mae"] is not None:
        print("  per_feature_masked_mae :", metrics["per_feature_masked_mae"])
        print("  per_feature_masked_rmse:", metrics["per_feature_masked_rmse"])

    # Plots
    if masked_abs_errors.size > 0:
        plot_and_save_hist(
            masked_abs_errors,
            title="Masked Abs Error Distribution (all features on masked nodes)",
            xlabel="Absolute Error",
            out_path=os.path.join(EVAL_OUT_DIR, "hist_masked_abs_error.png"),
            bins=120
        )

    if per_sample_max_node_mae.size > 0:
        plot_and_save_hist(
            per_sample_max_node_mae,
            title="Per-sample Max Node MAE (masked nodes only)",
            xlabel="Max Node MAE",
            out_path=os.path.join(EVAL_OUT_DIR, "hist_per_sample_max_node_mae.png"),
            bins=80
        )

    top_nodes = plot_and_save_topk_nodes(
        per_node_avg_mae=per_node_avg_mae,
        per_node_cnt=per_node_cnt,
        out_path=os.path.join(EVAL_OUT_DIR, "topk_hard_nodes.png"),
        k=30
    )

    # Save top nodes list
    top_nodes_path = os.path.join(EVAL_OUT_DIR, "topk_hard_nodes.json")
    with open(top_nodes_path, "w", encoding="utf-8") as f:
        json.dump(
            [{"node": n, "avg_mae": mae, "masked_count": cnt} for (n, mae, cnt) in top_nodes],
            f, ensure_ascii=False, indent=2
        )

    # Also print the single worst node (by avg)
    if len(top_nodes) > 0:
        worst = top_nodes[0]
        print("=" * 90)
        print(f"[Worst Node] node={worst[0]}, avg_masked_node_mae={worst[1]:.6f}, masked_count={worst[2]}")
        print(f"[Outputs saved] {EVAL_OUT_DIR}")
        print("  - val_metrics.json")
        print("  - hist_masked_abs_error.png")
        print("  - hist_per_sample_max_node_mae.png")
        print("  - topk_hard_nodes.png")
        print("  - topk_hard_nodes.json")
        print("=" * 90)
    else:
        print("[Warning] No valid masked nodes encountered (unexpected). Check mask_ratio or data loader.")

# %%
