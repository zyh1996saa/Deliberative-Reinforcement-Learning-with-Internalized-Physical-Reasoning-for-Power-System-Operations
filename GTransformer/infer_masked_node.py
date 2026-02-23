# In[]
# 04_infer_masked_node.py
# Inference for masked-node reconstruction with Graph Transformer (Torch)
# Dataset: H_{idx}.npy + Y_{idx}.npz
# Pipeline:
#   load one sample -> build adjacency -> fixed mask ratio -> model inference -> compare on masked nodes

from __future__ import annotations

import os
import re
import sys
sys.path.append(r"/home/user/Desktop/zyh/self-refl/GTransformer")

import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from scipy.sparse import load_npz

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
        raise RuntimeError(f"No valid samples found in {dataset_dir}")
    return indices


def load_one_sample(dataset_dir: str, idx: int, adj_mode: str = "binary", drop_diag: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      H: float32 [N, din]
      A: float32 [N, N]
    """
    h_path = os.path.join(dataset_dir, f"H_{idx}.npy")
    y_path = os.path.join(dataset_dir, f"Y_{idx}.npz")

    if not os.path.exists(h_path):
        raise FileNotFoundError(h_path)
    if not os.path.exists(y_path):
        raise FileNotFoundError(y_path)

    H = np.load(h_path).astype(np.float32)  # [N, din]
    Y = load_npz(y_path)
    A = ybus_to_adjacency(Y, mode=adj_mode, drop_diagonal=drop_diag).astype(np.float32)  # [N, N]
    return H, A


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """If keys start with 'module.', strip it (for DataParallel compatibility)."""
    if not state_dict:
        return state_dict
    has_module = any(k.startswith("module.") for k in state_dict.keys())
    if not has_module:
        return state_dict
    return {k[len("module."):]: v for k, v in state_dict.items()}


def build_model_from_checkpoint(ckpt_path: str, device: torch.device) -> nn.Module:
    """
    Build GTransformer from checkpoint cfg and load weights.
    Works for both DP/non-DP checkpoints.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    if "cfg" not in ckpt or "model" not in ckpt["cfg"]:
        raise RuntimeError("Checkpoint does not contain cfg['model']. Please check your ckpt format.")

    model_cfg = ckpt["cfg"]["model"]

    # Safety: only pass keys existing in GTConfig
    valid_keys = set(GTConfig.__annotations__.keys())
    filtered = {k: model_cfg[k] for k in model_cfg.keys() if k in valid_keys}

    cfg = GTConfig(**filtered)
    model = GTransformer(cfg).to(device)

    sd = ckpt.get("model", None)
    if sd is None:
        raise RuntimeError("Checkpoint does not contain 'model' state_dict.")

    try:
        model.load_state_dict(_strip_module_prefix(sd), strict=True)
    except RuntimeError:
        # maybe dp/non-dp mismatch, try raw
        try:
            model.load_state_dict(sd, strict=True)
        except RuntimeError as e:
            raise RuntimeError(f"Failed to load state_dict: {e}")

    model.eval()
    return model


def make_fixed_ratio_node_mask(
    N: int,
    mask_ratio: float,
    seed: int,
    device: torch.device,
    exact_count: bool = True
) -> torch.Tensor:
    """
    Return:
      node_mask: bool [N], True = masked node

    exact_count=True:
      exactly mask round(N*mask_ratio) nodes.
    """
    assert 0.0 < mask_ratio < 1.0
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))

    m = int(round(N * mask_ratio))
    m = max(1, min(m, N))

    if exact_count:
        perm = torch.randperm(N, generator=g)
        mask_idx = perm[:m]
        node_mask = torch.zeros(N, dtype=torch.bool)
        node_mask[mask_idx] = True
    else:
        node_mask = (torch.rand(N, generator=g) < mask_ratio)

    return node_mask.to(device)


# -----------------------------
# Metrics (masked nodes only)
# -----------------------------
def _safe_quantile_1d(x: torch.Tensor, q: float) -> float:
    """x: 1D tensor"""
    if x.numel() == 0:
        return 0.0
    # torch.quantile exists in recent PyTorch; keep as-is
    return float(torch.quantile(x, q).item())


def compute_masked_metrics(
    H_true: torch.Tensor,         # [N, din]
    H_pred: torch.Tensor,         # [N, din]
    node_mask: torch.Tensor       # [N] bool
) -> Dict[str, object]:
    """
    Metrics computed ONLY on masked nodes.

    Notes:
      - "masked_mse/mae": averaged over ALL masked entries (masked nodes × din features).
      - Feature-wise metrics: per-dimension error stats (length=din).
      - Node-wise RMSE: per-node aggregation over features, for masked nodes only.
      - Also reports global max abs error entry and its (node, feature) location.
    """
    assert H_true.shape == H_pred.shape
    H_true = H_true.float()
    H_pred = H_pred.float()

    N, din = H_true.shape
    masked_nodes = int(node_mask.sum().item())

    # If somehow no masked nodes (shouldn't happen with your mask generator), return zeros safely
    if masked_nodes <= 0:
        return {
            "masked_node_count": 0,
            "masked_mse": 0.0,
            "masked_mae": 0.0,
            "masked_node_rmse_mean": 0.0,
            "masked_node_rmse_p50": 0.0,
            "masked_node_rmse_p90": 0.0,
            "masked_node_rmse_p99": 0.0,
            "masked_node_rmse_max": 0.0,
            "masked_node_rmse_max_node": -1,
            "per_feature_mse": [0.0] * din,
            "per_feature_mae": [0.0] * din,
            "per_feature_rmse": [0.0] * din,
            "per_feature_abs_p50": [0.0] * din,
            "per_feature_abs_p90": [0.0] * din,
            "per_feature_abs_p99": [0.0] * din,
            "per_feature_abs_max": [0.0] * din,
            "global_abs_max": 0.0,
            "global_abs_max_node": -1,
            "global_abs_max_feature": -1,
        }

    # diff on all nodes
    diff = (H_pred - H_true)  # [N, din]

    # masked rows only
    diff_m = diff[node_mask]  # [M, din]
    abs_m = diff_m.abs()      # [M, din]

    # 1) overall masked entry metrics (average over M*din entries)
    masked_mse = float((diff_m ** 2).mean().item())
    masked_mae = float(abs_m.mean().item())

    # 2) feature-wise metrics (length din)
    per_feat_mse_t = (diff_m ** 2).mean(dim=0)     # [din]
    per_feat_mae_t = abs_m.mean(dim=0)             # [din]
    per_feat_rmse_t = torch.sqrt(per_feat_mse_t + 1e-12)

    per_feat_abs_p50 = []
    per_feat_abs_p90 = []
    per_feat_abs_p99 = []
    per_feat_abs_max = []

    for j in range(din):
        col = abs_m[:, j]
        per_feat_abs_p50.append(_safe_quantile_1d(col, 0.50))
        per_feat_abs_p90.append(_safe_quantile_1d(col, 0.90))
        per_feat_abs_p99.append(_safe_quantile_1d(col, 0.99))
        per_feat_abs_max.append(float(col.max().item()) if col.numel() > 0 else 0.0)

    # 3) node-wise RMSE (aggregate over features) for masked nodes
    per_node_mse = (diff ** 2).mean(dim=-1)               # [N]
    per_node_rmse = torch.sqrt(per_node_mse + 1e-12)      # [N]
    masked_rmse = per_node_rmse[node_mask]                # [M]

    rmse_mean = float(masked_rmse.mean().item())
    rmse_p50 = _safe_quantile_1d(masked_rmse, 0.50)
    rmse_p90 = _safe_quantile_1d(masked_rmse, 0.90)
    rmse_p99 = _safe_quantile_1d(masked_rmse, 0.99)
    max_rmse = float(masked_rmse.max().item())

    # index of max rmse node in original node space
    # (mask false -> -1, mask true -> rmse)
    max_node = int(torch.argmax(torch.where(node_mask, per_node_rmse, torch.tensor(-1.0, device=H_true.device))).item())

    # 4) global max abs error entry among masked nodes (node, feature)
    # abs_m: [M, din]
    flat_idx = int(torch.argmax(abs_m.reshape(-1)).item())
    max_abs = float(abs_m.reshape(-1)[flat_idx].item())
    m_idx = flat_idx // din          # index within masked subset
    f_idx = flat_idx % din           # feature dim
    # map masked-subset index back to global node index
    masked_node_indices = torch.nonzero(node_mask, as_tuple=False).view(-1)  # [M]
    g_node = int(masked_node_indices[m_idx].item())

    return {
        # overall masked entry
        "masked_node_count": masked_nodes,
        "masked_mse": masked_mse,
        "masked_mae": masked_mae,

        # node-wise distribution (masked nodes only)
        "masked_node_rmse_mean": rmse_mean,
        "masked_node_rmse_p50": rmse_p50,
        "masked_node_rmse_p90": rmse_p90,
        "masked_node_rmse_p99": rmse_p99,
        "masked_node_rmse_max": max_rmse,
        "masked_node_rmse_max_node": max_node,

        # feature-wise metrics (masked nodes only)
        "per_feature_mse": [float(x) for x in per_feat_mse_t.detach().cpu().tolist()],
        "per_feature_mae": [float(x) for x in per_feat_mae_t.detach().cpu().tolist()],
        "per_feature_rmse": [float(x) for x in per_feat_rmse_t.detach().cpu().tolist()],
        "per_feature_abs_p50": [float(x) for x in per_feat_abs_p50],
        "per_feature_abs_p90": [float(x) for x in per_feat_abs_p90],
        "per_feature_abs_p99": [float(x) for x in per_feat_abs_p99],
        "per_feature_abs_max": [float(x) for x in per_feat_abs_max],

        # global worst entry among masked nodes
        "global_abs_max": max_abs,
        "global_abs_max_node": g_node,
        "global_abs_max_feature": int(f_idx),
    }


# -----------------------------
# Inference
# -----------------------------
@torch.no_grad()
def infer_one_sample(
    dataset_dir: str,
    idx: int,
    ckpt_path: str,
    mask_ratio: float,
    mask_seed: int,
    device: Optional[torch.device] = None,
    amp: bool = True,
    adj_mode: str = "binary",
    drop_diag: bool = False,
    exact_mask_count: bool = True,
) -> Dict:
    """
    One-shot inference function.

    Steps:
      1) load (H, A)
      2) build fixed mask with ratio
      3) model forward to get node_pred
      4) stitch reconstructed H_recon (masked nodes use prediction)
      5) compare metrics ONLY on masked nodes
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(amp and device.type == "cuda")

    H_np, A_np = load_one_sample(dataset_dir, idx, adj_mode=adj_mode, drop_diag=drop_diag)
    if not np.isfinite(H_np).all():
        raise RuntimeError(f"H contains non-finite values at idx={idx}")
    if not np.isfinite(A_np).all():
        raise RuntimeError(f"A contains non-finite values at idx={idx}")

    N, din = H_np.shape

    node_mask = make_fixed_ratio_node_mask(
        N=N,
        mask_ratio=mask_ratio,
        seed=mask_seed,
        device=device,
        exact_count=exact_mask_count,
    )  # [N] bool

    feature_mask = (~node_mask).float().unsqueeze(-1).expand(N, din)  # [N, din]

    H = torch.tensor(H_np, dtype=torch.float32, device=device)  # [N, din]
    A = torch.tensor(A_np, dtype=torch.float32, device=device)  # [N, N]

    model = build_model_from_checkpoint(ckpt_path, device=device)

    with autocast(device_type="cuda", enabled=use_amp):
        out = model(H, A, feature_mask=feature_mask, return_embeddings=False)
        node_pred = out["node_pred"]  # may be float16 under autocast

    node_pred_f = node_pred.float()

    H_recon = H.clone()
    H_recon[node_mask] = node_pred_f[node_mask]

    metrics = compute_masked_metrics(H_true=H, H_pred=node_pred_f, node_mask=node_mask)

    return {
        "idx": int(idx),
        "H_true": H_np,
        "A": A_np,
        "node_mask": node_mask.detach().cpu().numpy().astype(bool),
        "node_pred": node_pred_f.detach().cpu().numpy().astype(np.float32),
        "H_recon": H_recon.detach().cpu().numpy().astype(np.float32),
        "metrics": metrics,
        "ckpt_path": ckpt_path,
        "mask_ratio": float(mask_ratio),
        "mask_seed": int(mask_seed),
        "adj_mode": adj_mode,
        "drop_diag": bool(drop_diag),
    }


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":

    WORKPATH = "/home/user/Desktop/zyh/self-refl/GTransformer"
    DATASET_DIR = "/data2/zyh/yantian752_260105"

    EXP_NAME = "pretrain_masked_node_only_v1"
    OUT_DIR = os.path.join(WORKPATH, "runs", EXP_NAME)
    CKPT_BEST = os.path.join(OUT_DIR, "ckpt_best.pt")
    CKPT_LATEST = os.path.join(OUT_DIR, "ckpt_latest.pt")
    CKPT_PATH = CKPT_BEST if os.path.exists(CKPT_BEST) else CKPT_LATEST

    # Inference settings
    SAMPLE_IDX = 60000
    MASK_RATIO = 0.5
    MASK_SEED = 20260105
    ADJ_MODE = "binary"
    DROP_DIAG = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = True if device.type == "cuda" else False

    set_all_seeds(20260105)

    result = infer_one_sample(
        dataset_dir=DATASET_DIR,
        idx=SAMPLE_IDX,
        ckpt_path=CKPT_PATH,
        mask_ratio=MASK_RATIO,
        mask_seed=MASK_SEED,
        device=device,
        amp=amp,
        adj_mode=ADJ_MODE,
        drop_diag=DROP_DIAG,
        exact_mask_count=True,
    )

    # =========================================================
    # 在主函数中明确定义：真实矩阵 / 预测矩阵（以及可选的重建矩阵）
    # =========================================================
    H_true_mat: np.ndarray = result["H_true"]      # [N, din] 真实节点特征
    H_pred_mat: np.ndarray = result["node_pred"]   # [N, din] 模型预测输出（所有节点的预测）
    H_recon_mat: np.ndarray = result["H_recon"]    # [N, din] 重建结果：masked用预测，unmasked用真实（可选但很实用）
    node_mask: np.ndarray = result["node_mask"]    # [N] bool

    # 可选：只取 masked 节点对应的真实/预测，用于进一步分析或画图
    H_true_masked: np.ndarray = H_true_mat[node_mask]  # [M, din]
    H_pred_masked: np.ndarray = H_pred_mat[node_mask]  # [M, din]

    # 基本一致性检查（建议保留）
    assert H_true_mat.shape == H_pred_mat.shape == H_recon_mat.shape, "H matrices shape mismatch."
    assert node_mask.ndim == 1 and node_mask.shape[0] == H_true_mat.shape[0], "node_mask shape mismatch."

    # 你后续如果要把矩阵传给别的代码/函数，这里就已经是明确变量了
    # =========================================================

    m = result["metrics"]
    print("=" * 90)
    print(f"[Infer] idx={result['idx']}, ckpt={result['ckpt_path']}")
    print(f"[Mask] ratio={result['mask_ratio']}, seed={result['mask_seed']}, masked_nodes={m['masked_node_count']}")
    print("-" * 90)

    print("[Overall masked-entry metrics] (average over masked nodes × features)")
    print(f"  masked_mse: {m['masked_mse']:.6e}")
    print(f"  masked_mae: {m['masked_mae']:.6e}")

    print("[Masked-node RMSE distribution] (per-node RMSE over features)")
    print(f"  rmse_mean: {m['masked_node_rmse_mean']:.6e}")
    print(f"  rmse_p50 : {m['masked_node_rmse_p50']:.6e}")
    print(f"  rmse_p90 : {m['masked_node_rmse_p90']:.6e}")
    print(f"  rmse_p99 : {m['masked_node_rmse_p99']:.6e}")
    print(f"  rmse_max : {m['masked_node_rmse_max']:.6e} (node={m['masked_node_rmse_max_node']})")

    print("[Global worst masked entry]")
    print(
        f"  max|error|={m['global_abs_max']:.6e} at node={m['global_abs_max_node']}, feature={m['global_abs_max_feature']}"
    )

    print("=" * 90)

    # Optional save examples:
    # np.save(os.path.join(OUT_DIR, f"H_true_{SAMPLE_IDX}.npy"), H_true_mat)
    # np.save(os.path.join(OUT_DIR, f"H_pred_{SAMPLE_IDX}.npy"), H_pred_mat)
    # np.save(os.path.join(OUT_DIR, f"H_recon_{SAMPLE_IDX}.npy"), H_recon_mat)


# %%
