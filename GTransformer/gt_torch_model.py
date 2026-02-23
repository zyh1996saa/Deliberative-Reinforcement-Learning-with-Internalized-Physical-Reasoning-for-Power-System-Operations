# In[]
# gt_torch_model.py
# Torch implementation of the paper's Graph Transformer (GTransformer) core:
# DyMPN + Graphical Multi-head Attention, designed to fit (H.npy, Y.npz) samples.
#
# Expected data format per sample:
#   - H_{idx}.npy : numpy array, shape [N, din], float
#   - Y_{idx}.npz : scipy sparse matrix (csr/csc), shape [N, N], complex or float (Ybus)
#
# Notes:
#   1) We derive adjacency A from Ybus by default: A_ij = 1 if |Y_ij|>0 else 0 (excluding diagonal optionally).
#   2) This file focuses on the MODEL. Training scripts can call:
#        - model.forward(H, A) for node representation / reconstruction
#        - model.predict_edges(z, edge_index) for edge (Y) reconstruction on selected edges

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.sparse import load_npz


# --------------------------
# I/O helpers for your dataset
# --------------------------

def load_HY_sample(datapath: str, idx: int) -> Tuple[np.ndarray, "scipy.sparse.spmatrix"]:
    """
    Load one sample from:
      H_{idx}.npy and Y_{idx}.npz

    Returns
    -------
    H : np.ndarray, shape [N, din]
    Y : scipy sparse matrix, shape [N, N] (Ybus)
    """
    h_path = os.path.join(datapath, f"H_{idx}.npy")
    y_path = os.path.join(datapath, f"Y_{idx}.npz")

    H = np.load(h_path)  # [N, din]
    Y = load_npz(y_path)  # sparse [N, N], may be complex

    return H, Y


def ybus_to_adjacency(
    Y,
    mode: str = "binary",
    drop_diagonal: bool = False,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Convert sparse Ybus to dense adjacency A.

    Parameters
    ----------
    mode:
      - "binary": A_ij = 1 if |Y_ij|>0 else 0
      - "abs":    A_ij = |Y_ij|
      - "real":   A_ij = Re(Y_ij)
      - "imag":   A_ij = Im(Y_ij)

    drop_diagonal:
      if True, set diagonal to 0 (self-loops can be added later in normalization)

    Returns
    -------
    A : np.ndarray float32, shape [N, N]
    """
    # Work in COO for easy access
    Y_coo = Y.tocoo()
    n = Y_coo.shape[0]

    A = np.zeros((n, n), dtype=np.float32)

    if mode == "binary":
        vals = np.ones_like(Y_coo.data, dtype=np.float32)
    elif mode == "abs":
        vals = np.abs(Y_coo.data).astype(np.float32)
    elif mode == "real":
        vals = np.real(Y_coo.data).astype(np.float32)
    elif mode == "imag":
        vals = np.imag(Y_coo.data).astype(np.float32)
    else:
        raise ValueError(f"Unknown mode={mode}. Use binary/abs/real/imag.")

    # threshold small values
    vals[np.abs(vals) < eps] = 0.0

    A[Y_coo.row, Y_coo.col] = vals

    if drop_diagonal:
        np.fill_diagonal(A, 0.0)

    return A


def normalize_adjacency_dense(A: torch.Tensor, add_self_loops: bool = True, eps: float = 1e-12) -> torch.Tensor:
    """
    Row-normalize adjacency: A_norm = D^{-1} (A + I)

    Supports:
      - A shape [N, N]
      - A shape [B, N, N]

    Returns same shape as input.
    """
    if add_self_loops:
        if A.dim() == 2:
            A = A + torch.eye(A.size(0), device=A.device, dtype=A.dtype)
        elif A.dim() == 3:
            b, n, _ = A.shape
            I = torch.eye(n, device=A.device, dtype=A.dtype).unsqueeze(0).expand(b, n, n)
            A = A + I
        else:
            raise ValueError(f"A must be 2D or 3D, got {A.dim()}D")

    # degree
    deg = A.sum(dim=-1, keepdim=True).clamp_min(eps)  # [N,1] or [B,N,1]
    return A / deg


def adjacency_to_edge_index(A: torch.Tensor, include_self: bool = False) -> torch.Tensor:
    """
    Build edge_index from dense adjacency A.

    A: [N, N] (single graph)
    Returns: edge_index [2, E]
    """
    if A.dim() != 2:
        raise ValueError("adjacency_to_edge_index expects A with shape [N, N].")

    mask = A != 0
    if not include_self:
        n = A.size(0)
        diag = torch.eye(n, device=A.device, dtype=torch.bool)
        mask = mask & (~diag)

    src, dst = mask.nonzero(as_tuple=True)
    return torch.stack([src, dst], dim=0)  # [2, E]


# --------------------------
# Model components
# --------------------------

@dataclass
class GTConfig:
    din: int
    d_model: int = 128
    n_heads: int = 8
    d_ff: int = 256
    n_layers: int = 3

    # DyMPN settings: K ~ Uniform[k_min, k_max] during training
    k_min: int = 1
    k_max: int = 5

    dropout: float = 0.1
    attn_dropout: float = 0.1

    # adjacency derived from Ybus
    adj_mode: str = "binary"  # binary/abs/real/imag


class DyMPN(nn.Module):
    """
    Dynamic Message Passing Network (DyMPN):
      - Randomly choose K steps during training (K in [k_min, k_max])
      - Each step: H <- ReLU( A_norm @ H @ W + b )

    This is a practical torch version aligned to the paper description of (4),
    using row-normalized adjacency (A_ij / D_i).
    """

    def __init__(self, d_model: int, k_min: int = 1, k_max: int = 5, dropout: float = 0.0):
        super().__init__()
        assert k_min >= 1 and k_max >= k_min
        self.d_model = d_model
        self.k_min = k_min
        self.k_max = k_max
        self.lin = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _sample_k(self) -> int:
        if self.training:
            return int(torch.randint(low=self.k_min, high=self.k_max + 1, size=(1,)).item())
        return self.k_max

    def forward(self, x: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        """
        x:      [B, N, d] or [N, d]
        A_norm: [B, N, N] or [N, N]
        """
        K = self._sample_k()

        if x.dim() == 2:
            # [N, d]
            for _ in range(K):
                agg = A_norm @ x  # [N, d]
                x = F.relu(self.lin(agg))
                x = self.dropout(x)
            return x

        if x.dim() == 3:
            # [B, N, d]
            if A_norm.dim() == 2:
                # share same adjacency across batch
                for _ in range(K):
                    agg = torch.matmul(A_norm, x)  # broadcast matmul -> [B, N, d]
                    x = F.relu(self.lin(agg))
                    x = self.dropout(x)
                return x

            # per-sample adjacency
            for _ in range(K):
                agg = torch.bmm(A_norm, x)  # [B, N, d]
                x = F.relu(self.lin(agg))
                x = self.dropout(x)
            return x

        raise ValueError(f"Unsupported x.dim()={x.dim()}")


class GraphMultiHeadAttention(nn.Module):
    """
    Graphical multi-head attention:
      - Build q/k/v from THREE DyMPNs (same structure, different weights)
      - Then standard scaled dot-product multi-head attention

    Output: [B, N, d_model]
    """

    def __init__(self, d_model: int, n_heads: int, k_min: int, k_max: int, dropout: float, attn_dropout: float):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # 3 DyMPNs for q/k/v
        self.dympn_q = DyMPN(d_model, k_min=k_min, k_max=k_max, dropout=dropout)
        self.dympn_k = DyMPN(d_model, k_min=k_min, k_max=k_max, dropout=dropout)
        self.dympn_v = DyMPN(d_model, k_min=k_min, k_max=k_max, dropout=dropout)

        # linear projections after DyMPN
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(
        self,
        x: torch.Tensor,         # [B, N, d]
        A_norm: torch.Tensor,    # [B, N, N] or [N, N]
        attn_bias: Optional[torch.Tensor] = None,  # [B, 1 or H, N, N] additive bias (optional)
        attn_mask: Optional[torch.Tensor] = None,  # [B, 1 or H, N, N] boolean mask (True=keep)
    ) -> torch.Tensor:
        B, N, _ = x.shape

        # DyMPN -> q/k/v bases
        q_base = self.dympn_q(x, A_norm)
        k_base = self.dympn_k(x, A_norm)
        v_base = self.dympn_v(x, A_norm)

        # Linear projections
        q = self.Wq(q_base)  # [B, N, d]
        k = self.Wk(k_base)
        v = self.Wv(v_base)

        # reshape to heads
        q = q.view(B, N, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, N, Dh]
        k = k.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        # scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B, H, N, N]

        if attn_bias is not None:
            # allow [B,1,N,N] or [B,H,N,N]
            scores = scores + attn_bias

        if attn_mask is not None:
            # boolean mask: True keeps, False masks out
            scores = scores.masked_fill(~attn_mask, -1e9)

        weights = torch.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        out = torch.matmul(weights, v)  # [B, H, N, Dh]
        out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)  # [B, N, d]

        out = self.out_proj(out)
        out = self.dropout(out)
        return out


class MPN(nn.Module):
    """
    A simple (fixed-step) message passing block used after attention in each layer
    to re-inject local topology information.
    """

    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.lin = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            if A_norm.dim() == 2:
                agg = torch.matmul(A_norm, x)
            else:
                agg = torch.bmm(A_norm, x)
            y = F.relu(self.lin(agg))
            return self.dropout(y)

        if x.dim() == 2:
            agg = A_norm @ x
            y = F.relu(self.lin(agg))
            return self.dropout(y)

        raise ValueError(f"Unsupported x.dim()={x.dim()}")


class GTransformerBlock(nn.Module):
    """
    One Graph Transformer block (practical implementation):

      x -> GraphMHA(DyMPN-q/k/v) -> Add&Norm
        -> MPN(local) -> concat([x, mpn(x)]) -> FF -> Add&Norm
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, k_min: int, k_max: int, dropout: float, attn_dropout: float):
        super().__init__()
        self.attn = GraphMultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            k_min=k_min,
            k_max=k_max,
            dropout=dropout,
            attn_dropout=attn_dropout,
        )
        self.ln1 = nn.LayerNorm(d_model)

        self.mpn = MPN(d_model, dropout=dropout)

        self.ff = nn.Sequential(
            nn.Linear(2 * d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, A_norm: torch.Tensor, attn_bias: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # attention
        attn_out = self.attn(x, A_norm, attn_bias=attn_bias, attn_mask=attn_mask)
        x = self.ln1(x + attn_out)

        # local MPN
        mpn_out = self.mpn(x, A_norm)

        # concat + FF
        cat = torch.cat([x, mpn_out], dim=-1)
        ff_out = self.ff(cat)
        x = self.ln2(x + ff_out)
        return x


class EdgePredictor(nn.Module):
    """
    Pairwise edge predictor:
      Given node embeddings z_i, z_j -> predict edge attributes (e.g., Re(Y_ij), Im(Y_ij)) or other.

    For masked-edge pretraining, you typically call:
      y_hat = edge_pred(z, edge_index)  # only on selected edges

    Output dimension default=2 for (real, imag).
    """

    def __init__(self, d_model: int, out_dim: int = 2, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        z: [B, N, d] or [N, d]
        edge_index: [2, E] with (src, dst)
        returns: [B, E, out_dim] or [E, out_dim]
        """
        src, dst = edge_index[0], edge_index[1]

        if z.dim() == 2:
            zi = z[src]  # [E, d]
            zj = z[dst]
            return self.mlp(torch.cat([zi, zj], dim=-1))

        if z.dim() == 3:
            zi = z[:, src, :]  # [B, E, d]
            zj = z[:, dst, :]
            return self.mlp(torch.cat([zi, zj], dim=-1))

        raise ValueError(f"Unsupported z.dim()={z.dim()}")


class GTransformer(nn.Module):
    """
    Full encoder + heads:
      - embed(H) -> stacked GTransformer blocks -> node embeddings z
      - node_head(z) -> reconstruct node features (masked feature prediction / general reconstruction)
      - edge_head(z, edge_index) -> reconstruct edge attributes on selected edges
    """

    def __init__(self, cfg: GTConfig):
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Linear(cfg.din, cfg.d_model)

        self.blocks = nn.ModuleList([
            GTransformerBlock(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                d_ff=cfg.d_ff,
                k_min=cfg.k_min,
                k_max=cfg.k_max,
                dropout=cfg.dropout,
                attn_dropout=cfg.attn_dropout,
            )
            for _ in range(cfg.n_layers)
        ])

        self.node_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.din),
        )

        # default predict (Re, Im) of Y_ij; you can set out_dim=1 if you only want abs/binary
        self.edge_head = EdgePredictor(cfg.d_model, out_dim=2, hidden=cfg.d_ff, dropout=cfg.dropout)

    def forward(
        self,
        H: torch.Tensor,                    # [B, N, din] or [N, din]
        A: torch.Tensor,                    # [B, N, N] or [N, N] dense adjacency (derived from Ybus)
        feature_mask: Optional[torch.Tensor] = None,  # [B, N, din] or [N, din], 1 keep / 0 mask
        attn_bias: Optional[torch.Tensor] = None,     # [B,1 or H,N,N]
        attn_mask: Optional[torch.Tensor] = None,     # [B,1 or H,N,N] boolean
        return_embeddings: bool = True,
    ) -> Dict[str, torch.Tensor]:
        # Apply feature mask by zeroing masked inputs (paper uses setting to zero)
        if feature_mask is not None:
            H_in = H * feature_mask
        else:
            H_in = H

        # Embed to d_model
        x = self.embed(H_in)

        # Normalize adjacency (row-normalize, optionally add self-loops)
        A_norm = normalize_adjacency_dense(A, add_self_loops=True)

        # Ensure batched
        if x.dim() == 2:
            x = x.unsqueeze(0)        # [1, N, d]
            A_norm = A_norm.unsqueeze(0) if A_norm.dim() == 2 else A_norm

            if attn_bias is not None and attn_bias.dim() == 3:
                attn_bias = attn_bias.unsqueeze(0)
            if attn_mask is not None and attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(0)

            squeezed = True
        else:
            squeezed = False

        # Blocks
        for blk in self.blocks:
            x = blk(x, A_norm, attn_bias=attn_bias, attn_mask=attn_mask)

        # Node reconstruction head
        node_pred = self.node_head(x)  # [B, N, din]

        out = {"node_pred": node_pred}
        if return_embeddings:
            out["z"] = x

        if squeezed:
            out = {k: v.squeeze(0) for k, v in out.items()}

        return out

    @torch.no_grad()
    def encode(self, H: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        return self.forward(H, A, return_embeddings=True)["z"]

    def predict_edges(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.edge_head(z, edge_index)


# --------------------------
# Minimal sanity check (optional)
# --------------------------
if __name__ == "__main__":
    # Adjust these to your environment when you actually run:
    DATAPATH = "/data2/zyh/yantian752_260105"
    SAMPLE_IDX = 0

    # Load numpy/scipy data
    H_np, Y_sp = load_HY_sample(DATAPATH, SAMPLE_IDX)
    A_np = ybus_to_adjacency(Y_sp, mode="binary", drop_diagonal=False)

    # Torch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H = torch.tensor(H_np, dtype=torch.float32, device=device)          # [N, din]
    A = torch.tensor(A_np, dtype=torch.float32, device=device)          # [N, N]

    cfg = GTConfig(
        din=H.size(-1),
        d_model=128,
        n_heads=8,
        d_ff=256,
        n_layers=3,
        k_min=1,
        k_max=5,
        dropout=0.1,
        attn_dropout=0.1,
        adj_mode="binary",
    )
    model = GTransformer(cfg).to(device)
    model.train()

    out = model(H, A, return_embeddings=True)
    print("node_pred:", out["node_pred"].shape)  # [N, din]
    print("z:", out["z"].shape)                  # [N, d_model]

    # edge prediction example on existing edges
    edge_index = adjacency_to_edge_index(A, include_self=False)
    y_hat = model.predict_edges(out["z"], edge_index)  # [E, 2]
    print("edge_pred:", y_hat.shape)

# %%
