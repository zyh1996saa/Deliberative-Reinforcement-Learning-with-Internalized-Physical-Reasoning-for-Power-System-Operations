# In[]
# -*- coding: utf-8 -*-
"""
run_trained_ppo_gt_interact_wo_thinking.py

Load a PPO+GT (Multi-Branch Actor-Critic) checkpoint and interact with PowerDispatchEnv continuously,
printing intermediate variables (state/decision/reward and their sub-items).

No argparse: edit parameters in __main__.

Assumptions:
- power_dispatch_env_withGT.py is available and provides:
    PowerDispatchEnv, EnvConfig, TimeseriesSchema,
    built_ppnet_for_pfcal, set_fc_state_with_acts
- The RL checkpoint is produced by your PPO training script and contains:
    ckpt["model"], ckpt["optimizer"], ckpt["normalizer"], ckpt["extra"]["gt_ckpt_path"] (optional)
"""

from __future__ import annotations

import os
import math
import time
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from power_dispatch_env_withGT import (
    PowerDispatchEnv,
    EnvConfig,
    TimeseriesSchema,
    built_ppnet_for_pfcal,
    set_fc_state_with_acts,
)

# ============================================================
# seeds / device
# ============================================================
def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_str: str) -> torch.device:
    device_str = (device_str or "auto").lower().strip()
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_str in ("cuda", "cpu"):
        return torch.device(device_str)
    raise ValueError(f"Unknown device: {device_str}. Use 'auto'/'cpu'/'cuda'.")


# ============================================================
# observation flatten + normalization (same as train)
# ============================================================
class RunningMeanStd:
    def __init__(self, shape: Tuple[int, ...], eps: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = eps

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta**2) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


class ObsFlattenerV2:
    """
    flat 顺序（固定）：
        bus_vm_pu
        bus_va_deg
        load_p_mw
        load_q_mvar
        forecast_p_mw
        forecast_q_mvar
        time_feat: time_scaled + sin + cos      [3]
        topology_id                              [1]
    gt_H 不放入 flat（单独传入）。
    """

    def __init__(self, obs_space: Any, *, n_actions: int, time_scale: float = 1e-3, time_period: int = 24):
        self.time_scale = float(time_scale)
        self.time_period = int(max(time_period, 1))
        self.n_actions = int(max(n_actions, 1))

        self.n_bus = int(np.prod(obs_space["bus_vm_pu"].shape))
        self.n_load = int(np.prod(obs_space["load_p_mw"].shape))
        H = int(obs_space["forecast_p_mw"].shape[0])
        self.horizon = int(H)

        self.flat_dim = (
            self.n_bus
            + self.n_bus
            + self.n_load
            + self.n_load
            + (self.horizon * self.n_load)
            + (self.horizon * self.n_load)
            + 3
            + 1
        )

        off = 0
        self.slices: Dict[str, slice] = {}

        def _add(name: str, length: int) -> None:
            nonlocal off
            self.slices[name] = slice(off, off + int(length))
            off += int(length)

        _add("bus_vm_pu", self.n_bus)
        _add("bus_va_deg", self.n_bus)
        _add("load_p_mw", self.n_load)
        _add("load_q_mvar", self.n_load)
        _add("forecast_p_mw", self.horizon * self.n_load)
        _add("forecast_q_mvar", self.horizon * self.n_load)
        _add("time_feat", 3)
        _add("topology_id", 1)

        assert off == self.flat_dim, (off, self.flat_dim)

    def flatten(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        parts: List[np.ndarray] = []
        parts.append(np.asarray(obs["bus_vm_pu"], dtype=np.float32).reshape(-1))
        parts.append(np.asarray(obs["bus_va_deg"], dtype=np.float32).reshape(-1))
        parts.append(np.asarray(obs["load_p_mw"], dtype=np.float32).reshape(-1))
        parts.append(np.asarray(obs["load_q_mvar"], dtype=np.float32).reshape(-1))
        parts.append(np.asarray(obs["forecast_p_mw"], dtype=np.float32).reshape(-1))
        parts.append(np.asarray(obs["forecast_q_mvar"], dtype=np.float32).reshape(-1))

        t = int(np.asarray(obs["time_index"]).reshape(-1)[0])
        t_scaled = np.array([t * self.time_scale], dtype=np.float32)
        phase = 2.0 * math.pi * ((t % self.time_period) / float(self.time_period))
        t_sin = np.array([math.sin(phase)], dtype=np.float32)
        t_cos = np.array([math.cos(phase)], dtype=np.float32)
        parts.append(np.concatenate([t_scaled, t_sin, t_cos], axis=0).astype(np.float32))

        topo = int(np.asarray(obs["topology_id"]).reshape(-1)[0])
        topo = int(np.clip(topo, 0, self.n_actions - 1))
        parts.append(np.array([float(topo)], dtype=np.float32))

        x = np.concatenate(parts, axis=0).astype(np.float32)
        if x.shape[0] != self.flat_dim:
            raise RuntimeError(f"[ObsFlattenerV2] flat_dim mismatch: got {x.shape[0]} expected {self.flat_dim}")
        return x


class MaskedObsNormalizer:
    def __init__(self, dim: int, mask: np.ndarray, clip: float = 10.0):
        self.dim = int(dim)
        mask = np.asarray(mask, dtype=np.bool_)
        if mask.shape != (self.dim,):
            raise ValueError(f"mask shape must be ({self.dim},), got {mask.shape}")
        self.mask = mask
        self.clip = float(clip)
        self.rms = RunningMeanStd((int(mask.sum()),))

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        self.rms.update(x[:, self.mask])

    def normalize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        out = x.copy()
        mean = self.rms.mean.astype(np.float32)
        var = self.rms.var.astype(np.float32)
        y = (x[:, self.mask] - mean) / (np.sqrt(var) + 1e-8)
        y = np.clip(y, -self.clip, self.clip)
        out[:, self.mask] = y.astype(np.float32)
        return out.astype(np.float32) if out.shape[0] > 1 else out.reshape(-1).astype(np.float32)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "dim": self.dim,
            "mask": self.mask.astype(np.uint8),
            "mean": self.rms.mean,
            "var": self.rms.var,
            "count": self.rms.count,
            "clip": self.clip,
        }

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        self.dim = int(d["dim"])
        self.mask = np.asarray(d["mask"], dtype=np.uint8).astype(bool)
        self.rms.mean = np.asarray(d["mean"], dtype=np.float64)
        self.rms.var = np.asarray(d["var"], dtype=np.float64)
        self.rms.count = float(d["count"])
        self.clip = float(d.get("clip", self.clip))


# ============================================================
# adjacency cache (same logic as train)
# ============================================================
class TopologyAdjacencyCache:
    def __init__(
        self,
        *,
        feeder_cluster: Any,
        base_net: Any,
        r_switch: float,
        x_switch: float,
        max_entries: int = 64,
    ):
        self.feeder_cluster = feeder_cluster
        self.base_net = base_net
        self.r_switch = float(r_switch)
        self.x_switch = float(x_switch)
        self.max_entries = int(max(1, max_entries))
        self._cache: "OrderedDict[int, torch.Tensor]" = OrderedDict()

        self._bus_ids = sorted(list(base_net.bus.index))
        self._bus_map = {int(b): i for i, b in enumerate(self._bus_ids)}
        self.n_bus = len(self._bus_ids)

    def _build_adj_np(self, topology_id: int) -> np.ndarray:
        net_line_repr = set_fc_state_with_acts(self.feeder_cluster, self.base_net, [int(topology_id)])
        net_cal = built_ppnet_for_pfcal(net_line_repr, self.r_switch, self.x_switch)

        n = self.n_bus
        A = np.zeros((n, n), dtype=np.float32)

        if hasattr(net_cal, "line") and len(net_cal.line) > 0:
            for _, row in net_cal.line.iterrows():
                fb = int(row["from_bus"])
                tb = int(row["to_bus"])
                if fb in self._bus_map and tb in self._bus_map:
                    i = self._bus_map[fb]
                    j = self._bus_map[tb]
                    A[i, j] = 1.0
                    A[j, i] = 1.0

        if hasattr(net_cal, "trafo") and len(net_cal.trafo) > 0:
            for _, row in net_cal.trafo.iterrows():
                hb = int(row["hv_bus"])
                lb = int(row["lv_bus"])
                if hb in self._bus_map and lb in self._bus_map:
                    i = self._bus_map[hb]
                    j = self._bus_map[lb]
                    A[i, j] = 1.0
                    A[j, i] = 1.0

        if hasattr(net_cal, "switch") and len(net_cal.switch) > 0:
            sw = net_cal.switch
            for _, row in sw.iterrows():
                try:
                    if str(row.get("et", "")) != "b":
                        continue
                    if not bool(row.get("closed", True)):
                        continue
                    b1 = int(row["bus"])
                    b2 = int(row["element"])
                    if b1 in self._bus_map and b2 in self._bus_map:
                        i = self._bus_map[b1]
                        j = self._bus_map[b2]
                        A[i, j] = 1.0
                        A[j, i] = 1.0
                except Exception:
                    continue

        return A

    def get(self, topology_id: int) -> torch.Tensor:
        tid = int(topology_id)
        if tid in self._cache:
            A = self._cache.pop(tid)
            self._cache[tid] = A
            return A

        A_np = self._build_adj_np(tid)
        A_t = torch.tensor(A_np, dtype=torch.float32, device=torch.device("cpu"))

        self._cache[tid] = A_t
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)
        return A_t


# ============================================================
# GT load + forward probing (same idea as train)
# ============================================================
def _strip_module_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(k.startswith("module.") for k in keys):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def _torch_load_trusted(path: str, map_location: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_gtransformer_checkpoint(path: str, device: torch.device):
    """
    Lazy import GT to avoid multiprocessing spawn issues in some setups.
    Returns: (gt_model, gt_cfg(dataclass-like), raw_ckpt_dict)
    """
    import sys
    import importlib

    base_dir = os.path.dirname(os.path.abspath(__file__))
    gt_dir = os.path.join(base_dir, "GTransformer")
    if gt_dir not in sys.path:
        sys.path.append(gt_dir)

    gt_mod = importlib.import_module("gt_torch_model")
    GTConfig = getattr(gt_mod, "GTConfig")
    GTransformer = getattr(gt_mod, "GTransformer")

    ckpt = _torch_load_trusted(path, map_location=device)

    cfg_dict = None
    if isinstance(ckpt.get("cfg", None), dict):
        c = ckpt["cfg"]
        if isinstance(c.get("model", None), dict):
            cfg_dict = c["model"]
        elif all(k in c for k in ["din", "d_model", "n_heads", "d_ff", "n_layers"]):
            cfg_dict = c

    if cfg_dict is None:
        cfg_dict = {
            "din": 6,
            "d_model": 128,
            "n_heads": 8,
            "d_ff": 256,
            "n_layers": 3,
            "k_min": 1,
            "k_max": 5,
            "dropout": 0.1,
            "attn_dropout": 0.1,
            "adj_mode": "binary",
        }

    gt_cfg = GTConfig(**cfg_dict)
    gt = GTransformer(gt_cfg).to(device)

    sd = ckpt.get("model", None)
    if isinstance(sd, dict):
        sd = _strip_module_prefix(sd)
        gt.load_state_dict(sd, strict=True)
    else:
        sd2 = _strip_module_prefix(ckpt)
        gt.load_state_dict(sd2, strict=False)

    return gt, gt_cfg, ckpt


@dataclass
class GTForwardSpec:
    use_return_embeddings: bool
    use_batched_adj: bool
    z_key: Optional[str]
    out_dim: int


def _extract_z(out: Any, z_key: Optional[str]) -> torch.Tensor:
    if isinstance(out, dict):
        if z_key is not None and z_key in out and torch.is_tensor(out[z_key]):
            return out[z_key]
        for k in ("z", "emb", "node_emb", "node_embeddings", "h", "hidden"):
            if k in out and torch.is_tensor(out[k]):
                return out[k]
        raise RuntimeError(f"GT output is dict but no embedding key found. keys={list(out.keys())}")
    if torch.is_tensor(out):
        return out
    raise RuntimeError(f"Unsupported GT output type: {type(out)}")


@torch.no_grad()
def probe_gt_forward(gt: nn.Module, H: torch.Tensor, A2: torch.Tensor) -> GTForwardSpec:
    gt.eval()
    B, N, _din = H.shape

    candidates: List[Tuple[bool, bool, Dict[str, Any]]] = [
        (True, False, {"return_embeddings": True}),
        (False, False, {}),
        (True, True, {"return_embeddings": True}),
        (False, True, {}),
    ]

    for use_ret, use_batched, kwargs in candidates:
        try:
            A = A2.unsqueeze(0).expand(B, -1, -1) if use_batched else A2
            out = gt(H, A, **kwargs) if use_ret else gt(H, A)
            z_key = "z" if isinstance(out, dict) and "z" in out else None
            z = _extract_z(out, z_key=z_key)
            if z.ndim != 3 or z.shape[0] != B or z.shape[1] != N:
                continue
            D = int(z.shape[2])

            out_key = None
            if isinstance(out, dict):
                for k in ("z", "emb", "node_emb", "node_embeddings", "h", "hidden"):
                    if k in out and torch.is_tensor(out[k]) and out[k].shape == z.shape:
                        out_key = k
                        break

            return GTForwardSpec(
                use_return_embeddings=use_ret,
                use_batched_adj=use_batched,
                z_key=out_key,
                out_dim=D,
            )
        except Exception:
            continue

    raise RuntimeError("Failed to probe a valid GT forward signature. Please verify GTransformer forward(H,A,...) interface.")


def gt_forward_with_spec(gt: nn.Module, spec: GTForwardSpec, H: torch.Tensor, A2: torch.Tensor) -> torch.Tensor:
    A = A2.unsqueeze(0).expand(H.shape[0], -1, -1) if spec.use_batched_adj else A2
    out = gt(H, A, return_embeddings=True) if spec.use_return_embeddings else gt(H, A)
    z = _extract_z(out, spec.z_key)
    if z.ndim != 3 or z.shape[0] != H.shape[0] or z.shape[1] != H.shape[1]:
        raise RuntimeError(f"GT embedding shape mismatch: got {tuple(z.shape)} expected ({H.shape[0]},{H.shape[1]},D)")
    return z


# ============================================================
# model
# ============================================================
class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, *, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, *, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.ln = nn.LayerNorm(dim)
        self.drop = nn.Dropout(p=float(dropout)) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.fc1(x))
        h = self.drop(h)
        h = self.fc2(h)
        return self.ln(x + h)


class MultiBranchActorCriticWithGT(nn.Module):
    def __init__(
        self,
        *,
        obs_dim: int,
        n_actions: int,
        slices: Dict[str, slice],
        gt: nn.Module,
        gt_spec: GTForwardSpec,
        adj_cache: TopologyAdjacencyCache,
        gt_pool: str = "mean",
        gt_proj_dim: int = 128,
        emb_dim_bus: int = 128,
        emb_dim_load: int = 128,
        emb_dim_fcst: int = 128,
        emb_dim_time: int = 32,
        topo_emb_dim: int = 32,
        enc_hidden: int = 256,
        fusion_hidden: int = 256,
        fusion_blocks: int = 2,
        dropout: float = 0.0,
        policy_temperature: float = 1.0,
        adj_cache_cuda_size: int = 16,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.slices = dict(slices)
        self.policy_temperature = float(max(policy_temperature, 1e-6))

        self.gt = gt
        self.gt_spec = gt_spec
        self.adj_cache = adj_cache
        self.gt_pool = str(gt_pool).lower().strip()
        if self.gt_pool not in ("mean", "sum"):
            raise ValueError("gt_pool must be 'mean' or 'sum'")

        self.enc_bus_vm = MLPEncoder(self._len("bus_vm_pu"), emb_dim_bus, enc_hidden, dropout=dropout)
        self.enc_bus_va = MLPEncoder(self._len("bus_va_deg"), emb_dim_bus, enc_hidden, dropout=dropout)
        self.enc_load_p = MLPEncoder(self._len("load_p_mw"), emb_dim_load, enc_hidden, dropout=dropout)
        self.enc_load_q = MLPEncoder(self._len("load_q_mvar"), emb_dim_load, enc_hidden, dropout=dropout)
        self.enc_fcst_p = MLPEncoder(self._len("forecast_p_mw"), emb_dim_fcst, enc_hidden, dropout=dropout)
        self.enc_fcst_q = MLPEncoder(self._len("forecast_q_mvar"), emb_dim_fcst, enc_hidden, dropout=dropout)
        self.enc_time = MLPEncoder(3, emb_dim_time, max(64, enc_hidden // 2), dropout=dropout)

        self.topo_emb = nn.Embedding(self.n_actions, topo_emb_dim)

        self.gt_proj = nn.Sequential(
            nn.Linear(int(gt_spec.out_dim), gt_proj_dim),
            nn.GELU(),
            nn.LayerNorm(gt_proj_dim),
        )

        fused_in = (
            2 * emb_dim_bus
            + 2 * emb_dim_load
            + 2 * emb_dim_fcst
            + emb_dim_time
            + topo_emb_dim
            + gt_proj_dim
        )

        self.fusion = nn.Sequential(
            nn.Linear(fused_in, fusion_hidden),
            nn.GELU(),
            nn.LayerNorm(fusion_hidden),
        )
        self.fusion_blocks = nn.ModuleList(
            [ResidualMLPBlock(fusion_hidden, fusion_hidden * 2, dropout=dropout) for _ in range(int(fusion_blocks))]
        )

        self.policy_head = nn.Linear(fusion_hidden, self.n_actions)
        self.value_head = nn.Linear(fusion_hidden, 1)

        self._adj_cache_cuda_size = int(max(0, adj_cache_cuda_size))
        self._adj_cuda: "OrderedDict[int, torch.Tensor]" = OrderedDict()

        self._init_weights()

    def _len(self, name: str) -> int:
        sl = self.slices[name]
        return int(sl.stop - sl.start)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def _pool_nodes(self, z: torch.Tensor) -> torch.Tensor:
        return z.sum(dim=1) if self.gt_pool == "sum" else z.mean(dim=1)

    def _get_adj_on_device(self, tid: int, device: torch.device) -> torch.Tensor:
        if device.type != "cuda" or self._adj_cache_cuda_size <= 0:
            return self.adj_cache.get(tid).to(device=device, non_blocking=True)

        tid = int(tid)
        if tid in self._adj_cuda:
            A = self._adj_cuda.pop(tid)
            self._adj_cuda[tid] = A
            return A

        A_cpu = self.adj_cache.get(tid)
        A = A_cpu.to(device=device, non_blocking=True)
        self._adj_cuda[tid] = A
        while len(self._adj_cuda) > self._adj_cache_cuda_size:
            self._adj_cuda.popitem(last=False)
        return A

    def _encode_graph_by_topology(self, H: torch.Tensor, topo_id: torch.Tensor) -> torch.Tensor:
        if H.ndim != 3:
            raise RuntimeError(f"gt_H must be (B,N,d), got {tuple(H.shape)}")
        B = int(H.shape[0])
        device = H.device

        out = torch.zeros((B, int(self.gt_spec.out_dim)), dtype=torch.float32, device=device)

        uniq = torch.unique(topo_id.detach().cpu())
        for tid_t in uniq:
            tid = int(tid_t.item())
            idx = (topo_id == tid_t.to(topo_id.device)).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue

            H_sub = H.index_select(0, idx)
            A = self._get_adj_on_device(tid, device)

            z = gt_forward_with_spec(self.gt, self.gt_spec, H_sub, A)
            g = self._pool_nodes(z)
            out.index_copy_(0, idx, g)

        return out

    def forward(self, obs: torch.Tensor, gt_H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if obs.ndim != 2 or obs.shape[1] != self.obs_dim:
            raise RuntimeError(f"obs shape must be (B,{self.obs_dim}), got {tuple(obs.shape)}")

        vm = obs[:, self.slices["bus_vm_pu"]]
        va = obs[:, self.slices["bus_va_deg"]]
        lp = obs[:, self.slices["load_p_mw"]]
        lq = obs[:, self.slices["load_q_mvar"]]
        fp = obs[:, self.slices["forecast_p_mw"]]
        fq = obs[:, self.slices["forecast_q_mvar"]]
        time_feat = obs[:, self.slices["time_feat"]]
        topo_raw = obs[:, self.slices["topology_id"]].squeeze(-1)
        topo_id = torch.clamp(topo_raw.round().long(), 0, self.n_actions - 1)

        e_vm = self.enc_bus_vm(vm)
        e_va = self.enc_bus_va(va)
        e_lp = self.enc_load_p(lp)
        e_lq = self.enc_load_q(lq)
        e_fp = self.enc_fcst_p(fp)
        e_fq = self.enc_fcst_q(fq)
        e_t = self.enc_time(time_feat)
        e_topo = self.topo_emb(topo_id)

        g = self._encode_graph_by_topology(gt_H, topo_id)
        g = self.gt_proj(g)

        z = torch.cat([e_vm, e_va, e_lp, e_lq, e_fp, e_fq, e_t, e_topo, g], dim=-1)
        z = self.fusion(z)
        for blk in self.fusion_blocks:
            z = blk(z)

        logits = self.policy_head(z) / self.policy_temperature
        value = self.value_head(z).squeeze(-1)
        return logits, value


# ============================================================
# checkpoint load (compatible with your trainer save format)
# ============================================================
def load_checkpoint_raw(path: str, device: torch.device) -> Dict[str, Any]:
    return _torch_load_trusted(path, map_location=device)


def apply_checkpoint(
    ckpt: Dict[str, Any],
    model: nn.Module,
    normalizer: MaskedObsNormalizer,
    device: torch.device,
) -> Dict[str, Any]:
    sd = ckpt.get("model", None)
    if not isinstance(sd, dict):
        raise ValueError("checkpoint missing 'model' state_dict")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[ckpt] missing keys: {missing[:20]}{' ...' if len(missing) > 20 else ''}")
    if unexpected:
        print(f"[ckpt] unexpected keys: {unexpected[:20]}{' ...' if len(unexpected) > 20 else ''}")

    nd = ckpt.get("normalizer", None)
    if isinstance(nd, dict):
        normalizer.load_state_dict(nd)
    else:
        print("[ckpt] warning: checkpoint has no 'normalizer' dict; normalization may be inconsistent.")

    meta = {
        "step": int(ckpt.get("step", 0)),
        "update": int(ckpt.get("update", 0)),
        "episode_count": int(ckpt.get("episode_count", 0)),
        "hparams": dict(ckpt.get("hparams", {})),
        "extra": dict(ckpt.get("extra", {})),
    }
    return meta


# ============================================================
# pretty printing helpers
# ============================================================
def _stat_1d(x: np.ndarray) -> str:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return "empty"
    return f"mean={float(x.mean()):.4f} std={float(x.std()):.4f} min={float(x.min()):.4f} max={float(x.max()):.4f}"


def _head(x: np.ndarray, k: int = 6) -> str:
    x = np.asarray(x).reshape(-1)
    k = int(max(0, k))
    return "[" + ", ".join([f"{float(v):.4f}" for v in x[:k].tolist()]) + (", ..." if x.size > k else "") + "]"


@torch.no_grad()
def select_action(
    model: nn.Module,
    obs_norm: np.ndarray,
    gt_H: np.ndarray,
    device: torch.device,
    *,
    deterministic: bool,
) -> Tuple[int, Dict[str, Any]]:
    obs_t = torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)
    gt_t = torch.tensor(gt_H, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.inference_mode():
        logits, value = model(obs_t, gt_t)
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        if deterministic:
            a = int(torch.argmax(probs).item())
        else:
            dist = torch.distributions.Categorical(probs=probs)
            a = int(dist.sample().item())

    # topk for printing
    k = min(8, probs.numel())
    topv, topi = torch.topk(probs, k=k, largest=True, sorted=True)
    return a, {
        "value": float(value.squeeze(0).item()),
        "probs_topi": topi.detach().cpu().numpy().astype(int).tolist(),
        "probs_topv": topv.detach().cpu().numpy().astype(float).tolist(),
        "logits": logits.squeeze(0).detach().cpu().numpy().astype(np.float32),
        "probs": probs.detach().cpu().numpy().astype(np.float32),
    }


def _print_step_debug(
    step_idx: int,
    obs: Dict[str, np.ndarray],
    obs_flat: np.ndarray,
    obs_norm: np.ndarray,
    act: int,
    pol: Dict[str, Any],
    reward: float,
    info: Dict[str, Any],
) -> None:
    # 1) state
    t = int(np.asarray(obs["time_index"]).reshape(-1)[0])
    topo = int(np.asarray(obs["topology_id"]).reshape(-1)[0])
    print(f"\n[step {step_idx:05d}] t={t} topo_id={topo}  action={act}  reward={reward:.6f}  V={pol['value']:.4f}")

    # Raw observation highlights
    print("  [state/raw] bus_vm_pu:", _stat_1d(obs["bus_vm_pu"]), "head=", _head(obs["bus_vm_pu"]))
    print("  [state/raw] bus_va_deg:", _stat_1d(obs["bus_va_deg"]), "head=", _head(obs["bus_va_deg"]))
    print("  [state/raw] load_p_mw:", _stat_1d(obs["load_p_mw"]), "head=", _head(obs["load_p_mw"]))
    print("  [state/raw] load_q_mvar:", _stat_1d(obs["load_q_mvar"]), "head=", _head(obs["load_q_mvar"]))
    fp = np.asarray(obs["forecast_p_mw"], dtype=np.float32)
    fq = np.asarray(obs["forecast_q_mvar"], dtype=np.float32)
    print("  [state/raw] forecast_p_mw:", f"shape={tuple(fp.shape)}", "row0_head=", _head(fp[0] if fp.ndim == 2 else fp))
    print("  [state/raw] forecast_q_mvar:", f"shape={tuple(fq.shape)}", "row0_head=", _head(fq[0] if fq.ndim == 2 else fq))
    gt_H = np.asarray(obs.get("gt_H", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32)
    if gt_H.size:
        # summarize each feature column
        col_mean = gt_H.mean(axis=0)
        print("  [state/raw] gt_H:", f"shape={tuple(gt_H.shape)}", "col_mean=", _head(col_mean, k=min(6, col_mean.size)))
    else:
        print("  [state/raw] gt_H: empty or disabled")

    # flat/norm
    print("  [state/flat] dim=", int(obs_flat.size), "flat_head=", _head(obs_flat, k=10))
    print("  [state/norm] dim=", int(obs_norm.size), "norm_head=", _head(obs_norm, k=10))

    # 2) decision
    print("  [decision] top_probs:", list(zip(pol["probs_topi"], [round(v, 6) for v in pol["probs_topv"]])))

    # 3) reward decomposition (if available)
    # Your env stores reward and sub-terms in info dict; names match your implementation.
    keys = [
        "reward",
        "loss_mw", "v_viol", "line_viol", "trafo_bal", "switch_cost",
        "loss_term", "v_term", "line_term", "trafo_term", "switch_term",
        "pf_failed",
        "t", "action", "prev_action",
    ]
    present = {k: info.get(k, None) for k in keys if k in info}
    if present:
        # stable ordering
        s = "  [reward/info] " + " ".join([f"{k}={present[k]}" for k in keys if k in present])
        print(s)


def run_interaction(
    *,
    env: PowerDispatchEnv,
    flattener: ObsFlattenerV2,
    normalizer: MaskedObsNormalizer,
    model: MultiBranchActorCriticWithGT,
    device: torch.device,
    n_episodes: int,
    max_steps_per_episode: int,
    deterministic: bool,
    print_every: int,
) -> None:
    model.eval()

    total_steps = 0
    for ep in range(int(n_episodes)):
        obs, info0 = env.reset(seed=1000 * ep + 123, options=None)

        ep_ret = 0.0
        ep_len = 0
        t0 = time.time()

        for step in range(int(max_steps_per_episode)):
            obs_flat = flattener.flatten(obs)
            obs_norm = normalizer.normalize(obs_flat)  # do NOT update in evaluation
            gt_H = np.asarray(obs.get("gt_H", None), dtype=np.float32)

            act, pol = select_action(model, obs_norm, gt_H, device, deterministic=deterministic)

            next_obs, reward, terminated, truncated, info = env.step(act)
            ep_ret += float(reward)
            ep_len += 1
            total_steps += 1

            if print_every > 0 and (total_steps % int(print_every) == 0):
                _print_step_debug(total_steps, obs, obs_flat, obs_norm, act, pol, float(reward), info if isinstance(info, dict) else {})

            obs = next_obs
            if bool(terminated) or bool(truncated):
                break

        dt = time.time() - t0
        print(f"\n[episode {ep+1:04d}] len={ep_len} return={ep_ret:.6f} wall={dt:.2f}s avg_step={dt/max(ep_len,1):.4f}s")


# ============================================================
# entry (edit params here)
# ============================================================
if __name__ == "__main__":
    # ------------------
    # Paths
    # ------------------
    RL_CKPT_PATH = "./runs_ppo_mp_multienc_withGT_fast_260107/ppo_gt_ckpt_final.pt"
    GT_CKPT_PATH = ""  # optional; if empty, try checkpoint["extra"]["gt_ckpt_path"]

    # ------------------
    # Runtime
    # ------------------
    SEED = 0
    DEVICE = "cuda"  # "cuda"/"cpu"/"auto"
    ALLOW_TF32 = True

    # Interaction control
    N_EPISODES = 3
    MAX_STEPS_PER_EPISODE = 5000
    DETERMINISTIC = True
    PRINT_EVERY_ENV_STEPS = 1  # set to 10/50 if too verbose

    # Model hyperparams (must match your training run)
    ENC_HIDDEN = 256
    FUSION_HIDDEN = 256
    FUSION_BLOCKS = 2
    DROPOUT = 0.0
    POLICY_TEMPERATURE = 1.0
    EMB_BUS = 128
    EMB_LOAD = 128
    EMB_FCST = 128
    EMB_TIME = 32
    TOPO_EMB = 32
    GT_PROJ_DIM = 128
    GT_POOL = "mean"
    ADJ_CACHE_SIZE = 64
    ADJ_CACHE_CUDA_SIZE = 16

    # ------------------
    # Setup
    # ------------------
    set_global_seeds(SEED)
    device = get_device(DEVICE)

    if device.type == "cuda" and ALLOW_TF32:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    env_cfg = EnvConfig(seed=SEED)
    env = PowerDispatchEnv(env_cfg)

    n_actions = int(env.action_space.n)
    time_period = int(getattr(env_cfg, "episode_len", 24))
    flattener = ObsFlattenerV2(env.observation_space, n_actions=n_actions, time_scale=1e-3, time_period=time_period)

    obs0, info0 = env.reset(seed=SEED, options=None)
    gt_H0 = np.asarray(obs0.get("gt_H", None), dtype=np.float32)
    if gt_H0.ndim != 2:
        raise RuntimeError(f"env obs['gt_H'] must be (N,D); got {gt_H0.shape}")
    n_bus = int(gt_H0.shape[0])
    gt_din = int(gt_H0.shape[1])

    # Normalizer mask: do not normalize sin/cos and topology_id (same as train)
    mask = np.ones((flattener.flat_dim,), dtype=bool)
    sl_time = flattener.slices["time_feat"]
    mask[sl_time.start + 1 : sl_time.stop] = False
    mask[flattener.slices["topology_id"]] = False
    normalizer = MaskedObsNormalizer(flattener.flat_dim, mask=mask, clip=10.0)

    # Load RL ckpt first (to fetch gt_ckpt_path fallback)
    ckpt = load_checkpoint_raw(RL_CKPT_PATH, device=device)
    extra = dict(ckpt.get("extra", {})) if isinstance(ckpt.get("extra", {}), dict) else {}

    if not GT_CKPT_PATH:
        GT_CKPT_PATH = str(extra.get("gt_ckpt_path", "") or "")
    if not GT_CKPT_PATH:
        raise ValueError("GT_CKPT_PATH is empty. Set it explicitly or ensure RL checkpoint contains extra['gt_ckpt_path'].")

    # Load GT (initial weights), later overwritten by RL model state_dict
    gt, gt_cfg, gt_raw = load_gtransformer_checkpoint(GT_CKPT_PATH, device=device)

    # Build adjacency cache (use env_cfg r/x if present; otherwise try config746sys)
    r_switch = float(getattr(env_cfg, "r_switch", float("nan")))
    x_switch = float(getattr(env_cfg, "x_switch", float("nan")))
    if not (np.isfinite(r_switch) and np.isfinite(x_switch)):
        try:
            import config746sys  # type: ignore
            r_switch = float(getattr(config746sys, "r_switch"))
            x_switch = float(getattr(config746sys, "x_switch"))
        except Exception as e:
            raise RuntimeError("Cannot resolve r_switch/x_switch from EnvConfig or config746sys.") from e

    adj_cache = TopologyAdjacencyCache(
        feeder_cluster=env.feeder_cluster,
        base_net=env.base_net,
        r_switch=r_switch,
        x_switch=x_switch,
        max_entries=int(ADJ_CACHE_SIZE),
    )

    topo0 = int(np.asarray(obs0["topology_id"]).reshape(-1)[0])
    A0 = adj_cache.get(topo0).to(device=device)
    H_probe = torch.from_numpy(np.asarray(gt_H0[None, :, :], dtype=np.float32)).to(device=device)
    spec = probe_gt_forward(gt, H_probe, A0)
    print(f"[GT probe] use_return_embeddings={spec.use_return_embeddings} use_batched_adj={spec.use_batched_adj} z_key={spec.z_key} out_dim={spec.out_dim}")

    model = MultiBranchActorCriticWithGT(
        obs_dim=flattener.flat_dim,
        n_actions=n_actions,
        slices=flattener.slices,
        gt=gt,
        gt_spec=spec,
        adj_cache=adj_cache,
        gt_pool=GT_POOL,
        gt_proj_dim=GT_PROJ_DIM,
        emb_dim_bus=EMB_BUS,
        emb_dim_load=EMB_LOAD,
        emb_dim_fcst=EMB_FCST,
        emb_dim_time=EMB_TIME,
        topo_emb_dim=TOPO_EMB,
        enc_hidden=ENC_HIDDEN,
        fusion_hidden=FUSION_HIDDEN,
        fusion_blocks=FUSION_BLOCKS,
        dropout=DROPOUT,
        policy_temperature=POLICY_TEMPERATURE,
        adj_cache_cuda_size=ADJ_CACHE_CUDA_SIZE,
    ).to(device)

    meta = apply_checkpoint(ckpt, model, normalizer, device)
    print(f"[loaded] rl_ckpt={RL_CKPT_PATH}")
    print(f"[loaded] step={meta['step']} update={meta['update']} episode_count={meta['episode_count']}")
    print(f"[loaded] gt_ckpt={GT_CKPT_PATH}")

    # Main interaction loop
    run_interaction(
        env=env,
        flattener=flattener,
        normalizer=normalizer,
        model=model,
        device=device,
        n_episodes=N_EPISODES,
        max_steps_per_episode=MAX_STEPS_PER_EPISODE,
        deterministic=DETERMINISTIC,
        print_every=PRINT_EVERY_ENV_STEPS,
    )

# %%
