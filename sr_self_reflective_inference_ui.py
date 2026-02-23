# In[]
# -*- coding: utf-8 -*-
"""
sr_self_reflective_inference_ui.py

Self-Reflective (SR) inference with look-ahead beam search (H=3) + baseline comparison.
Includes:
  - CLI-style run_episode() for SR and baseline
  - Gradio UI:
      A) Debug mode (step-by-step): expand beam search, show candidates, optionally verify via env simulation, execute action
      B) Auto mode: run full episode, export CSV

No argparse. Edit params in __main__.
"""

from __future__ import annotations

import os
import math
import time
import json
import copy
import csv
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gradio as gr

# ------------------------------
# import your env
# ------------------------------
from power_dispatch_env_withGT_dimrisk import (
    PowerDispatchEnv,
    EnvConfig,
    TimeseriesSchema,
    built_ppnet_for_pfcal,
    set_fc_state_with_acts,
)

# ============================================================
# small utilities (logging)
# ============================================================

def now_str() -> str:
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def fmt_sec(s: float) -> str:
    s = float(s)
    if s < 60:
        return f"{s:.3f}s"
    m = int(s // 60)
    r = s - 60 * m
    if m < 60:
        return f"{m}m{r:.0f}s"
    h = int(m // 60)
    m2 = m - 60 * h
    return f"{h}h{m2}m"

class PrintLogger:
    """Collect logs in-memory (for UI) + print to stdout."""
    def __init__(self):
        self.lines: List[str] = []

    def log(self, msg: str) -> None:
        line = f"[{now_str()}] {msg}"
        print(line)
        self.lines.append(line)

    def text(self, last_n: int = 400) -> str:
        if len(self.lines) <= last_n:
            return "\n".join(self.lines)
        return "\n".join(self.lines[-last_n:])

# ============================================================
# Obs flatten + normalizer (same ordering as your training)
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
    Flat order (fixed):
      bus_vm_pu
      bus_va_deg
      load_p_mw
      load_q_mvar
      forecast_p_mw
      forecast_q_mvar
      time_feat: time_scaled + sin + cos [3]
      topology_id [1]
    gt_H not included in flat.
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
            self.n_bus + self.n_bus +
            self.n_load + self.n_load +
            (self.horizon * self.n_load) + (self.horizon * self.n_load) +
            3 + 1
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
        assert off == self.flat_dim

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
# Topology adjacency cache (same as training, minimal)
# ============================================================

from collections import OrderedDict
import importlib

def _load_switch_params() -> Tuple[float, float]:
    m = importlib.import_module("config746sys")
    r_switch = float(getattr(m, "r_switch"))
    x_switch = float(getattr(m, "x_switch"))
    return r_switch, x_switch

class TopologyAdjacencyCache:
    def __init__(self, *, feeder_cluster: Any, base_net: Any, r_switch: float, x_switch: float, max_entries: int = 64):
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
# GT loader + forward spec (compatible with your training)
# ============================================================

def _strip_module_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(k.startswith("module.") for k in keys):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict

def _torch_load_trusted(path: str, map_location: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)

def load_gtransformer_checkpoint(path: str, device: torch.device):
    import sys
    base_dir = os.getcwd()
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
        # fallback
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
        raise RuntimeError(f"GT output dict but no embedding key found. keys={list(out.keys())}")
    if torch.is_tensor(out):
        return out
    raise RuntimeError(f"Unsupported GT output type: {type(out)}")

@torch.no_grad()
def probe_gt_forward(gt: nn.Module, H: torch.Tensor, A2: torch.Tensor) -> GTForwardSpec:
    gt.eval()
    B, N, _din = H.shape
    candidates = [(True, False), (False, False), (True, True), (False, True)]
    for use_ret, use_batched in candidates:
        try:
            A = A2.unsqueeze(0).expand(B, -1, -1) if use_batched else A2
            out = gt(H, A, return_embeddings=True) if use_ret else gt(H, A)
            z = _extract_z(out, z_key="z" if isinstance(out, dict) and "z" in out else None)
            if z.ndim != 3:
                continue
            if z.shape[0] != B or z.shape[1] != N:
                continue
            D = int(z.shape[2])
            z_key = None
            if isinstance(out, dict):
                for k in ("z", "emb", "node_emb", "node_embeddings", "h", "hidden"):
                    if k in out and torch.is_tensor(out[k]) and out[k].shape == z.shape:
                        z_key = k
                        break
            return GTForwardSpec(use_return_embeddings=use_ret, use_batched_adj=use_batched, z_key=z_key, out_dim=D)
        except Exception:
            continue
    raise RuntimeError("Failed to probe GT forward signature. Check gt_torch_model interface.")

def gt_forward_with_spec(gt: nn.Module, spec: GTForwardSpec, H: torch.Tensor, A2: torch.Tensor) -> torch.Tensor:
    A = A2.unsqueeze(0).expand(H.shape[0], -1, -1) if spec.use_batched_adj else A2
    out = gt(H, A, return_embeddings=True) if spec.use_return_embeddings else gt(H, A)
    z = _extract_z(out, spec.z_key)
    return z

# ============================================================
# Actor-Critic + SR head (must match your training architecture)
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

class SelfReflectionDecoder(nn.Module):
    """
    Predict normalized costs conditioned on action.
      - total_cost_norm_hat
      - risk_cost_norm_hat
    """
    def __init__(self, phi_dim: int, n_actions: int, act_emb_dim: int = 64, hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.n_actions = int(n_actions)
        self.act_emb = nn.Embedding(self.n_actions, int(act_emb_dim))
        in_dim = int(phi_dim) + int(act_emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, int(hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(int(hidden), int(hidden)),
            nn.GELU(),
            nn.LayerNorm(int(hidden)),
        )
        self.head_total = nn.Linear(int(hidden), 1)
        self.head_risk = nn.Linear(int(hidden), 1)

        nn.init.orthogonal_(self.head_total.weight, gain=0.01)
        nn.init.orthogonal_(self.head_risk.weight, gain=0.01)
        nn.init.constant_(self.head_total.bias, 0.0)
        nn.init.constant_(self.head_risk.bias, 0.0)

    def forward(self, phi: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if a.ndim != 1:
            a = a.reshape(-1)
        a = torch.clamp(a.long(), 0, self.n_actions - 1)
        e = self.act_emb(a)
        x = torch.cat([phi, e], dim=-1)
        h = self.mlp(x)
        total = self.head_total(h).squeeze(-1)
        risk = self.head_risk(h).squeeze(-1)
        return total, risk

class MultiBranchActorCriticWithGTAndSR(nn.Module):
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
        sr_act_emb_dim: int = 64,
        sr_hidden: int = 256,
        sr_dropout: float = 0.0,
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
            2 * emb_dim_bus +
            2 * emb_dim_load +
            2 * emb_dim_fcst +
            emb_dim_time +
            topo_emb_dim +
            gt_proj_dim
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

        self.sr = SelfReflectionDecoder(
            phi_dim=fusion_hidden,
            n_actions=self.n_actions,
            act_emb_dim=sr_act_emb_dim,
            hidden=sr_hidden,
            dropout=sr_dropout,
        )

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

    def forward_with_phi(self, obs: torch.Tensor, gt_H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        phi = self.fusion(z)
        for blk in self.fusion_blocks:
            phi = blk(phi)

        logits = self.policy_head(phi) / self.policy_temperature
        value = self.value_head(phi).squeeze(-1)
        return logits, value, phi

# ============================================================
# SR reward helper + label extraction helper
# ============================================================

def _get_float(info: Dict[str, Any], keys: List[str], default: float = 0.0) -> float:
    for k in keys:
        if k in info:
            try:
                return float(info.get(k, default))
            except Exception:
                continue
    return float(default)

def extract_cost_labels_yuan_from_info(info: Dict[str, Any]) -> Tuple[float, float]:
    """Match your training extraction logic (single env)."""
    if "total_cost_yuan" in info:
        tot = float(info.get("total_cost_yuan", 0.0))
    else:
        tot = (
            _get_float(info, ["import_cost_yuan"], 0.0)
            + _get_float(info, ["loss_cost_yuan"], 0.0)
            + _get_float(info, ["v_cost_yuan"], 0.0)
            + _get_float(info, ["line_cost_yuan"], 0.0)
            + _get_float(info, ["trafo_cost_yuan"], 0.0)
            + _get_float(info, ["risk_cost_yuan", "risk_term", "risk_cost"], 0.0)
            + _get_float(info, ["switch_cost_yuan"], 0.0)
        )
    risk = _get_float(info, ["risk_cost_yuan", "risk_term", "risk_cost"], 0.0)
    return tot, risk

def sr_reward_from_pred(
    total_hat_norm: torch.Tensor,
    risk_hat_norm: torch.Tensor,
    *,
    total_scale_yuan: float,
    risk_scale_yuan: float,
    reward_scale: float,
    lambda_risk: float,
) -> torch.Tensor:
    """
    Match your training shaping:
      r_hat_rl = - (total_hat_norm*total_scale + lambda_risk*risk_hat_norm*risk_scale) * reward_scale
    """
    return -(
        total_hat_norm * float(total_scale_yuan)
        + float(lambda_risk) * risk_hat_norm * float(risk_scale_yuan)
    ) * float(reward_scale)

# ============================================================
# Core: side-effect-free simulation aligned with env.step()
# ============================================================

def simulate_one_step_no_side_effect(
    env: PowerDispatchEnv,
    *,
    cur_t: int,
    prev_action: int,
    action: int,
) -> Tuple[Dict[str, np.ndarray], float, Dict[str, Any]]:
    """
    env.step() does:
      t <- t+1
      solve at time t with action, prev_action provided
    Here we simulate that transition without modifying env._t / env._prev_action.

    NOTE: env._solve_and_build_obs sets env._net_cal internally; we accept this benign side effect.
    """
    next_t = int(cur_t) + 1
    net_line_repr = set_fc_state_with_acts(env.feeder_cluster, env.base_net, [int(action)])
    obs_next, info = env._solve_and_build_obs(net_line_repr, next_t, prev_action=int(prev_action), action=int(action))
    reward = float(info.get("reward", 0.0))
    return obs_next, reward, info

# ============================================================
# Beam Search data structure
# ============================================================

@dataclass
class BeamItem:
    actions: List[int]
    sr_return: float
    sr_risk_yuan: float
    # optional verified true rollout
    true_return: Optional[float] = None
    true_risk_yuan: Optional[float] = None
    # per-step debug details (strings)
    details: Optional[List[str]] = None

# ============================================================
# Candidate generation (policy)
# ============================================================

@torch.no_grad()
def propose_candidates_from_policy(
    model: MultiBranchActorCriticWithGTAndSR,
    obs_flat_norm: np.ndarray,
    gt_H: np.ndarray,
    *,
    device: torch.device,
    K: int,
    temperature: float = 1.0,
    logger: Optional[PrintLogger] = None,
) -> Tuple[List[int], np.ndarray]:
    """
    Return:
      - candidates (unique) up to K
      - probs over actions (for printing)
    """
    obs_t = torch.from_numpy(obs_flat_norm.reshape(1, -1)).to(device=device, dtype=torch.float32)
    gt_t = torch.from_numpy(gt_H.reshape(1, gt_H.shape[0], gt_H.shape[1])).to(device=device, dtype=torch.float32)
    logits, _value, _phi = model.forward_with_phi(obs_t, gt_t)
    logits = logits / max(float(temperature), 1e-6)
    dist = torch.distributions.Categorical(logits=logits)
    probs = dist.probs.detach().cpu().numpy().reshape(-1)

    nA = probs.shape[0]
    K = int(max(1, min(K, nA)))

    # mixture: topk + sampling to encourage diversity
    topk = min(K, nA)
    top_ids = list(np.argsort(-probs)[:topk].astype(int))

    cand_set = []
    for a in top_ids:
        cand_set.append(int(a))

    # add stochastic samples until reaching K unique
    tries = 0
    while len(cand_set) < K and tries < 10 * K:
        a = int(dist.sample().item())
        if a not in cand_set:
            cand_set.append(a)
        tries += 1

    if logger is not None:
        show_top = min(10, nA)
        top_pairs = [(int(i), float(probs[i])) for i in np.argsort(-probs)[:show_top]]
        logger.log(f"[policy] propose K={K} candidates. top_probs={top_pairs}")
        logger.log(f"[policy] candidates(unique)={cand_set}")

    return cand_set, probs

# ============================================================
# Beam Search with SR scoring (look-ahead H=3)
# ============================================================

@torch.no_grad()
def beam_search_sr(
    env: PowerDispatchEnv,
    model: MultiBranchActorCriticWithGTAndSR,
    flattener: ObsFlattenerV2,
    normalizer: MaskedObsNormalizer,
    *,
    cur_obs: Dict[str, np.ndarray],
    cur_gt_H: np.ndarray,
    cur_t: int,
    prev_action: int,
    look_ahead: int = 3,
    beam_width: int = 6,
    cand_K: int = 10,
    gamma: float = 0.99,
    device: torch.device,
    # SR reward parameters
    total_scale_yuan: float,
    risk_scale_yuan: float,
    reward_scale: float,
    lambda_risk: float,
    # optional risk budget (yuan) constraint
    risk_budget_yuan: Optional[float] = None,
    # verbose logger
    logger: Optional[PrintLogger] = None,
) -> List[BeamItem]:
    """
    We expand in *real env time*:
      node at time t has observation obs(t) (after solving previous action)
      choosing action a leads to t+1 reward/obs.
    For SR scoring:
      at node (obs(t), gt_H(t)), we compute phi(t),
      then sr(phi(t), a) -> predicted cost at t+1 (aligned to training rollout).
    For deeper rollout:
      we simulate obs(t+1) using env._solve_and_build_obs (no side effect on t variables),
      then repeat.

    Return: list of BeamItem sorted by sr_return descending (higher is better).
    """
    assert look_ahead >= 1

    t0 = time.perf_counter()

    # initial node
    obs_flat = flattener.flatten(cur_obs)
    obs_flat_norm = normalizer.normalize(obs_flat)
    # propose candidates at root (for printing)
    root_cands, _ = propose_candidates_from_policy(
        model, obs_flat_norm, cur_gt_H, device=device, K=cand_K, logger=logger
    )

    # beam holds tuples: (obs_dict, gtH, t, prev_action, actions_seq, sr_return, sr_risk_yuan, details_list)
    beam = [(
        cur_obs, cur_gt_H, int(cur_t), int(prev_action),
        [], 0.0, 0.0, []
    )]

    for depth in range(look_ahead):
        if logger is not None:
            logger.log(f"[beam] ===== expand depth={depth+1}/{look_ahead} | current beam_size={len(beam)} =====")

        new_beam = []
        for bi, (obs_d, gtH_d, t_d, prev_a_d, seq, sr_ret, sr_risk, details) in enumerate(beam):
            # prepare policy candidates for this node
            obs_flat_d = flattener.flatten(obs_d)
            obs_flat_d_norm = normalizer.normalize(obs_flat_d)

            cands, _probs = propose_candidates_from_policy(
                model, obs_flat_d_norm, gtH_d, device=device, K=cand_K,
                logger=(logger if (depth == 0 and bi == 0) else None)  # only root prints to avoid huge spam
            )

            # compute phi once
            obs_t = torch.from_numpy(obs_flat_d_norm.reshape(1, -1)).to(device=device, dtype=torch.float32)
            gt_t = torch.from_numpy(gtH_d.reshape(1, gtH_d.shape[0], gtH_d.shape[1])).to(device=device, dtype=torch.float32)
            logits, _val, phi = model.forward_with_phi(obs_t, gt_t)
            phi = phi.reshape(1, -1)

            for a in cands:
                a_t = torch.tensor([int(a)], device=device, dtype=torch.int64)
                total_hat_norm, risk_hat_norm = model.sr(phi, a_t)

                r_hat = sr_reward_from_pred(
                    total_hat_norm, risk_hat_norm,
                    total_scale_yuan=total_scale_yuan,
                    risk_scale_yuan=risk_scale_yuan,
                    reward_scale=reward_scale,
                    lambda_risk=lambda_risk,
                ).item()

                # predicted risk in yuan (for budget)
                pred_risk_yuan = float(risk_hat_norm.item()) * float(risk_scale_yuan)

                # simulate next obs (physics) for deeper nodes
                obs_next, _r_true, info_next = simulate_one_step_no_side_effect(
                    env, cur_t=t_d, prev_action=prev_a_d, action=int(a)
                )
                gt_next = np.asarray(obs_next.get("gt_H", None), dtype=np.float32)
                if gt_next.ndim != 2:
                    raise RuntimeError("obs_next missing gt_H or wrong shape.")

                # accumulate discounted SR return
                sr_ret2 = float(sr_ret) + (float(gamma) ** depth) * float(r_hat)
                sr_risk2 = float(sr_risk) + (float(gamma) ** depth) * float(pred_risk_yuan)

                # risk budget check (optional)
                if risk_budget_yuan is not None:
                    if sr_risk2 > float(risk_budget_yuan) + 1e-9:
                        continue

                seq2 = seq + [int(a)]
                det2 = details + [
                    f"[d{depth+1}] a={int(a)} r_hat={r_hat:+.4e} pred_risk_yuan={pred_risk_yuan:+.4e} -> sr_return={sr_ret2:+.4e}"
                ]

                new_beam.append((obs_next, gt_next, t_d + 1, int(a), seq2, sr_ret2, sr_risk2, det2))

        # keep top beam_width by sr_ret2 (descending)
        new_beam.sort(key=lambda x: x[5], reverse=True)
        new_beam = new_beam[:int(max(1, beam_width))]
        beam = new_beam

        if logger is not None:
            logger.log(f"[beam] depth={depth+1} expanded -> kept beam_size={len(beam)} (beam_width={beam_width})")
            for i, b in enumerate(beam[:min(6, len(beam))]):
                logger.log(f"[beam]   rank#{i+1} seq={b[4]} sr_return={b[5]:+.4e} sr_risk_yuan={b[6]:+.4e}")

        if len(beam) == 0:
            if logger is not None:
                logger.log("[beam] all candidates pruned (risk_budget or empty). stop early.")
            break

    # package BeamItem list
    items: List[BeamItem] = []
    for (obs_d, gtH_d, t_d, prev_a_d, seq, sr_ret, sr_risk, det) in beam:
        items.append(BeamItem(actions=seq, sr_return=float(sr_ret), sr_risk_yuan=float(sr_risk), details=det))

    items.sort(key=lambda b: b.sr_return, reverse=True)

    if logger is not None:
        logger.log(f"[beam] DONE. total_dt={fmt_sec(time.perf_counter()-t0)} best_seq={items[0].actions if items else None}")

    return items

# ============================================================
# Verify candidate sequence via "true env simulation" (no side effect)
# ============================================================

def verify_sequence_true_return(
    env: PowerDispatchEnv,
    *,
    start_t: int,
    start_prev_action: int,
    start_obs: Dict[str, np.ndarray],
    actions: List[int],
    gamma: float,
    logger: Optional[PrintLogger] = None,
) -> Tuple[float, float, List[str]]:
    """
    Roll out actions using env._solve_and_build_obs (aligned to env.step time shift),
    compute discounted true return and discounted true risk_yuan.
    """
    t = int(start_t)
    prev_a = int(start_prev_action)
    obs = start_obs
    ret = 0.0
    risk_acc = 0.0
    details = []
    for i, a in enumerate(actions):
        obs_next, r_true, info = simulate_one_step_no_side_effect(env, cur_t=t, prev_action=prev_a, action=int(a))
        tot_yuan, risk_yuan = extract_cost_labels_yuan_from_info(info)
        ret += (float(gamma) ** i) * float(r_true)
        risk_acc += (float(gamma) ** i) * float(risk_yuan)
        details.append(f"[true][d{i+1}] a={int(a)} r_true={r_true:+.4e} total_cost_yuan={tot_yuan:+.4e} risk_yuan={risk_yuan:+.4e}")
        t += 1
        prev_a = int(a)
        obs = obs_next
    if logger is not None:
        logger.log(f"[verify] seq={actions} -> true_return={ret:+.4e} true_risk_yuan={risk_acc:+.4e}")
    return float(ret), float(risk_acc), details

# ============================================================
# Load PPO+SR checkpoint and build everything
# ============================================================

def build_inference_stack(
    *,
    ppo_sr_ckpt_path: str,
    gt_pretrain_ckpt_path: str,
    device_str: str,
    logger: Optional[PrintLogger] = None,
) -> Dict[str, Any]:
    """
    Returns dict containing:
      env, flattener, normalizer, model, device, scales, reward_scale, lambda_risk, etc.
    """
    device = torch.device(device_str if device_str in ("cpu", "cuda") else ("cuda" if torch.cuda.is_available() else "cpu"))
    if logger is not None:
        logger.log(f"[init] loading PPO+SR checkpoint: {ppo_sr_ckpt_path}")
        logger.log(f"[init] device={device}")

    ckpt = _torch_load_trusted(ppo_sr_ckpt_path, map_location=device)
    norm_sd = ckpt.get("normalizer", None)
    model_sd = ckpt.get("model", None)
    train_meta = ckpt.get("train_meta", {}) if isinstance(ckpt.get("train_meta", {}), dict) else {}

    cfg_train = train_meta.get("cfg_train", {}) if isinstance(train_meta.get("cfg_train", {}), dict) else {}
    sr_cfg = train_meta.get("sr_cfg", {}) if isinstance(train_meta.get("sr_cfg", {}), dict) else {}

    # reward_scale used in training
    reward_scale = float(cfg_train.get("reward_scale", 1.0e-6))
    lambda_risk = float(sr_cfg.get("lambda_risk", 0.0))

    # scales snapshot (if present); fallback to init values
    total_scale_yuan = float(train_meta.get("total_scale_yuan", sr_cfg.get("total_scale_init_yuan", 1.0e6)))
    risk_scale_yuan = float(train_meta.get("risk_scale_yuan", sr_cfg.get("risk_scale_init_yuan", 1.0e5)))

    if logger is not None:
        logger.log(f"[init] reward_scale={reward_scale:g} lambda_risk={lambda_risk:g}")
        logger.log(f"[init] scales: total_scale_yuan={total_scale_yuan:.3e} risk_scale_yuan={risk_scale_yuan:.3e}")

    # build env (single env)
    env_cfg = EnvConfig(seed=int(cfg_train.get("seed", 0)))
    env = PowerDispatchEnv(env_cfg)
    n_actions = int(env.action_space.n)
    time_period = int(getattr(env_cfg, "episode_len", 24))

    flattener = ObsFlattenerV2(env.observation_space, n_actions=n_actions, time_scale=float(cfg_train.get("time_scale", 1e-3)), time_period=time_period)

    # get n_bus, gt_din
    obs0, _info0 = env.reset(seed=int(cfg_train.get("seed", 0)), options=None)
    gt0 = np.asarray(obs0.get("gt_H", None), dtype=np.float32)
    if gt0.ndim != 2:
        raise RuntimeError("env obs must contain gt_H (N,D)")
    n_bus, gt_din = int(gt0.shape[0]), int(gt0.shape[1])

    # normalizer skeleton + load state
    mask = np.ones((flattener.flat_dim,), dtype=bool)
    sl_time = flattener.slices["time_feat"]
    mask[sl_time.start + 1: sl_time.stop] = False
    mask[flattener.slices["topology_id"]] = False
    normalizer = MaskedObsNormalizer(flattener.flat_dim, mask=mask, clip=float(cfg_train.get("obs_clip", 10.0)))
    if isinstance(norm_sd, dict):
        normalizer.load_state_dict(norm_sd)
        if logger is not None:
            logger.log("[init] normalizer loaded from checkpoint.")
    else:
        if logger is not None:
            logger.log("[init][warn] checkpoint has no normalizer; using fresh normalizer (may degrade inference).")

    # load GT pretrain to construct module + probe forward
    if logger is not None:
        logger.log(f"[init] load GT pretrain to build GT module: {gt_pretrain_ckpt_path}")
    gt, gt_cfg, _ = load_gtransformer_checkpoint(gt_pretrain_ckpt_path, device=device)
    r_switch, x_switch = _load_switch_params()
    adj_cache = TopologyAdjacencyCache(
        feeder_cluster=env.feeder_cluster,
        base_net=env.base_net,
        r_switch=r_switch,
        x_switch=x_switch,
        max_entries=int(cfg_train.get("model", {}).get("adj_cache_size", 64)) if isinstance(cfg_train.get("model", {}), dict) else 64
    )
    topo0 = int(np.asarray(obs0["topology_id"]).reshape(-1)[0])
    A0 = adj_cache.get(topo0).to(device=device)
    H_probe = torch.from_numpy(gt0.reshape(1, n_bus, gt_din)).to(device=device, dtype=torch.float32)
    spec = probe_gt_forward(gt, H_probe, A0)
    if logger is not None:
        logger.log(f"[init] GT probe: use_return_embeddings={spec.use_return_embeddings} use_batched_adj={spec.use_batched_adj} z_key={spec.z_key} out_dim={spec.out_dim}")

    # model hyperparams from cfg_train
    mcfg = cfg_train.get("model", {}) if isinstance(cfg_train.get("model", {}), dict) else {}
    # sr hyperparams from sr_cfg
    sr_act_emb_dim = int(sr_cfg.get("act_emb_dim", 64))
    sr_hidden = int(sr_cfg.get("hidden", 256))
    sr_dropout = float(sr_cfg.get("dropout", 0.0))

    model = MultiBranchActorCriticWithGTAndSR(
        obs_dim=flattener.flat_dim,
        n_actions=n_actions,
        slices=flattener.slices,
        gt=gt,
        gt_spec=spec,
        adj_cache=adj_cache,
        gt_pool=str(mcfg.get("gt_pool", "mean")),
        gt_proj_dim=int(mcfg.get("gt_proj_dim", 128)),
        enc_hidden=int(mcfg.get("enc_hidden", 256)),
        fusion_hidden=int(mcfg.get("fusion_hidden", 256)),
        fusion_blocks=int(mcfg.get("fusion_blocks", 2)),
        dropout=float(mcfg.get("dropout", 0.0)),
        policy_temperature=float(mcfg.get("policy_temperature", 1.0)),
        emb_dim_bus=int(mcfg.get("emb_dim_bus", 128)),
        emb_dim_load=int(mcfg.get("emb_dim_load", 128)),
        emb_dim_fcst=int(mcfg.get("emb_dim_fcst", 128)),
        emb_dim_time=int(mcfg.get("emb_dim_time", 32)),
        topo_emb_dim=int(mcfg.get("topo_emb_dim", 32)),
        adj_cache_cuda_size=int(mcfg.get("adj_cache_cuda_size", 16)),
        sr_act_emb_dim=sr_act_emb_dim,
        sr_hidden=sr_hidden,
        sr_dropout=sr_dropout,
    ).to(device)
    model.eval()

    if not isinstance(model_sd, dict):
        raise RuntimeError("checkpoint missing 'model' state_dict")

    missing, unexpected = model.load_state_dict(model_sd, strict=False)
    if logger is not None:
        logger.log(f"[init] model state loaded. missing={len(missing)} unexpected={len(unexpected)}")
        if len(missing) > 0:
            logger.log(f"[init][warn] missing keys (show up to 20): {missing[:20]}")
        if len(unexpected) > 0:
            logger.log(f"[init][warn] unexpected keys (show up to 20): {unexpected[:20]}")

    return {
        "env": env,
        "flattener": flattener,
        "normalizer": normalizer,
        "model": model,
        "device": device,
        "n_actions": n_actions,
        "reward_scale": reward_scale,
        "lambda_risk": lambda_risk,
        "total_scale_yuan": total_scale_yuan,
        "risk_scale_yuan": risk_scale_yuan,
        "seed": int(cfg_train.get("seed", 0)),
        "time_scale": float(cfg_train.get("time_scale", 1e-3)),
    }

# ============================================================
# Baseline & SR episode runner (non-UI)
# ============================================================

@torch.no_grad()
def sample_action_baseline(
    model: MultiBranchActorCriticWithGTAndSR,
    flattener: ObsFlattenerV2,
    normalizer: MaskedObsNormalizer,
    obs: Dict[str, np.ndarray],
    gtH: np.ndarray,
    *,
    device: torch.device,
    logger: Optional[PrintLogger] = None,
) -> Tuple[int, float, float]:
    """Sample one action from policy. Also compute SR predicted immediate r_hat for that action (for printing)."""
    obs_flat = flattener.flatten(obs)
    obs_norm = normalizer.normalize(obs_flat)
    obs_t = torch.from_numpy(obs_norm.reshape(1, -1)).to(device=device, dtype=torch.float32)
    gt_t = torch.from_numpy(gtH.reshape(1, gtH.shape[0], gtH.shape[1])).to(device=device, dtype=torch.float32)

    logits, _v, phi = model.forward_with_phi(obs_t, gt_t)
    dist = torch.distributions.Categorical(logits=logits)
    a = int(dist.sample().item())

    total_hat_norm, risk_hat_norm = model.sr(phi, torch.tensor([a], device=device, dtype=torch.int64))
    # These scales are not known here; caller usually prints only action.
    return a, float(total_hat_norm.item()), float(risk_hat_norm.item())

def run_episode_compare(
    stack: Dict[str, Any],
    *,
    look_ahead: int,
    beam_width: int,
    cand_K: int,
    gamma: float,
    max_steps: Optional[int],
    risk_budget_yuan: Optional[float] = None,
    verbose: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Run two episodes from identical reset:
      - baseline: policy sample
      - SR: beam search (H=look_ahead)
    Return: (baseline_records, sr_records)
    """
    env0: PowerDispatchEnv = stack["env"]
    # create two env instances to ensure identical dynamics (same ts, same config)
    env_base = PowerDispatchEnv(copy.deepcopy(env0.cfg))
    env_sr = PowerDispatchEnv(copy.deepcopy(env0.cfg))

    seed = int(stack.get("seed", 0))
    obs_b, info_b = env_base.reset(seed=seed, options=None)
    obs_s, info_s = env_sr.reset(seed=seed, options=None)

    logger = PrintLogger()
    if verbose:
        logger.log("[compare] start baseline vs SR (two separate env instances, same seed reset).")

    model = stack["model"]
    flattener = stack["flattener"]
    normalizer = stack["normalizer"]
    device = stack["device"]

    reward_scale = float(stack["reward_scale"])
    lambda_risk = float(stack["lambda_risk"])
    total_scale_yuan = float(stack["total_scale_yuan"])
    risk_scale_yuan = float(stack["risk_scale_yuan"])

    # env internal time trackers
    tb = int(getattr(env_base, "_t", 0))
    tsr = int(getattr(env_sr, "_t", 0))
    pab = int(getattr(env_base, "_prev_action", 0))
    pas = int(getattr(env_sr, "_prev_action", 0))

    baseline_records: List[Dict[str, Any]] = []
    sr_records: List[Dict[str, Any]] = []

    step_i = 0
    done_b = False
    done_s = False

    while True:
        step_i += 1
        if max_steps is not None and step_i > int(max_steps):
            break

        # ---------- baseline ----------
        if not done_b:
            gtHb = np.asarray(obs_b["gt_H"], dtype=np.float32)
            a_b, _t_hat, _r_hat = sample_action_baseline(
                model, flattener, normalizer, obs_b, gtHb, device=device
            )
            obs_b2, r_b, term_b, trunc_b, info_b2 = env_base.step(int(a_b))
            done_b = bool(term_b or trunc_b)

            tot_b, risk_b = extract_cost_labels_yuan_from_info(info_b2)
            baseline_records.append({
                "step": step_i,
                "t_prev": tb,
                "t_exec": tb + 1,
                "prev_action": pab,
                "action": int(a_b),
                "reward_true": float(r_b),
                "total_cost_yuan": float(tot_b),
                "risk_yuan": float(risk_b),
                "done": int(done_b),
            })
            obs_b = obs_b2
            tb = int(getattr(env_base, "_t", tb + 1))
            pab = int(getattr(env_base, "_prev_action", a_b))

        # ---------- SR beam search ----------
        if not done_s:
            gtHs = np.asarray(obs_s["gt_H"], dtype=np.float32)

            # compute SR immediate r_hat for chosen best action as well (for diff print)
            beam_items = beam_search_sr(
                env_sr, model, flattener, normalizer,
                cur_obs=obs_s, cur_gt_H=gtHs, cur_t=tsr, prev_action=pas,
                look_ahead=look_ahead, beam_width=beam_width, cand_K=cand_K, gamma=gamma, device=device,
                total_scale_yuan=total_scale_yuan, risk_scale_yuan=risk_scale_yuan,
                reward_scale=reward_scale, lambda_risk=lambda_risk,
                risk_budget_yuan=risk_budget_yuan,
                logger=(logger if verbose else None),
            )
            if len(beam_items) == 0:
                # fallback: sample action
                a_s, _, _ = sample_action_baseline(model, flattener, normalizer, obs_s, gtHs, device=device)
                chosen = [int(a_s)]
                sr_return = float("nan")
            else:
                chosen = beam_items[0].actions
                sr_return = float(beam_items[0].sr_return)

            a0 = int(chosen[0])

            # compute SR r_hat for the executed action at current node (depth=1 term)
            # (for printing: compare r_true at t+1 vs predicted r_hat)
            obs_flat = flattener.flatten(obs_s)
            obs_norm = normalizer.normalize(obs_flat)
            obs_t = torch.from_numpy(obs_norm.reshape(1, -1)).to(device=device, dtype=torch.float32)
            gt_t = torch.from_numpy(gtHs.reshape(1, gtHs.shape[0], gtHs.shape[1])).to(device=device, dtype=torch.float32)
            _logits, _v, phi = model.forward_with_phi(obs_t, gt_t)
            total_hat_norm, risk_hat_norm = model.sr(phi, torch.tensor([a0], device=device, dtype=torch.int64))
            r_hat_immediate = sr_reward_from_pred(
                total_hat_norm, risk_hat_norm,
                total_scale_yuan=total_scale_yuan,
                risk_scale_yuan=risk_scale_yuan,
                reward_scale=reward_scale,
                lambda_risk=lambda_risk,
            ).item()

            obs_s2, r_s, term_s, trunc_s, info_s2 = env_sr.step(int(a0))
            done_s = bool(term_s or trunc_s)

            tot_s, risk_s = extract_cost_labels_yuan_from_info(info_s2)
            sr_records.append({
                "step": step_i,
                "t_prev": tsr,
                "t_exec": tsr + 1,
                "prev_action": pas,
                "action": int(a0),
                "reward_true": float(r_s),
                "reward_hat_immediate": float(r_hat_immediate),
                "reward_true_minus_hat": float(r_s - r_hat_immediate),
                "sr_return_lookahead": float(sr_return),
                "total_cost_yuan": float(tot_s),
                "risk_yuan": float(risk_s),
                "done": int(done_s),
            })

            if verbose:
                logger.log(
                    f"[SR step={step_i}] exec a0={a0} | r_true={r_s:+.4e} r_hat={r_hat_immediate:+.4e} "
                    f"diff(true-hat)={r_s-r_hat_immediate:+.4e} | best_sr_return(lookahead)={sr_return:+.4e}"
                )

            obs_s = obs_s2
            tsr = int(getattr(env_sr, "_t", tsr + 1))
            pas = int(getattr(env_sr, "_prev_action", a0))

        if done_b and done_s:
            break

    return baseline_records, sr_records

# ============================================================
# CSV helpers
# ============================================================

def write_csv(path: str, rows: List[Dict[str, Any]]) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if len(rows) == 0:
        # create empty file
        with open(path, "w", encoding="utf-8-sig") as f:
            f.write("")
        return path
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path

# ============================================================
# Gradio UI (Debug + Auto)
# ============================================================

@dataclass
class UIState:
    stack: Optional[Dict[str, Any]] = None
    obs: Optional[Dict[str, np.ndarray]] = None
    info: Optional[Dict[str, Any]] = None
    cur_t: int = 0
    prev_action: int = 0
    done: bool = False
    beam: Optional[List[BeamItem]] = None
    step_records: Optional[List[Dict[str, Any]]] = None
    logger: Optional[PrintLogger] = None

def ui_reset(st: UIState, seed: int) -> Tuple[UIState, str, List[List[Any]]]:
    st.logger = st.logger or PrintLogger()
    if st.stack is None:
        st.logger.log("[ui][reset] stack is None. Please click 'Init/Reload Model' first.")
        return st, st.logger.text(), []

    env: PowerDispatchEnv = st.stack["env"]
    obs, info = env.reset(seed=int(seed), options=None)

    st.obs = obs
    st.info = info if isinstance(info, dict) else {}
    st.cur_t = int(getattr(env, "_t", 0))
    st.prev_action = int(getattr(env, "_prev_action", 0))
    st.done = False
    st.beam = None
    st.step_records = []

    st.logger.log(f"[ui][reset] env reset done. t={st.cur_t} prev_action={st.prev_action}")
    return st, st.logger.text(), []

def ui_init_reload(
    st: UIState,
    ppo_ckpt_path: str,
    gt_ckpt_path: str,
    device_str: str,
) -> Tuple[UIState, str]:
    st.logger = st.logger or PrintLogger()
    st.logger.log("[ui][init] initializing inference stack ...")
    try:
        stack = build_inference_stack(
            ppo_sr_ckpt_path=ppo_ckpt_path,
            gt_pretrain_ckpt_path=gt_ckpt_path,
            device_str=device_str,
            logger=st.logger,
        )
        st.stack = stack
        st.logger.log("[ui][init] stack ready.")
    except Exception as e:
        st.logger.log(f"[ui][init][ERROR] {repr(e)}")
        st.stack = None
    return st, st.logger.text()

def ui_expand_beam(
    st: UIState,
    look_ahead: int,
    beam_width: int,
    cand_K: int,
    gamma: float,
    risk_budget_yuan: float,
    use_risk_budget: bool,
) -> Tuple[UIState, str, List[List[Any]]]:
    st.logger = st.logger or PrintLogger()
    if st.stack is None or st.obs is None or st.done:
        st.logger.log("[ui][beam] invalid state (stack/obs/done).")
        return st, st.logger.text(), []

    stack = st.stack
    env: PowerDispatchEnv = stack["env"]
    model = stack["model"]
    flattener = stack["flattener"]
    normalizer = stack["normalizer"]
    device = stack["device"]

    gtH = np.asarray(st.obs["gt_H"], dtype=np.float32)

    rb = float(risk_budget_yuan) if use_risk_budget else None

    st.logger.log(
        f"[ui][beam] expand beam: look_ahead={look_ahead} beam_width={beam_width} cand_K={cand_K} gamma={gamma} risk_budget_yuan={rb}"
    )

    beam_items = beam_search_sr(
        env, model, flattener, normalizer,
        cur_obs=st.obs, cur_gt_H=gtH, cur_t=st.cur_t, prev_action=st.prev_action,
        look_ahead=int(look_ahead), beam_width=int(beam_width), cand_K=int(cand_K),
        gamma=float(gamma), device=device,
        total_scale_yuan=float(stack["total_scale_yuan"]),
        risk_scale_yuan=float(stack["risk_scale_yuan"]),
        reward_scale=float(stack["reward_scale"]),
        lambda_risk=float(stack["lambda_risk"]),
        risk_budget_yuan=rb,
        logger=st.logger,
    )
    st.beam = beam_items

    # build table rows
    table: List[List[Any]] = []
    for i, b in enumerate(beam_items[:30]):
        table.append([
            i,
            str(b.actions),
            float(b.sr_return),
            float(b.sr_risk_yuan),
            "" if b.true_return is None else float(b.true_return),
            "" if b.true_risk_yuan is None else float(b.true_risk_yuan),
        ])

    return st, st.logger.text(), table

def ui_verify_candidate(
    st: UIState,
    cand_index: int,
    gamma: float,
) -> Tuple[UIState, str, List[List[Any]]]:
    st.logger = st.logger or PrintLogger()
    if st.stack is None or st.obs is None or st.beam is None:
        st.logger.log("[ui][verify] need beam results first.")
        return st, st.logger.text(), []

    if cand_index < 0 or cand_index >= len(st.beam):
        st.logger.log(f"[ui][verify] invalid cand_index={cand_index}")
        return st, st.logger.text(), []

    env: PowerDispatchEnv = st.stack["env"]
    b = st.beam[cand_index]

    st.logger.log(f"[ui][verify] verifying candidate idx={cand_index} seq={b.actions}")
    true_ret, true_risk, det = verify_sequence_true_return(
        env,
        start_t=st.cur_t,
        start_prev_action=st.prev_action,
        start_obs=st.obs,
        actions=b.actions,
        gamma=float(gamma),
        logger=st.logger,
    )
    b.true_return = float(true_ret)
    b.true_risk_yuan = float(true_risk)
    # attach details (append)
    if b.details is None:
        b.details = []
    b.details.extend(det)

    # rebuild table
    table: List[List[Any]] = []
    for i, bb in enumerate(st.beam[:30]):
        table.append([
            i,
            str(bb.actions),
            float(bb.sr_return),
            float(bb.sr_risk_yuan),
            "" if bb.true_return is None else float(bb.true_return),
            "" if bb.true_risk_yuan is None else float(bb.true_risk_yuan),
        ])
    return st, st.logger.text(), table

def ui_execute_action(
    st: UIState,
    action: int,
    gamma: float,
    record_sr_hat: bool,
) -> Tuple[UIState, str, str]:
    st.logger = st.logger or PrintLogger()
    if st.stack is None or st.obs is None or st.done:
        st.logger.log("[ui][step] invalid state.")
        return st, st.logger.text(), "N/A"

    stack = st.stack
    env: PowerDispatchEnv = stack["env"]
    model = stack["model"]
    flattener = stack["flattener"]
    normalizer = stack["normalizer"]
    device = stack["device"]

    a = int(action)

    # compute immediate r_hat for this action (optional)
    r_hat = None
    if record_sr_hat:
        gtH = np.asarray(st.obs["gt_H"], dtype=np.float32)
        obs_flat = flattener.flatten(st.obs)
        obs_norm = normalizer.normalize(obs_flat)
        obs_t = torch.from_numpy(obs_norm.reshape(1, -1)).to(device=device, dtype=torch.float32)
        gt_t = torch.from_numpy(gtH.reshape(1, gtH.shape[0], gtH.shape[1])).to(device=device, dtype=torch.float32)
        _logits, _v, phi = model.forward_with_phi(obs_t, gt_t)
        total_hat_norm, risk_hat_norm = model.sr(phi, torch.tensor([a], device=device, dtype=torch.int64))
        r_hat = sr_reward_from_pred(
            total_hat_norm, risk_hat_norm,
            total_scale_yuan=float(stack["total_scale_yuan"]),
            risk_scale_yuan=float(stack["risk_scale_yuan"]),
            reward_scale=float(stack["reward_scale"]),
            lambda_risk=float(stack["lambda_risk"]),
        ).item()

    obs2, r_true, term, trunc, info2 = env.step(a)
    done = bool(term or trunc)

    tot_yuan, risk_yuan = extract_cost_labels_yuan_from_info(info2)
    st.step_records = st.step_records or []
    st.step_records.append({
        "step": len(st.step_records) + 1,
        "t_prev": st.cur_t,
        "t_exec": st.cur_t + 1,
        "prev_action": st.prev_action,
        "action": a,
        "reward_true": float(r_true),
        "reward_hat_immediate": "" if r_hat is None else float(r_hat),
        "diff_true_minus_hat": "" if r_hat is None else float(r_true - r_hat),
        "total_cost_yuan": float(tot_yuan),
        "risk_yuan": float(risk_yuan),
        "done": int(done),
    })

    st.logger.log(
        f"[ui][step] exec action={a} | r_true={r_true:+.4e} "
        + (f"r_hat={r_hat:+.4e} diff={r_true-r_hat:+.4e} " if r_hat is not None else "")
        + f"| total_cost_yuan={tot_yuan:+.4e} risk_yuan={risk_yuan:+.4e} done={done}"
    )

    st.obs = obs2
    st.info = info2
    st.cur_t = int(getattr(env, "_t", st.cur_t + 1))
    st.prev_action = int(getattr(env, "_prev_action", a))
    st.done = done
    st.beam = None  # clear beam after stepping

    status = f"t={st.cur_t} prev_action={st.prev_action} done={st.done}"
    return st, st.logger.text(), status

def ui_execute_best_from_beam(st: UIState, gamma: float, record_sr_hat: bool) -> Tuple[UIState, str, str]:
    st.logger = st.logger or PrintLogger()
    if st.beam is None or len(st.beam) == 0:
        st.logger.log("[ui][best] no beam results. Expand beam first.")
        return st, st.logger.text(), "N/A"
    a0 = int(st.beam[0].actions[0])
    st.logger.log(f"[ui][best] execute best a0={a0} from seq={st.beam[0].actions}")
    return ui_execute_action(st, a0, gamma=gamma, record_sr_hat=record_sr_hat)

def ui_auto_run(
    st: UIState,
    mode: str,
    look_ahead: int,
    beam_width: int,
    cand_K: int,
    gamma: float,
    risk_budget_yuan: float,
    use_risk_budget: bool,
    max_steps: int,
) -> Tuple[UIState, str, str, str]:
    st.logger = st.logger or PrintLogger()
    if st.stack is None:
        st.logger.log("[ui][auto] init stack first.")
        return st, st.logger.text(), "N/A", ""

    stack = st.stack

    # run single episode from fresh reset
    env = stack["env"]
    seed = int(stack.get("seed", 0))
    obs, info = env.reset(seed=seed, options=None)
    cur_t = int(getattr(env, "_t", 0))
    prev_action = int(getattr(env, "_prev_action", 0))
    done = False

    model = stack["model"]
    flattener = stack["flattener"]
    normalizer = stack["normalizer"]
    device = stack["device"]

    rb = float(risk_budget_yuan) if use_risk_budget else None

    records: List[Dict[str, Any]] = []

    st.logger.log(f"[ui][auto] mode={mode} reset done. start t={cur_t} prev_action={prev_action}")

    for step_i in range(1, int(max_steps) + 1):
        if done:
            break

        gtH = np.asarray(obs["gt_H"], dtype=np.float32)

        if mode.lower().startswith("baseline"):
            a, _, _ = sample_action_baseline(model, flattener, normalizer, obs, gtH, device=device)
            chosen = [int(a)]
            sr_return = float("nan")
        else:
            beam_items = beam_search_sr(
                env, model, flattener, normalizer,
                cur_obs=obs, cur_gt_H=gtH, cur_t=cur_t, prev_action=prev_action,
                look_ahead=int(look_ahead),
                beam_width=int(beam_width),
                cand_K=int(cand_K),
                gamma=float(gamma),
                device=device,
                total_scale_yuan=float(stack["total_scale_yuan"]),
                risk_scale_yuan=float(stack["risk_scale_yuan"]),
                reward_scale=float(stack["reward_scale"]),
                lambda_risk=float(stack["lambda_risk"]),
                risk_budget_yuan=rb,
                logger=None,  # auto mode不刷爆日志
            )
            if len(beam_items) == 0:
                a, _, _ = sample_action_baseline(model, flattener, normalizer, obs, gtH, device=device)
                chosen = [int(a)]
                sr_return = float("nan")
            else:
                chosen = beam_items[0].actions
                sr_return = float(beam_items[0].sr_return)

        a0 = int(chosen[0])

        # immediate r_hat for record
        obs_flat = flattener.flatten(obs)
        obs_norm = normalizer.normalize(obs_flat)
        obs_t = torch.from_numpy(obs_norm.reshape(1, -1)).to(device=device, dtype=torch.float32)
        gt_t = torch.from_numpy(gtH.reshape(1, gtH.shape[0], gtH.shape[1])).to(device=device, dtype=torch.float32)
        _logits, _v, phi = model.forward_with_phi(obs_t, gt_t)
        total_hat_norm, risk_hat_norm = model.sr(phi, torch.tensor([a0], device=device, dtype=torch.int64))
        r_hat = sr_reward_from_pred(
            total_hat_norm, risk_hat_norm,
            total_scale_yuan=float(stack["total_scale_yuan"]),
            risk_scale_yuan=float(stack["risk_scale_yuan"]),
            reward_scale=float(stack["reward_scale"]),
            lambda_risk=float(stack["lambda_risk"]),
        ).item()

        obs2, r_true, term, trunc, info2 = env.step(a0)
        done = bool(term or trunc)

        tot_y, risk_y = extract_cost_labels_yuan_from_info(info2)

        records.append({
            "step": step_i,
            "mode": mode,
            "t_prev": cur_t,
            "t_exec": cur_t + 1,
            "prev_action": prev_action,
            "action": a0,
            "reward_true": float(r_true),
            "reward_hat_immediate": float(r_hat),
            "diff_true_minus_hat": float(r_true - r_hat),
            "sr_return_lookahead": float(sr_return),
            "total_cost_yuan": float(tot_y),
            "risk_yuan": float(risk_y),
            "done": int(done),
        })

        obs = obs2
        cur_t = int(getattr(env, "_t", cur_t + 1))
        prev_action = int(getattr(env, "_prev_action", a0))

    # save csv
    out_dir = "./sr_inference_outputs"
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"auto_{mode.replace(' ','_')}.csv")
    write_csv(csv_path, records)

    # summary
    ret = float(np.sum([r["reward_true"] for r in records])) if records else 0.0
    diff_mean = float(np.mean([r["diff_true_minus_hat"] for r in records])) if records else 0.0
    st.logger.log(f"[ui][auto] finished steps={len(records)} return_true_sum={ret:+.4e} mean(true-hat)={diff_mean:+.4e} csv={csv_path}")

    st.step_records = records
    st.done = True
    status = f"auto done. steps={len(records)} return_sum={ret:+.4e} mean(true-hat)={diff_mean:+.4e}"
    return st, st.logger.text(), status, csv_path

def build_gradio_app() -> gr.Blocks:
    st = UIState(logger=PrintLogger())

    with gr.Blocks(title="Self-Reflective SR Beam Search Debugger", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Self-Reflective Dispatch Inference (SR + Look-ahead Beam Search)\n"
            "- Debug: step-by-step beam search + optional environment verification\n"
            "- Auto: run full episode and export CSV\n"
        )

        with gr.Row():
            ppo_ckpt = gr.Textbox(label="PPO+SR checkpoint path", value="./runs_ppo_gt_sr_merged/ppo_gt_sr_merged_ckpt_final.pt")
            gt_ckpt = gr.Textbox(label="GT pretrain checkpoint path", value="./GTransformer/runs/pretrain_masked_node_only_v1/ckpt_best.pt")
            device = gr.Dropdown(label="device", choices=["cuda", "cpu"], value="cuda")

        init_btn = gr.Button("Init / Reload Model", variant="primary")
        log_box = gr.Textbox(label="Logs (tail)", value="", lines=18)

        init_btn.click(
            fn=lambda ppo_path, gt_path, dev: ui_init_reload(st, ppo_path, gt_path, dev),
            inputs=[ppo_ckpt, gt_ckpt, device],
            outputs=[gr.State(), log_box],
        )

        with gr.Tabs():
            with gr.Tab("A. Debug Mode (step-by-step)"):
                with gr.Row():
                    seed_in = gr.Number(label="reset seed", value=0, precision=0)
                    reset_btn = gr.Button("Reset Env", variant="secondary")
                    status = gr.Textbox(label="Env status", value="N/A")

                with gr.Row():
                    look_ahead = gr.Slider(label="look-ahead H", minimum=1, maximum=5, step=1, value=3)
                    beam_width = gr.Slider(label="beam width B", minimum=1, maximum=20, step=1, value=6)
                    cand_K = gr.Slider(label="candidate K per node", minimum=1, maximum=30, step=1, value=10)
                    gamma = gr.Slider(label="discount gamma", minimum=0.5, maximum=1.0, step=0.01, value=0.99)

                with gr.Row():
                    use_risk_budget = gr.Checkbox(label="use risk budget constraint", value=False)
                    risk_budget_yuan = gr.Number(label="risk budget (yuan)", value=1.0e9, precision=2)

                expand_btn = gr.Button("1) Expand Beam Search", variant="primary")
                beam_table = gr.Dataframe(
                    headers=["idx", "action_seq", "sr_return", "sr_risk_yuan", "true_return(verified)", "true_risk_yuan(verified)"],
                    datatype=["number", "str", "number", "number", "str", "str"],
                    row_count=10,
                    col_count=6,
                    interactive=False,
                )

                with gr.Row():
                    cand_idx = gr.Number(label="candidate idx to verify", value=0, precision=0)
                    verify_btn = gr.Button("2) Verify via env simulation (selected seq)", variant="secondary")

                with gr.Row():
                    record_sr_hat = gr.Checkbox(label="record & print immediate SR r_hat", value=True)
                    exec_best_btn = gr.Button("3) Execute best action (beam[0].actions[0])", variant="primary")

                with gr.Row():
                    manual_a = gr.Number(label="manual action id", value=0, precision=0)
                    exec_manual_btn = gr.Button("Execute manual action", variant="secondary")

                reset_btn.click(
                    fn=lambda sd: ui_reset(st, int(sd)),
                    inputs=[seed_in],
                    outputs=[gr.State(), log_box, beam_table],
                )

                expand_btn.click(
                    fn=lambda H, B, K, g, rb, use_rb: ui_expand_beam(st, int(H), int(B), int(K), float(g), float(rb), bool(use_rb)),
                    inputs=[look_ahead, beam_width, cand_K, gamma, risk_budget_yuan, use_risk_budget],
                    outputs=[gr.State(), log_box, beam_table],
                )

                verify_btn.click(
                    fn=lambda idx, g: ui_verify_candidate(st, int(idx), float(g)),
                    inputs=[cand_idx, gamma],
                    outputs=[gr.State(), log_box, beam_table],
                )

                exec_best_btn.click(
                    fn=lambda g, rec: ui_execute_best_from_beam(st, float(g), bool(rec)),
                    inputs=[gamma, record_sr_hat],
                    outputs=[gr.State(), log_box, status],
                )

                exec_manual_btn.click(
                    fn=lambda a, g, rec: ui_execute_action(st, int(a), float(g), bool(rec)),
                    inputs=[manual_a, gamma, record_sr_hat],
                    outputs=[gr.State(), log_box, status],
                )

            with gr.Tab("B. Auto Mode (full episode + CSV)"):
                with gr.Row():
                    mode = gr.Dropdown(label="mode", choices=["SR (beam search)", "Baseline (policy sample)"], value="SR (beam search)")
                    max_steps = gr.Slider(label="max_steps", minimum=1, maximum=200, step=1, value=48)

                with gr.Row():
                    run_btn = gr.Button("Run Auto Episode", variant="primary")
                    auto_status = gr.Textbox(label="Summary", value="N/A")
                    csv_out = gr.File(label="Download CSV")

                run_btn.click(
                    fn=lambda m, H, B, K, g, rb, use_rb, ms: ui_auto_run(
                        st, m, int(H), int(B), int(K), float(g), float(rb), bool(use_rb), int(ms)
                    ),
                    inputs=[mode, look_ahead, beam_width, cand_K, gamma, risk_budget_yuan, use_risk_budget, max_steps],
                    outputs=[gr.State(), log_box, auto_status, csv_out],
                )

        gr.Markdown(
            "### Tips\n"
            "- Debug 模式：先 Reset，再 Expand Beam，然后可 Verify 某条候选序列，最后 Execute best 或手动动作。\n"
            "- Auto 模式：直接跑完整 episode，输出 CSV。\n"
        )

    return demo

# ============================================================
# main (edit params here)
# ============================================================

if __name__ == "__main__":
    # -------------------------
    # USER EDIT AREA (no argparse)
    # -------------------------
    PPO_SR_CKPT_PATH = "./runs_ppo_gt_sr_merged/ppo_gt_sr_merged_ckpt_final.pt"
    GT_PRETRAIN_CKPT_PATH = "./GTransformer/runs/pretrain_masked_node_only_v1/ckpt_best.pt"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # If you want to quickly run baseline vs SR in terminal (optional):
    RUN_COMPARE_ONCE = False
    COMPARE_LOOKAHEAD = 3
    COMPARE_BEAM_WIDTH = 6
    COMPARE_CAND_K = 10
    COMPARE_GAMMA = 0.99
    COMPARE_MAX_STEPS = 48

    # -------------------------
    # build UI
    # -------------------------
    app = build_gradio_app()
    # launch
    app.launch(server_name="0.0.0.0", server_port=7860, show_error=True)

    # -------------------------
    # optional compare in terminal
    # -------------------------
    if RUN_COMPARE_ONCE:
        lg = PrintLogger()
        stack = build_inference_stack(
            ppo_sr_ckpt_path=PPO_SR_CKPT_PATH,
            gt_pretrain_ckpt_path=GT_PRETRAIN_CKPT_PATH,
            device_str=DEVICE,
            logger=lg,
        )
        baseline_rows, sr_rows = run_episode_compare(
            stack,
            look_ahead=COMPARE_LOOKAHEAD,
            beam_width=COMPARE_BEAM_WIDTH,
            cand_K=COMPARE_CAND_K,
            gamma=COMPARE_GAMMA,
            max_steps=COMPARE_MAX_STEPS,
            risk_budget_yuan=None,
            verbose=True,
        )
        out_dir = "./sr_inference_outputs"
        os.makedirs(out_dir, exist_ok=True)
        write_csv(os.path.join(out_dir, "baseline.csv"), baseline_rows)
        write_csv(os.path.join(out_dir, "sr.csv"), sr_rows)
        print("Saved to ./sr_inference_outputs/baseline.csv and sr.csv")
