# In[]
# -*- coding: utf-8 -*-
"""
eval_policy_and_estimators.py

目标：
1) 统一量纲（元）下评估两个估计器：
   A) SR decoder 预测的 step cost（元） vs env 真实 step cost（元）
   B) Critic value 预测的“折扣 return_used” vs Monte-Carlo 折扣回报，并换算成元

2) 若干 episode 检测 policy 是否有效：greedy policy vs random baseline
3) 绘制趋势图（episode 维度）并保存 png

无 argparse；所有参数在 __main__ 中直接改。
"""

from __future__ import annotations

import os
import math
import json
import time
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")  # 适合服务器/无 GUI
import matplotlib.pyplot as plt

# --------------------------
# 依赖你的环境与系统
# --------------------------
from power_dispatch_env_withGT_dimrisk import (
    PowerDispatchEnv,
    EnvConfig,
    TimeseriesSchema,
    built_ppnet_for_pfcal,
    set_fc_state_with_acts,
)

# 你训练脚本里用到的 r/x switch 参数来自 config746sys
import importlib


# ============================================================
# 基础工具
# ============================================================

def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device(device_str: str) -> torch.device:
    s = (device_str or "auto").lower().strip()
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if s in ("cuda", "cpu"):
        return torch.device(s)
    raise ValueError(f"Unknown device: {device_str}")

def safe_mean(x: List[float]) -> float:
    return float(np.mean(x)) if len(x) > 0 else float("nan")

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _load_switch_params() -> Tuple[float, float]:
    m = importlib.import_module("config746sys")
    r_switch = float(getattr(m, "r_switch"))
    x_switch = float(getattr(m, "x_switch"))
    return r_switch, x_switch

def _get_float(info: Dict[str, Any], keys: List[str], default: float = 0.0) -> float:
    for k in keys:
        if k in info:
            try:
                return float(info.get(k, default))
            except Exception:
                continue
    return float(default)

def extract_costs_yuan_from_info(info: Dict[str, Any]) -> Tuple[float, float, int]:
    """
    返回：total_cost_yuan, risk_cost_yuan, pf_failed(0/1)
    """
    total = _get_float(info, ["total_cost_yuan"], 0.0)
    risk = _get_float(info, ["risk_cost_yuan", "risk_term", "risk_cost"], 0.0)
    pf_failed = 1 if bool(info.get("pf_failed", False)) else 0
    return float(total), float(risk), int(pf_failed)


# ============================================================
# Observation flatten + normalizer（复用你训练脚本逻辑）
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

class ObsFlattenerV2:
    """
    与训练脚本一致的 flatten 顺序：
      bus_vm_pu
      bus_va_deg
      load_p_mw
      load_q_mvar
      forecast_p_mw
      forecast_q_mvar
      time_feat [3]
      topology_id [1]
    gt_H 不进 flat，单独传。
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


# ============================================================
# GT + Adj cache + ActorCritic + SR（与训练脚本对齐）
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
    Lazy import: ./GTransformer/gt_torch_model.py
    """
    import sys
    base_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    gt_dir = os.path.join(base_dir, "GTransformer")
    if gt_dir not in sys.path:
        sys.path.append(gt_dir)

    gt_mod = __import__("gt_torch_model")
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
            "din": 6, "d_model": 128, "n_heads": 8, "d_ff": 256, "n_layers": 3,
            "k_min": 1, "k_max": 5, "dropout": 0.1, "attn_dropout": 0.1, "adj_mode": "binary",
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
        raise RuntimeError(f"GT output dict has no embedding. keys={list(out.keys())}")
    if torch.is_tensor(out):
        return out
    raise RuntimeError(f"Unsupported GT output type: {type(out)}")

@torch.no_grad()
def probe_gt_forward(gt: nn.Module, H: torch.Tensor, A2: torch.Tensor) -> GTForwardSpec:
    gt.eval()
    B, N, _ = H.shape
    candidates: List[Tuple[bool, bool]] = [(True, False), (False, False), (True, True), (False, True)]
    for use_ret, use_batched in candidates:
        try:
            A = A2.unsqueeze(0).expand(B, -1, -1) if use_batched else A2
            out = gt(H, A, return_embeddings=True) if use_ret else gt(H, A)
            z = _extract_z(out, z_key="z" if isinstance(out, dict) and "z" in out else None)
            if z.ndim != 3 or z.shape[0] != B or z.shape[1] != N:
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
    raise RuntimeError("Failed to probe GT forward signature.")

def gt_forward_with_spec(gt: nn.Module, spec: GTForwardSpec, H: torch.Tensor, A2: torch.Tensor) -> torch.Tensor:
    A = A2.unsqueeze(0).expand(H.shape[0], -1, -1) if spec.use_batched_adj else A2
    out = gt(H, A, return_embeddings=True) if spec.use_return_embeddings else gt(H, A)
    z = _extract_z(out, spec.z_key)
    if z.ndim != 3 or z.shape[0] != H.shape[0] or z.shape[1] != H.shape[1]:
        raise RuntimeError(f"GT embedding shape mismatch: got {tuple(z.shape)} expected ({H.shape[0]},{H.shape[1]},D)")
    return z

class TopologyAdjacencyCache:
    def __init__(self, *, feeder_cluster: Any, base_net: Any, r_switch: float, x_switch: float, max_entries: int = 64):
        from collections import OrderedDict
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
                fb = int(row["from_bus"]); tb = int(row["to_bus"])
                if fb in self._bus_map and tb in self._bus_map:
                    i = self._bus_map[fb]; j = self._bus_map[tb]
                    A[i, j] = 1.0; A[j, i] = 1.0

        if hasattr(net_cal, "trafo") and len(net_cal.trafo) > 0:
            for _, row in net_cal.trafo.iterrows():
                hb = int(row["hv_bus"]); lb = int(row["lv_bus"])
                if hb in self._bus_map and lb in self._bus_map:
                    i = self._bus_map[hb]; j = self._bus_map[lb]
                    A[i, j] = 1.0; A[j, i] = 1.0

        if hasattr(net_cal, "switch") and len(net_cal.switch) > 0:
            sw = net_cal.switch
            for _, row in sw.iterrows():
                try:
                    if str(row.get("et", "")) != "b":
                        continue
                    if not bool(row.get("closed", True)):
                        continue
                    b1 = int(row["bus"]); b2 = int(row["element"])
                    if b1 in self._bus_map and b2 in self._bus_map:
                        i = self._bus_map[b1]; j = self._bus_map[b2]
                        A[i, j] = 1.0; A[j, i] = 1.0
                except Exception:
                    continue
        return A

    def get(self, topology_id: int) -> torch.Tensor:
        from collections import OrderedDict
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

class SelfReflectionDecoder(nn.Module):
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
        from collections import OrderedDict

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

    def _len(self, name: str) -> int:
        sl = self.slices[name]
        return int(sl.stop - sl.start)

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
# 评估指标
# ============================================================

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return {k: float("nan") for k in ["mae", "rmse", "mape", "r2"]}
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    denom = np.maximum(np.abs(y_true), 1e-12)
    mape = float(np.mean(np.abs(err) / denom))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true))**2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}

def discounted_return(x: List[float], gamma: float) -> float:
    g = float(gamma)
    acc = 0.0
    pw = 1.0
    for v in x:
        acc += pw * float(v)
        pw *= g
    return float(acc)


# ============================================================
# 核心评估逻辑
# ============================================================

@dataclass
class EvalConfig:
    ckpt_path: str
    gt_ckpt_path: str
    device: str = "cuda"
    seed: int = 0

    # eval episode 设置
    n_eval_episodes: int = 20
    eval_mode: str = "greedy"  # "greedy" 或 "sample"
    compare_random_baseline: bool = True
    max_steps_per_episode: Optional[int] = None  # None 则用 env_cfg.episode_len

    # reward/return 相关（与你训练一致的量纲）
    reward_scale: float = 1e-6          # env.reward = -total_cost_yuan; 训练里又乘 reward_scale
    lambda_risk: float = 0.0           # 你的 SR 配置里是 0
    alpha_eval: float = 0.0            # 若想复现训练 return_used，则设为训练末 alpha（从 ckpt meta 读到也行）
    gamma: float = 0.99

    # SR 反标准化尺度（若 ckpt meta 有则自动覆盖）
    total_scale_yuan: float = 1.0e6
    risk_scale_yuan: float = 1.0e5

    # 输出
    out_dir: str = "./eval_outputs"
    save_step_scatter: bool = True


def build_env_and_tools(time_scale: float, seed: int) -> Tuple[PowerDispatchEnv, ObsFlattenerV2, MaskedObsNormalizer]:
    env_cfg = EnvConfig(seed=seed)
    env = PowerDispatchEnv(env_cfg)
    n_actions = int(env.action_space.n)
    time_period = int(getattr(env_cfg, "episode_len", 24))

    flattener = ObsFlattenerV2(env.observation_space, n_actions=n_actions, time_scale=time_scale, time_period=time_period)

    # normalizer mask：不 normalize sin/cos & topology_id
    mask = np.ones((flattener.flat_dim,), dtype=bool)
    sl_time = flattener.slices["time_feat"]
    mask[sl_time.start + 1 : sl_time.stop] = False
    mask[flattener.slices["topology_id"]] = False
    normalizer = MaskedObsNormalizer(flattener.flat_dim, mask=mask, clip=10.0)
    return env, flattener, normalizer


def build_model_from_ckpt(
    eval_cfg: EvalConfig,
    env: PowerDispatchEnv,
    flattener: ObsFlattenerV2,
    normalizer: MaskedObsNormalizer,
) -> Tuple[MultiBranchActorCriticWithGTAndSR, Dict[str, Any]]:
    device = get_device(eval_cfg.device)

    ckpt = _torch_load_trusted(eval_cfg.ckpt_path, map_location=device)
    meta = ckpt.get("train_meta", {}) if isinstance(ckpt, dict) else {}

    # 读 meta 覆盖关键参数（若存在）
    if isinstance(meta, dict):
        if "cfg_train" in meta and isinstance(meta["cfg_train"], dict):
            ct = meta["cfg_train"]
            if "reward_scale" in ct:
                eval_cfg.reward_scale = float(ct["reward_scale"])
        if "sr_cfg" in meta and isinstance(meta["sr_cfg"], dict):
            sc = meta["sr_cfg"]
            if "lambda_risk" in sc:
                eval_cfg.lambda_risk = float(sc["lambda_risk"])
        if "alpha_now" in meta:
            # 你如果想用训练末 shaping，直接把 alpha_eval 改成 meta 的 alpha_now
            pass
        if "total_scale_yuan" in meta:
            eval_cfg.total_scale_yuan = float(meta["total_scale_yuan"])
        if "risk_scale_yuan" in meta:
            eval_cfg.risk_scale_yuan = float(meta["risk_scale_yuan"])

    # 加载 normalizer
    if "normalizer" in ckpt:
        normalizer.load_state_dict(ckpt["normalizer"])

    # 构建 GT / adj_cache / probe spec
    gt, gt_cfg, _ = load_gtransformer_checkpoint(eval_cfg.gt_ckpt_path, device=device)

    # 用 env reset 拿到 gt_H 形状
    obs0, _ = env.reset(seed=eval_cfg.seed, options=None)
    gt_H0 = np.asarray(obs0.get("gt_H", None), dtype=np.float32)
    if gt_H0.ndim != 2:
        raise RuntimeError(f"env obs['gt_H'] must be (N,D); got {gt_H0.shape}")
    n_bus, gt_din = int(gt_H0.shape[0]), int(gt_H0.shape[1])

    r_switch, x_switch = _load_switch_params()
    adj_cache = TopologyAdjacencyCache(
        feeder_cluster=env.feeder_cluster,
        base_net=env.base_net,
        r_switch=r_switch,
        x_switch=x_switch,
        max_entries=64,
    )

    # probe
    topo0 = int(np.asarray(obs0["topology_id"]).reshape(-1)[0])
    A0 = adj_cache.get(topo0).to(device=device)
    H_probe = torch.from_numpy(gt_H0[None, ...]).to(device=device, dtype=torch.float32)
    H_probe = H_probe.expand(2, -1, -1).contiguous()  # B=2
    spec = probe_gt_forward(gt, H_probe, A0)

    # 尝试从 ckpt meta 里恢复模型尺寸（若没有则用你训练时常用的大配置默认）
    # 这里用“稳妥默认”，避免因 meta 缺字段导致无法评估
    model_cfg = {}
    if isinstance(meta, dict) and isinstance(meta.get("cfg_train", None), dict):
        m = meta["cfg_train"].get("model", None)
        if isinstance(m, dict):
            model_cfg = m

    def _g(k: str, default: int) -> int:
        try:
            return int(model_cfg.get(k, default))
        except Exception:
            return int(default)

    def _gf(k: str, default: float) -> float:
        try:
            return float(model_cfg.get(k, default))
        except Exception:
            return float(default)

    # SR cfg 也可能在 meta 里
    sr_cfg = {}
    if isinstance(meta, dict) and isinstance(meta.get("sr_cfg", None), dict):
        sr_cfg = meta["sr_cfg"]

    sr_act_emb_dim = int(sr_cfg.get("act_emb_dim", 64)) if isinstance(sr_cfg, dict) else 64
    sr_hidden = int(sr_cfg.get("hidden", 256)) if isinstance(sr_cfg, dict) else 256
    sr_dropout = float(sr_cfg.get("dropout", 0.0)) if isinstance(sr_cfg, dict) else 0.0

    n_actions = int(env.action_space.n)

    model = MultiBranchActorCriticWithGTAndSR(
        obs_dim=flattener.flat_dim,
        n_actions=n_actions,
        slices=flattener.slices,
        gt=gt,
        gt_spec=spec,
        adj_cache=adj_cache,
        gt_pool=str(model_cfg.get("gt_pool", "mean")),
        gt_proj_dim=_g("gt_proj_dim", 128),
        enc_hidden=_g("enc_hidden", 256 * 4),
        fusion_hidden=_g("fusion_hidden", 256 * 4),
        fusion_blocks=_g("fusion_blocks", 2),
        dropout=_gf("dropout", 0.0),
        policy_temperature=_gf("policy_temperature", 1.0),
        emb_dim_bus=_g("emb_dim_bus", 128 * 4),
        emb_dim_load=_g("emb_dim_load", 128 * 4),
        emb_dim_fcst=_g("emb_dim_fcst", 128 * 4),
        emb_dim_time=_g("emb_dim_time", 32 * 4),
        topo_emb_dim=_g("topo_emb_dim", 32 * 4),
        adj_cache_cuda_size=_g("adj_cache_cuda_size", 16),
        sr_act_emb_dim=sr_act_emb_dim,
        sr_hidden=sr_hidden,
        sr_dropout=sr_dropout,
    ).to(device)

    # load weights
    sd = ckpt.get("model", None)
    if not isinstance(sd, dict):
        raise RuntimeError("Checkpoint has no 'model' state_dict.")
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, meta


@torch.no_grad()
def select_action(
    model: MultiBranchActorCriticWithGTAndSR,
    obs_norm: np.ndarray,
    gt_H: np.ndarray,
    device: torch.device,
    mode: str,
) -> Tuple[int, Dict[str, float]]:
    obs_t = torch.from_numpy(obs_norm[None, :]).to(device=device, dtype=torch.float32)
    gt_t = torch.from_numpy(gt_H[None, :, :]).to(device=device, dtype=torch.float32)
    logits, value, phi = model.forward_with_phi(obs_t, gt_t)
    logits = logits.squeeze(0)
    value = value.squeeze(0)

    dist = torch.distributions.Categorical(logits=logits)
    if mode == "sample":
        a = dist.sample()
    else:
        a = torch.argmax(logits, dim=-1)
    logp = dist.log_prob(a)
    ent = dist.entropy()

    a_i = int(a.item())
    aux = {
        "value": float(value.item()),
        "logp": float(logp.item()),
        "entropy": float(ent.item()),
    }
    return a_i, aux


def run_eval_episodes(
    eval_cfg: EvalConfig,
    env: PowerDispatchEnv,
    flattener: ObsFlattenerV2,
    normalizer: MaskedObsNormalizer,
    model: MultiBranchActorCriticWithGTAndSR,
    policy_kind: str,
) -> Dict[str, Any]:
    device = get_device(eval_cfg.device)
    n_actions = int(env.action_space.n)
    max_steps = eval_cfg.max_steps_per_episode
    if max_steps is None:
        max_steps = int(getattr(env.cfg, "episode_len", 24))

    # step-level buffers
    step_true_total: List[float] = []
    step_true_risk: List[float] = []
    step_pred_total: List[float] = []
    step_pred_risk: List[float] = []
    step_pf_failed: List[int] = []

    # 用于 Critic return_used 评估：按 episode 存每步 reward_used & value
    ep_returns_used_yuan: List[float] = []
    ep_value0_used_yuan: List[float] = []  # episode 起点的 V(s0) -> 元
    ep_total_cost_yuan: List[float] = []
    ep_pf_failed_rate: List[float] = []
    ep_sr_mae_total: List[float] = []

    # 如果 policy_kind 是 random，不用模型
    use_model = (policy_kind != "random")

    for ep in range(eval_cfg.n_eval_episodes):
        obs, _ = env.reset(seed=eval_cfg.seed + ep * 1000, options=None)
        x = flattener.flatten(obs)
        gt_H = np.asarray(obs["gt_H"], dtype=np.float32)

        # 注意：normalizer 来自训练 ckpt；评估时通常只 normalize，不 update（避免漂移）
        obs_norm = normalizer.normalize(x)

        # per-episode buffers
        rewards_used_step: List[float] = []
        values_step: List[float] = []
        costs_total_step: List[float] = []
        pf_failed_step: List[int] = []
        sr_abs_err_total_step: List[float] = []

        # 记录 episode 起点 value 预测（用于 estimator B 的 episode-level 对比）
        if use_model:
            a0, aux0 = select_action(model, obs_norm, gt_H, device, mode=eval_cfg.eval_mode)
            v0 = aux0["value"]
            # 先不真正执行 a0；这里只拿 V(s0)
            # value 是 reward units（训练时用的 reward_used），换算到“元”的“折扣 cost-to-go”：
            # reward_used = -cost_used_yuan * reward_scale  =>  cost_used_yuan = -reward_used / reward_scale
            v0_used_yuan = -float(v0) / float(eval_cfg.reward_scale)
        else:
            v0_used_yuan = float("nan")

        for t in range(max_steps):
            if use_model:
                a, aux = select_action(model, obs_norm, gt_H, device, mode=eval_cfg.eval_mode)
                values_step.append(aux["value"])
            else:
                a = int(np.random.randint(0, n_actions))
                values_step.append(float("nan"))

            # ------ SR 预测（估计器 A）------
            if use_model:
                obs_t = torch.from_numpy(obs_norm[None, :]).to(device=device, dtype=torch.float32)
                gt_t = torch.from_numpy(gt_H[None, :, :]).to(device=device, dtype=torch.float32)
                logits, value, phi = model.forward_with_phi(obs_t, gt_t)
                act_t = torch.tensor([a], device=device, dtype=torch.int64)
                pred_total_norm, pred_risk_norm = model.sr(phi, act_t)
                pred_total_yuan = float(pred_total_norm.item()) * float(eval_cfg.total_scale_yuan)
                pred_risk_yuan = float(pred_risk_norm.item()) * float(eval_cfg.risk_scale_yuan)
            else:
                pred_total_yuan = 0.0
                pred_risk_yuan = 0.0

            # ------ env step（真实）------
            obs2, reward_raw, terminated, truncated, info = env.step(a)
            true_total_yuan, true_risk_yuan, pf_failed = extract_costs_yuan_from_info(info)

            # 训练里 env reward_raw = -total_cost_yuan（单位：元），之后乘 reward_scale
            # 这里构造 reward_env_scaled / reward_used（单位：reward units），用于 estimator B
            reward_env_scaled = float(reward_raw) * float(eval_cfg.reward_scale)  # = -true_total_yuan * reward_scale

            # 训练里 r_hat_rl = -(pred_total_yuan + lambda_risk*pred_risk_yuan)*reward_scale
            r_hat_rl = -(pred_total_yuan + float(eval_cfg.lambda_risk) * pred_risk_yuan) * float(eval_cfg.reward_scale)

            reward_used = reward_env_scaled + float(eval_cfg.alpha_eval) * float(r_hat_rl)

            # 记录
            step_true_total.append(true_total_yuan)
            step_true_risk.append(true_risk_yuan)
            step_pred_total.append(pred_total_yuan)
            step_pred_risk.append(pred_risk_yuan)
            step_pf_failed.append(pf_failed)

            rewards_used_step.append(reward_used)
            costs_total_step.append(true_total_yuan)
            pf_failed_step.append(pf_failed)
            sr_abs_err_total_step.append(abs(pred_total_yuan - true_total_yuan))

            # next
            obs = obs2
            x = flattener.flatten(obs)
            gt_H = np.asarray(obs["gt_H"], dtype=np.float32)
            obs_norm = normalizer.normalize(x)

            if bool(terminated) or bool(truncated):
                break

        # episode-level 汇总
        ep_total_cost = float(np.sum(costs_total_step)) if len(costs_total_step) else float("nan")
        ep_pf_rate = float(np.mean(pf_failed_step)) if len(pf_failed_step) else float("nan")
        ep_sr_mae = float(np.mean(sr_abs_err_total_step)) if len(sr_abs_err_total_step) else float("nan")

        # estimator B：Monte-Carlo 折扣 return_used（reward units） -> 元
        # MC_return_used_reward = sum gamma^t reward_used_t
        mc_return_used_reward = discounted_return(rewards_used_step, gamma=eval_cfg.gamma)
        mc_return_used_yuan = -float(mc_return_used_reward) / float(eval_cfg.reward_scale)

        ep_total_cost_yuan.append(ep_total_cost)
        ep_pf_failed_rate.append(ep_pf_rate)
        ep_sr_mae_total.append(ep_sr_mae)
        ep_returns_used_yuan.append(mc_return_used_yuan)
        ep_value0_used_yuan.append(v0_used_yuan)

    # step-level metrics（估计器 A）
    m_total = regression_metrics(np.array(step_true_total), np.array(step_pred_total))
    m_risk = regression_metrics(np.array(step_true_risk), np.array(step_pred_risk))

    # episode-level metrics（估计器 B）
    m_v = regression_metrics(np.array(ep_returns_used_yuan), np.array(ep_value0_used_yuan))

    out = {
        "policy_kind": policy_kind,
        "step_metrics_total_cost": m_total,
        "step_metrics_risk_cost": m_risk,
        "episode_metrics_value_used_yuan": m_v,
        "episode_total_cost_yuan": ep_total_cost_yuan,
        "episode_pf_failed_rate": ep_pf_failed_rate,
        "episode_sr_mae_total": ep_sr_mae_total,
        "episode_mc_return_used_yuan": ep_returns_used_yuan,
        "episode_v0_used_yuan": ep_value0_used_yuan,
        "step_true_total_cost_yuan": step_true_total,
        "step_pred_total_cost_yuan": step_pred_total,
    }
    return out


def plot_episode_trends(out_dir: str, results: Dict[str, Dict[str, Any]]) -> None:
    """
    results: {"greedy": {...}, "random": {...}} 或只含 "greedy"
    """
    ensure_dir(out_dir)

    def _plot_one(metric_key: str, title: str, ylabel: str, fname: str):
        plt.figure()
        for k, r in results.items():
            y = r.get(metric_key, None)
            if y is None:
                continue
            x = np.arange(1, len(y) + 1)
            plt.plot(x, y, label=k)
        plt.xlabel("Episode")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=160)
        plt.close()

    _plot_one("episode_total_cost_yuan", "Episode Total Cost (Yuan)", "Total Cost (yuan)", "trend_total_cost_yuan.png")
    _plot_one("episode_pf_failed_rate", "Episode PF-Failed Rate", "pf_failed rate", "trend_pf_failed_rate.png")
    _plot_one("episode_sr_mae_total", "SR |TotalCost| MAE per Episode", "MAE (yuan)", "trend_sr_mae_total.png")
    _plot_one("episode_mc_return_used_yuan", "MC Discounted Cost-to-go (Used Reward) in Yuan", "Discounted cost-to-go (yuan)", "trend_mc_return_used_yuan.png")
    _plot_one("episode_v0_used_yuan", "Critic V(s0) Predicted Cost-to-go in Yuan", "Pred cost-to-go (yuan)", "trend_v0_used_yuan.png")


def plot_step_scatter(out_dir: str, tag: str, y_true: List[float], y_pred: List[float]) -> None:
    ensure_dir(out_dir)
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[m]
    y_pred = y_pred[m]
    if y_true.size == 0:
        return
    plt.figure()
    plt.scatter(y_true, y_pred, s=6, alpha=0.6)
    mn = float(min(np.min(y_true), np.min(y_pred)))
    mx = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([mn, mx], [mn, mx], linewidth=1.0)
    plt.xlabel("True total_cost_yuan")
    plt.ylabel("Pred total_cost_yuan (SR)")
    plt.title(f"Step Scatter: SR total cost (yuan) [{tag}]")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"scatter_sr_total_cost_{tag}.png"), dpi=160)
    plt.close()


def save_json(out_dir: str, name: str, obj: Any) -> None:
    ensure_dir(out_dir)
    path = os.path.join(out_dir, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ============================================================
# main
# ============================================================

if __name__ == "__main__":
    # -------------------------
    # 参数：按你的实际路径修改
    # -------------------------
    E = EvalConfig(
        ckpt_path="./runs_ppo_gt_sr_merged/ppo_gt_sr_merged_ckpt_final.pt",
        gt_ckpt_path="./GTransformer/runs/pretrain_masked_node_only_v1/ckpt_best.pt",
        device="cuda",
        seed=0,

        n_eval_episodes=20,
        eval_mode="greedy",               # "greedy" 或 "sample"
        compare_random_baseline=True,
        max_steps_per_episode=None,

        reward_scale=1e-6,
        lambda_risk=0.0,
        alpha_eval=0.0,                   # 如果你要评估“训练时 return_used”，可改成训练末 alpha（比如 0.2）
        gamma=0.99,

        total_scale_yuan=1e6,
        risk_scale_yuan=1e5,

        out_dir="./eval_outputs",
        save_step_scatter=True,
    )

    ensure_dir(E.out_dir)
    set_global_seeds(E.seed)
    device = get_device(E.device)

    # -------------------------
    # build env + tools
    # -------------------------
    # time_scale 需要与你训练一致
    TIME_SCALE = 1e-3
    env, flattener, normalizer = build_env_and_tools(time_scale=TIME_SCALE, seed=E.seed)

    # -------------------------
    # build model from ckpt
    # -------------------------
    model, meta = build_model_from_ckpt(E, env, flattener, normalizer)

    # 若你希望 alpha_eval 使用训练末的 alpha（更贴近 critic 训练目标），打开这两行：
    # if isinstance(meta, dict) and "alpha_now" in meta:
    #     E.alpha_eval = float(meta["alpha_now"])

    # -------------------------
    # eval greedy/sample policy
    # -------------------------
    results: Dict[str, Dict[str, Any]] = {}

    print(f"[eval] policy=trained({E.eval_mode}), episodes={E.n_eval_episodes}, alpha_eval={E.alpha_eval}, reward_scale={E.reward_scale}")
    r_trained = run_eval_episodes(E, env, flattener, normalizer, model, policy_kind="trained")
    results["trained"] = r_trained

    # baseline random
    if E.compare_random_baseline:
        print(f"[eval] policy=random, episodes={E.n_eval_episodes}")
        r_rand = run_eval_episodes(E, env, flattener, normalizer, model, policy_kind="random")
        results["random"] = r_rand

    # -------------------------
    # summary print
    # -------------------------
    def _pfx(d: Dict[str, float]) -> str:
        return f"mae={d.get('mae', float('nan')):.3e} rmse={d.get('rmse', float('nan')):.3e} mape={d.get('mape', float('nan')):.3e} r2={d.get('r2', float('nan')):.3f}"

    for k, r in results.items():
        print("=" * 80)
        print(f"[{k}] step SR total_cost_yuan metrics: {_pfx(r['step_metrics_total_cost'])}")
        print(f"[{k}] step SR risk_cost_yuan  metrics: {_pfx(r['step_metrics_risk_cost'])}")
        print(f"[{k}] episode Critic V(s0) vs MC(return_used_yuan): {_pfx(r['episode_metrics_value_used_yuan'])}")
        print(f"[{k}] episode total_cost_yuan mean={np.mean(r['episode_total_cost_yuan']):.6g}")
        print(f"[{k}] episode pf_failed_rate mean={np.mean(r['episode_pf_failed_rate']):.6g}")
        print(f"[{k}] episode SR-MAE(total) mean={np.mean(r['episode_sr_mae_total']):.6g}")

    # -------------------------
    # save json + plots
    # -------------------------
    save_json(E.out_dir, "eval_results.json", results)
    plot_episode_trends(E.out_dir, results)

    if E.save_step_scatter:
        for k, r in results.items():
            plot_step_scatter(E.out_dir, tag=k,
                              y_true=r.get("step_true_total_cost_yuan", []),
                              y_pred=r.get("step_pred_total_cost_yuan", []))

    print(f"[done] outputs saved to: {E.out_dir}")

# In[]
# -*- coding: utf-8 -*-
"""
plot_episode_curves.py

Read episode_metrics.csv from your training run and plot curves vs episode.
No argparse; edit parameters in __main__.
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if window <= 1:
        return x
    window = int(window)
    kernel = np.ones(window, dtype=np.float64) / float(window)
    # pad to keep same length (reflect padding is more stable than zero padding)
    pad = window // 2
    if x.size < 2:
        return x
    xp = np.pad(x, (pad, window - 1 - pad), mode="edge")
    return np.convolve(xp, kernel, mode="valid")


def safe_series(df: pd.DataFrame, key: str):
    return df[key] if key in df.columns else None


def plot_one_curve(ax, episodes, y, label, smooth_window=1, linewidth=1.0, alpha=0.35):
    y = np.asarray(y, dtype=np.float64)
    ax.plot(episodes, y, label=f"{label} (raw)", linewidth=linewidth, alpha=alpha)
    if smooth_window and smooth_window > 1:
        ys = moving_average(y, smooth_window)
        ax.plot(episodes, ys, label=f"{label} (ma{smooth_window})", linewidth=linewidth + 0.8, alpha=0.95)


def main():
    # =======================
    # Parameters (edit here)
    # =======================
    RUN_DIR = "./runs_ppo_gt_sr_merged"
    CSV_NAME = "episode_metrics.csv"
    OUT_DIR = os.path.join(RUN_DIR, "plots")

    # env_id filter:
    #   None  -> use all envs (aggregated by episode index)
    #   int   -> filter a specific env_id (e.g., 0)
    ENV_ID = None

    # smoothing window for moving average (episodes)
    SMOOTH_WINDOW = 25

    # Which curves to plot
    PLOT_TOTAL_COST = True
    PLOT_PF_FAILED = True
    PLOT_RET_USED = True

    # =======================
    # Load
    # =======================
    csv_path = os.path.join(RUN_DIR, CSV_NAME)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # Basic required column
    if "episode" not in df.columns:
        raise ValueError("CSV missing required column: 'episode'")

    # Optional filter by env_id
    if ENV_ID is not None:
        if "env_id" not in df.columns:
            raise ValueError("ENV_ID is set but CSV has no column 'env_id'")
        df = df[df["env_id"] == int(ENV_ID)].copy()

    # Sort by episode (important)
    df = df.sort_values("episode").reset_index(drop=True)

    episodes = df["episode"].to_numpy(dtype=np.int64)

    # Some runs may have duplicated episode indices (multi-env logging or restarts).
    # If duplicates exist, we aggregate by episode with mean.
    if len(np.unique(episodes)) != len(episodes):
        g = df.groupby("episode", as_index=False).mean(numeric_only=True)
        df = g.sort_values("episode").reset_index(drop=True)
        episodes = df["episode"].to_numpy(dtype=np.int64)

    # =======================
    # Plot: total_cost_yuan
    # =======================
    if PLOT_TOTAL_COST:
        s = safe_series(df, "total_cost_yuan")
        if s is None:
            print("[warn] 'total_cost_yuan' not found in CSV; skip total cost plot.")
        else:
            y = s.to_numpy(dtype=np.float64)
            fig, ax = plt.subplots(figsize=(10, 4))
            plot_one_curve(ax, episodes, y, "total_cost_yuan", smooth_window=SMOOTH_WINDOW)
            ax.set_title("Episode Total Cost (yuan)")
            ax.set_xlabel("episode")
            ax.set_ylabel("yuan")
            ax.grid(True, alpha=0.3)
            ax.legend()
            out = os.path.join(OUT_DIR, f"trend_total_cost_yuan_env{ENV_ID if ENV_ID is not None else 'all'}.png")
            fig.tight_layout()
            fig.savefig(out, dpi=200)
            plt.close(fig)
            print(f"[ok] saved: {out}")

    # =======================
    # Plot: pf_failed (if exists)
    # In your CSV you have pf_failed as int per episode end.
    # =======================
    if PLOT_PF_FAILED:
        s = safe_series(df, "pf_failed")
        if s is None:
            print("[warn] 'pf_failed' not found in CSV; skip pf_failed plot.")
        else:
            y = s.to_numpy(dtype=np.float64)
            fig, ax = plt.subplots(figsize=(10, 4))
            plot_one_curve(ax, episodes, y, "pf_failed", smooth_window=SMOOTH_WINDOW)
            ax.set_title("Episode PF Failed")
            ax.set_xlabel("episode")
            ax.set_ylabel("pf_failed (0/1 or rate)")
            ax.grid(True, alpha=0.3)
            ax.legend()
            out = os.path.join(OUT_DIR, f"trend_pf_failed_env{ENV_ID if ENV_ID is not None else 'all'}.png")
            fig.tight_layout()
            fig.savefig(out, dpi=200)
            plt.close(fig)
            print(f"[ok] saved: {out}")

    # =======================
    # Plot: ret_used (if exists)
    # =======================
    if PLOT_RET_USED:
        s = safe_series(df, "ret_used")
        if s is None:
            print("[warn] 'ret_used' not found in CSV; skip ret_used plot.")
        else:
            y = s.to_numpy(dtype=np.float64)
            fig, ax = plt.subplots(figsize=(10, 4))
            plot_one_curve(ax, episodes, y, "ret_used", smooth_window=SMOOTH_WINDOW)
            ax.set_title("Episode Return Used (training reward space)")
            ax.set_xlabel("episode")
            ax.set_ylabel("ret_used")
            ax.grid(True, alpha=0.3)
            ax.legend()
            out = os.path.join(OUT_DIR, f"trend_ret_used_env{ENV_ID if ENV_ID is not None else 'all'}.png")
            fig.tight_layout()
            fig.savefig(out, dpi=200)
            plt.close(fig)
            print(f"[ok] saved: {out}")

    # =======================
    # Optional: plot cost components if present
    # =======================
    # You logged many cost terms; you can extend here similarly:
    # keys = ["import_cost_yuan", "loss_cost_yuan", "v_cost_yuan", "line_cost_yuan", "trafo_cost_yuan", "switch_cost_yuan"]
    # then plot each on same figure.

    print("[done]")


if __name__ == "__main__":
    main()


# %%
