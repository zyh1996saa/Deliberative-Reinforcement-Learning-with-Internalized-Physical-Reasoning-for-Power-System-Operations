# -*- coding: utf-8 -*-
"""
run_trained_ppo_gt_interact_with_lookahead_search_v2.py

Evaluate a trained PPO+GT (Multi-Branch Actor-Critic) policy on PowerDispatchEnv with:
  (1) Baseline: pure forward pass (greedy argmax)
  (2) Look-ahead: physics-guided beam search (receding horizon) using environment mechanistic rollout

Outputs to OUT_DIR (default: ./eval_lookahead/):
  baseline_steps.csv
  search_steps.csv
  action_diff.csv
  summary.json
  plots/*.png

No argparse: edit parameters in __main__.
"""

from __future__ import annotations

import os
import sys
import math
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import OrderedDict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F


from datetime import datetime


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ============================================================
# Optional: robust alias for env dependency
# ============================================================
def _ensure_config746sys_alias() -> None:
    """
    power_dispatch_env_withGT.py imports `config746sys as cfg`.
    If config746sys is absent but new746_system_v0713 exists, alias it.
    """
    try:
        import config746sys  # noqa: F401
        return
    except Exception:
        pass

    try:
        import importlib

        cfg_mod = importlib.import_module("new746_system_v0713")
        sys.modules["config746sys"] = cfg_mod
    except Exception:
        return


_ensure_config746sys_alias()

from power_dispatch_env_withGT import (  # noqa: E402
    PowerDispatchEnv,
    EnvConfig,
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
# observation flatten + normalization (must match training)
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
        M2 = m_a + m_b + (delta ** 2) * self.count * batch_count / tot_count
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
      time_feat: time_scaled + sin + cos [3]
      topology_id [1]
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

        if off != self.flat_dim:
            raise RuntimeError(f"[ObsFlattenerV2] flat_dim mismatch: got {off} expected {self.flat_dim}")

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
# adjacency cache
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
# GT load + probing
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
            out_key = "z" if isinstance(out, dict) and "z" in out else None
            z = _extract_z(out, z_key=out_key)
            if z.ndim != 3 or z.shape[0] != B or z.shape[1] != N:
                continue
            D = int(z.shape[2])
            pick_key = None
            if isinstance(out, dict):
                for k in ("z", "emb", "node_emb", "node_embeddings", "h", "hidden"):
                    if k in out and torch.is_tensor(out[k]) and out[k].shape == z.shape:
                        pick_key = k
                        break
            return GTForwardSpec(use_return_embeddings=use_ret, use_batched_adj=use_batched, z_key=pick_key, out_dim=D)
        except Exception:
            continue
    raise RuntimeError("Failed to probe a valid GT forward signature. Verify GTransformer forward(H,A,...) interface.")


def gt_forward_with_spec(gt: nn.Module, spec: GTForwardSpec, H: torch.Tensor, A2: torch.Tensor) -> torch.Tensor:
    A = A2.unsqueeze(0).expand(H.shape[0], -1, -1) if spec.use_batched_adj else A2
    out = gt(H, A, return_embeddings=True) if spec.use_return_embeddings else gt(H, A)
    z = _extract_z(out, spec.z_key)
    if z.ndim != 3 or z.shape[0] != H.shape[0] or z.shape[1] != H.shape[1]:
        raise RuntimeError(f"GT embedding shape mismatch: got {tuple(z.shape)} expected ({H.shape[0]},{H.shape[1]},D)")
    return z


# ============================================================
# Actor-Critic model
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
# checkpoint load
# ============================================================
def load_checkpoint_raw(path: str, device: torch.device) -> Dict[str, Any]:
    return _torch_load_trusted(path, map_location=device)


def apply_checkpoint(ckpt: Dict[str, Any], model: nn.Module, normalizer: MaskedObsNormalizer) -> Dict[str, Any]:
    sd = ckpt.get("model", None)
    if not isinstance(sd, dict):
        raise ValueError("checkpoint missing 'model' state_dict")
    model.load_state_dict(sd, strict=False)

    nd = ckpt.get("normalizer", None)
    if isinstance(nd, dict):
        normalizer.load_state_dict(nd)

    meta = {
        "step": int(ckpt.get("step", 0)),
        "update": int(ckpt.get("update", 0)),
        "episode_count": int(ckpt.get("episode_count", 0)),
        "hparams": dict(ckpt.get("hparams", {})),
        "extra": dict(ckpt.get("extra", {})),
    }
    return meta


# ============================================================
# policy forward helper
# ============================================================
@torch.no_grad()
def policy_forward(
    model: nn.Module,
    flattener: ObsFlattenerV2,
    normalizer: MaskedObsNormalizer,
    obs: Dict[str, np.ndarray],
    device: torch.device,
) -> Dict[str, Any]:
    obs_flat = flattener.flatten(obs)
    obs_norm = normalizer.normalize(obs_flat)
    gt_H = np.asarray(obs.get("gt_H", None), dtype=np.float32)

    obs_t = torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)
    gt_t = torch.tensor(gt_H, dtype=torch.float32, device=device).unsqueeze(0)

    logits, value = model(obs_t, gt_t)
    probs = torch.softmax(logits, dim=-1).squeeze(0)

    return {
        "probs": probs.detach().cpu().numpy().astype(np.float32),
        "value": float(value.squeeze(0).item()),
    }


def topk_actions(probs: np.ndarray, k: int) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float32).reshape(-1)
    k = int(min(max(1, k), probs.size))
    idx = np.argpartition(-probs, kth=k - 1)[:k]
    idx = idx[np.argsort(-probs[idx])]
    return idx.astype(int)


# ============================================================
# look-ahead beam search planner
# ============================================================
@dataclass
class SearchNode:
    obs: Dict[str, np.ndarray]
    t: int
    prev_action: int
    step_in_episode: int
    score: float
    path: List[int]



class LookaheadBeamSearchPlanner:
    """
    Physics-guided beam search with receding-horizon rollout.

    Key performance note:
      - Pandapower `runpp` is expensive. With (H=3, K=32, B=8) a naive rollout may
        require O(32 + 8*32*2)=544 power-flows per *decision step* in the worst case.
      - This planner therefore caches PF results by (t, action) and reuses them across
        different prev_action values, because only switch_cost depends on prev_action.
    """

    def __init__(
        self,
        *,
        env: PowerDispatchEnv,
        model: nn.Module,
        flattener: ObsFlattenerV2,
        normalizer: MaskedObsNormalizer,
        device: torch.device,
        horizon: int,
        beam_width: int,
        topk: int,
        gamma: float,
        reject_pf_failed: bool = True,
        plan_log: bool = True,
        log_depth_every: int = 1,
        log_expand_every: int = 0,
    ):
        self.env = env
        self.model = model
        self.flattener = flattener
        self.normalizer = normalizer
        self.device = device

        self.horizon = int(max(1, horizon))
        self.beam_width = int(max(1, beam_width))
        self.topk = int(max(1, topk))
        self.gamma = float(gamma)
        self.reject_pf_failed = bool(reject_pf_failed)

        # logging controls
        self.plan_log = bool(plan_log)
        self.log_depth_every = int(max(1, log_depth_every))
        self.log_expand_every = int(max(0, log_expand_every))

        # caches
        self._pi_cache: Dict[Tuple[int, int], Dict[str, Any]] = {}   # (t, topo) -> policy forward
        self._pf_cache: Dict[Tuple[int, int], Tuple[Dict[str, np.ndarray], Dict[str, Any]]] = {}  # (t, action) -> (obs, info_phy)
        self._sim_cache: Dict[Tuple[int, int, int], Tuple[Dict[str, np.ndarray], Dict[str, Any]]] = {}  # (t, prev_action, action) -> (obs, info_full)

        # per-plan stats (reset every plan call)
        self._stats: Dict[str, int] = {}

    def clear_caches(self) -> None:
        self._pi_cache.clear()
        self._pf_cache.clear()
        self._sim_cache.clear()

    def _policy_cached(self, obs: Dict[str, np.ndarray]) -> Dict[str, Any]:
        t = int(np.asarray(obs["time_index"]).reshape(-1)[0])
        topo = int(np.asarray(obs["topology_id"]).reshape(-1)[0])
        key = (t, topo)
        if key in self._pi_cache:
            return self._pi_cache[key]
        out = policy_forward(self.model, self.flattener, self.normalizer, obs, self.device)
        self._pi_cache[key] = out
        return out

    def _pf_cached(self, *, t: int, action: int) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Cache PF-dependent results by (t, action). These are independent of prev_action.
        We compute them once by calling env._solve_and_build_obs(prev_action=None),
        which makes switch_cost=0 so reward excludes switch term.
        """
        key = (int(t), int(action))
        if key in self._pf_cache:
            self._stats["pf_hit"] += 1
            return self._pf_cache[key]
        self._stats["pf_miss"] += 1

        net_line_repr = set_fc_state_with_acts(self.env.feeder_cluster, self.env.base_net, [int(action)])
        obs_phy, info_phy = self.env._solve_and_build_obs(  # type: ignore[attr-defined]
            net_line_repr,
            int(t),
            prev_action=None,
            action=int(action),
        )
        if not isinstance(info_phy, dict):
            info_phy = {}
        self._pf_cache[key] = (obs_phy, info_phy)
        return obs_phy, info_phy

    def _simulate_one_step(
        self,
        *,
        t: int,
        prev_action: int,
        step_in_episode: int,
        action: int,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], bool]:
        """
        Return: (next_obs, info, truncated)
        - truncated here is episode_len based; terminated is not simulated.
        """
        t_next = int(t) + 1
        step_next = int(step_in_episode) + 1
        truncated = step_next >= int(self.env.cfg.episode_len)

        key = (t_next, int(prev_action), int(action))
        if key in self._sim_cache:
            self._stats["sim_hit"] += 1
            next_obs, info = self._sim_cache[key]
            return next_obs, info, truncated

        self._stats["sim_miss"] += 1

        # 1) PF-dependent terms (independent of prev_action)
        next_obs, info_phy = self._pf_cached(t=t_next, action=int(action))

        # 2) Switch cost depends on prev_action
        sw_cost, sw_stats = self.env._calc_switch_cost(int(prev_action), int(action))  # type: ignore[attr-defined]

        # 3) Recompose reward: reward_phy already excludes switch_term
        w_sw = float(getattr(self.env.cfg, "w_switch", 0.0))
        reward_phy = float(info_phy.get("reward", 0.0))
        reward = float(reward_phy - w_sw * float(sw_cost))
        switch_term = float(w_sw * float(sw_cost))

        info_full = dict(info_phy)
        info_full.update(sw_stats if isinstance(sw_stats, dict) else {})
        info_full.update(
            {
                "t": int(t_next),
                "action": int(action),
                "prev_action": int(prev_action),
                "switch_cost": float(sw_cost),
                "switch_term": float(switch_term),
                "reward": float(reward),
            }
        )

        self._sim_cache[key] = (next_obs, info_full)
        return next_obs, info_full, truncated

    def plan(
        self,
        *,
        obs: Dict[str, np.ndarray],
        t: int,
        prev_action: int,
        step_in_episode: int,
    ) -> Tuple[int, Dict[str, Any]]:
        import time

        # reset per-plan stats
        self._stats = {"pf_hit": 0, "pf_miss": 0, "sim_hit": 0, "sim_miss": 0, "expanded": 0}

        t0 = time.time()
        root = self._policy_cached(obs)
        root_probs = root["probs"]
        root_topk = topk_actions(root_probs, self.topk)

        beam: List[SearchNode] = [
            SearchNode(
                obs=obs,
                t=int(t),
                prev_action=int(prev_action),
                step_in_episode=int(step_in_episode),
                score=0.0,
                path=[],
            )
        ]
        best_leaf: Optional[SearchNode] = None

        for depth in range(self.horizon):
            new_nodes: List[SearchNode] = []

            for node_i, node in enumerate(beam):
                if node.step_in_episode >= int(self.env.cfg.episode_len):
                    if best_leaf is None or node.score > best_leaf.score:
                        best_leaf = node
                    continue

                if depth == 0:
                    cand_actions = root_topk
                else:
                    pi = self._policy_cached(node.obs)
                    cand_actions = topk_actions(pi["probs"], self.topk)

                for a in cand_actions:
                    self._stats["expanded"] += 1

                    next_obs, info, truncated = self._simulate_one_step(
                        t=node.t,
                        prev_action=node.prev_action,
                        step_in_episode=node.step_in_episode,
                        action=int(a),
                    )
                    if self.reject_pf_failed and bool(info.get("pf_failed", False)):
                        continue

                    r = float(info.get("reward", 0.0))
                    score = float(node.score + (self.gamma ** depth) * r)

                    child = SearchNode(
                        obs=next_obs,
                        t=int(node.t) + 1,
                        prev_action=int(a),
                        step_in_episode=int(node.step_in_episode) + 1,
                        score=score,
                        path=node.path + [int(a)],
                    )
                    new_nodes.append(child)

                    if truncated:
                        if best_leaf is None or child.score > best_leaf.score:
                            best_leaf = child

                    if self.plan_log and self.log_expand_every > 0 and (self._stats["expanded"] % self.log_expand_every == 0):
                        elapsed = time.time() - t0
                        log(
                            f"[planner] depth={depth+1}/{self.horizon} expanded={self._stats['expanded']} "
                            f"pf(miss/hit)={self._stats['pf_miss']}/{self._stats['pf_hit']} "
                            f"sim(miss/hit)={self._stats['sim_miss']}/{self._stats['sim_hit']} "
                            f"elapsed={elapsed:.1f}s"
                        )

            if not new_nodes:
                break

            new_nodes.sort(key=lambda n: n.score, reverse=True)
            beam = new_nodes[: self.beam_width]

            if self.plan_log and ((depth + 1) % self.log_depth_every == 0):
                elapsed = time.time() - t0
                log(
                    f"[planner] depth={depth+1}/{self.horizon} done | kept={len(beam)} "
                    f"candidates_generated={len(new_nodes)} expanded={self._stats['expanded']} "
                    f"pf(miss/hit)={self._stats['pf_miss']}/{self._stats['pf_hit']} "
                    f"sim(miss/hit)={self._stats['sim_miss']}/{self._stats['sim_hit']} "
                    f"elapsed={elapsed:.1f}s"
                )

        if beam:
            beam_best = max(beam, key=lambda n: n.score)
            if best_leaf is None or beam_best.score > best_leaf.score:
                best_leaf = beam_best

        if best_leaf is None or len(best_leaf.path) == 0:
            a = int(np.argmax(root_probs))
            return a, {
                "fallback": True,
                "best_path": [a],
                "root_topk": root_topk.tolist(),
                "plan_stats": dict(self._stats),
                "plan_time_s": float(time.time() - t0),
            }

        a0 = int(best_leaf.path[0])
        return a0, {
            "fallback": False,
            "best_path": best_leaf.path,
            "best_score": float(best_leaf.score),
            "root_topk": root_topk.tolist(),
            "plan_stats": dict(self._stats),
            "plan_time_s": float(time.time() - t0),
        }


# ============================================================
# evaluation utilities
# ============================================================
def run_episode(
    *,
    env: PowerDispatchEnv,
    policy_fn,
    episode_seed: int,
    max_steps: int,
    policy_name: str,
    step_log_every: int = 1,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run one episode and collect step-level diagnostics.

    Important:
      - Lookahead planning may take substantial time; therefore we log step-level progress
        (which step, which t, planning time, env step time).
    """
    import time

    obs, _ = env.reset(seed=int(episode_seed), options=None)
    rows: List[Dict[str, Any]] = []
    ep_return = 0.0
    pf_failed = 0

    t_ep0 = time.time()

    for step in range(int(max_steps)):
        t_obs = int(np.asarray(obs["time_index"]).reshape(-1)[0])
        topo = int(np.asarray(obs["topology_id"]).reshape(-1)[0])

        # --- plan action (can be slow) ---
        t_plan0 = time.time()
        action, dbg = policy_fn(env=env, obs=obs)
        t_plan = time.time() - t_plan0

        # --- env transition ---
        t_step0 = time.time()
        next_obs, reward, terminated, truncated, info = env.step(int(action))
        t_env = time.time() - t_step0

        reward = float(reward)
        ep_return += reward
        pf_failed += int(bool(info.get("pf_failed", False)))

        if step_log_every > 0 and ((step + 1) % int(step_log_every) == 0):
            log(
                f"[{policy_name}] step={step+1:02d}/{max_steps} t={t_obs} topo={topo} "
                f"action={int(action)} r={reward:.4f} cum={ep_return:.4f} "
                f"plan={t_plan:.2f}s env={t_env:.2f}s"
            )

        rows.append(
            {
                "episode_seed": int(episode_seed),
                "step": int(step + 1),
                "t_obs": int(t_obs),
                "topo_obs": int(topo),
                "action": int(action),
                "reward": float(reward),
                "cum_return": float(ep_return),
                "pf_failed": int(bool(info.get("pf_failed", False))),
                "loss_mw": float(info.get("loss_mw", np.nan)),
                "v_viol": float(info.get("v_viol", np.nan)),
                "line_viol": float(info.get("line_viol", np.nan)),
                "trafo_bal": float(info.get("trafo_bal", np.nan)),
                "switch_cost": float(info.get("switch_cost", np.nan)),
                "plan_time_s": float(t_plan),
                "env_time_s": float(t_env),
                "dbg": json.dumps(dbg, ensure_ascii=False),
            }
        )

        obs = next_obs
        if bool(terminated) or bool(truncated):
            break

    df = pd.DataFrame(rows)
    wall = time.time() - t_ep0
    summary = {
        "episode_seed": int(episode_seed),
        "steps": int(len(df)),
        "return": float(ep_return),
        "pf_failed_rate": float(pf_failed / max(1, len(df))),
        "mean_v_viol": float(df["v_viol"].mean()) if len(df) else float("nan"),
        "mean_line_viol": float(df["line_viol"].mean()) if len(df) else float("nan"),
        "sum_switch_cost": float(df["switch_cost"].sum()) if len(df) else float("nan"),
        "wall_s": float(wall),
        "avg_step_s": float(wall / max(1, len(df))),
        "avg_plan_s": float(df["plan_time_s"].mean()) if len(df) else float("nan"),
    }
    return df, summary


def build_action_diff(baseline_df: pd.DataFrame, search_df: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["episode_seed", "step"]
    b = baseline_df[key_cols + ["t_obs", "action", "reward", "cum_return", "v_viol", "line_viol", "switch_cost", "pf_failed"]].copy()
    s = search_df[key_cols + ["action", "reward", "cum_return", "v_viol", "line_viol", "switch_cost", "pf_failed"]].copy()
    m = b.merge(s, on=key_cols, how="outer", suffixes=("_base", "_search"))
    m["action_diff"] = (m["action_base"] != m["action_search"]).astype(int)
    m["reward_gap"] = m["reward_search"] - m["reward_base"]
    m["cum_return_gap"] = m["cum_return_search"] - m["cum_return_base"]
    m["v_viol_gap"] = m["v_viol_search"] - m["v_viol_base"]
    m["line_viol_gap"] = m["line_viol_search"] - m["line_viol_base"]
    m["switch_cost_gap"] = m["switch_cost_search"] - m["switch_cost_base"]
    m["pf_failed_gap"] = m["pf_failed_search"] - m["pf_failed_base"]
    return m


def make_plots(out_dir: str, baseline_df: pd.DataFrame, search_df: pd.DataFrame, baseline_summaries: List[Dict[str, Any]], search_summaries: List[Dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    seeds = sorted(set(baseline_df["episode_seed"].unique().tolist()) | set(search_df["episode_seed"].unique().tolist()))
    for i, seed in enumerate(seeds, start=1):
        b = baseline_df[baseline_df["episode_seed"] == seed].sort_values("step")
        s = search_df[search_df["episode_seed"] == seed].sort_values("step")
        if len(b) == 0 or len(s) == 0:
            continue

        plt.figure()
        plt.plot(b["step"].to_numpy(), b["cum_return"].to_numpy(), label="baseline")
        plt.plot(s["step"].to_numpy(), s["cum_return"].to_numpy(), label="lookahead")
        plt.xlabel("Step")
        plt.ylabel("Cumulative return")
        plt.title(f"Episode {i:04d} (seed={seed})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"ep{i:04d}_returns.png"), dpi=160)
        plt.close()

        plt.figure()
        plt.plot(b["step"].to_numpy(), b["v_viol"].to_numpy(), label="v_viol baseline")
        plt.plot(s["step"].to_numpy(), s["v_viol"].to_numpy(), label="v_viol lookahead")
        plt.plot(b["step"].to_numpy(), b["line_viol"].to_numpy(), label="line_viol baseline")
        plt.plot(s["step"].to_numpy(), s["line_viol"].to_numpy(), label="line_viol lookahead")
        plt.xlabel("Step")
        plt.ylabel("Violation (mean)")
        plt.title(f"Episode {i:04d} constraints (seed={seed})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"ep{i:04d}_constraints.png"), dpi=160)
        plt.close()

        plt.figure()
        plt.plot(b["step"].to_numpy(), b["action"].to_numpy(), label="action baseline")
        plt.plot(s["step"].to_numpy(), s["action"].to_numpy(), label="action lookahead")
        for st in range(1, min(len(b), len(s)) + 1):
            ab = int(b[b["step"] == st]["action"].iloc[0])
            as_ = int(s[s["step"] == st]["action"].iloc[0])
            if ab != as_:
                plt.axvline(st, linewidth=0.5, alpha=0.4)
        plt.xlabel("Step")
        plt.ylabel("Action id")
        plt.title(f"Episode {i:04d} actions (seed={seed})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"ep{i:04d}_actions.png"), dpi=160)
        plt.close()

    b_ret = np.array([d["return"] for d in baseline_summaries], dtype=float)
    s_ret = np.array([d["return"] for d in search_summaries], dtype=float)

    plt.figure()
    plt.boxplot([b_ret, s_ret], labels=["baseline", "lookahead"])
    plt.ylabel("Episode return")
    plt.title("Overall return distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "overall_return_box.png"), dpi=160)
    plt.close()


# ============================================================
# entry (edit params here)
# ============================================================
if __name__ == "__main__":
    # ------------------
    # Paths
    # ------------------
    RL_CKPT_PATH = "./runs_ppo_mp_multienc_withGT_fast_260107/ppo_gt_ckpt_final.pt"
    GT_CKPT_PATH = ""  # if empty, try checkpoint["extra"]["gt_ckpt_path"]
    OUT_DIR = "./eval_lookahead/"

    # ------------------
    # Runtime
    # ------------------
    DEVICE = "cuda"  # "cuda"/"cpu"/"auto"
    ALLOW_TF32 = True
    BASE_SEED = 123
    N_EPISODES = 10

    # ------------------
    # Look-ahead search (paper-aligned)
    # ------------------
    LOOKAHEAD_H = 3
    TOPK_K = 32
    BEAM_WIDTH = 8
    GAMMA_FALLBACK = 0.99
    REJECT_PF_FAILED = True

    # ------------------
    # Model hyperparams (must match training)
    # ------------------
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

    os.makedirs(OUT_DIR, exist_ok=True)
    set_global_seeds(BASE_SEED)
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

    env_cfg = EnvConfig(seed=BASE_SEED)
    env_base = PowerDispatchEnv(env_cfg)
    env_search = PowerDispatchEnv(env_cfg)

    n_actions = int(env_base.action_space.n)
    time_period = int(getattr(env_cfg, "episode_len", 24))
    flattener = ObsFlattenerV2(env_base.observation_space, n_actions=n_actions, time_scale=1e-3, time_period=time_period)

    obs0, _ = env_base.reset(seed=BASE_SEED, options=None)
    gt_H0 = np.asarray(obs0.get("gt_H", None), dtype=np.float32)
    if gt_H0.ndim != 2:
        raise RuntimeError(f"env obs['gt_H'] must be (N,D); got {gt_H0.shape}")

    mask = np.ones((flattener.flat_dim,), dtype=bool)
    sl_time = flattener.slices["time_feat"]
    mask[sl_time.start + 1 : sl_time.stop] = False
    mask[flattener.slices["topology_id"]] = False
    normalizer_base = MaskedObsNormalizer(flattener.flat_dim, mask=mask, clip=10.0)
    normalizer_search = MaskedObsNormalizer(flattener.flat_dim, mask=mask, clip=10.0)

    ckpt = load_checkpoint_raw(RL_CKPT_PATH, device=device)
    extra = dict(ckpt.get("extra", {})) if isinstance(ckpt.get("extra", {}), dict) else {}
    if not GT_CKPT_PATH:
        GT_CKPT_PATH = str(extra.get("gt_ckpt_path", "") or "")
    if not GT_CKPT_PATH:
        raise ValueError("GT_CKPT_PATH is empty. Set it explicitly or ensure RL checkpoint contains extra['gt_ckpt_path'].")

    hparams = dict(ckpt.get("hparams", {})) if isinstance(ckpt.get("hparams", {}), dict) else {}
    gamma = float(hparams.get("gamma", GAMMA_FALLBACK))

    gt, gt_cfg, gt_raw = load_gtransformer_checkpoint(GT_CKPT_PATH, device=device)

    r_switch = float(getattr(env_cfg, "r_switch", float("nan")))
    x_switch = float(getattr(env_cfg, "x_switch", float("nan")))
    if not (np.isfinite(r_switch) and np.isfinite(x_switch)):
        import config746sys  # type: ignore
        r_switch = float(getattr(config746sys, "r_switch"))
        x_switch = float(getattr(config746sys, "x_switch"))

    adj_cache = TopologyAdjacencyCache(
        feeder_cluster=env_base.feeder_cluster,
        base_net=env_base.base_net,
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
    meta = apply_checkpoint(ckpt, model, normalizer_base)
    normalizer_search.load_state_dict(normalizer_base.state_dict())
    model.eval()

    max_steps = int(env_cfg.episode_len)

    def baseline_policy(*, env: PowerDispatchEnv, obs: Dict[str, np.ndarray]) -> Tuple[int, Dict[str, Any]]:
        out = policy_forward(model, flattener, normalizer_base, obs, device)
        probs = out["probs"]
        a = int(np.argmax(probs))
        return a, {"policy": "baseline_argmax", "value": float(out["value"]), "top8": topk_actions(probs, min(8, probs.size)).tolist()}

    planner = LookaheadBeamSearchPlanner(
        env=env_search,
        model=model,
        flattener=flattener,
        normalizer=normalizer_search,
        device=device,
        horizon=LOOKAHEAD_H,
        beam_width=BEAM_WIDTH,
        topk=TOPK_K,
        gamma=gamma,
        reject_pf_failed=REJECT_PF_FAILED,
        plan_log=True,
        log_depth_every=1,
        log_expand_every=0,
    )

    def lookahead_policy(*, env: PowerDispatchEnv, obs: Dict[str, np.ndarray]) -> Tuple[int, Dict[str, Any]]:
        t = int(getattr(env, "_t"))
        prev_action = int(getattr(env, "_prev_action"))
        step_in_episode = int(getattr(env, "_step_in_episode"))
        a, dbg = planner.plan(obs=obs, t=t, prev_action=prev_action, step_in_episode=step_in_episode)
        dbg.update({"policy": "lookahead_beam", "H": LOOKAHEAD_H, "K": TOPK_K, "B": BEAM_WIDTH, "gamma": gamma})
        return a, dbg

    baseline_dfs, search_dfs = [], []
    baseline_summaries, search_summaries = [], []

    for ep in range(int(N_EPISODES)):
        seed = int(BASE_SEED + ep)
        log(f"---- Episode {ep+1:02d}/{N_EPISODES} | seed={seed} ----")

        log("Running baseline (forward argmax) ...")
        df_b, sum_b = run_episode(
            env=env_base,
            policy_fn=baseline_policy,
            episode_seed=seed,
            max_steps=max_steps,
            policy_name="baseline",
            step_log_every=1,
        )
        baseline_dfs.append(df_b)
        baseline_summaries.append(sum_b)
        log(
            f"Baseline done. steps={sum_b['steps']} return={sum_b['return']:.6f} "
            f"pf_failed_rate={sum_b['pf_failed_rate']:.4f} wall={sum_b['wall_s']:.2f}s"
        )

        planner.clear_caches()
        log("Running lookahead (beam search + physics rollout) ...")
        df_s, sum_s = run_episode(
            env=env_search,
            policy_fn=lookahead_policy,
            episode_seed=seed,
            max_steps=max_steps,
            policy_name="lookahead",
            step_log_every=1,
        )
        search_dfs.append(df_s)
        search_summaries.append(sum_s)
        log(
            f"Lookahead done. steps={sum_s['steps']} return={sum_s['return']:.6f} "
            f"pf_failed_rate={sum_s['pf_failed_rate']:.4f} wall={sum_s['wall_s']:.2f}s"
        )

        log(f"[episode {ep+1:02d}] base_ret={sum_b['return']:.6f} lookahead_ret={sum_s['return']:.6f}")

    baseline_df = pd.concat(baseline_dfs, ignore_index=True)
    search_df = pd.concat(search_dfs, ignore_index=True)

    baseline_df.to_csv(os.path.join(OUT_DIR, "baseline_steps.csv"), index=False, encoding="utf-8-sig")
    search_df.to_csv(os.path.join(OUT_DIR, "search_steps.csv"), index=False, encoding="utf-8-sig")

    diff_df = build_action_diff(baseline_df, search_df)
    diff_df.to_csv(os.path.join(OUT_DIR, "action_diff.csv"), index=False, encoding="utf-8-sig")

    summary = {
        "meta": {
            "rl_ckpt": RL_CKPT_PATH,
            "gt_ckpt": GT_CKPT_PATH,
            "device": str(device),
            "gamma": float(gamma),
            "episodes": int(N_EPISODES),
            "steps_per_episode": int(max_steps),
            "lookahead": {"H": LOOKAHEAD_H, "K": TOPK_K, "B": BEAM_WIDTH, "reject_pf_failed": REJECT_PF_FAILED},
        },
        "baseline": baseline_summaries,
        "lookahead": search_summaries,
        "diff_stats": {
            "action_diff_rate": float(diff_df["action_diff"].mean()) if len(diff_df) else float("nan"),
            "mean_reward_gap": float(diff_df["reward_gap"].mean()) if len(diff_df) else float("nan"),
            "mean_final_return_gap": float(diff_df.groupby("episode_seed")["cum_return_gap"].last().mean()) if len(diff_df) else float("nan"),
        },
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    make_plots(OUT_DIR, baseline_df, search_df, baseline_summaries, search_summaries)

    log(f"[done] results written to: {OUT_DIR}")
