# -*- coding: utf-8 -*-
"""
train_ppo_power_dispatch_multiproc_multiencoder_withGT_sr_merged.py

在你给出的“merged 修复版”基础上，新增“尽可能详细”的执行日志与耗时统计，目标包括：
- 明确代码执行到哪一步（阶段/子阶段/循环进度）
- 每一步（rollout step / optimizer batch / reset burst / save）的关键结果摘要
- 打印当前时间、阶段耗时、累计总用时、ETA 等
- 同步落盘：stdout + 追加写入 run_log.txt（不覆盖旧日志）

无 argparse；所有参数仍在 __main__ 中直接编辑。

注意：
- 默认会输出非常多日志（尤其 n_envs=64、steps_per_env=32 时），如需减少打印量：
  直接在 __main__ 修改 cfg_train.log_* 相关参数即可。
"""

from __future__ import annotations

import os
import time
import math
import json
import random
import traceback
import importlib
import multiprocessing as mp
from multiprocessing.connection import wait as mp_wait
from multiprocessing import shared_memory
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from collections import OrderedDict
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

# Env imports must remain at top-level for worker spawn
from power_dispatch_env_withGT_dimrisk import (
    PowerDispatchEnv,
    EnvConfig,
    TimeseriesSchema,
    built_ppnet_for_pfcal,
    set_fc_state_with_acts,
)

import sys
import platform
from datetime import datetime

# ============================================================
# utils: seeds / device / pretty time / notebook detection
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

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _in_notebook() -> bool:
    try:
        from IPython import get_ipython  # type: ignore
        ip = get_ipython()
        if ip is None:
            return False
        return ip.__class__.__name__.startswith("ZMQ")
    except Exception:
        return False

def eta_str(elapsed: float, frac_done: float) -> str:
    frac_done = float(np.clip(frac_done, 1e-6, 1.0))
    total = elapsed / frac_done
    remain = max(0.0, total - elapsed)
    return f"elapsed={fmt_sec(elapsed)} remain={fmt_sec(remain)}"

# ============================================================
# verbose logger (stdout + file append), with timers
# ============================================================

class VerboseLogger:
    """
    轻量 logger：
    - level: 0=ERROR, 1=WARN, 2=INFO, 3=DEBUG, 4=TRACE
    - 同时写 stdout 与文件（追加，不覆盖）
    """
    LEVELS = {"ERROR": 0, "WARN": 1, "INFO": 2, "DEBUG": 3, "TRACE": 4}

    def __init__(self, log_path: str, level: int = 2, flush: bool = True):
        self.log_path = str(log_path)
        self.level = int(level)
        self.flush = bool(flush)
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        self._t0 = time.perf_counter()

    def _write(self, s: str) -> None:
        print(s)
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(s + "\n")
                if self.flush:
                    f.flush()
        except Exception:
            pass

    def log(self, lvl_name: str, msg: str) -> None:
        lvl = int(self.LEVELS.get(lvl_name.upper(), 2))
        if lvl > self.level:
            return
        dt = time.perf_counter() - self._t0
        line = f"[{now_str()}][+{fmt_sec(dt)}][{lvl_name.upper():5s}] {msg}"
        self._write(line)

    def error(self, msg: str) -> None:
        self.log("ERROR", msg)

    def warn(self, msg: str) -> None:
        self.log("WARN", msg)

    def info(self, msg: str) -> None:
        self.log("INFO", msg)

    def debug(self, msg: str) -> None:
        self.log("DEBUG", msg)

    def trace(self, msg: str) -> None:
        self.log("TRACE", msg)

@contextmanager
def stage_timer(logger: VerboseLogger, name: str, level: str = "INFO", extra: str = ""):
    t0 = time.perf_counter()
    logger.log(level, f"[stage:start] {name}" + (f" | {extra}" if extra else ""))
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        logger.log(level, f"[stage:end]   {name} | dt={fmt_sec(dt)}")

class ImportLogger:
    """
    记录当前进程导入的模块（sys.modules），并按“新增模块”增量写入 txt。
    记录字段：module_name, file, version（若可得）
    """
    def __init__(self, path: str):
        self.path = str(path)
        self.seen = set()  # 已记录过的模块名

    def _get_mod_file(self, m) -> str:
        try:
            f = getattr(m, "__file__", None)
            return str(f) if f else ""
        except Exception:
            return ""

    def _get_mod_version(self, m) -> str:
        try:
            v = getattr(m, "__version__", None)
            return str(v) if v is not None else ""
        except Exception:
            return ""

    def dump(self, *, tag: str, only_new: bool = True) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        modules = sys.modules

        lines = []
        lines.append("=" * 80)
        lines.append(f"[{now}] tag={tag}")
        lines.append(f"python={sys.version.replace(os.linesep, ' ')}")
        lines.append(f"platform={platform.platform()}")
        try:
            import numpy as _np
            lines.append(f"numpy_version={getattr(_np, '__version__', '')}")
        except Exception:
            pass
        try:
            import torch as _torch
            lines.append(f"torch_version={getattr(_torch, '__version__', '')}")
        except Exception:
            pass
        lines.append("-" * 80)

        names = sorted(list(modules.keys()))
        new_count = 0
        for name in names:
            if only_new and (name in self.seen):
                continue
            m = modules.get(name, None)
            if m is None:
                continue
            f = self._get_mod_file(m)
            v = self._get_mod_version(m)
            lines.append(f"{name}\t{v}\t{f}")
            self.seen.add(name)
            new_count += 1

        lines.append(f"[summary] new_modules_written={new_count} total_seen={len(self.seen)}")
        lines.append("")

        with open(self.path, "a", encoding="utf-8") as fp:
            fp.write("\n".join(lines))

# ============================================================
# IO helpers: atomic json + append csv (do not overwrite)
# ============================================================

def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass
    os.replace(tmp, path)

def append_row_csv(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    import csv
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)

# ============================================================
# observation: dict -> flat + (masked) normalization
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
    gt_H is NOT included in flat; passed separately.
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
# SR cost scaler (关键修复：把 1e6 量级目标变为归一化目标)
# ============================================================

class RobustScaleEMA:
    """
    用分位数 + EMA 估计尺度，避免 SR 目标巨数导致学习退化。
    - 每次 update 用当前 batch 的 q 分位数更新
    - scale = EMA(scale, q_value)
    """
    def __init__(self, init_scale: float, q: float = 0.95, ema: float = 0.98, min_scale: float = 1.0):
        self.scale = float(max(init_scale, min_scale))
        self.q = float(np.clip(q, 0.5, 0.999))
        self.ema = float(np.clip(ema, 0.0, 0.9999))
        self.min_scale = float(min_scale)
        self._inited = True

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size == 0:
            return
        v = float(np.quantile(np.abs(x), self.q))
        v = max(v, self.min_scale)
        self.scale = self.ema * self.scale + (1.0 - self.ema) * v

    def get(self) -> float:
        return float(max(self.scale, self.min_scale))

# ============================================================
# topology adjacency cache (on-demand, LRU)
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

def _load_switch_params() -> Tuple[float, float]:
    m = importlib.import_module("config746sys")
    r_switch = float(getattr(m, "r_switch"))
    x_switch = float(getattr(m, "x_switch"))
    return r_switch, x_switch

# ============================================================
# model blocks + GT checkpoint loader + forward adapter
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
    Lazy import GT to avoid worker spawn import issues.
    Returns: (gt_model, gt_cfg(dataclass), raw_ckpt_dict)
    """
    import sys

    if "__file__" in globals():
        base_dir = os.path.dirname(os.path.abspath(__file__))
    else:
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
    candidates: List[Tuple[bool, bool]] = [
        (True, False),
        (False, False),
        (True, True),
        (False, True),
    ]
    for use_ret, use_batched in candidates:
        try:
            if use_batched:
                A = A2.unsqueeze(0).expand(B, -1, -1)
            else:
                A = A2
            if use_ret:
                out = gt(H, A, return_embeddings=True)
            else:
                out = gt(H, A)
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
            return GTForwardSpec(
                use_return_embeddings=use_ret,
                use_batched_adj=use_batched,
                z_key=z_key,
                out_dim=D,
            )
        except Exception:
            continue
    raise RuntimeError(
        "Failed to probe a valid GT forward signature. "
        "Please verify gt_torch_model.GTransformer forward(H,A,...) interface and output."
    )

def gt_forward_with_spec(gt: nn.Module, spec: GTForwardSpec, H: torch.Tensor, A2: torch.Tensor) -> torch.Tensor:
    if spec.use_batched_adj:
        A = A2.unsqueeze(0).expand(H.shape[0], -1, -1)
    else:
        A = A2
    if spec.use_return_embeddings:
        out = gt(H, A, return_embeddings=True)
    else:
        out = gt(H, A)
    z = _extract_z(out, spec.z_key)
    if z.ndim != 3 or z.shape[0] != H.shape[0] or z.shape[1] != H.shape[1]:
        raise RuntimeError(
            f"GT embedding shape mismatch: got {tuple(z.shape)} expected ({H.shape[0]},{H.shape[1]},D)"
        )
    return z

# ============================================================
# Multi-Branch Actor-Critic with GT (base)
# ============================================================

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
        phi = self.fusion(z)
        for blk in self.fusion_blocks:
            phi = blk(phi)

        logits = self.policy_head(phi) / self.policy_temperature
        value = self.value_head(phi).squeeze(-1)
        return logits, value

# ============================================================
# multiproc VecEnv with shared memory for large payloads
# ============================================================

def _restore_env_cfg(env_cfg_dict: Dict[str, Any]) -> EnvConfig:
    ts_schema = env_cfg_dict.get("timeseries_schema", None)
    if isinstance(ts_schema, dict):
        env_cfg_dict["timeseries_schema"] = TimeseriesSchema(**ts_schema)
    return EnvConfig(**env_cfg_dict)

def _env_worker_shared(
    conn,
    idx: int,
    env_cfg_dict: Dict[str, Any],
    time_scale: float,
    shm_obs_name: str,
    shm_gt_name: str,
    obs_shape: Tuple[int, int],
    gt_shape: Tuple[int, int, int],
) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    shm_obs = None
    shm_gt = None
    try:
        env_cfg = _restore_env_cfg(env_cfg_dict)
        env = PowerDispatchEnv(env_cfg)
        n_actions = int(env.action_space.n)

        flattener = ObsFlattenerV2(
            env.observation_space,
            n_actions=n_actions,
            time_scale=time_scale,
            time_period=int(getattr(env_cfg, "episode_len", 24)),
        )
        gt_din_env = int(getattr(env_cfg, "gt_node_feat_dim", 6))

        shm_obs = shared_memory.SharedMemory(name=shm_obs_name, create=False)
        shm_gt = shared_memory.SharedMemory(name=shm_gt_name, create=False)
        obs_buf = np.ndarray(obs_shape, dtype=np.float32, buffer=shm_obs.buf)
        gt_buf = np.ndarray(gt_shape, dtype=np.float32, buffer=shm_gt.buf)

        if flattener.flat_dim != obs_shape[1]:
            raise RuntimeError(f"[worker {idx}] obs_dim mismatch: flattener={flattener.flat_dim} shm={obs_shape[1]}")
        if gt_buf.shape[2] != gt_din_env:
            raise RuntimeError(f"[worker {idx}] gt_din mismatch: shm_gt={gt_buf.shape} env_gt_din={gt_din_env}")

        conn.send(("ready", {"idx": int(idx), "pid": int(os.getpid()), "n_actions": int(n_actions), "flat_dim": int(flattener.flat_dim)}))

        while True:
            cmd, data = conn.recv()
            if cmd == "reset":
                seed, options = data
                t0 = time.time()
                obs, info = env.reset(seed=seed, options=options)
                t_reset = time.time() - t0

                x = flattener.flatten(obs)
                gt_H0 = obs.get("gt_H", None)
                if gt_H0 is None:
                    raise KeyError("obs missing key 'gt_H' (env must provide it).")
                gt_H = np.asarray(gt_H0, dtype=np.float32)
                if gt_H.ndim != 2 or gt_H.shape[0] != gt_shape[1] or gt_H.shape[1] != gt_din_env:
                    raise ValueError(f"gt_H shape mismatch: got {gt_H.shape} expected ({gt_shape[1]},{gt_din_env})")

                np.copyto(obs_buf[idx], x, casting="no")
                np.copyto(gt_buf[idx], gt_H, casting="no")

                info_out = info if isinstance(info, dict) else {}
                info_out["_reset_sec"] = float(t_reset)
                conn.send(("ok", info_out))

            elif cmd == "step":
                action = int(data)
                obs, reward, terminated, truncated, info = env.step(action)

                x = flattener.flatten(obs)
                gt_H0 = obs.get("gt_H", None)
                if gt_H0 is None:
                    raise KeyError("obs missing key 'gt_H' (env must provide it).")
                gt_H = np.asarray(gt_H0, dtype=np.float32)
                if gt_H.ndim != 2 or gt_H.shape[0] != gt_shape[1] or gt_H.shape[1] != gt_din_env:
                    raise ValueError(f"gt_H shape mismatch: got {gt_H.shape} expected ({gt_shape[1]},{gt_din_env})")

                np.copyto(obs_buf[idx], x, casting="no")
                np.copyto(gt_buf[idx], gt_H, casting="no")

                conn.send(("ok", (float(reward), bool(terminated), bool(truncated), info if isinstance(info, dict) else {})))

            elif cmd == "ping":
                conn.send(("ok", "pong"))
            elif cmd == "close":
                break
            else:
                conn.send(("err", f"Unknown cmd: {cmd}"))
                break

    except Exception:
        try:
            conn.send(("init_err", "Worker failed:\n" + traceback.format_exc()))
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass
        try:
            if shm_obs is not None:
                shm_obs.close()
        except Exception:
            pass
        try:
            if shm_gt is not None:
                shm_gt.close()
        except Exception:
            pass

class SubprocVecEnv:
    """
    Shared-memory SubprocVecEnv:
    - obs and gt_H communicated through shared memory arrays (bulk payloads)
    - Pipe only carries small scalars and info dict
    - Uses wait() to receive replies
    """
    def __init__(
        self,
        n_envs: int,
        env_cfg: EnvConfig,
        time_scale: float,
        base_seed: int,
        obs_dim: int,
        n_bus: int,
        gt_din: int,
        *,
        ctx=None,
        start_timeout: float = 60.0,
        heartbeat_sec: float = 5.0,
        step_timeout_sec: float = 300.0,
        strict_timeout: bool = True,
        verbose: bool = True,
    ):
        self.n_envs = int(n_envs)
        self.time_scale = float(time_scale)
        self.ctx = ctx or mp.get_context()
        self.start_timeout = float(start_timeout)
        self.heartbeat_sec = float(max(0.1, heartbeat_sec))
        self.step_timeout_sec = float(max(1.0, step_timeout_sec))
        self.strict_timeout = bool(strict_timeout)
        self.verbose = bool(verbose)

        self.obs_dim = int(obs_dim)
        self.n_bus = int(n_bus)
        self.gt_din = int(gt_din)

        base_dict = asdict(env_cfg)

        obs_nbytes = int(self.n_envs * self.obs_dim * 4)
        gt_nbytes = int(self.n_envs * self.n_bus * self.gt_din * 4)
        self._shm_obs = shared_memory.SharedMemory(create=True, size=obs_nbytes)
        self._shm_gt = shared_memory.SharedMemory(create=True, size=gt_nbytes)
        self._obs_buf = np.ndarray((self.n_envs, self.obs_dim), dtype=np.float32, buffer=self._shm_obs.buf)
        self._gt_buf = np.ndarray((self.n_envs, self.n_bus, self.gt_din), dtype=np.float32, buffer=self._shm_gt.buf)
        self._obs_buf.fill(0.0)
        self._gt_buf.fill(0.0)

        self._parent_conns: List[Any] = []
        self._procs: List[Any] = []

        for i in range(self.n_envs):
            parent_conn, child_conn = self.ctx.Pipe(duplex=True)
            self._parent_conns.append(parent_conn)

            env_cfg_i = dict(base_dict)
            env_cfg_i["seed"] = int(base_seed + i * 1000)

            p = self.ctx.Process(
                target=_env_worker_shared,
                args=(
                    child_conn,
                    i,
                    env_cfg_i,
                    self.time_scale,
                    self._shm_obs.name,
                    self._shm_gt.name,
                    self._obs_buf.shape,
                    self._gt_buf.shape,
                ),
                daemon=True,
            )
            p.start()
            self._procs.append(p)
            try:
                child_conn.close()
            except Exception:
                pass

        for i, (p, conn) in enumerate(zip(self._procs, self._parent_conns)):
            if not conn.poll(self.start_timeout):
                raise RuntimeError(
                    f"[SubprocVecEnv] worker[{i}] no READY within {self.start_timeout}s. "
                    f"alive={p.is_alive()} exitcode={p.exitcode}."
                )
            tag, payload = conn.recv()
            if tag == "init_err":
                raise RuntimeError(f"[SubprocVecEnv] worker[{i}] init_err:\n{payload}")
            if tag != "ready":
                raise RuntimeError(f"[SubprocVecEnv] worker[{i}] expected READY, got {tag}: {payload}")

    def _recv_one_checked(self, conn, who: str):
        try:
            tag, payload = conn.recv()
        except (EOFError, ConnectionResetError) as e:
            states = [f"env[{i}]: alive={p.is_alive()} exitcode={p.exitcode}" for i, p in enumerate(self._procs)]
            raise RuntimeError(f"[SubprocVecEnv] {who} recv failed: {repr(e)} | " + " | ".join(states))
        if tag == "ok":
            return payload
        raise RuntimeError(f"[SubprocVecEnv] worker error from {who}:\n{payload}")

    def _recv_many_checked(self, conns: List[Any], who_prefix: str) -> List[Any]:
        pending = set(range(len(conns)))
        results: List[Any] = [None] * len(conns)
        start = time.time()
        last_beat = start
        local_map = {conns[i]: i for i in range(len(conns))}
        while pending:
            now = time.time()
            if now - start > self.step_timeout_sec:
                msg = f"[SubprocVecEnv][TIMEOUT] waiting {who_prefix} for {now - start:.1f}s; pending={sorted(list(pending))}"
                if self.verbose:
                    print(msg)
                if self.strict_timeout:
                    raise TimeoutError(msg)
                start = now

            ready = mp_wait([conns[i] for i in pending], timeout=self.heartbeat_sec)
            if not ready:
                if self.verbose and (time.time() - last_beat >= self.heartbeat_sec):
                    print(f"[SubprocVecEnv][heartbeat] waiting {who_prefix}: pending={sorted(list(pending))}")
                    last_beat = time.time()
                continue

            for conn in ready:
                li = local_map[conn]
                payload = self._recv_one_checked(conn, who=f"{who_prefix}[{li}]")
                results[li] = payload
                if li in pending:
                    pending.remove(li)
        return results

    def reset(
        self,
        seeds: Optional[List[Optional[int]]] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        if seeds is None:
            seeds = [None] * self.n_envs
        if len(seeds) != self.n_envs:
            raise ValueError("reset seeds length mismatch")
        for conn, sd in zip(self._parent_conns, seeds):
            conn.send(("reset", (sd, options)))
        infos_payload = self._recv_many_checked(self._parent_conns, who_prefix="reset")
        infos: List[Dict[str, Any]] = [(p if isinstance(p, dict) else {}) for p in infos_payload]
        return self._obs_buf.copy(), self._gt_buf.copy(), infos

    def step(
        self,
        actions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        actions = np.asarray(actions, dtype=np.int64).reshape(-1)
        if actions.shape[0] != self.n_envs:
            raise ValueError("step actions length mismatch")
        for conn, a in zip(self._parent_conns, actions):
            conn.send(("step", int(a)))
        payloads = self._recv_many_checked(self._parent_conns, who_prefix="step")

        rewards: List[float] = []
        terms: List[bool] = []
        truncs: List[bool] = []
        infos: List[Dict[str, Any]] = []
        for p in payloads:
            r, term, trunc, info = p
            rewards.append(float(r))
            terms.append(bool(term))
            truncs.append(bool(trunc))
            infos.append(info if isinstance(info, dict) else {})

        return (
            self._obs_buf.copy(),
            self._gt_buf.copy(),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(terms, dtype=np.bool_),
            np.asarray(truncs, dtype=np.bool_),
            infos,
        )

    def reset_many(
        self,
        idxs: List[int],
        seeds: Optional[List[Optional[int]]] = None,
        options: Optional[dict] = None,
        *,
        who_prefix: str = "reset_many",
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        if not idxs:
            return (
                np.zeros((0, self.obs_dim), dtype=np.float32),
                np.zeros((0, self.n_bus, self.gt_din), dtype=np.float32),
                [],
            )
        idxs2 = [int(i) for i in idxs]
        for i in idxs2:
            if i < 0 or i >= self.n_envs:
                raise IndexError(f"reset_many idx out of range: {i} (n_envs={self.n_envs})")
        if seeds is None:
            seeds2: List[Optional[int]] = [None] * len(idxs2)
        else:
            if len(seeds) != len(idxs2):
                raise ValueError("reset_many seeds length mismatch")
            seeds2 = list(seeds)

        conns = [self._parent_conns[i] for i in idxs2]
        for conn, sd in zip(conns, seeds2):
            conn.send(("reset", (sd, options)))
        infos_payload = self._recv_many_checked(conns, who_prefix=f"{who_prefix}(n={len(idxs2)})")
        infos: List[Dict[str, Any]] = [(p if isinstance(p, dict) else {}) for p in infos_payload]
        obs = self._obs_buf[idxs2].copy()
        gt = self._gt_buf[idxs2].copy()
        return obs, gt, infos

    def close(self) -> None:
        for conn in self._parent_conns:
            try:
                conn.send(("close", None))
            except Exception:
                pass
        for conn in self._parent_conns:
            try:
                conn.close()
            except Exception:
                pass
        for p in self._procs:
            try:
                p.join(timeout=2.0)
            except Exception:
                pass
            if p.is_alive():
                try:
                    p.terminate()
                except Exception:
                    pass
        try:
            self._shm_obs.close()
        except Exception:
            pass
        try:
            self._shm_gt.close()
        except Exception:
            pass
        try:
            self._shm_obs.unlink()
        except Exception:
            pass
        try:
            self._shm_gt.unlink()
        except Exception:
            pass

# ============================================================
# PPO buffer + GAE (store gt_H on CPU directly)
# ============================================================

class RolloutBufferGT:
    def __init__(
        self,
        n_steps: int,
        n_envs: int,
        obs_dim: int,
        n_bus: int,
        gt_din: int,
        device: torch.device,
        *,
        store_gt_on_cpu: bool = True,
    ):
        self.n_steps = int(n_steps)
        self.n_envs = int(n_envs)
        self.obs_dim = int(obs_dim)
        self.n_bus = int(n_bus)
        self.gt_din = int(gt_din)
        self.device = device
        self.store_gt_on_cpu = bool(store_gt_on_cpu)

        self.obs = torch.zeros((n_steps, n_envs, obs_dim), dtype=torch.float32, device=device)
        if self.store_gt_on_cpu:
            pin = (device.type == "cuda")
            self.gt_H = torch.zeros((n_steps, n_envs, n_bus, gt_din), dtype=torch.float32, device="cpu", pin_memory=pin)
        else:
            self.gt_H = torch.zeros((n_steps, n_envs, n_bus, gt_din), dtype=torch.float32, device=device)

        self.actions = torch.zeros((n_steps, n_envs), dtype=torch.int64, device=device)
        self.rewards = torch.zeros((n_steps, n_envs), dtype=torch.float32, device=device)
        self.dones = torch.zeros((n_steps, n_envs), dtype=torch.float32, device=device)
        self.values = torch.zeros((n_steps, n_envs), dtype=torch.float32, device=device)
        self.logprobs = torch.zeros((n_steps, n_envs), dtype=torch.float32, device=device)

        self.advantages = torch.zeros((n_steps, n_envs), dtype=torch.float32, device=device)
        self.returns = torch.zeros((n_steps, n_envs), dtype=torch.float32, device=device)

        self._pos = 0

    def add(
        self,
        obs: torch.Tensor,
        gt_H_np: np.ndarray,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
        logprob: torch.Tensor,
    ) -> None:
        i = self._pos
        self.obs[i].copy_(obs)

        gt_cpu_t = torch.from_numpy(np.asarray(gt_H_np, dtype=np.float32, order="C"))
        if self.store_gt_on_cpu:
            self.gt_H[i].copy_(gt_cpu_t)
        else:
            self.gt_H[i].copy_(gt_cpu_t.to(self.device, non_blocking=True))

        self.actions[i].copy_(action)
        self.rewards[i].copy_(reward)
        self.dones[i].copy_(done)
        self.values[i].copy_(value)
        self.logprobs[i].copy_(logprob)
        self._pos += 1

    def compute_returns_and_advantages(self, last_values: torch.Tensor, gamma: float, gae_lambda: float) -> None:
        gae = torch.zeros((self.n_envs,), dtype=torch.float32, device=self.device)
        for t in reversed(range(self.n_steps)):
            next_nonterminal = 1.0 - self.dones[t]
            next_values = last_values if t == self.n_steps - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_values * next_nonterminal - self.values[t]
            gae = delta + gamma * gae_lambda * next_nonterminal * gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values

    def get_batches(self, batch_size: int, *, shuffle: bool = True):
        n = self.n_steps * self.n_envs
        obs = self.obs.reshape(n, self.obs_dim)
        gt_H = self.gt_H.reshape(n, self.n_bus, self.gt_din)
        actions = self.actions.reshape(n)
        logprobs = self.logprobs.reshape(n)
        values = self.values.reshape(n)
        returns = self.returns.reshape(n)
        adv = self.advantages.reshape(n)

        idxs = np.arange(n)
        if shuffle:
            np.random.shuffle(idxs)
        for start in range(0, n, batch_size):
            mb = idxs[start : start + batch_size]
            yield (obs[mb], gt_H[mb], actions[mb], logprobs[mb], values[mb], returns[mb], adv[mb])

# ============================================================
# PPO hyperparams / train config
# ============================================================

@dataclass
class PPOHyperParams:
    total_steps: int = 200_000
    n_steps: int = 2048
    n_envs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    n_epochs: int = 10
    batch_size: int = 256
    lr: float = 3e-4
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = 0.2
    ent_coef: float = 0.01
    ent_coef_final: float = 0.003
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = 0.05
    reward_clip: Optional[float] = None
    lr_anneal: bool = True
    normalize_adv: bool = True
    save_every_update: int = 10
    rollout_print_every: int = 0
    print_episode_end: bool = True
    print_ppo_epoch: bool = False

@dataclass
class ModelConfig:
    enc_hidden: int = 256
    fusion_hidden: int = 256
    fusion_blocks: int = 2
    dropout: float = 0.0
    policy_temperature: float = 1.0
    emb_dim_bus: int = 128
    emb_dim_load: int = 128
    emb_dim_fcst: int = 128
    emb_dim_time: int = 32
    topo_emb_dim: int = 32
    gt_proj_dim: int = 128
    gt_pool: str = "mean"
    adj_cache_size: int = 64
    adj_cache_cuda_size: int = 16

@dataclass
class TrainConfig:
    seed: int = 0
    device: str = "auto"
    time_scale: float = 1e-3
    obs_clip: float = 10.0

    # PPO reward scale：仅作用于 env reward -> PPO reward
    reward_scale: float = 1.0

    rollout_debug_every: int = 64
    start_method: str = "spawn"
    gt_ckpt_path: str = ""
    gt_lr: float = 1e-5
    gt_train: bool = True
    rollout_inference_mode: bool = True
    allow_tf32: bool = True
    hparams: PPOHyperParams = PPOHyperParams()
    model: ModelConfig = ModelConfig()
    out_dir: str = "./runs_ppo_gt_sr_merged"
    resume_path: Optional[str] = None

    import_log_every_update: int = 5

    # ============ 新增：日志控制（无 argparse，直接改这里） ============
    # 0=ERROR,1=WARN,2=INFO,3=DEBUG,4=TRACE
    log_level: int = 3

    # rollout 每 step 打印间隔（1=每一步都打印）
    log_rollout_every_step: int = 1

    # rollout 打印“动作分布”的间隔（避免太重）；0=不打印
    log_action_hist_every_step: int = 8

    # optimizer 每多少个 batch 打印一次；1=每 batch
    log_opt_every_batch: int = 10

    # optimizer 是否额外打印 grad_norm（会增加计算开销）
    log_grad_norm: bool = True

    # 每次 rollout step 是否打印一条“info keys 的样例”；0=不打印
    log_info_sample_every_step: int = 16

    # run_log.txt 的文件名
    run_log_name: str = "run_log.txt"

# ============================================================
# SR decoder + SR training config (输出在“归一化cost空间”)
# ============================================================

class SelfReflectionDecoder(nn.Module):
    """
    Shared trunk + 2 heads:
      - total_cost_norm_hat  （预测 total_cost_yuan / total_scale_yuan）
      - risk_cost_norm_hat   （预测 risk_cost_yuan / risk_scale_yuan）
    conditioned on action.
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

@dataclass
class SRConfig:
    enabled: bool = True

    # phase schedule
    warmup_updates: int = 20
    warmup_sr_epochs: int = 5

    # shaping reward: r_used = r_env_scaled + alpha * r_hat_rl
    alpha_final: float = 0.2
    alpha_ramp_updates: int = 30

    # in-mind reward: r_hat_rl = -(total_hat_norm*total_scale + lambda_risk*risk_hat_norm*risk_scale) * reward_scale
    lambda_risk: float = 0.0

    # SR alignment loss
    loss_type: str = "huber"   # {"huber","mse"}
    huber_delta: float = 1.0

    loss_w_total: float = 1.0
    loss_w_risk: float = 1.0

    # SR optimizer
    lr: float = 1e-4
    grad_clip: float = 10.0

    # SR net size
    act_emb_dim: int = 64
    hidden: int = 256
    dropout: float = 0.0

    # label scale (yuan -> normalized)
    total_scale_init_yuan: float = 1.0e6
    risk_scale_init_yuan: float = 1.0e5
    scale_q: float = 0.95
    scale_ema: float = 0.98
    scale_min: float = 1.0

    # logging
    print_every_update: int = 1

class MultiBranchActorCriticWithGTAndSR(MultiBranchActorCriticWithGT):
    def __init__(self, *args, sr_cfg: SRConfig, **kwargs):
        super().__init__(*args, **kwargs)
        self.sr_cfg = sr_cfg

        fusion_hidden = None
        if isinstance(self.fusion, nn.Sequential) and len(self.fusion) > 0 and isinstance(self.fusion[0], nn.Linear):
            fusion_hidden = int(self.fusion[0].out_features)
        if fusion_hidden is None:
            fusion_hidden = int(kwargs.get("fusion_hidden", 256))

        self.sr = SelfReflectionDecoder(
            phi_dim=fusion_hidden,
            n_actions=int(self.n_actions),
            act_emb_dim=int(sr_cfg.act_emb_dim),
            hidden=int(sr_cfg.hidden),
            dropout=float(sr_cfg.dropout),
        )

    def forward_with_phi(self, obs: torch.Tensor, gt_H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        phi = self.fusion(z)
        for blk in self.fusion_blocks:
            phi = blk(phi)

        logits = self.policy_head(phi) / self.policy_temperature
        value = self.value_head(phi).squeeze(-1)
        return logits, value, phi

# ============================================================
# Buffer: store SR normalized labels
# ============================================================

class RolloutBufferSR(RolloutBufferGT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n_steps, n_envs = self.n_steps, self.n_envs
        dev = self.device
        self.cost_total_norm = torch.zeros((n_steps, n_envs), dtype=torch.float32, device=dev)
        self.cost_risk_norm = torch.zeros((n_steps, n_envs), dtype=torch.float32, device=dev)

    def add_sr_labels(self, cost_total_norm: torch.Tensor, cost_risk_norm: torch.Tensor) -> None:
        i = self._pos - 1
        self.cost_total_norm[i].copy_(cost_total_norm)
        self.cost_risk_norm[i].copy_(cost_risk_norm)

    def get_batches_sr(self, batch_size: int, *, shuffle: bool = True):
        n = self.n_steps * self.n_envs
        idxs = np.arange(n)
        if shuffle:
            np.random.shuffle(idxs)

        obs = self.obs.reshape(n, self.obs_dim)
        gt_H = self.gt_H.reshape(n, self.n_bus, self.gt_din)
        actions = self.actions.reshape(n)
        old_logp = self.logprobs.reshape(n)
        old_val = self.values.reshape(n)
        returns = self.returns.reshape(n)
        adv = self.advantages.reshape(n)
        ctot = self.cost_total_norm.reshape(n)
        crisk = self.cost_risk_norm.reshape(n)

        for start in range(0, n, batch_size):
            mb = idxs[start : start + batch_size]
            yield (obs[mb], gt_H[mb], actions[mb], old_logp[mb], old_val[mb], returns[mb], adv[mb], ctot[mb], crisk[mb])

# ============================================================
# SR label extraction + losses + alpha schedule
# ============================================================

def _get_float(info: Dict[str, Any], keys: List[str], default: float = 0.0) -> float:
    for k in keys:
        if k in info:
            try:
                return float(info.get(k, default))
            except Exception:
                continue
    return float(default)

def extract_cost_labels_yuan(
    infos: List[Dict[str, Any]],
    n_envs: int,
    *,
    verbose_once: bool,
    tag: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    total_cost_yuan：优先读取 total_cost_yuan；否则用若干项拼出来（保底）
    risk_cost_yuan：兼容 risk_cost_yuan / risk_term / risk_cost
    """
    tot = np.zeros((n_envs,), dtype=np.float32)
    risk = np.zeros((n_envs,), dtype=np.float32)
    missing_tot, missing_risk = 0, 0

    for i, info in enumerate(infos):
        if "total_cost_yuan" in info:
            tot[i] = float(info.get("total_cost_yuan", 0.0))
        else:
            missing_tot += 1
            tot[i] = (
                _get_float(info, ["import_cost_yuan"], 0.0)
                + _get_float(info, ["loss_cost_yuan"], 0.0)
                + _get_float(info, ["v_cost_yuan"], 0.0)
                + _get_float(info, ["line_cost_yuan"], 0.0)
                + _get_float(info, ["trafo_cost_yuan"], 0.0)
                + _get_float(info, ["risk_cost_yuan", "risk_term", "risk_cost"], 0.0)
                + _get_float(info, ["switch_cost_yuan"], 0.0)
            )

        if ("risk_cost_yuan" in info) or ("risk_term" in info) or ("risk_cost" in info):
            risk[i] = _get_float(info, ["risk_cost_yuan", "risk_term", "risk_cost"], 0.0)
        else:
            missing_risk += 1
            risk[i] = 0.0

    if verbose_once and (missing_tot > 0 or missing_risk > 0):
        print(f"[SR][labels][{tag}] missing total_cost_yuan {missing_tot}/{n_envs}, missing risk_cost_yuan-like {missing_risk}/{n_envs}")
    return tot, risk

def sr_alignment_loss(
    sr_cfg: SRConfig,
    pred_total_norm: torch.Tensor,
    pred_risk_norm: torch.Tensor,
    y_total_norm: torch.Tensor,
    y_risk_norm: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if sr_cfg.loss_type.lower() == "mse":
        lt = F.mse_loss(pred_total_norm, y_total_norm)
        lr = F.mse_loss(pred_risk_norm, y_risk_norm)
    else:
        lt = F.huber_loss(pred_total_norm, y_total_norm, delta=float(sr_cfg.huber_delta))
        lr = F.huber_loss(pred_risk_norm, y_risk_norm, delta=float(sr_cfg.huber_delta))
    loss = float(sr_cfg.loss_w_total) * lt + float(sr_cfg.loss_w_risk) * lr
    return lt, lr, loss

def compute_alpha(update_i: int, sr_cfg: SRConfig) -> float:
    if update_i <= int(sr_cfg.warmup_updates):
        return 0.0
    ramp_u = int(sr_cfg.alpha_ramp_updates)
    if ramp_u <= 0:
        return float(sr_cfg.alpha_final)
    t = update_i - int(sr_cfg.warmup_updates)
    frac = float(np.clip(t / float(ramp_u), 0.0, 1.0))
    return float(sr_cfg.alpha_final) * frac

# ============================================================
# checkpoint (SR version)
# ============================================================

def save_checkpoint_sr(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    normalizer: MaskedObsNormalizer,
    step: int,
    update: int,
    episode_count: int,
    train_meta: Dict[str, Any],
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "normalizer": normalizer.state_dict(),
        "step": int(step),
        "update": int(update),
        "episode_count": int(episode_count),
        "train_meta": dict(train_meta),
    }
    torch.save(payload, path)

# ============================================================
# training loop (SR warmup + joint) + logging (VERBOSE)
# ============================================================

def _action_hist(actions_np: np.ndarray, n_actions: int) -> str:
    actions_np = np.asarray(actions_np, dtype=np.int64).reshape(-1)
    if actions_np.size == 0:
        return "[]"
    cnt = np.bincount(actions_np, minlength=n_actions)
    topk = min(8, n_actions)
    idx = np.argsort(-cnt)[:topk]
    pairs = [(int(i), int(cnt[i])) for i in idx if cnt[i] > 0]
    return str(pairs)

def _safe_keys_sample(info: Dict[str, Any], max_keys: int = 16) -> str:
    if not isinstance(info, dict):
        return "{}"
    keys = list(info.keys())
    keys = keys[:max_keys]
    return "{" + ", ".join(keys) + ("..." if len(info.keys()) > max_keys else "") + "}"

def train(cfg_train: TrainConfig, sr_cfg: SRConfig) -> None:
    # ---- setup loggers ----
    os.makedirs(cfg_train.out_dir, exist_ok=True)
    run_log_path = os.path.join(cfg_train.out_dir, cfg_train.run_log_name)
    logger = VerboseLogger(run_log_path, level=int(cfg_train.log_level), flush=True)

    logger.info("train() entered.")
    logger.info(f"python={sys.version.replace(os.linesep,' ')}")
    logger.info(f"platform={platform.platform()}")
    logger.info(f"cwd={os.getcwd()}")
    logger.info(f"out_dir={cfg_train.out_dir}")
    logger.info(f"start_method={cfg_train.start_method}")

    import_log_path = os.path.join(cfg_train.out_dir, "imports_log.txt")
    import_logger = ImportLogger(import_log_path)
    import_logger.dump(tag="train_start", only_new=False)

    with stage_timer(logger, "set_seeds_and_device", "INFO", extra=f"seed={cfg_train.seed} device={cfg_train.device}"):
        set_global_seeds(cfg_train.seed)
        mp_ctx = mp.get_context(cfg_train.start_method)
        device = get_device(cfg_train.device)

    if device.type == "cuda" and cfg_train.allow_tf32:
        with stage_timer(logger, "enable_tf32", "DEBUG"):
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

    tb_dir = os.path.join(cfg_train.out_dir, "tb")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)

    episode_csv = os.path.join(cfg_train.out_dir, "episode_metrics.csv")
    sr_csv = os.path.join(cfg_train.out_dir, "sr_metrics.csv")

    with stage_timer(logger, "env_init_eval_env_and_flattener", "INFO"):
        env_cfg = EnvConfig(seed=cfg_train.seed)
        eval_env = PowerDispatchEnv(env_cfg)
        n_actions = int(eval_env.action_space.n)
        time_period = int(getattr(env_cfg, "episode_len", 24))

        flattener = ObsFlattenerV2(
            eval_env.observation_space,
            n_actions=n_actions,
            time_scale=cfg_train.time_scale,
            time_period=time_period,
        )

        obs0, _ = eval_env.reset(seed=cfg_train.seed, options=None)
        gt_H0 = np.asarray(obs0.get("gt_H", None), dtype=np.float32)
        if gt_H0.ndim != 2:
            raise RuntimeError(f"env obs['gt_H'] must be (N,D); got {gt_H0.shape}")
        n_bus, gt_din = int(gt_H0.shape[0]), int(gt_H0.shape[1])

        mask = np.ones((flattener.flat_dim,), dtype=bool)
        sl_time = flattener.slices["time_feat"]
        mask[sl_time.start + 1 : sl_time.stop] = False  # do not normalize sin/cos
        mask[flattener.slices["topology_id"]] = False    # do not normalize discrete topo id
        normalizer = MaskedObsNormalizer(flattener.flat_dim, mask=mask, clip=cfg_train.obs_clip)

    h = cfg_train.hparams
    n_envs = int(h.n_envs)
    if h.n_steps % n_envs != 0:
        raise ValueError(f"h.n_steps ({h.n_steps}) must be divisible by h.n_envs ({n_envs}).")
    steps_per_env = int(h.n_steps // n_envs)

    logger.info(f"device={device} | n_envs={n_envs} steps_per_env={steps_per_env} total_steps={h.total_steps}")
    logger.info(f"obs_dim={flattener.flat_dim} n_actions={n_actions} n_bus={n_bus} gt_din={gt_din}")
    logger.info(
        f"SR schedule: warmup_updates={sr_cfg.warmup_updates} warmup_sr_epochs={sr_cfg.warmup_sr_epochs} "
        f"alpha_final={sr_cfg.alpha_final:g} alpha_ramp_updates={sr_cfg.alpha_ramp_updates}"
    )
    logger.info(
        f"SR scale: init total_scale_yuan={sr_cfg.total_scale_init_yuan:g} risk_scale_yuan={sr_cfg.risk_scale_init_yuan:g} "
        f"(q={sr_cfg.scale_q}, ema={sr_cfg.scale_ema}, min={sr_cfg.scale_min})"
    )
    logger.info(f"reward_scale={cfg_train.reward_scale:g} | rollout_inference_mode={cfg_train.rollout_inference_mode} | gt_train={cfg_train.gt_train}")

    total_scale_est = RobustScaleEMA(
        init_scale=float(sr_cfg.total_scale_init_yuan),
        q=float(sr_cfg.scale_q),
        ema=float(sr_cfg.scale_ema),
        min_scale=float(sr_cfg.scale_min),
    )
    risk_scale_est = RobustScaleEMA(
        init_scale=float(sr_cfg.risk_scale_init_yuan),
        q=float(sr_cfg.scale_q),
        ema=float(sr_cfg.scale_ema),
        min_scale=float(sr_cfg.scale_min),
    )

    with stage_timer(logger, "create_subproc_vecenv", "INFO", extra=f"n_envs={n_envs} start_timeout=120s"):
        vecenv = SubprocVecEnv(
            n_envs=n_envs,
            env_cfg=env_cfg,
            time_scale=cfg_train.time_scale,
            base_seed=cfg_train.seed,
            obs_dim=int(flattener.flat_dim),
            n_bus=int(n_bus),
            gt_din=int(gt_din),
            ctx=mp_ctx,
            start_timeout=120.0,
            heartbeat_sec=5.0,
            step_timeout_sec=300.0,
            strict_timeout=True,
            verbose=True,
        )

    with stage_timer(logger, "vecenv_initial_reset", "INFO"):
        obs_raw, gt_H_raw, infos0 = vecenv.reset(seeds=[cfg_train.seed + i * 1000 for i in range(n_envs)], options=None)
        if infos0 and isinstance(infos0[0], dict):
            logger.debug(f"reset info sample keys: {_safe_keys_sample(infos0[0])}")
        logger.info(f"obs_raw.shape={tuple(obs_raw.shape)} gt_H_raw.shape={tuple(gt_H_raw.shape)}")

    if not cfg_train.gt_ckpt_path:
        raise ValueError("gt_ckpt_path is empty. Set TrainConfig.gt_ckpt_path to pretrained GT checkpoint path.")

    with stage_timer(logger, "load_gtransformer_checkpoint", "INFO", extra=f"path={cfg_train.gt_ckpt_path}"):
        gt, gt_cfg, _ = load_gtransformer_checkpoint(cfg_train.gt_ckpt_path, device)
    import_logger.dump(tag="after_load_gt", only_new=True)
    if not cfg_train.gt_train:
        for p in gt.parameters():
            p.requires_grad_(False)

    with stage_timer(logger, "build_adj_cache_and_probe_gt", "INFO"):
        r_switch, x_switch = _load_switch_params()
        adj_cache = TopologyAdjacencyCache(
            feeder_cluster=eval_env.feeder_cluster,
            base_net=eval_env.base_net,
            r_switch=r_switch,
            x_switch=x_switch,
            max_entries=int(cfg_train.model.adj_cache_size),
        )

        topo0 = int(obs_raw[0, flattener.slices["topology_id"]][0].round())
        A0 = adj_cache.get(topo0).to(device=device)
        H_probe = torch.from_numpy(gt_H_raw[:2]).to(device=device, dtype=torch.float32)
        spec = probe_gt_forward(gt, H_probe, A0)

    logger.info(f"GT probe: use_return_embeddings={spec.use_return_embeddings} use_batched_adj={spec.use_batched_adj} z_key={spec.z_key} out_dim={spec.out_dim}")

    with stage_timer(logger, "build_model_and_optimizer", "INFO"):
        model = MultiBranchActorCriticWithGTAndSR(
            obs_dim=flattener.flat_dim,
            n_actions=n_actions,
            slices=flattener.slices,
            gt=gt,
            gt_spec=spec,
            adj_cache=adj_cache,
            gt_pool=cfg_train.model.gt_pool,
            gt_proj_dim=cfg_train.model.gt_proj_dim,
            enc_hidden=cfg_train.model.enc_hidden,
            fusion_hidden=cfg_train.model.fusion_hidden,
            fusion_blocks=cfg_train.model.fusion_blocks,
            dropout=cfg_train.model.dropout,
            policy_temperature=cfg_train.model.policy_temperature,
            emb_dim_bus=cfg_train.model.emb_dim_bus,
            emb_dim_load=cfg_train.model.emb_dim_load,
            emb_dim_fcst=cfg_train.model.emb_dim_fcst,
            emb_dim_time=cfg_train.model.emb_dim_time,
            topo_emb_dim=cfg_train.model.topo_emb_dim,
            adj_cache_cuda_size=cfg_train.model.adj_cache_cuda_size,
            sr_cfg=sr_cfg,
        ).to(device)

        base_lr = float(h.lr)
        gt_lr = float(cfg_train.gt_lr)
        sr_lr = float(sr_cfg.lr)

        ac_params: List[nn.Parameter] = []
        gt_params: List[nn.Parameter] = []
        sr_params: List[nn.Parameter] = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("gt.") or name.startswith("gt_proj."):
                gt_params.append(p)
            elif name.startswith("sr."):
                sr_params.append(p)
            else:
                ac_params.append(p)

        optimizer = torch.optim.Adam(
            [
                {"params": ac_params, "lr": base_lr, "eps": 1e-5},
                {"params": gt_params, "lr": gt_lr, "eps": 1e-5},
                {"params": sr_params, "lr": sr_lr, "eps": 1e-5},
            ]
        )

    with stage_timer(logger, "write_train_meta_json", "INFO"):
        write_json(
            os.path.join(cfg_train.out_dir, "train_meta_sr_merged.json"),
            {
                "train_cfg": asdict(cfg_train),
                "sr_cfg": asdict(sr_cfg),
                "env_cfg": asdict(env_cfg),
                "gt_cfg": asdict(gt_cfg) if hasattr(gt_cfg, "__dict__") else str(gt_cfg),
                "gt_forward_spec": asdict(spec),
                "obs_dim": int(flattener.flat_dim),
                "n_actions": int(n_actions),
                "n_bus": int(n_bus),
                "gt_din": int(gt_din),
            },
        )

    # ---- training state ----
    global_step, update_i, episode_count = 0, 0, 0
    n_updates = int(math.ceil(h.total_steps / h.n_steps))
    wall0 = time.perf_counter()

    # episode bookkeeping
    ep_return_used = np.zeros((n_envs,), dtype=np.float64)
    ep_return_env_scaled = np.zeros((n_envs,), dtype=np.float64)
    ep_return_shaping = np.zeros((n_envs,), dtype=np.float64)
    ep_len = np.zeros((n_envs,), dtype=np.int32)
    last_info: List[Dict[str, Any]] = [{} for _ in range(n_envs)]

    warned_labels = False
    printed_scale_once = False

    logger.info(f"training loop start: n_updates={n_updates} h.n_steps={h.n_steps} (steps_per_env={steps_per_env}, n_envs={n_envs})")

    try:
        for _ in range(n_updates):
            update_i += 1
            t_up0 = time.perf_counter()
            elapsed_total = time.perf_counter() - wall0
            frac_done = min(1.0, global_step / max(1, h.total_steps))

            is_warmup = (update_i <= int(sr_cfg.warmup_updates))
            alpha_now = compute_alpha(update_i, sr_cfg)

            frac = 1.0 - (update_i - 1) / max(n_updates - 1, 1)
            lr_ac_now = float(h.lr) * frac if h.lr_anneal else float(h.lr)
            lr_gt_now = float(cfg_train.gt_lr) * frac if h.lr_anneal else float(cfg_train.gt_lr)
            lr_sr_now = float(sr_cfg.lr) * frac if h.lr_anneal else float(sr_cfg.lr)

            if is_warmup:
                optimizer.param_groups[0]["lr"] = 0.0
                optimizer.param_groups[1]["lr"] = 0.0
                optimizer.param_groups[2]["lr"] = lr_sr_now
            else:
                optimizer.param_groups[0]["lr"] = lr_ac_now
                optimizer.param_groups[1]["lr"] = lr_gt_now
                optimizer.param_groups[2]["lr"] = lr_sr_now

            ent_coef_now = h.ent_coef_final + (h.ent_coef - h.ent_coef_final) * frac
            mode_str = "WARMUP_SR" if is_warmup else "JOINT"

            logger.info(
                f"[update {update_i:04d}/{n_updates}] mode={mode_str} global_step={global_step}/{h.total_steps} "
                f"alpha={alpha_now:.6g} ent_coef={ent_coef_now:.6g} lr(ac/gt/sr)={optimizer.param_groups[0]['lr']:.3e}/"
                f"{optimizer.param_groups[1]['lr']:.3e}/{optimizer.param_groups[2]['lr']:.3e} | {eta_str(elapsed_total, frac_done)}"
            )

            buffer = RolloutBufferSR(
                steps_per_env,
                n_envs,
                flattener.flat_dim,
                n_bus,
                gt_din,
                device=device,
                store_gt_on_cpu=True,
            )

            rollout_r_env_mean: List[float] = []
            rollout_r_hat_mean: List[float] = []
            rollout_r_used_mean: List[float] = []

            # ---- rollout ----
            t_roll0 = time.perf_counter()
            # detailed timers accumulators
            acc_norm = 0.0
            acc_forward = 0.0
            acc_sr = 0.0
            acc_env_step = 0.0
            acc_labels = 0.0
            acc_buffer = 0.0
            acc_reset = 0.0

            logger.debug(f"[rollout] start steps_per_env={steps_per_env} (each step advances global_step by n_envs={n_envs})")

            for step in range(steps_per_env):
                step_t0 = time.perf_counter()

                # --- normalize ---
                t0 = time.perf_counter()
                normalizer.update(obs_raw)
                obs_norm = normalizer.normalize(obs_raw)
                acc_norm += (time.perf_counter() - t0)

                # --- move tensors ---
                obs_t = torch.from_numpy(obs_norm).to(device=device, dtype=torch.float32)
                gt_t = torch.from_numpy(gt_H_raw).to(device=device, dtype=torch.float32)

                # --- policy forward + sample ---
                t0 = time.perf_counter()
                if cfg_train.rollout_inference_mode:
                    with torch.inference_mode():
                        logits, values, phi = model.forward_with_phi(obs_t, gt_t)
                        dist = torch.distributions.Categorical(logits=logits)
                        actions = dist.sample()
                        logprobs = dist.log_prob(actions)
                else:
                    logits, values, phi = model.forward_with_phi(obs_t, gt_t)
                    dist = torch.distributions.Categorical(logits=logits)
                    actions = dist.sample()
                    logprobs = dist.log_prob(actions)
                acc_forward += (time.perf_counter() - t0)

                # --- SR forward ---
                t0 = time.perf_counter()
                if sr_cfg.enabled:
                    total_hat_norm, risk_hat_norm = model.sr(phi, actions)
                else:
                    total_hat_norm = torch.zeros((n_envs,), device=device, dtype=torch.float32)
                    risk_hat_norm = torch.zeros((n_envs,), device=device, dtype=torch.float32)
                acc_sr += (time.perf_counter() - t0)

                # --- env step ---
                t0 = time.perf_counter()
                next_obs_raw, next_gt_H_raw, rewards_raw, terms, truncs, infos = vecenv.step(actions.detach().cpu().numpy())
                acc_env_step += (time.perf_counter() - t0)

                dones = np.asarray(terms | truncs, dtype=np.float32)

                # stash infos for episode end logging
                for i in range(n_envs):
                    last_info[i] = infos[i] if isinstance(infos[i], dict) else {}

                rs_scale = float(getattr(cfg_train, "reward_scale", 1.0))
                rewards_env_scaled = rewards_raw.astype(np.float32) * np.float32(rs_scale)

                # --- labels & scales ---
                t0 = time.perf_counter()
                tot_yuan, risk_yuan = extract_cost_labels_yuan(
                    infos, n_envs, verbose_once=(not warned_labels), tag=("warmup" if is_warmup else "joint")
                )
                warned_labels = True

                total_scale_est.update(tot_yuan)
                risk_scale_est.update(risk_yuan)
                total_scale = total_scale_est.get()
                risk_scale = risk_scale_est.get()

                tot_norm = (tot_yuan / max(total_scale, 1e-6)).astype(np.float32)
                risk_norm = (risk_yuan / max(risk_scale, 1e-6)).astype(np.float32)
                acc_labels += (time.perf_counter() - t0)

                # --- shaping ---
                if sr_cfg.enabled:
                    r_hat_rl_t = -(
                        total_hat_norm * float(total_scale) +
                        float(sr_cfg.lambda_risk) * risk_hat_norm * float(risk_scale)
                    ) * float(rs_scale)
                    r_hat_rl_np = r_hat_rl_t.detach().cpu().numpy().astype(np.float32)
                else:
                    r_hat_rl_np = np.zeros((n_envs,), dtype=np.float32)

                rewards_used = rewards_env_scaled + float(alpha_now) * r_hat_rl_np
                if h.reward_clip is not None:
                    rc = float(h.reward_clip)
                    rewards_used = np.clip(rewards_used, -rc, rc)

                rollout_r_env_mean.append(float(np.mean(rewards_env_scaled)))
                rollout_r_hat_mean.append(float(np.mean(r_hat_rl_np)))
                rollout_r_used_mean.append(float(np.mean(rewards_used)))

                # --- store transition ---
                t0 = time.perf_counter()
                buffer.add(
                    obs=obs_t,
                    gt_H_np=gt_H_raw,
                    action=actions.to(device),
                    reward=torch.from_numpy(rewards_used).to(device=device, dtype=torch.float32),
                    done=torch.from_numpy(dones).to(device=device, dtype=torch.float32),
                    value=values.detach(),
                    logprob=logprobs.detach(),
                )
                buffer.add_sr_labels(
                    cost_total_norm=torch.from_numpy(tot_norm).to(device=device, dtype=torch.float32),
                    cost_risk_norm=torch.from_numpy(risk_norm).to(device=device, dtype=torch.float32),
                )
                acc_buffer += (time.perf_counter() - t0)

                # --- episode bookkeeping & reset burst ---
                done_envs: List[int] = []
                done_seeds: List[Optional[int]] = []
                for i in range(n_envs):
                    ep_return_used[i] += float(rewards_used[i])
                    ep_return_env_scaled[i] += float(rewards_env_scaled[i])
                    ep_return_shaping[i] += float(alpha_now) * float(r_hat_rl_np[i])
                    ep_len[i] += 1

                    if dones[i] > 0.5:
                        episode_count += 1
                        info_i = last_info[i] if isinstance(last_info[i], dict) else {}
                        row = {
                            "episode": int(episode_count),
                            "env_id": int(i),
                            "update": int(update_i),
                            "global_step": int(global_step),
                            "mode": str(mode_str),
                            "alpha": float(alpha_now),
                            "reward_scale": float(rs_scale),
                            "ret_used": float(ep_return_used[i]),
                            "ret_env_scaled": float(ep_return_env_scaled[i]),
                            "ret_shaping": float(ep_return_shaping[i]),
                            "ep_len": int(ep_len[i]),
                            "total_cost_yuan": _get_float(info_i, ["total_cost_yuan"], float("nan")),
                            "loss_term": _get_float(info_i, ["loss_term"], float("nan")),
                            "v_term": _get_float(info_i, ["v_term"], float("nan")),
                            "line_term": _get_float(info_i, ["line_term"], float("nan")),
                            "switch_term": _get_float(info_i, ["switch_term"], float("nan")),
                            "trafo_term": _get_float(info_i, ["trafo_term"], float("nan")),
                            "risk_term": _get_float(info_i, ["risk_term"], float("nan")),
                            "import_cost_yuan": _get_float(info_i, ["import_cost_yuan"], float("nan")),
                            "loss_cost_yuan": _get_float(info_i, ["loss_cost_yuan"], float("nan")),
                            "v_cost_yuan": _get_float(info_i, ["v_cost_yuan"], float("nan")),
                            "line_cost_yuan": _get_float(info_i, ["line_cost_yuan"], float("nan")),
                            "trafo_cost_yuan": _get_float(info_i, ["trafo_cost_yuan"], float("nan")),
                            "risk_cost_yuan_like": _get_float(info_i, ["risk_cost_yuan", "risk_term", "risk_cost"], float("nan")),
                            "switch_cost_yuan": _get_float(info_i, ["switch_cost_yuan"], float("nan")),
                            "pf_failed": int(bool(info_i.get("pf_failed", False))) if isinstance(info_i, dict) else 0,
                            "total_scale_yuan": float(total_scale),
                            "risk_scale_yuan": float(risk_scale),
                        }
                        append_row_csv(episode_csv, row)

                        writer.add_scalar("episode/ret_used", float(ep_return_used[i]), episode_count)
                        writer.add_scalar("episode/ret_env_scaled", float(ep_return_env_scaled[i]), episode_count)
                        writer.add_scalar("episode/ret_shaping", float(ep_return_shaping[i]), episode_count)
                        writer.add_scalar("episode/len", int(ep_len[i]), episode_count)
                        if not math.isnan(row["total_cost_yuan"]):
                            writer.add_scalar("episode/total_cost_yuan", float(row["total_cost_yuan"]), episode_count)
                        writer.add_scalar("scale/total_scale_yuan", float(total_scale), episode_count)
                        writer.add_scalar("scale/risk_scale_yuan", float(risk_scale), episode_count)

                        if h.print_episode_end:
                            logger.info(f"[episode {episode_count:06d}] env={i} ret_used={ep_return_used[i]:.6g} len={int(ep_len[i])} total_cost_yuan={row['total_cost_yuan']}")

                        ep_return_used[i] = 0.0
                        ep_return_env_scaled[i] = 0.0
                        ep_return_shaping[i] = 0.0
                        ep_len[i] = 0

                        done_envs.append(i)
                        done_seeds.append(int(cfg_train.seed + i * 1000 + global_step))

                if done_envs:
                    t0 = time.perf_counter()
                    logger.debug(f"[reset-burst] start n={len(done_envs)} envs={done_envs}")
                    obs_r, gt_r, infos_r = vecenv.reset_many(done_envs, seeds=done_seeds, options=None, who_prefix="reset_after_done")
                    next_obs_raw[np.asarray(done_envs, dtype=np.int64)] = obs_r
                    next_gt_H_raw[np.asarray(done_envs, dtype=np.int64)] = gt_r
                    dt_reset = time.perf_counter() - t0
                    acc_reset += dt_reset

                    reset_secs = [float(d.get("_reset_sec", float("nan"))) for d in infos_r]
                    if reset_secs:
                        rmin = float(np.nanmin(reset_secs))
                        rmax = float(np.nanmax(reset_secs))
                        ravg = float(np.nanmean(reset_secs))
                        logger.debug(f"[reset-burst] end dt={fmt_sec(dt_reset)} reset_sec(min/mean/max)={rmin:.4f}/{ravg:.4f}/{rmax:.4f}")

                obs_raw, gt_H_raw = next_obs_raw, next_gt_H_raw
                global_step += n_envs

                # --- verbose per-step logging ---
                if cfg_train.log_rollout_every_step and ((step + 1) % int(cfg_train.log_rollout_every_step) == 0):
                    dt_step = time.perf_counter() - step_t0
                    ndone = int(np.sum(dones > 0.5))
                    msg = (
                        f"[rollout step {step+1:04d}/{steps_per_env}] "
                        f"dt={fmt_sec(dt_step)} ndone={ndone}/{n_envs} "
                        f"r_env(mean/min/max)={np.mean(rewards_env_scaled):.3e}/{np.min(rewards_env_scaled):.3e}/{np.max(rewards_env_scaled):.3e} "
                        f"r_hat(mean)={np.mean(r_hat_rl_np):.3e} r_used(mean)={np.mean(rewards_used):.3e} "
                        f"scale(total/risk)={total_scale:.3e}/{risk_scale:.3e} alpha={alpha_now:.3e}"
                    )
                    logger.debug(msg)

                if cfg_train.log_action_hist_every_step and ((step + 1) % int(cfg_train.log_action_hist_every_step) == 0):
                    ah = _action_hist(actions.detach().cpu().numpy(), n_actions)
                    logger.trace(f"[rollout step {step+1:04d}] action_hist(top)={ah}")

                if cfg_train.log_info_sample_every_step and infos and ((step + 1) % int(cfg_train.log_info_sample_every_step) == 0):
                    logger.trace(f"[rollout step {step+1:04d}] info[0] keys sample: {_safe_keys_sample(infos[0])}")

                if (not printed_scale_once) and (global_step > n_envs * 128):
                    printed_scale_once = True
                    logger.info(f"[SR scale snapshot] total_scale_yuan≈{total_scale_est.get():.3e} risk_scale_yuan≈{risk_scale_est.get():.3e} reward_scale={cfg_train.reward_scale:g}")

            t_roll = time.perf_counter() - t_roll0
            logger.info(
                f"[rollout done] dt={fmt_sec(t_roll)} "
                f"avg_step_dt={fmt_sec(t_roll/max(1,steps_per_env))} "
                f"(norm={fmt_sec(acc_norm)} forward={fmt_sec(acc_forward)} sr={fmt_sec(acc_sr)} env_step={fmt_sec(acc_env_step)} "
                f"labels={fmt_sec(acc_labels)} buffer={fmt_sec(acc_buffer)} reset={fmt_sec(acc_reset)})"
            )

            # bootstrap value + GAE
            with stage_timer(logger, "compute_gae", "DEBUG"):
                normalizer.update(obs_raw)
                obs_norm = normalizer.normalize(obs_raw)
                with torch.no_grad():
                    last_values = model.forward_with_phi(
                        torch.from_numpy(obs_norm).to(device=device, dtype=torch.float32),
                        torch.from_numpy(gt_H_raw).to(device=device, dtype=torch.float32),
                    )[1]
                buffer.compute_returns_and_advantages(last_values, gamma=h.gamma, gae_lambda=h.gae_lambda)
                if h.normalize_adv:
                    adv = buffer.advantages
                    buffer.advantages = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

            # ---- optimization ----
            policy_losses: List[float] = []
            value_losses: List[float] = []
            entropies: List[float] = []
            approx_kls: List[float] = []
            clip_fracs: List[float] = []
            sr_losses: List[float] = []
            sr_total_losses: List[float] = []
            sr_risk_losses: List[float] = []
            stopped_by_kl = False

            t_opt0 = time.perf_counter()

            if is_warmup:
                sr_epochs = int(max(1, sr_cfg.warmup_sr_epochs))
                logger.info(f"[opt warmup] sr_epochs={sr_epochs} batch_size={h.batch_size}")
                for ep in range(sr_epochs):
                    ep_sr_losses: List[float] = []
                    ep_sr_t: List[float] = []
                    ep_sr_r: List[float] = []
                    bcount = 0

                    for obs_b, gtH_b, act_b, _old_logp_b, _old_val_b, _ret_b, _adv_b, ctot_norm_b, crisk_norm_b in buffer.get_batches_sr(h.batch_size, shuffle=True):
                        bcount += 1
                        if torch.is_tensor(gtH_b) and gtH_b.device.type == "cpu":
                            gtH_b = gtH_b.to(device=device, non_blocking=True)

                        _, _, phi = model.forward_with_phi(obs_b, gtH_b)
                        pred_total_norm, pred_risk_norm = model.sr(phi, act_b)
                        lt, lr, loss_sr = sr_alignment_loss(sr_cfg, pred_total_norm, pred_risk_norm, ctot_norm_b, crisk_norm_b)

                        optimizer.zero_grad(set_to_none=True)
                        loss_sr.backward()

                        if cfg_train.log_grad_norm:
                            with torch.no_grad():
                                gn = 0.0
                                for p in model.sr.parameters():
                                    if p.grad is None:
                                        continue
                                    gn += float(torch.sum(p.grad.detach() ** 2).item())
                                gn = math.sqrt(max(gn, 0.0))
                        else:
                            gn = float("nan")

                        nn.utils.clip_grad_norm_(model.sr.parameters(), float(sr_cfg.grad_clip))
                        optimizer.step()

                        ep_sr_losses.append(float(loss_sr.item()))
                        ep_sr_t.append(float(lt.item()))
                        ep_sr_r.append(float(lr.item()))

                        if cfg_train.log_opt_every_batch and (bcount % int(cfg_train.log_opt_every_batch) == 0):
                            logger.debug(
                                f"[warmup opt] ep={ep+1:02d}/{sr_epochs} batch={bcount:04d} "
                                f"sr_loss={float(loss_sr.item()):.4e} lt={float(lt.item()):.4e} lr={float(lr.item()):.4e} "
                                f"grad_norm(sr)={gn:.4e}"
                            )

                    sr_losses.extend(ep_sr_losses)
                    sr_total_losses.extend(ep_sr_t)
                    sr_risk_losses.extend(ep_sr_r)

                    logger.info(
                        f"[warmup opt] ep={ep+1:02d}/{sr_epochs} "
                        f"sr_loss_mean={float(np.mean(ep_sr_losses)):.4e} "
                        f"lt_mean={float(np.mean(ep_sr_t)):.4e} lr_mean={float(np.mean(ep_sr_r)):.4e} "
                        f"batches={bcount}"
                    )

            else:
                logger.info(f"[opt joint] n_epochs={h.n_epochs} batch_size={h.batch_size} target_kl={h.target_kl}")
                for epoch in range(int(h.n_epochs)):
                    epoch_kls: List[float] = []
                    epoch_clip: List[float] = []
                    bcount = 0

                    for obs_b, gtH_b, act_b, old_logp_b, old_val_b, ret_b, adv_b, ctot_norm_b, crisk_norm_b in buffer.get_batches_sr(h.batch_size, shuffle=True):
                        bcount += 1
                        if torch.is_tensor(gtH_b) and gtH_b.device.type == "cpu":
                            gtH_b = gtH_b.to(device=device, non_blocking=True)

                        logits, value, phi = model.forward_with_phi(obs_b, gtH_b)
                        dist = torch.distributions.Categorical(logits=logits)
                        logp = dist.log_prob(act_b)
                        entropy = dist.entropy().mean()

                        ratio = torch.exp(logp - old_logp_b)
                        clip_frac = torch.mean((torch.abs(ratio - 1.0) > h.clip_range).float()).item()

                        pg_loss1 = -adv_b * ratio
                        pg_loss2 = -adv_b * torch.clamp(ratio, 1.0 - h.clip_range, 1.0 + h.clip_range)
                        policy_loss = torch.mean(torch.max(pg_loss1, pg_loss2))

                        if h.clip_range_vf is not None:
                            v_clipped = old_val_b + torch.clamp(value - old_val_b, -h.clip_range_vf, h.clip_range_vf)
                            v_loss1 = (value - ret_b).pow(2)
                            v_loss2 = (v_clipped - ret_b).pow(2)
                            value_loss = 0.5 * torch.mean(torch.max(v_loss1, v_loss2))
                        else:
                            value_loss = 0.5 * F.mse_loss(value, ret_b)

                        pred_total_norm, pred_risk_norm = model.sr(phi, act_b)
                        lt, lr, loss_sr = sr_alignment_loss(sr_cfg, pred_total_norm, pred_risk_norm, ctot_norm_b, crisk_norm_b)

                        loss = policy_loss + h.vf_coef * value_loss - ent_coef_now * entropy + loss_sr

                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()

                        if cfg_train.log_grad_norm:
                            with torch.no_grad():
                                gn = 0.0
                                for p in model.parameters():
                                    if p.grad is None:
                                        continue
                                    gn += float(torch.sum(p.grad.detach() ** 2).item())
                                gn = math.sqrt(max(gn, 0.0))
                        else:
                            gn = float("nan")

                        nn.utils.clip_grad_norm_(model.parameters(), h.max_grad_norm)
                        optimizer.step()

                        with torch.no_grad():
                            approx_kl = torch.mean(old_logp_b - logp).item()

                        policy_losses.append(float(policy_loss.item()))
                        value_losses.append(float(value_loss.item()))
                        entropies.append(float(entropy.item()))
                        approx_kls.append(float(approx_kl))
                        clip_fracs.append(float(clip_frac))
                        sr_losses.append(float(loss_sr.item()))
                        sr_total_losses.append(float(lt.item()))
                        sr_risk_losses.append(float(lr.item()))

                        epoch_kls.append(float(approx_kl))
                        epoch_clip.append(float(clip_frac))

                        if cfg_train.log_opt_every_batch and (bcount % int(cfg_train.log_opt_every_batch) == 0):
                            logger.debug(
                                f"[joint opt] epoch={epoch+1:02d}/{h.n_epochs} batch={bcount:04d} "
                                f"loss={float(loss.item()):.4e} pi={float(policy_loss.item()):.4e} v={float(value_loss.item()):.4e} "
                                f"ent={float(entropy.item()):.4e} kl={float(approx_kl):.4e} clip={float(clip_frac):.3f} "
                                f"sr={float(loss_sr.item()):.4e} (lt={float(lt.item()):.4e} lr={float(lr.item()):.4e}) "
                                f"grad_norm(all)={gn:.4e}"
                            )

                    if h.print_ppo_epoch and epoch_kls:
                        logger.info(f"[ppo epoch] {epoch+1:02d}/{h.n_epochs} kl_mean={float(np.mean(epoch_kls)):.4f} clip_mean={float(np.mean(epoch_clip)):.3f} batches={bcount}")

                    if h.target_kl is not None and len(epoch_kls) > 0:
                        if float(np.mean(epoch_kls)) > 1.5 * float(h.target_kl):
                            stopped_by_kl = True
                            logger.warn(f"[early stop] epoch={epoch+1:02d} mean_kl={float(np.mean(epoch_kls)):.6g} > 1.5*target_kl={1.5*float(h.target_kl):.6g}")
                            break

            t_opt = time.perf_counter() - t_opt0

            # ---- summary & ETA ----
            elapsed_total = time.perf_counter() - wall0
            frac_done = min(1.0, global_step / max(1, h.total_steps))
            fps = float(h.n_steps / max(time.perf_counter() - t_up0, 1e-6))

            sr_loss_m = float(np.mean(sr_losses)) if sr_losses else 0.0
            sr_t_m = float(np.mean(sr_total_losses)) if sr_total_losses else 0.0
            sr_r_m = float(np.mean(sr_risk_losses)) if sr_risk_losses else 0.0
            pi_loss_m = float(np.mean(policy_losses)) if policy_losses else float("nan")
            v_loss_m = float(np.mean(value_losses)) if value_losses else float("nan")
            ent_m = float(np.mean(entropies)) if entropies else float("nan")
            kl_m = float(np.mean(approx_kls)) if approx_kls else float("nan")
            clip_m = float(np.mean(clip_fracs)) if clip_fracs else float("nan")

            logger.info(
                f"[update {update_i:04d}] mode={mode_str} dt_rollout={fmt_sec(t_roll)} dt_opt={fmt_sec(t_opt)} fps={fps:.1f} | {eta_str(elapsed_total, frac_done)}"
            )
            logger.info(
                f"[update {update_i:04d}] rollout stats: r_env_mean={np.mean(rollout_r_env_mean):.3e} "
                f"r_hat_mean={np.mean(rollout_r_hat_mean):.3e} r_used_mean={np.mean(rollout_r_used_mean):.3e} "
                f"alpha={alpha_now:.3e} sr_loss_mean={sr_loss_m:.4e} scale(total/risk)={total_scale_est.get():.3e}/{risk_scale_est.get():.3e}"
            )
            if not is_warmup:
                logger.info(
                    f"[update {update_i:04d}] ppo stats: pi={pi_loss_m:.4f} v={v_loss_m:.4f} ent={ent_m:.4f} "
                    f"kl={kl_m:.4f} clip={clip_m:.3f} stopped_by_kl={stopped_by_kl}"
                )

            # ---- log SR training curve (CSV + TB) ----
            cur_total_scale = total_scale_est.get()
            cur_risk_scale = risk_scale_est.get()

            append_row_csv(sr_csv, {
                "update": int(update_i),
                "global_step": int(global_step),
                "mode": str(mode_str),
                "alpha": float(alpha_now),
                "reward_scale": float(cfg_train.reward_scale),
                "sr_loss": float(sr_loss_m),
                "sr_total_loss": float(sr_t_m),
                "sr_risk_loss": float(sr_r_m),
                "total_scale_yuan": float(cur_total_scale),
                "risk_scale_yuan": float(cur_risk_scale),
                "r_env_mean": float(np.mean(rollout_r_env_mean) if rollout_r_env_mean else 0.0),
                "r_hat_mean": float(np.mean(rollout_r_hat_mean) if rollout_r_hat_mean else 0.0),
                "r_used_mean": float(np.mean(rollout_r_used_mean) if rollout_r_used_mean else 0.0),
                "dt_rollout_sec": float(t_roll),
                "dt_opt_sec": float(t_opt),
                "fps": float(fps),
            })

            writer.add_scalar("sr/loss", float(sr_loss_m), update_i)
            writer.add_scalar("sr/loss_total", float(sr_t_m), update_i)
            writer.add_scalar("sr/loss_risk", float(sr_r_m), update_i)
            writer.add_scalar("sr/alpha", float(alpha_now), update_i)
            writer.add_scalar("rollout/r_env_mean", float(np.mean(rollout_r_env_mean)), update_i)
            writer.add_scalar("rollout/r_hat_mean", float(np.mean(rollout_r_hat_mean)), update_i)
            writer.add_scalar("rollout/r_used_mean", float(np.mean(rollout_r_used_mean)), update_i)
            writer.add_scalar("scale/total_scale_yuan_update", float(cur_total_scale), update_i)
            writer.add_scalar("scale/risk_scale_yuan_update", float(cur_risk_scale), update_i)
            writer.add_scalar("time/dt_rollout_sec", float(t_roll), update_i)
            writer.add_scalar("time/dt_opt_sec", float(t_opt), update_i)
            writer.add_scalar("time/fps", float(fps), update_i)

            # ---- import log (incremental) ----
            if cfg_train.import_log_every_update and (update_i % int(cfg_train.import_log_every_update) == 0):
                import_logger.dump(tag=f"update_{update_i:04d}", only_new=True)

            # ---- save ckpt (do not overwrite) ----
            if h.save_every_update and (update_i % int(h.save_every_update) == 0):
                ckpt_path = os.path.join(cfg_train.out_dir, f"ppo_gt_sr_merged_ckpt_update{update_i:04d}.pt")
                with stage_timer(logger, "save_checkpoint", "INFO", extra=f"path={ckpt_path}"):
                    save_checkpoint_sr(
                        ckpt_path,
                        model,
                        optimizer,
                        normalizer,
                        global_step,
                        update_i,
                        episode_count,
                        train_meta={
                            "cfg_train": asdict(cfg_train),
                            "sr_cfg": asdict(sr_cfg),
                            "alpha_now": float(alpha_now),
                            "mode": mode_str,
                            "gt_ckpt_path": cfg_train.gt_ckpt_path,
                            "total_scale_yuan": float(cur_total_scale),
                            "risk_scale_yuan": float(cur_risk_scale),
                        },
                    )

            if global_step >= h.total_steps:
                logger.info("global_step reached total_steps; breaking training loop.")
                break

        final_path = os.path.join(cfg_train.out_dir, "ppo_gt_sr_merged_ckpt_final.pt")
        with stage_timer(logger, "final_save_checkpoint", "INFO", extra=f"path={final_path}"):
            save_checkpoint_sr(
                final_path,
                model,
                optimizer,
                normalizer,
                global_step,
                update_i,
                episode_count,
                train_meta={
                    "cfg_train": asdict(cfg_train),
                    "sr_cfg": asdict(sr_cfg),
                    "alpha_now": float(compute_alpha(update_i, sr_cfg)),
                    "mode": "FINAL",
                    "gt_ckpt_path": cfg_train.gt_ckpt_path,
                    "total_scale_yuan": float(total_scale_est.get()),
                    "risk_scale_yuan": float(risk_scale_est.get()),
                },
            )

        logger.info(f"training finished: updates={update_i} global_step={global_step} episode_count={episode_count} total_time={fmt_sec(time.perf_counter()-wall0)}")

    finally:
        logger.warn("entering finally: closing vecenv/writer.")
        try:
            vecenv.close()
        except Exception as e:
            logger.error(f"vecenv.close() failed: {repr(e)}")
        try:
            writer.flush()
            writer.close()
        except Exception as e:
            logger.error(f"writer close failed: {repr(e)}")

# ============================================================
# entry (edit params here)
# ============================================================

if __name__ == "__main__":
    # -------------------------
    # 核心参数（按你的现状设置）
    # -------------------------
    GT_CKPT_PATH = "./GTransformer/runs/pretrain_masked_node_only_v1/ckpt_best.pt"

    cfg_train = TrainConfig(
        seed=0,
        device="cuda",
        time_scale=1e-3,
        obs_clip=10.0,

        reward_scale=1e-6,

        rollout_debug_every=64,
        start_method="spawn",
        gt_ckpt_path=GT_CKPT_PATH,
        gt_lr=1e-5,
        gt_train=True,
        rollout_inference_mode=True,
        allow_tf32=True,
        hparams=PPOHyperParams(
            total_steps=2048 * 100,
            n_steps=2048,
            n_envs=64,
            n_epochs=10,
            batch_size=256,
            lr=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=0.2,
            ent_coef=0.01,
            ent_coef_final=0.003,
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=0.05,
            reward_clip=None,
            lr_anneal=True,
            normalize_adv=True,
            save_every_update=10,
            rollout_print_every=0,
            print_episode_end=True,
            print_ppo_epoch=False,
        ),
        model=ModelConfig(
            enc_hidden=256 * 4,
            fusion_hidden=256 * 4,
            fusion_blocks=2,
            dropout=0.0,
            policy_temperature=1.0,
            emb_dim_bus=128 * 4,
            emb_dim_load=128 * 4,
            emb_dim_fcst=128 * 4,
            emb_dim_time=32 * 4,
            topo_emb_dim=32 * 4,
            gt_proj_dim=128,
            gt_pool="mean",
            adj_cache_size=64,
            adj_cache_cuda_size=16,
        ),
        out_dir="./runs_ppo_gt_sr_merged",
        resume_path=None,

        # ========= 日志参数：默认尽可能详细 =========
        log_level=3,                 # 3=DEBUG；如果要更极端可设为 4=TRACE
        log_rollout_every_step=1,    # 每 step 打印一行（非常多）
        log_action_hist_every_step=8,
        log_opt_every_batch=10,
        log_grad_norm=True,
        log_info_sample_every_step=16,
        run_log_name="run_log.txt",
    )

    sr_cfg = SRConfig(
        enabled=True,
        warmup_updates=20,
        warmup_sr_epochs=5,
        alpha_final=0.2,
        alpha_ramp_updates=30,
        lambda_risk=0.0,
        loss_type="huber",
        huber_delta=1.0,
        lr=1e-4,
        grad_clip=10.0,
        total_scale_init_yuan=1.0e6,
        risk_scale_init_yuan=1.0e5,
        scale_q=0.95,
        scale_ema=0.98,
        scale_min=1.0,
        act_emb_dim=64,
        hidden=256,
        dropout=0.0,
        print_every_update=1,
    )

    if _in_notebook() and os.name == "posix" and cfg_train.start_method == "spawn":
        print("[warn] Detected notebook environment; spawn often fails. Auto-switch start_method to 'fork' for this run.")
        cfg_train.start_method = "fork"

    try:
        mp.set_start_method(cfg_train.start_method, force=True)
    except RuntimeError:
        pass

    train(cfg_train, sr_cfg)
