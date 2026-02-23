# In[]
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation / validation script for the PPO power-dispatch RL agent (NO argparse版).

你需要做的事情
--------------
在本文件顶部的“USER CONFIG”区域，直接修改变量即可运行。
不使用 argparse；所有参数都通过独立变量显式定义。

功能
----
1) 完整交互评估：reset -> step -> done，支持 deterministic(argmax) / stochastic(sample)
2) 指标统计：回合回报、PF失败率、loss/viol/switch/trafo等均值
3) 性能曲线：
   - 优先读取 run_dir 下的 eval_update.csv / train_update.csv
   - 否则遍历 ppo_ckpt_updateXXXX.pt 并评估后绘图
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt


# ======================================================================================
# USER CONFIG 
# ======================================================================================

# ======================================================================================
# USER CONFIG
# ======================================================================================

# 训练运行目录：包含 checkpoints / logs / train_meta.json
RUN_DIR = "./runs_ppo_mp_multienc_251231"

# checkpoint 路径：
# - ""：自动选择 RUN_DIR/ppo_ckpt_final.pt，若不存在则选最大 update 的 ppo_ckpt_update*.pt
# - 或者写相对路径（相对 RUN_DIR）/绝对路径
CKPT = ""

# 训练脚本路径（用于动态 import：Env/Flattener/Normalizer/Model/TimeseriesSchema）
# - 相对路径：优先从 RUN_DIR 下查找，其次从本文件所在目录查找
TRAIN_SCRIPT = "train_ppo_power_dispatch_multiproc_multiencoder.py"

# 设备： "auto" | "cpu" | "cuda" | "cuda:0"
DEVICE = "auto"

# 每次评估跑多少个 episode
EPISODES = 20

# True=argmax；False=从 Categorical(logits) 采样
DETERMINISTIC = True

# 评估 episode 的 base seed（每个 episode 会在此基础上加偏移）
BASE_SEED = 123

# 是否额外跑一个 verbose episode，并保存 trace JSON
INTERACTIVE_EPISODE = False
TRACE_PATH = ""  # 留空则保存到 RUN_DIR/interactive_episode_trace.json

# 性能曲线来源： "auto" | "logs" | "ckpt"
CURVE_FROM = "auto"

# 当 CURVE_FROM="logs" 时：画哪一列
# - eval_update.csv 常用：eval_return_mean / eval_pf_failed_rate ...
# - train_update.csv 常用：recent_ep_return_mean ...
METRIC = "eval_return_mean"

# 保存曲线图（相对路径默认在 RUN_DIR 下）
SAVE_PLOT = "eval_curve.png"

# 当 CURVE_FROM="ckpt" 时：保存逐 checkpoint 评估得到的曲线 CSV（相对路径默认在 RUN_DIR 下）
SAVE_CURVE_CSV = "eval_curve_from_ckpt.csv"



# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

def _abs(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv_rows(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(dict(r))
    return rows


def _try_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def import_module_from_path(py_path: str, module_name: str = "_train_module") -> Any:
    """
    Import a python module by file path, without requiring it to be on sys.path.

    This is used so that the evaluation script can reuse the *exact* model and
    preprocessing classes defined in the training script.
    """
    py_path = _abs(py_path)
    if not os.path.exists(py_path):
        raise FileNotFoundError(f"Training script not found: {py_path}")
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from: {py_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod   # 或你自定义的 module_name
    spec.loader.exec_module(mod)
    return mod



def find_latest_checkpoint(run_dir: str) -> str:
    """
    Prefer ppo_ckpt_final.pt; otherwise pick the maximum update checkpoint.
    """
    run_dir = _abs(run_dir)
    final_path = os.path.join(run_dir, "ppo_ckpt_final.pt")
    if os.path.exists(final_path):
        return final_path

    pat = re.compile(r"ppo_ckpt_update(\d+)\.pt$")
    best_u = -1
    best_p = None
    for fn in os.listdir(run_dir):
        m = pat.match(fn)
        if not m:
            continue
        u = int(m.group(1))
        if u > best_u:
            best_u = u
            best_p = os.path.join(run_dir, fn)
    if best_p is None:
        raise FileNotFoundError(f"No checkpoint found under: {run_dir}")
    return best_p


def list_update_checkpoints(run_dir: str) -> List[Tuple[int, str]]:
    """
    Return sorted list of (update, ckpt_path) for ppo_ckpt_update*.pt under run_dir.
    """
    run_dir = _abs(run_dir)
    pat = re.compile(r"ppo_ckpt_update(\d+)\.pt$")
    items: List[Tuple[int, str]] = []
    for fn in os.listdir(run_dir):
        m = pat.match(fn)
        if not m:
            continue
        items.append((int(m.group(1)), os.path.join(run_dir, fn)))
    items.sort(key=lambda x: x[0])
    return items


def get_device(device_str: str) -> torch.device:
    if device_str.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


# --------------------------------------------------------------------------------------
# Environment + preprocessing compatibility helpers
# --------------------------------------------------------------------------------------

def ensure_topology_id(
    obs: Dict[str, Any],
    *,
    info: Optional[Dict[str, Any]] = None,
    last_action: Optional[int] = None
) -> Dict[str, Any]:
    """
    Training code's ObsFlattenerV2 expects obs['topology_id'].
    In some env versions, 'topology_id' is not part of the observation dict.
    This helper injects it deterministically, using info['action'] if available,
    otherwise last_action, otherwise 0.

    NOTE: If your env already provides topology_id, this function is a no-op.
    """
    if "topology_id" in obs:
        return obs
    topo = None
    if info is not None:
        topo = info.get("action", None)
    if topo is None:
        topo = last_action
    if topo is None:
        topo = 0
    out = dict(obs)
    out["topology_id"] = np.array([int(topo)], dtype=np.int32)
    return out


# --------------------------------------------------------------------------------------
# Core evaluation
# --------------------------------------------------------------------------------------

@torch.no_grad()
def rollout_episode(
    env: Any,
    flattener: Any,
    normalizer: Any,
    model: torch.nn.Module,
    device: torch.device,
    *,
    seed: int,
    deterministic: bool,
    max_steps: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run one episode and return detailed metrics, including step-by-step info if verbose.
    """
    obs, info = env.reset(seed=int(seed), options=None)
    last_action: Optional[int] = None

    obs = ensure_topology_id(obs, info=info, last_action=last_action)
    x = flattener.flatten(obs)
    x = normalizer.normalize(x)

    ep_ret = 0.0
    ep_len = 0

    # step-level parts
    parts_loss: List[float] = []
    parts_v: List[float] = []
    parts_line: List[float] = []
    parts_sw: List[float] = []
    parts_tf: List[float] = []
    step_trace: List[Dict[str, Any]] = []

    done = False
    while not done:
        if max_steps is not None and ep_len >= int(max_steps):
            break

        obs_t = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
        logits, _v = model(obs_t)

        if deterministic:
            a = torch.argmax(logits, dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()

        action = int(a.item())
        last_action = action

        obs, r, term, trunc, info = env.step(action)
        done = bool(term or trunc)

        ep_ret += float(r)
        ep_len += 1

        # collect per-step metrics (env provides these in info)
        parts_loss.append(float(info.get("loss_mw", 0.0)))
        parts_v.append(float(info.get("v_viol", 0.0)))
        parts_line.append(float(info.get("line_viol", 0.0)))
        parts_sw.append(float(info.get("switch_cost", 0.0)))
        parts_tf.append(float(info.get("trafo_balance", 0.0)))

        if verbose:
            step_trace.append(
                {
                    "t": int(info.get("t", ep_len - 1)),
                    "action": int(action),
                    "reward": float(info.get("reward", r)),
                    "pf_failed": bool(info.get("pf_failed", False)),
                    "loss_mw": _try_float(info.get("loss_mw", float("nan"))),
                    "v_viol": _try_float(info.get("v_viol", float("nan"))),
                    "line_viol": _try_float(info.get("line_viol", float("nan"))),
                    "switch_cost": _try_float(info.get("switch_cost", float("nan"))),
                    "trafo_balance": _try_float(info.get("trafo_balance", float("nan"))),
                }
            )

        obs = ensure_topology_id(obs, info=info, last_action=last_action)
        x = flattener.flatten(obs)
        x = normalizer.normalize(x)

    def _m(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    return {
        "ep_return": float(ep_ret),
        "ep_len": int(ep_len),
        "pf_failed": float(bool(info.get("pf_failed", False))) if isinstance(info, dict) else 0.0,
        "loss_mw_mean": _m(parts_loss),
        "v_viol_mean": _m(parts_v),
        "line_viol_mean": _m(parts_line),
        "switch_cost_mean": _m(parts_sw),
        "trafo_balance_mean": _m(parts_tf),
        "trace": step_trace,
    }


@torch.no_grad()
def evaluate_policy(
    env: Any,
    flattener: Any,
    normalizer: Any,
    model: torch.nn.Module,
    device: torch.device,
    *,
    n_episodes: int,
    deterministic: bool = True,
    base_seed: int = 123,
) -> Dict[str, float]:
    """
    Multi-episode evaluation and aggregate metrics (mean/std).
    """
    rets: List[float] = []
    lens: List[int] = []
    pf_failed: List[float] = []
    loss_mw_mean: List[float] = []
    v_viol_mean: List[float] = []
    line_viol_mean: List[float] = []
    switch_cost_mean: List[float] = []
    trafo_balance_mean: List[float] = []

    for ep in range(int(n_episodes)):
        m = rollout_episode(
            env,
            flattener,
            normalizer,
            model,
            device,
            seed=base_seed + ep * 1000,
            deterministic=deterministic,
            verbose=False,
        )
        rets.append(float(m["ep_return"]))
        lens.append(int(m["ep_len"]))
        pf_failed.append(float(m["pf_failed"]))
        loss_mw_mean.append(float(m["loss_mw_mean"]))
        v_viol_mean.append(float(m["v_viol_mean"]))
        line_viol_mean.append(float(m["line_viol_mean"]))
        switch_cost_mean.append(float(m["switch_cost_mean"]))
        trafo_balance_mean.append(float(m["trafo_balance_mean"]))

    return {
        "eval_return_mean": float(np.mean(rets)) if rets else float("nan"),
        "eval_return_std": float(np.std(rets)) if rets else float("nan"),
        "eval_len_mean": float(np.mean(lens)) if lens else float("nan"),
        "eval_pf_failed_rate": float(np.mean(pf_failed)) if pf_failed else float("nan"),
        "eval_loss_mw_mean": float(np.mean(loss_mw_mean)) if loss_mw_mean else float("nan"),
        "eval_v_viol_mean": float(np.mean(v_viol_mean)) if v_viol_mean else float("nan"),
        "eval_line_viol_mean": float(np.mean(line_viol_mean)) if line_viol_mean else float("nan"),
        "eval_switch_cost_mean": float(np.mean(switch_cost_mean)) if switch_cost_mean else float("nan"),
        "eval_trafo_balance_mean": float(np.mean(trafo_balance_mean)) if trafo_balance_mean else float("nan"),
    }


# --------------------------------------------------------------------------------------
# Checkpoint / run loading
# --------------------------------------------------------------------------------------

def _torch_load_trusted(path: str, device: torch.device) -> Dict[str, Any]:
    # PyTorch 2.6+ may default weights_only=True; if you trust the ckpt, use weights_only=False.
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _restore_env_cfg_dict(train_mod, env_cfg_dict):
    d = dict(env_cfg_dict)
    ts_schema = d.get("timeseries_schema", None)
    if isinstance(ts_schema, dict):
        d["timeseries_schema"] = train_mod.TimeseriesSchema(**ts_schema)
    return train_mod.EnvConfig(**d)

def build_env_and_model_from_run(
    train_mod: Any,
    *,
    run_dir: str,
    ckpt_path: str,
    device: torch.device,
) -> Tuple[Any, Any, Any, torch.nn.Module, Dict[str, Any]]:
    """
    Reconstruct env/flattener/normalizer/model using training metadata + checkpoint.
    """
    run_dir = _abs(run_dir)
    ckpt_path = _abs(ckpt_path)

    # -------- read meta if exists --------
    meta_path = os.path.join(run_dir, "train_meta.json")
    meta: Dict[str, Any] = {}
    if os.path.exists(meta_path):
        meta = _read_json(meta_path)

    # -------- build env --------
    # Prefer using env_cfg recorded during training; otherwise EnvConfig() defaults.
    env_cfg_obj = None
    if "env_cfg" in meta and isinstance(meta["env_cfg"], dict):
        try:
            env_cfg_obj = _restore_env_cfg_dict(train_mod, meta["env_cfg"])
        except Exception:
            env_cfg_obj = None

    env = train_mod.PowerDispatchEnv(env_cfg_obj)

    # -------- dimensions --------
    n_actions = int(meta.get("n_actions", getattr(env, "n_actions", env.action_space.n)))
    time_scale = float(meta.get("train_cfg", {}).get("time_scale", 1.0))
    time_period = int(meta.get("env_cfg", {}).get("episode_len", getattr(env.cfg, "episode_len", 24)))

    flattener = train_mod.ObsFlattenerV2(
        env.observation_space,
        n_actions=n_actions,
        time_scale=time_scale,
        time_period=time_period
    )

    # normalization mask: do NOT normalize time features nor topology_id (same as training)
    mask = np.ones((int(flattener.flat_dim),), dtype=np.bool_)
    for k in ("time_feat", "topology_id"):
        if k in flattener.slices:
            sl = flattener.slices[k]
            mask[sl] = False

    normalizer = train_mod.MaskedObsNormalizer(
        int(flattener.flat_dim),
        mask=mask,
        clip=float(meta.get("train_cfg", {}).get("obs_clip", 10.0))
    )

    # -------- model --------
    # Build model using recorded slices if available; otherwise reuse flattener.slices.
    slices = flattener.slices
    if "slices" in meta and isinstance(meta["slices"], dict):
        try:
            slices = {k: slice(int(v[0]), int(v[1])) for k, v in meta["slices"].items()}
        except Exception:
            slices = flattener.slices

    # Prefer recorded model hyperparams from train_cfg.model, if present.
    model_kwargs: Dict[str, Any] = dict(
        obs_dim=int(flattener.flat_dim),
        n_actions=int(n_actions),
        slices=slices,
    )
    if isinstance(meta.get("train_cfg", {}).get("model", None), dict):
        model_kwargs.update(meta["train_cfg"]["model"])

    model = train_mod.MultiBranchActorCritic(**model_kwargs).to(device)
    model.eval()

    # -------- load checkpoint --------
    ckpt = _torch_load_trusted(ckpt_path, device=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        # If user saved torchscript or raw state dict, try to load directly
        model.load_state_dict(ckpt)

    if isinstance(ckpt, dict) and isinstance(ckpt.get("normalizer", None), dict):
        normalizer.load_state_dict(ckpt["normalizer"])

    extra = {
        "ckpt": ckpt,
        "meta": meta,
    }
    return env, flattener, normalizer, model, extra


# --------------------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------------------

def plot_curve(
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    save_path: str,
    ys2: Optional[Sequence[float]] = None,
    ylabel2: Optional[str] = None,
) -> None:
    """
    Single figure, optionally with a secondary y-axis.
    """
    if len(xs) == 0:
        raise ValueError("Empty curve data; cannot plot.")

    plt.figure()
    ax1 = plt.gca()
    ax1.plot(xs, ys, marker="o")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)

    if ys2 is not None:
        ax2 = ax1.twinx()
        ax2.plot(xs, list(ys2), marker="x")
        ax2.set_ylabel(ylabel2 or "metric2")

    _ensure_dir(save_path)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


# --------------------------------------------------------------------------------------
# Main (no argparse)
# --------------------------------------------------------------------------------------

def main() -> None:
    run_dir = _abs(RUN_DIR)
    device = get_device(DEVICE)

    # Ensure repo root is on sys.path (helpful if env depends on repo-local packages)
    repo_root = os.path.abspath(os.path.dirname(__file__))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # Import training module for shared definitions
    train_script_path = TRAIN_SCRIPT
    if not os.path.isabs(train_script_path):
        # prefer from run_dir; fallback to current directory
        cand1 = os.path.join(run_dir, train_script_path)
        cand2 = os.path.join(repo_root, train_script_path)
        train_script_path = cand1 if os.path.exists(cand1) else cand2
    train_mod = import_module_from_path(train_script_path, module_name="_ppo_train_mod")

    # Resolve checkpoint
    ckpt_path = CKPT.strip()
    if ckpt_path:
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(run_dir, ckpt_path)
        ckpt_path = _abs(ckpt_path)
    else:
        ckpt_path = find_latest_checkpoint(run_dir)

    # -----------------------------
    # 1) Evaluate a single checkpoint (always)
    # -----------------------------
    env, flattener, normalizer, model, extra = build_env_and_model_from_run(
        train_mod, run_dir=run_dir, ckpt_path=ckpt_path, device=device
    )

    ckpt_update = -1
    ckpt_step = -1
    if isinstance(extra.get("ckpt", None), dict):
        ckpt_update = int(extra["ckpt"].get("update", -1))
        ckpt_step = int(extra["ckpt"].get("step", -1))
    print(f"[load] ckpt={ckpt_path} | update={ckpt_update} step={ckpt_step} | device={device}")

    metrics = evaluate_policy(
        env, flattener, normalizer, model, device,
        n_episodes=int(EPISODES),
        deterministic=bool(DETERMINISTIC),
        base_seed=int(BASE_SEED),
    )
    print("[eval] " + " ".join([f"{k}={v:.6g}" for k, v in metrics.items()]))

    # Optional interactive episode trace
    if bool(INTERACTIVE_EPISODE):
        trace = rollout_episode(
            env, flattener, normalizer, model, device,
            seed=int(BASE_SEED) + 999_999,
            deterministic=bool(DETERMINISTIC),
            verbose=True,
        )
        trace_path = TRACE_PATH.strip()
        if not trace_path:
            trace_path = os.path.join(run_dir, "interactive_episode_trace.json")
        if not os.path.isabs(trace_path):
            trace_path = os.path.join(run_dir, trace_path)
        trace_path = _abs(trace_path)
        _ensure_dir(trace_path)
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(trace, f, ensure_ascii=False, indent=2)
        print(f"[trace] saved interactive trace to: {trace_path} (steps={len(trace.get('trace', []))})")

    # -----------------------------
    # 2) Build / plot performance curve
    # -----------------------------
    curve_from = CURVE_FROM
    eval_csv = os.path.join(run_dir, "eval_update.csv")
    train_csv = os.path.join(run_dir, "train_update.csv")

    if curve_from == "auto":
        if os.path.exists(eval_csv) or os.path.exists(train_csv):
            curve_from = "logs"
        else:
            curve_from = "ckpt"

    save_plot = SAVE_PLOT
    if not os.path.isabs(save_plot):
        save_plot = os.path.join(run_dir, save_plot)
    save_plot = _abs(save_plot)

    if curve_from == "logs":
        # Prefer eval_update.csv (clean eval metrics); fallback to train_update.csv
        src = eval_csv if os.path.exists(eval_csv) else train_csv
        rows = _read_csv_rows(src)
        if not rows:
            raise RuntimeError(f"No rows found in CSV: {src}")

        xs = [int(r.get("update", 0)) for r in rows]
        metric_key = METRIC
        ys = [_try_float(r.get(metric_key, float("nan"))) for r in rows]

        # Secondary metric heuristic for eval logs: pf_failed_rate
        ys2 = None
        ylabel2 = None
        if src.endswith("eval_update.csv") and "eval_pf_failed_rate" in rows[0] and metric_key != "eval_pf_failed_rate":
            ys2 = [_try_float(r.get("eval_pf_failed_rate", float("nan"))) for r in rows]
            ylabel2 = "eval_pf_failed_rate"

        plot_curve(
            xs, ys,
            xlabel="update",
            ylabel=metric_key,
            title=f"Performance curve from logs ({os.path.basename(src)})",
            save_path=save_plot,
            ys2=ys2,
            ylabel2=ylabel2,
        )
        print(f"[plot] saved plot to: {save_plot}")

    elif curve_from == "ckpt":
        items = list_update_checkpoints(run_dir)
        if not items:
            raise RuntimeError(f"No ppo_ckpt_update*.pt found under: {run_dir}")

        curve_rows: List[Dict[str, Any]] = []
        xs_u: List[int] = []
        ys_ret: List[float] = []
        ys_pf: List[float] = []

        # Evaluate each checkpoint in order
        for upd, p in items:
            env_i, flattener_i, normalizer_i, model_i, _extra_i = build_env_and_model_from_run(
                train_mod, run_dir=run_dir, ckpt_path=p, device=device
            )
            m = evaluate_policy(
                env_i, flattener_i, normalizer_i, model_i, device,
                n_episodes=int(EPISODES),
                deterministic=True,  # curve 默认 deterministic，便于可比
                base_seed=int(BASE_SEED),
            )
            row = {"update": int(upd), **m}
            curve_rows.append(row)

            xs_u.append(int(upd))
            ys_ret.append(float(m["eval_return_mean"]))
            ys_pf.append(float(m["eval_pf_failed_rate"]))

            print(f"[curve] update={upd:04d} eval_return_mean={m['eval_return_mean']:.6g} pf_failed_rate={m['eval_pf_failed_rate']:.6g}")

        # Save curve CSV
        save_curve_csv = SAVE_CURVE_CSV
        if not os.path.isabs(save_curve_csv):
            save_curve_csv = os.path.join(run_dir, save_curve_csv)
        save_curve_csv = _abs(save_curve_csv)
        _ensure_dir(save_curve_csv)
        with open(save_curve_csv, "w", encoding="utf-8", newline="") as f:
            fieldnames = list(curve_rows[0].keys())
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in curve_rows:
                w.writerow(r)
        print(f"[curve] saved evaluated curve CSV to: {save_curve_csv}")

        # Plot
        plot_curve(
            xs_u, ys_ret,
            xlabel="update",
            ylabel="eval_return_mean",
            title="Performance curve from checkpoint evaluation",
            save_path=save_plot,
            ys2=ys_pf,
            ylabel2="eval_pf_failed_rate",
        )
        print(f"[plot] saved plot to: {save_plot}")

    else:
        raise ValueError(f"Unknown CURVE_FROM={CURVE_FROM!r} (expected 'auto'|'logs'|'ckpt')")


if __name__ == "__main__":
    main()

# In[]
import numpy as np
import torch

def _summarize_array(a: np.ndarray, name: str = "", max_show: int = 6) -> str:
    a = np.asarray(a)
    flat = a.reshape(-1)
    if flat.size == 0:
        return f"{name}: shape={a.shape} empty"
    head = flat[:max_show]
    return (
        f"{name}: shape={a.shape} dtype={a.dtype} "
        f"min={float(np.min(flat)):.4g} max={float(np.max(flat)):.4g} mean={float(np.mean(flat)):.4g} "
        f"head={np.array2string(head, precision=3, separator=',')}"
    )

def _summarize_obs_dict(obs: dict, max_show: int = 6) -> str:
    # obs 是 env 的原始 observation dict；通常很大，这里只做摘要打印
    lines = []
    for k in sorted(obs.keys()):
        v = obs[k]
        if isinstance(v, (int, float, bool, np.number)):
            lines.append(f"{k}: {v}")
        else:
            try:
                arr = np.asarray(v)
                lines.append(_summarize_array(arr, name=k, max_show=max_show))
            except Exception:
                lines.append(f"{k}: <unprintable type={type(v)}>")
    return "\n    ".join(lines)

@torch.no_grad()
def run_interactive_episode_print(
    env,
    flattener,
    normalizer,
    model,
    device,
    *,
    seed: int = 123,
    deterministic: bool = True,
    max_steps: int = 200,
    print_raw_obs: bool = True,
    print_flat_obs: bool = True,
    flat_head: int = 12,
):
    """
    打印式交互回合：s_t -> a_t -> r_t -> s_{t+1}，直到 done 或 max_steps。
    """
    obs, info = env.reset(seed=int(seed), options=None)
    last_action = None

    # 有些 env 版本 obs 里没有 topology_id，这里补齐（与你原脚本一致）
    obs = ensure_topology_id(obs, info=info, last_action=last_action)

    print("\n==================== INTERACTIVE EPISODE START ====================")
    print(f"[reset] seed={seed}")
    if print_raw_obs:
        print("[state s0] raw obs summary:\n    " + _summarize_obs_dict(obs))

    for t in range(int(max_steps)):
        # 1) state -> (flatten+normalize) -> model
        x = flattener.flatten(obs)                # (obs_dim,)
        x_n = normalizer.normalize(x)            # (obs_dim,)

        if print_flat_obs:
            head = x_n[:flat_head]
            print(f"[s{t}] flat_norm: dim={x_n.shape[0]} head={np.array2string(head, precision=3, separator=',')}")

        obs_t = torch.tensor(x_n, dtype=torch.float32, device=device).unsqueeze(0)  # (1, obs_dim)
        logits, value = model(obs_t)  # logits: (1, n_actions), value: (1,) or (1,)

        # 2) choose action
        if deterministic:
            a_t = int(torch.argmax(logits, dim=-1).item())
            a_mode = "argmax"
        else:
            dist = torch.distributions.Categorical(logits=logits)
            a_t = int(dist.sample().item())
            a_mode = "sample"

        # 3) env step
        next_obs, r, terminated, truncated, info = env.step(a_t)
        done = bool(terminated or truncated)

        # 补齐 topology_id（下一状态）
        next_obs = ensure_topology_id(next_obs, info=info, last_action=a_t)

        # 4) print transition
        print(
            f"[t={t:03d}] a({a_mode})={a_t:3d} | r={float(r): .6g} | done={int(done)} "
            f"| term={int(bool(terminated))} trunc={int(bool(truncated))}"
        )

        # 可选：打印 info 中你关心的分解指标（你的 env info 里通常有这些键）
        if isinstance(info, dict):
            keys = ["pf_failed", "loss_mw", "v_viol", "line_viol", "switch_cost", "trafo_balance"]
            info_str = " ".join([f"{k}={info.get(k, None)}" for k in keys])
            print(f"         info: {info_str}")

        if print_raw_obs:
            print("[next_state] raw obs summary:\n    " + _summarize_obs_dict(next_obs))

        # 5) advance
        obs = next_obs
        last_action = a_t

        if done:
            print("==================== INTERACTIVE EPISODE END (done) ====================\n")
            return

    print("==================== INTERACTIVE EPISODE END (max_steps) ====================\n")

if __name__=="__main__" :
    run_dir = _abs(RUN_DIR)
    device = get_device(DEVICE)

    # Ensure repo root is on sys.path (helpful if env depends on repo-local packages)
    repo_root = os.path.abspath(os.path.dirname(__file__))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # Import training module for shared definitions
    train_script_path = TRAIN_SCRIPT
    if not os.path.isabs(train_script_path):
        # prefer from run_dir; fallback to current directory
        cand1 = os.path.join(run_dir, train_script_path)
        cand2 = os.path.join(repo_root, train_script_path)
        train_script_path = cand1 if os.path.exists(cand1) else cand2
    train_mod = import_module_from_path(train_script_path, module_name="_ppo_train_mod")

    # Resolve checkpoint
    ckpt_path = CKPT.strip()
    if ckpt_path:
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(run_dir, ckpt_path)
        ckpt_path = _abs(ckpt_path)
    else:
        ckpt_path = find_latest_checkpoint(run_dir)

    # -----------------------------
    # 1) Evaluate a single checkpoint (always)
    # -----------------------------
    env, flattener, normalizer, model, extra = build_env_and_model_from_run(
        train_mod, run_dir=run_dir, ckpt_path=ckpt_path, device=device
    )

    ckpt_update = -1
    ckpt_step = -1
    if isinstance(extra.get("ckpt", None), dict):
        ckpt_update = int(extra["ckpt"].get("update", -1))
        ckpt_step = int(extra["ckpt"].get("step", -1))
    print(f"[load] ckpt={ckpt_path} | update={ckpt_update} step={ckpt_step} | device={device}")
 
    run_interactive_episode_print(
        env=env,
        flattener=flattener,
        normalizer=normalizer,
        model=model,
        device=device,
        seed=BASE_SEED + 999_999,
        deterministic=DETERMINISTIC,
        max_steps=200,
        print_raw_obs=True,     # 打印原始 dict state 摘要（较长）
        print_flat_obs=True,    # 打印 flatten+normalize 后向量头部（更紧凑）
        flat_head=12,
    )
# %%
