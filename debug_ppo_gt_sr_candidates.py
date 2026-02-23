# In[]
# -*- coding: utf-8 -*-
"""
debug_ppo_gt_sr_candidates.py

调试 PPO+GT+SR 训练结果：
- 固定一个 reset 后的初始状态 s_t
- 策略网络生成若干候选动作 a（输出概率/对数概率）
- SR 估计每个候选动作的 cost_hat(¥) 与 r_hat(¥)
- 机理仿真在同一 s_t 上评估每个候选动作在 t+1 的真实 reward(¥)
- 输出 r_hat 与真实 reward 的误差，以及候选动作采样分布信息

无 argparse；参数在 __main__ 里直接改。
"""

from __future__ import annotations

import os
import json
import math
import copy
from dataclasses import asdict
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

from power_dispatch_env_withGT_dimrisk import (
    PowerDispatchEnv,
    EnvConfig,
    TimeseriesSchema,
    built_ppnet_for_pfcal,
    set_fc_state_with_acts,
)

# 你训练脚本里的一些组件（建议直接从训练脚本复制过来，保持完全一致）
# ------------------------------------------------------------
# 这里假设你把以下类/函数也放在同目录可 import，或者你直接复制进本文件：
# - ObsFlattenerV2
# - MaskedObsNormalizer
# - TopologyAdjacencyCache
# - load_gtransformer_checkpoint
# - probe_gt_forward, GTForwardSpec
# - MultiBranchActorCriticWithGTAndSR, SRConfig, TrainConfig (如果你不想引入 dataclass，可最小化重建)
#
# 为了让脚本自洽，我这里直接从你训练文件同名实现“最小依赖导入”：
from train_ppo_power_dispatch_multiproc_multiencoder_withGT_sr_merged import (
    ObsFlattenerV2,
    MaskedObsNormalizer,
    TopologyAdjacencyCache,
    load_gtransformer_checkpoint,
    probe_gt_forward,
    MultiBranchActorCriticWithGTAndSR,
    SRConfig,
    TrainConfig,
    ModelConfig,
    PPOHyperParams,
    _load_switch_params,
    get_device,
)

# ------------------------------------------------------------

def _torch_load_trusted(path: str, map_location: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)

def load_sr_checkpoint(
    ckpt_path: str,
    device: torch.device,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    返回：
      ckpt: torch.load 的原始 dict
      train_cfg_dict: ckpt['train_meta']['cfg_train']（若存在）
      sr_cfg_dict: ckpt['train_meta']['sr_cfg']（若存在）
    """
    ckpt = _torch_load_trusted(ckpt_path, map_location=device)
    meta = ckpt.get("train_meta", {}) if isinstance(ckpt, dict) else {}
    train_cfg_dict = meta.get("cfg_train", {}) if isinstance(meta, dict) else {}
    sr_cfg_dict = meta.get("sr_cfg", {}) if isinstance(meta, dict) else {}
    return ckpt, train_cfg_dict, sr_cfg_dict

def build_env_and_flattener(seed: int, time_scale: float) -> Tuple[PowerDispatchEnv, ObsFlattenerV2]:
    env_cfg = EnvConfig(seed=seed)
    env = PowerDispatchEnv(env_cfg)
    n_actions = int(env.action_space.n)
    time_period = int(getattr(env_cfg, "episode_len", 24))
    flattener = ObsFlattenerV2(
        env.observation_space,
        n_actions=n_actions,
        time_scale=time_scale,
        time_period=time_period,
    )
    return env, flattener

def build_normalizer_like_training(flattener: ObsFlattenerV2, obs_clip: float) -> MaskedObsNormalizer:
    mask = np.ones((flattener.flat_dim,), dtype=bool)
    sl_time = flattener.slices["time_feat"]
    mask[sl_time.start + 1 : sl_time.stop] = False  # 不归一化 sin/cos
    mask[flattener.slices["topology_id"]] = False    # 不归一化离散拓扑
    return MaskedObsNormalizer(flattener.flat_dim, mask=mask, clip=obs_clip)

@torch.inference_mode()
def policy_candidates(
    model: MultiBranchActorCriticWithGTAndSR,
    obs_norm_1: np.ndarray,      # (1, obs_dim)
    gt_H_1: np.ndarray,          # (1, n_bus, gt_din)
    device: torch.device,
    n_samples: int,
    topk: int,
    temperature: float = 1.0,
) -> Dict[str, Any]:
    """
    返回候选动作及其概率信息：
      - topk_actions, topk_probs
      - sampled_actions, sampled_probs, sampled_logprobs
      - full_probs（可选：很大时不返回全量）
      - entropy
    """
    obs_t = torch.from_numpy(obs_norm_1).to(device=device, dtype=torch.float32)
    gt_t = torch.from_numpy(gt_H_1).to(device=device, dtype=torch.float32)

    logits, _value, phi = model.forward_with_phi(obs_t, gt_t)

    # 可额外温度（与 model.policy_temperature 不同，这里是调试用的二次温度）
    if temperature is not None and abs(float(temperature) - 1.0) > 1e-9:
        logits = logits / float(temperature)

    dist = torch.distributions.Categorical(logits=logits)
    probs = dist.probs.squeeze(0)  # (A,)
    entropy = float(dist.entropy().mean().item())

    # top-k
    k = int(min(max(1, topk), probs.shape[0]))
    topk_probs, topk_actions = torch.topk(probs, k=k, largest=True, sorted=True)
    topk_actions_np = topk_actions.detach().cpu().numpy().astype(np.int64)
    topk_probs_np = topk_probs.detach().cpu().numpy().astype(np.float64)

    # multinomial sampling（允许重复，便于估计采样频次；如需不重复可用 torch.multinomial(replacement=False)）
    n_samples = int(max(1, n_samples))
    sampled_actions = torch.multinomial(probs, num_samples=n_samples, replacement=True)
    sampled_logprobs = torch.log(probs[sampled_actions] + 1e-30)
    sampled_probs = probs[sampled_actions]

    out = {
        "phi": phi.detach(),  # (1, hidden)
        "probs": probs.detach().cpu().numpy().astype(np.float64),
        "entropy": entropy,
        "topk_actions": topk_actions_np,
        "topk_probs": topk_probs_np,
        "sampled_actions": sampled_actions.detach().cpu().numpy().astype(np.int64),
        "sampled_probs": sampled_probs.detach().cpu().numpy().astype(np.float64),
        "sampled_logprobs": sampled_logprobs.detach().cpu().numpy().astype(np.float64),
    }
    return out

@torch.inference_mode()
def sr_predict_for_actions(
    model: MultiBranchActorCriticWithGTAndSR,
    phi_1: torch.Tensor,                 # (1, hidden)
    actions: np.ndarray,                 # (K,)
    lambda_risk: float,
) -> Dict[str, np.ndarray]:
    """
    SR 输出是 cost_hat(¥)，并给出 r_hat_yuan。
    """
    a = torch.from_numpy(actions.astype(np.int64)).to(device=phi_1.device)
    phiK = phi_1.expand(a.shape[0], -1)  # (K, hidden)
    total_hat, risk_hat = model.sr(phiK, a)
    total_hat_np = total_hat.detach().cpu().numpy().astype(np.float64)
    risk_hat_np = risk_hat.detach().cpu().numpy().astype(np.float64)
    r_hat_yuan = -(total_hat_np + float(lambda_risk) * risk_hat_np)
    return {
        "total_cost_yuan_hat": total_hat_np,
        "risk_cost_yuan_hat": risk_hat_np,
        "r_hat_yuan": r_hat_yuan,
    }

def mechanistic_eval_one_step_from_state(
    env: PowerDispatchEnv,
    t_now: int,
    prev_action: int,
    action: int,
) -> Dict[str, Any]:
    """
    在不改变 env 主状态的前提下，评估“从 s_t 采取 action 后，在 t+1 的真实 reward / cost”。

    训练时：phi 来自 t（reset/step 前），label 来自 step 后（t+1）。
    因此这里用 t_next = t_now + 1，并把 prev_action 固定为当前状态的 prev_action。
    """
    t_next = int(t_now) + 1
    net_line_repr = set_fc_state_with_acts(env.feeder_cluster, env.base_net, [int(action)])
    # 直接调用内部方法，不更新 env._t / _prev_action 等主状态
    _obs_next, info = env._solve_and_build_obs(
        net_line_repr,
        t_next,
        prev_action=int(prev_action),
        action=int(action),
    )
    # info["reward"] 在 dimensioned 模式下就是 -total_cost_yuan（¥）
    return info

def format_rows(rows: List[Dict[str, Any]], max_rows: int = 50) -> str:
    """
    简单文本表格，避免引入 pandas 依赖；你也可以自行换成 DataFrame 输出。
    """
    if not rows:
        return "(empty)"
    cols = list(rows[0].keys())
    rows2 = rows[: int(max_rows)]
    # col widths
    widths = {c: max(len(c), max(len(f"{r.get(c, '')}") for r in rows2)) for c in cols}
    def _line(items: List[str]) -> str:
        return " | ".join(s.ljust(widths[c]) for s, c in zip(items, cols))
    out = []
    out.append(_line(cols))
    out.append("-+-".join("-" * widths[c] for c in cols))
    for r in rows2:
        out.append(_line([f"{r.get(c, '')}" for c in cols]))
    if len(rows) > len(rows2):
        out.append(f"... ({len(rows)-len(rows2)} more rows)")
    return "\n".join(out)

def unique_with_counts(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vals, cnt = np.unique(x.astype(np.int64), return_counts=True)
    order = np.argsort(-cnt)
    return vals[order], cnt[order]


if __name__ == "__main__":
    # =========================
    # 1) 需要你改的参数
    # =========================
    RUN_DIR = "./runs_ppo_gt_sr_warmup_joint"
    CKPT_PATH = os.path.join(RUN_DIR, "ppo_gt_sr_ckpt_update0120.pt")  # 你要调试哪一个 ckpt
    GT_CKPT_PATH = "./GTransformer/runs/pretrain_masked_node_only_v1/ckpt_best.pt"

    DEVICE_STR = "cuda"  # "cpu" or "cuda" or "auto"
    SEED = 0

    # 候选动作生成
    TOPK = 20
    N_SAMPLES = 200          # 从策略分布重复采样次数（用于统计采样频次）
    TEMPERATURE_DEBUG = 1.0  # 调试用二次温度，不改模型结构，只影响采样尖锐度

    # 对齐到训练尺度时需要
    REWARD_SCALE = 1e-6      # 你训练里 cfg_train.reward_scale
    # 注意 alpha 是训练时 shaping 系数（ckpt meta 里一般记录了 alpha_now，但这里你可手动设）
    ALPHA_FOR_TRAIN_SCALE = 1e-6

    # =========================
    # 2) 加载 ckpt + meta（用于恢复 normalizer 等）
    # =========================
    device = get_device(DEVICE_STR)
    ckpt, train_cfg_dict, sr_cfg_dict = load_sr_checkpoint(CKPT_PATH, device=device)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise RuntimeError(f"Bad checkpoint format: {CKPT_PATH}")

    # SR 配置（lambda_risk 影响 r_hat 的定义）
    lambda_risk = float(sr_cfg_dict.get("lambda_risk", 0.0)) if isinstance(sr_cfg_dict, dict) else 0.0

    # =========================
    # 3) 建 env / flattener / normalizer，并恢复 normalizer 状态
    # =========================
    # time_scale / obs_clip 尽量从 ckpt meta 恢复；没有就用训练默认
    time_scale = float(train_cfg_dict.get("time_scale", 1e-3)) if isinstance(train_cfg_dict, dict) else 1e-3
    obs_clip = float(train_cfg_dict.get("obs_clip", 10.0)) if isinstance(train_cfg_dict, dict) else 10.0

    env, flattener = build_env_and_flattener(seed=SEED, time_scale=time_scale)

    # reset 得到初始工况 s_t
    obs_dict, info0 = env.reset(seed=SEED, options=None)
    gt_H = np.asarray(obs_dict["gt_H"], dtype=np.float32)
    x = flattener.flatten(obs_dict).reshape(1, -1).astype(np.float32)

    n_actions = int(env.action_space.n)
    n_bus = int(gt_H.shape[0])
    gt_din = int(gt_H.shape[1])

    # normalizer
    normalizer = build_normalizer_like_training(flattener, obs_clip=obs_clip)
    if "normalizer" in ckpt and isinstance(ckpt["normalizer"], dict):
        normalizer.load_state_dict(ckpt["normalizer"])

    # 归一化后的 obs
    obs_norm = normalizer.normalize(x)  # (1, obs_dim)
    gt_H_1 = gt_H.reshape(1, n_bus, gt_din).astype(np.float32)

    # 当前时间与 prev_action（用于“同一 s_t 的机理仿真评估”）
    t_now = int(obs_dict["time_index"].reshape(-1)[0])
    prev_action = int(info0.get("action", int(obs_dict["topology_id"].reshape(-1)[0])))

    print(f"[init] device={device} n_actions={n_actions} n_bus={n_bus} gt_din={gt_din}")
    print(f"[state] t_now={t_now} prev_action={prev_action} reward_mode={info0.get('reward_mode', 'unknown')}")

    # =========================
    # 4) 构建 GT / adj_cache / probe spec / 构建模型并加载权重
    # =========================
    gt, gt_cfg, _ = load_gtransformer_checkpoint(GT_CKPT_PATH, device)

    r_switch, x_switch = _load_switch_params()
    adj_cache = TopologyAdjacencyCache(
        feeder_cluster=env.feeder_cluster,
        base_net=env.base_net,
        r_switch=r_switch,
        x_switch=x_switch,
        max_entries=64,
    )
    # 归一化后的 obs
    obs_norm = normalizer.normalize(x)          # 可能返回 (obs_dim,)
    obs_norm = np.asarray(obs_norm, dtype=np.float32).reshape(1, -1)
    topo0 = int(obs_norm[0, flattener.slices["topology_id"]][0].round())
    A0 = adj_cache.get(topo0).to(device=device)
    H_probe = torch.from_numpy(gt_H_1).to(device=device, dtype=torch.float32)
    spec = probe_gt_forward(gt, H_probe, A0)
    print(f"[GT probe] use_return_embeddings={spec.use_return_embeddings} use_batched_adj={spec.use_batched_adj} z_key={spec.z_key} out_dim={spec.out_dim}")

    # 模型超参尽量从 meta 恢复，否则用训练默认
    model_cfg_dict = train_cfg_dict.get("model", {}) if isinstance(train_cfg_dict, dict) else {}
    def _g(k: str, default: Any) -> Any:
        return model_cfg_dict.get(k, default) if isinstance(model_cfg_dict, dict) else default

    sr_cfg = SRConfig(**sr_cfg_dict) if isinstance(sr_cfg_dict, dict) and sr_cfg_dict else SRConfig()

    model = MultiBranchActorCriticWithGTAndSR(
        obs_dim=flattener.flat_dim,
        n_actions=n_actions,
        slices=flattener.slices,
        gt=gt,
        gt_spec=spec,
        adj_cache=adj_cache,
        gt_pool=_g("gt_pool", "mean"),
        gt_proj_dim=int(_g("gt_proj_dim", 128)),
        enc_hidden=int(_g("enc_hidden", 256)),
        fusion_hidden=int(_g("fusion_hidden", 256)),
        fusion_blocks=int(_g("fusion_blocks", 2)),
        dropout=float(_g("dropout", 0.0)),
        policy_temperature=float(_g("policy_temperature", 1.0)),
        emb_dim_bus=int(_g("emb_dim_bus", 128)),
        emb_dim_load=int(_g("emb_dim_load", 128)),
        emb_dim_fcst=int(_g("emb_dim_fcst", 128)),
        emb_dim_time=int(_g("emb_dim_time", 32)),
        topo_emb_dim=int(_g("topo_emb_dim", 32)),
        adj_cache_cuda_size=int(_g("adj_cache_cuda_size", 16)),
        sr_cfg=sr_cfg,
    ).to(device)

    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    # =========================
    # 5) 在该 s_t 上生成候选动作 + SR 评估 + 机理仿真评估
    # =========================
    cand = policy_candidates(
        model=model,
        obs_norm_1=obs_norm.astype(np.float32),
        gt_H_1=gt_H_1.astype(np.float32),
        device=device,
        n_samples=N_SAMPLES,
        topk=TOPK,
        temperature=TEMPERATURE_DEBUG,
    )

    phi_1 = cand["phi"]  # torch.Tensor (1, hidden)
    probs = cand["probs"]  # (A,)

    topk_actions = cand["topk_actions"]
    topk_probs = cand["topk_probs"]

    sampled_actions = cand["sampled_actions"]
    sampled_probs = cand["sampled_probs"]
    sampled_logprobs = cand["sampled_logprobs"]

    # 你可以选择“候选集合”的定义：这里用 topk ∪ unique(sampled)
    pool_actions = np.unique(np.concatenate([topk_actions, np.unique(sampled_actions)], axis=0)).astype(np.int64)

    # SR 对每个候选动作输出 (¥)
    sr_out = sr_predict_for_actions(
        model=model,
        phi_1=phi_1,
        actions=pool_actions,
        lambda_risk=lambda_risk,
    )

    # 机理仿真对每个候选动作输出真实 reward(¥) 与成本(¥)
    mech_infos: List[Dict[str, Any]] = []
    for a in pool_actions.tolist():
        info = mechanistic_eval_one_step_from_state(env, t_now=t_now, prev_action=prev_action, action=int(a))
        mech_infos.append(info)

    # =========================
    # 6) 汇总对比表
    # =========================
    rows: List[Dict[str, Any]] = []
    for i, a in enumerate(pool_actions.tolist()):
        p = float(probs[int(a)])
        logp = math.log(p + 1e-30)

        total_hat = float(sr_out["total_cost_yuan_hat"][i])
        risk_hat = float(sr_out["risk_cost_yuan_hat"][i])
        r_hat_yuan = float(sr_out["r_hat_yuan"][i])

        info = mech_infos[i]
        reward_yuan = float(info.get("reward", float("nan")))  # dimensioned 模式下 reward=-total_cost_yuan
        total_cost_yuan = float(info.get("total_cost_yuan", float("nan")))
        risk_cost_yuan = float(info.get("risk_cost_yuan", float("nan"))) if "risk_cost_yuan" in info else float(info.get("risk_term", 0.0))

        # 误差（¥尺度）
        err_r_yuan = r_hat_yuan - reward_yuan

        # 如果你要看训练尺度对齐：
        reward_train_scaled = REWARD_SCALE * reward_yuan
        rhat_train_scaled = ALPHA_FOR_TRAIN_SCALE * r_hat_yuan
        err_train_scaled = rhat_train_scaled - reward_train_scaled

        rows.append({
            "a": a,
            "p(a)": f"{p:.4e}",
            "logp": f"{logp:.4f}",
            "SR_total_hat(¥)": f"{total_hat:.3e}",
            "SR_risk_hat(¥)": f"{risk_hat:.3e}",
            "SR_r_hat(¥)": f"{r_hat_yuan:.3e}",
            "MECH_total(¥)": f"{total_cost_yuan:.3e}",
            "MECH_risk(¥)": f"{risk_cost_yuan:.3e}",
            "MECH_reward(¥)": f"{reward_yuan:.3e}",
            "Δr(¥)": f"{err_r_yuan:.3e}",
            "train_r": f"{reward_train_scaled:.3e}",
            "train_rhat": f"{rhat_train_scaled:.3e}",
            "Δtrain": f"{err_train_scaled:.3e}",
        })

    # 按概率从大到小排序
    rows_sorted = sorted(rows, key=lambda d: float(d["p(a)"]), reverse=True)

    print("\n[candidates summary]")
    print(f"entropy={cand['entropy']:.4f}  temperature_debug={TEMPERATURE_DEBUG}  lambda_risk={lambda_risk}")
    print(format_rows(rows_sorted, max_rows=80))

    # =========================
    # 7) 采样概率 / 频次分析
    # =========================
    uniq_a, cnt = unique_with_counts(sampled_actions)
    freq_rows = []
    for a, c in zip(uniq_a.tolist(), cnt.tolist()):
        freq_rows.append({
            "a": int(a),
            "count": int(c),
            "freq": f"{(c / float(N_SAMPLES)):.4f}",
            "p(a)": f"{float(probs[int(a)]):.4e}",
        })
    freq_rows = sorted(freq_rows, key=lambda d: d["count"], reverse=True)

    print("\n[sampling frequency over repeated multinomial draws]")
    print(f"N_SAMPLES={N_SAMPLES}")
    print(format_rows(freq_rows, max_rows=60))

    # 也给一个 topk 概率质量覆盖度
    topk_mass = float(np.sum(topk_probs))
    print("\n[top-k mass]")
    print(f"TOPK={TOPK}  sum_p(topk)={topk_mass:.6f}")

    print("\n[done]")

# %%
