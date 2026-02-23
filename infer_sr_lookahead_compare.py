# In[]
# -*- coding: utf-8 -*-
"""
infer_sr_lookahead_compare.py

Self-Reflection (SR) + look-ahead(3) planning inference vs baseline sampling.


Baseline:
  - sample 1 action from policy distribution and execute in real env.

SR look-ahead:
  - at each decision step:
      * sample K candidate actions from policy at current state
      * run beam search depth=H (default 3)
      * evaluate each action by SR predicted costs:
            r_hat = -(total_cost_hat + lambda_risk * risk_cost_hat)
      * choose best sequence by discounted sum of r_hat
      * execute first action of best sequence in real env

This script prints detailed logs of:
  - root policy distribution info (top actions)
  - sampled candidates
  - per-node SR predicted costs and r_hat
  - beam keep/prune decisions
  - chosen action and predicted return
  - real reward, predicted r_hat, and their difference
  - episode summary and baseline vs SR difference

No argparse: edit parameters in __main__.
"""

from __future__ import annotations

import os
import json
import math
import copy
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Reuse your training definitions (model classes, GT loading, flattener, normalizer, cache, env config)
from train_ppo_power_dispatch_multiproc_multiencoder_withGT_sr_merged import (
    ObsFlattenerV2,
    MaskedObsNormalizer,
    TopologyAdjacencyCache,
    load_gtransformer_checkpoint,
    probe_gt_forward,
    MultiBranchActorCriticWithGTAndSR,
    SRConfig,
    ModelConfig,
    _load_switch_params,
)
from power_dispatch_env_withGT_dimrisk import PowerDispatchEnv, EnvConfig


# ============================================================
# Helpers: checkpoint load
# ============================================================

def _torch_load_trusted(path: str, map_location: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_meta_json(meta_path: str) -> Dict[str, Any]:
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Helpers: env snapshot/restore (for simulator rollouts)
# ============================================================

def snapshot_env_state(env: PowerDispatchEnv) -> Dict[str, Any]:
    """
    Snapshot minimal internal state to reproduce transitions deterministically.
    Notes:
      - We assume same EnvConfig and same timeseries are loaded in new env.
      - Restore RNG state to ensure consistency where randomness exists.
    """
    st = {
        "_t0": int(getattr(env, "_t0", 0)),
        "_t": int(getattr(env, "_t", 0)),
        "_step_in_episode": int(getattr(env, "_step_in_episode", 0)),
        "_prev_action": None if getattr(env, "_prev_action", None) is None else int(getattr(env, "_prev_action")),
        "_ts_scale": float(getattr(env, "_ts_scale", 1.0)),
        "_ts_scale_determined": bool(getattr(env, "_ts_scale_determined", False)),
        "_ts_scale_msg_printed": bool(getattr(env, "_ts_scale_msg_printed", False)),
        "_rng_state": None,
    }
    try:
        rng = getattr(env, "_rng", None)
        if rng is not None and hasattr(rng, "bit_generator"):
            st["_rng_state"] = copy.deepcopy(rng.bit_generator.state)
    except Exception:
        st["_rng_state"] = None
    return st


def restore_env_state(env: PowerDispatchEnv, st: Dict[str, Any]) -> None:
    setattr(env, "_t0", int(st.get("_t0", 0)))
    setattr(env, "_t", int(st.get("_t", 0)))
    setattr(env, "_step_in_episode", int(st.get("_step_in_episode", 0)))
    pa = st.get("_prev_action", None)
    setattr(env, "_prev_action", None if pa is None else int(pa))
    setattr(env, "_ts_scale", float(st.get("_ts_scale", 1.0)))
    setattr(env, "_ts_scale_determined", bool(st.get("_ts_scale_determined", False)))
    setattr(env, "_ts_scale_msg_printed", bool(st.get("_ts_scale_msg_printed", False)))

    rs = st.get("_rng_state", None)
    try:
        rng = getattr(env, "_rng", None)
        if rs is not None and rng is not None and hasattr(rng, "bit_generator"):
            rng.bit_generator.state = copy.deepcopy(rs)
    except Exception:
        pass


def make_sim_env_from_real(env_real: PowerDispatchEnv) -> PowerDispatchEnv:
    """
    Create a new env with same config and restore state from env_real snapshot.
    """
    cfg = env_real.cfg
    env_sim = PowerDispatchEnv(cfg)
    st = snapshot_env_state(env_real)
    restore_env_state(env_sim, st)
    return env_sim


# ============================================================
# Helpers: printing
# ============================================================

def fmt_topk_from_logits(logits_1d: torch.Tensor, k: int = 8) -> List[Tuple[int, float]]:
    """
    Returns [(action, prob), ...] sorted by prob desc.
    """
    probs = torch.softmax(logits_1d, dim=-1)
    k = int(min(k, probs.numel()))
    vals, idx = torch.topk(probs, k=k)
    out = []
    for a, p in zip(idx.tolist(), vals.tolist()):
        out.append((int(a), float(p)))
    return out


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


# ============================================================
# SR look-ahead planner (beam search)
# ============================================================

@torch.inference_mode()
def sr_predict_cost_and_rhat(
    model: MultiBranchActorCriticWithGTAndSR,
    normalizer: MaskedObsNormalizer,
    obs_flat_np: np.ndarray,
    gt_H_np: np.ndarray,
    action: int,
    device: torch.device,
    lambda_risk: float,
) -> Tuple[float, float, float]:
    """
    Return (total_cost_hat, risk_cost_hat, r_hat).
      r_hat = -(total_hat + lambda_risk * risk_hat)
    """
    obs_norm = normalizer.normalize(obs_flat_np.astype(np.float32))
    obs_t = torch.from_numpy(obs_norm.reshape(1, -1)).to(device=device, dtype=torch.float32)
    gt_t = torch.from_numpy(gt_H_np.reshape(1, gt_H_np.shape[0], gt_H_np.shape[1])).to(device=device, dtype=torch.float32)

    logits, value, phi = model.forward_with_phi(obs_t, gt_t)
    a_t = torch.tensor([int(action)], device=device, dtype=torch.int64)
    total_hat, risk_hat = model.sr(phi, a_t)

    total_hat_f = float(total_hat.item())
    risk_hat_f = float(risk_hat.item())
    r_hat = -(total_hat_f + float(lambda_risk) * risk_hat_f)
    return total_hat_f, risk_hat_f, float(r_hat)


@torch.inference_mode()
def sample_k_actions_from_policy(
    model: MultiBranchActorCriticWithGTAndSR,
    normalizer: MaskedObsNormalizer,
    obs_flat_np: np.ndarray,
    gt_H_np: np.ndarray,
    device: torch.device,
    k: int,
) -> Tuple[List[int], torch.Tensor]:
    """
    Sample k actions (with replacement) from policy at given state.
    Return (actions_list, logits_1d).
    """
    obs_norm = normalizer.normalize(obs_flat_np.astype(np.float32))
    obs_t = torch.from_numpy(obs_norm.reshape(1, -1)).to(device=device, dtype=torch.float32)
    gt_t = torch.from_numpy(gt_H_np.reshape(1, gt_H_np.shape[0], gt_H_np.shape[1])).to(device=device, dtype=torch.float32)

    logits, _, _phi = model.forward_with_phi(obs_t, gt_t)
    logits_1d = logits.squeeze(0)
    dist = torch.distributions.Categorical(logits=logits_1d)
    acts = dist.sample((int(k),)).tolist()
    acts = [int(a) for a in acts]
    return acts, logits_1d


def plan_action_sr_lookahead_beam(
    *,
    env_real: PowerDispatchEnv,
    model: MultiBranchActorCriticWithGTAndSR,
    flattener: ObsFlattenerV2,
    normalizer: MaskedObsNormalizer,
    obs_flat_np: np.ndarray,
    gt_H_np: np.ndarray,
    device: torch.device,
    depth: int = 3,
    gamma: float = 0.99,
    k_candidates: int = 12,
    beam_width: int = 6,
    lambda_risk: float = 0.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Beam search over action sequences using SR r_hat.
    We simulate next states by creating env copies and stepping them.

    Returns dict with:
      chosen_action, best_seq, best_return_hat, root_candidates, debug_nodes...
    """
    depth = int(max(1, depth))
    beam_width = int(max(1, beam_width))
    k_candidates = int(max(1, k_candidates))
    gamma = float(gamma)

    if verbose:
        print("\n[SR-PLAN] ========= START planning =========")
        print(f"[SR-PLAN] depth={depth} gamma={gamma} k_candidates={k_candidates} beam_width={beam_width} lambda_risk={lambda_risk}")

    # Print current env state
    if verbose:
        st0 = snapshot_env_state(env_real)
        print(f"[SR-PLAN] env_state: t0={st0['_t0']} t={st0['_t']} step_in_ep={st0['_step_in_episode']} prev_action={st0['_prev_action']} ts_scale={st0['_ts_scale']}")

    # Root: sample candidates from policy
    root_actions, root_logits = sample_k_actions_from_policy(
        model=model,
        normalizer=normalizer,
        obs_flat_np=obs_flat_np,
        gt_H_np=gt_H_np,
        device=device,
        k=k_candidates,
    )

    # Also print policy top-k
    if verbose:
        topk = fmt_topk_from_logits(root_logits, k=10)
        print(f"[SR-PLAN] policy top-10 probs: {topk}")
        # unique candidates frequency
        uniq, cnt = np.unique(np.asarray(root_actions, dtype=int), return_counts=True)
        freq = sorted([(int(u), int(c)) for u, c in zip(uniq, cnt)], key=lambda x: (-x[1], x[0]))
        print(f"[SR-PLAN] sampled root candidates (k={k_candidates}) freq: {freq}")

    # Each node carries: (seq_actions, return_hat, env_snapshot, obs_flat, gt_H)
    nodes: List[Dict[str, Any]] = []
    env_snap0 = snapshot_env_state(env_real)
    for a0 in root_actions:
        # r_hat at root for action a0
        tot_hat, risk_hat, r_hat = sr_predict_cost_and_rhat(
            model=model,
            normalizer=normalizer,
            obs_flat_np=obs_flat_np,
            gt_H_np=gt_H_np,
            action=int(a0),
            device=device,
            lambda_risk=lambda_risk,
        )
        node = {
            "seq": [int(a0)],
            "G": float(r_hat),  # cumulative discounted
            "last_r_hat": float(r_hat),
            "last_total_hat": float(tot_hat),
            "last_risk_hat": float(risk_hat),
            "snap": copy.deepcopy(env_snap0),
            "obs_flat": obs_flat_np.copy(),
            "gt_H": gt_H_np.copy(),
            "depth_done": 1,
        }
        nodes.append(node)

    # Keep top beam_width by G
    nodes.sort(key=lambda x: x["G"], reverse=True)
    nodes = nodes[:beam_width]

    if verbose:
        print("[SR-PLAN] after root eval -> keep beam:")
        for i, nd in enumerate(nodes):
            print(f"  beam[{i}] a0={nd['seq'][0]} G={nd['G']:.6e} r_hat0={nd['last_r_hat']:.6e} total_hat0={nd['last_total_hat']:.6e} risk_hat0={nd['last_risk_hat']:.6e}")

    # Expand depths 2..depth
    for d in range(2, depth + 1):
        if verbose:
            print(f"\n[SR-PLAN] ----- EXPAND depth={d}/{depth} -----")

        new_nodes: List[Dict[str, Any]] = []

        for bi, nd in enumerate(nodes):
            # Build simulator env at this node state
            env_sim = PowerDispatchEnv(env_real.cfg)
            restore_env_state(env_sim, nd["snap"])

            # Reconstruct the "current observation" of this node:
            # We already have obs_flat/gt_H stored for this node.
            # Now expand by sampling actions from policy at this node state.
            cand_actions_d, logits_d = sample_k_actions_from_policy(
                model=model,
                normalizer=normalizer,
                obs_flat_np=nd["obs_flat"],
                gt_H_np=nd["gt_H"],
                device=device,
                k=k_candidates,
            )

            if verbose:
                topk_d = fmt_topk_from_logits(logits_d, k=6)
                uniq, cnt = np.unique(np.asarray(cand_actions_d, dtype=int), return_counts=True)
                freq = sorted([(int(u), int(c)) for u, c in zip(uniq, cnt)], key=lambda x: (-x[1], x[0]))
                print(f"[SR-PLAN] expand from beam[{bi}] seq={nd['seq']} G={nd['G']:.6e}")
                print(f"          policy top-6 probs at this node: {topk_d}")
                print(f"          sampled candidates freq: {freq}")

            for a in cand_actions_d:
                # Evaluate this action by SR at current node obs
                tot_hat, risk_hat, r_hat = sr_predict_cost_and_rhat(
                    model=model,
                    normalizer=normalizer,
                    obs_flat_np=nd["obs_flat"],
                    gt_H_np=nd["gt_H"],
                    action=int(a),
                    device=device,
                    lambda_risk=lambda_risk,
                )

                # Simulate env step to get next obs for further depth
                obs_next, rew_real, term, trunc, info = env_sim.step(int(a))
                done = bool(term or trunc)

                obs_flat_next = flattener.flatten(obs_next)
                gt_H_next = np.asarray(obs_next["gt_H"], dtype=np.float32)

                # Update discounted return_hat
                G_new = float(nd["G"] + (gamma ** (d - 1)) * float(r_hat))

                # Snapshot env state after step for further expansion
                snap_next = snapshot_env_state(env_sim)

                new_node = {
                    "seq": nd["seq"] + [int(a)],
                    "G": G_new,
                    "last_r_hat": float(r_hat),
                    "last_total_hat": float(tot_hat),
                    "last_risk_hat": float(risk_hat),
                    "snap": snap_next,
                    "obs_flat": obs_flat_next,
                    "gt_H": gt_H_next,
                    "depth_done": d,
                    "done": done,

                    # purely for debugging prints (not used for ranking):
                    "sim_real_reward_last": float(rew_real),
                    "sim_info_last": info if isinstance(info, dict) else {},
                }
                new_nodes.append(new_node)

                if verbose:
                    print(
                        f"            cand a={int(a)} | r_hat={r_hat:.6e} (total_hat={tot_hat:.3e}, risk_hat={risk_hat:.3e}) "
                        f"| G_new={G_new:.6e} | sim_real_r={float(rew_real):.6e} done={done}"
                    )

                # Important: reset env_sim back to this node state for next candidate expansion
                # because we stepped env_sim once.
                # Re-restore snapshot and continue.
                restore_env_state(env_sim, nd["snap"])

        # prune
        new_nodes.sort(key=lambda x: x["G"], reverse=True)
        nodes = new_nodes[:beam_width]

        if verbose:
            print(f"[SR-PLAN] after prune depth={d}, keep top-{beam_width}:")
            for i, nd in enumerate(nodes):
                print(f"  beam[{i}] seq={nd['seq']} G={nd['G']:.6e} last_r_hat={nd['last_r_hat']:.6e}")

    best = max(nodes, key=lambda x: x["G"])
    chosen_action = int(best["seq"][0])

    if verbose:
        print("\n[SR-PLAN] ========= DONE planning =========")
        print(f"[SR-PLAN] best_seq={best['seq']} best_return_hat={best['G']:.6e} -> chosen_action={chosen_action}")

    return {
        "chosen_action": chosen_action,
        "best_seq": best["seq"],
        "best_return_hat": float(best["G"]),
        "root_candidates": root_actions,
    }


# ============================================================
# Baseline action sampling
# ============================================================

@torch.inference_mode()
def baseline_sample_action(
    model: MultiBranchActorCriticWithGTAndSR,
    normalizer: MaskedObsNormalizer,
    obs_flat_np: np.ndarray,
    gt_H_np: np.ndarray,
    device: torch.device,
) -> Dict[str, Any]:
    obs_norm = normalizer.normalize(obs_flat_np.astype(np.float32))
    obs_t = torch.from_numpy(obs_norm.reshape(1, -1)).to(device=device, dtype=torch.float32)
    gt_t = torch.from_numpy(gt_H_np.reshape(1, gt_H_np.shape[0], gt_H_np.shape[1])).to(device=device, dtype=torch.float32)

    logits, value, phi = model.forward_with_phi(obs_t, gt_t)
    logits_1d = logits.squeeze(0)
    dist = torch.distributions.Categorical(logits=logits_1d)
    a = int(dist.sample().item())
    logp = float(dist.log_prob(torch.tensor(a, device=device)).item())
    ent = float(dist.entropy().item())

    topk = fmt_topk_from_logits(logits_1d, k=8)
    return {
        "action": a,
        "logp": logp,
        "entropy": ent,
        "topk": topk,
    }


# ============================================================
# Main evaluation loop
# ============================================================

def evaluate_compare(
    *,
    ckpt_path: str,
    meta_json_path: str,
    device_str: str,
    n_episodes: int,
    base_seed: int,
    # planning hyperparams
    lookahead_depth: int,
    gamma: float,
    k_candidates: int,
    beam_width: int,
    lambda_risk: float,
    # printing
    max_steps_per_ep: Optional[int] = None,
) -> None:
    device = torch.device(device_str if device_str in ("cpu", "cuda") else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INIT] device={device}")
    print(f"[INIT] ckpt_path={ckpt_path}")
    print(f"[INIT] meta_json_path={meta_json_path}")

    meta = load_meta_json(meta_json_path)

    # Build env (for shapes & adjacency cache)
    env_cfg = EnvConfig(seed=base_seed)
    env_tmp = PowerDispatchEnv(env_cfg)
    obs0, info0 = env_tmp.reset(seed=base_seed, options=None)

    n_actions = int(env_tmp.action_space.n)
    time_period = int(getattr(env_cfg, "episode_len", 24))

    flattener = ObsFlattenerV2(
        env_tmp.observation_space,
        n_actions=n_actions,
        time_scale=float(meta.get("train_cfg", {}).get("time_scale", 1e-3)),
        time_period=time_period,
    )
    gt_H0 = np.asarray(obs0["gt_H"], dtype=np.float32)
    n_bus, gt_din = int(gt_H0.shape[0]), int(gt_H0.shape[1])

    print(f"[ENV] n_actions={n_actions} obs_dim={flattener.flat_dim} n_bus={n_bus} gt_din={gt_din} episode_len={env_cfg.episode_len}")

    # Load GT checkpoint path from meta if present, else from your known location
    gt_ckpt_path = None
    try:
        gt_ckpt_path = meta.get("train_cfg", {}).get("gt_ckpt_path", None)
    except Exception:
        gt_ckpt_path = None
    if not gt_ckpt_path:
        # fallback: you can hardcode if needed
        gt_ckpt_path = "./GTransformer/runs/pretrain_masked_node_only_v1/ckpt_best.pt"
    print(f"[GT] gt_ckpt_path={gt_ckpt_path}")

    # Load GT and probe forward spec
    gt, gt_cfg, _ = load_gtransformer_checkpoint(gt_ckpt_path, device)

    r_switch, x_switch = _load_switch_params()
    adj_cache = TopologyAdjacencyCache(
        feeder_cluster=env_tmp.feeder_cluster,
        base_net=env_tmp.base_net,
        r_switch=r_switch,
        x_switch=x_switch,
        max_entries=int(meta.get("train_cfg", {}).get("model", {}).get("adj_cache_size", 64)) if isinstance(meta.get("train_cfg", {}).get("model", {}), dict) else 64,
    )

    topo0 = int(np.round(flattener.flatten(obs0)[flattener.slices["topology_id"]][0]))
    A0 = adj_cache.get(topo0).to(device=device)
    H_probe = torch.from_numpy(np.stack([gt_H0, gt_H0], axis=0)).to(device=device, dtype=torch.float32)
    spec = probe_gt_forward(gt, H_probe, A0)
    print(f"[GT probe] use_return_embeddings={spec.use_return_embeddings} use_batched_adj={spec.use_batched_adj} z_key={spec.z_key} out_dim={spec.out_dim}")

    # Build model config (prefer meta if available, else fallback to your defaults)
    model_cfg = ModelConfig()
    try:
        mc = meta.get("train_cfg", {}).get("model", None)
        if isinstance(mc, dict):
            for k, v in mc.items():
                if hasattr(model_cfg, k):
                    setattr(model_cfg, k, v)
    except Exception:
        pass

    # Build SR config (prefer meta if you stored it; else set typical values)
    sr_cfg = SRConfig(
        enabled=True,
        lambda_risk=float(lambda_risk),
    )

    # Instantiate model
    model = MultiBranchActorCriticWithGTAndSR(
        obs_dim=flattener.flat_dim,
        n_actions=n_actions,
        slices=flattener.slices,
        gt=gt,
        gt_spec=spec,
        adj_cache=adj_cache,
        gt_pool=model_cfg.gt_pool,
        gt_proj_dim=model_cfg.gt_proj_dim,
        enc_hidden=model_cfg.enc_hidden,
        fusion_hidden=model_cfg.fusion_hidden,
        fusion_blocks=model_cfg.fusion_blocks,
        dropout=model_cfg.dropout,
        policy_temperature=model_cfg.policy_temperature,
        emb_dim_bus=model_cfg.emb_dim_bus,
        emb_dim_load=model_cfg.emb_dim_load,
        emb_dim_fcst=model_cfg.emb_dim_fcst,
        emb_dim_time=model_cfg.emb_dim_time,
        topo_emb_dim=model_cfg.topo_emb_dim,
        adj_cache_cuda_size=model_cfg.adj_cache_cuda_size,
        sr_cfg=sr_cfg,
    ).to(device)
    model.eval()

    # Load checkpoint (model + normalizer)
    ckpt = _torch_load_trusted(ckpt_path, map_location=device)
    if "model" not in ckpt or "normalizer" not in ckpt:
        raise RuntimeError(f"Checkpoint missing keys. found={list(ckpt.keys())}")

    print("[CKPT] loading model state_dict...")
    model.load_state_dict(ckpt["model"], strict=True)
    print("[CKPT] model loaded.")

    # normalizer: rebuild a compatible object then load_state_dict
    # mask construction same as training: normalize all except sin/cos and topology_id
    mask = np.ones((flattener.flat_dim,), dtype=bool)
    sl_time = flattener.slices["time_feat"]
    mask[sl_time.start + 1 : sl_time.stop] = False
    mask[flattener.slices["topology_id"]] = False
    normalizer = MaskedObsNormalizer(flattener.flat_dim, mask=mask, clip=float(meta.get("train_cfg", {}).get("obs_clip", 10.0)))
    print("[CKPT] loading normalizer state...")
    normalizer.load_state_dict(ckpt["normalizer"])
    print("[CKPT] normalizer loaded.")

    # Evaluation
    ep_returns_base: List[float] = []
    ep_returns_sr: List[float] = []
    per_step_records: List[Dict[str, Any]] = []

    for ep in range(int(n_episodes)):
        seed_ep = int(base_seed + ep * 10000)
        print("\n" + "=" * 120)
        print(f"[EVAL] EPISODE {ep+1}/{n_episodes} seed={seed_ep}")

        env_base = PowerDispatchEnv(env_cfg)
        env_sr = PowerDispatchEnv(env_cfg)

        obs_b, info_b = env_base.reset(seed=seed_ep, options=None)
        obs_s, info_s = env_sr.reset(seed=seed_ep, options=None)

        # sanity check: initial t0/t should match
        sb = snapshot_env_state(env_base)
        ss = snapshot_env_state(env_sr)
        print(f"[EVAL] baseline env t0/t/step={sb['_t0']}/{sb['_t']}/{sb['_step_in_episode']} prev_action={sb['_prev_action']}")
        print(f"[EVAL] SR       env t0/t/step={ss['_t0']}/{ss['_t']}/{ss['_step_in_episode']} prev_action={ss['_prev_action']}")

        done_b = False
        done_s = False
        Rb = 0.0
        Rs = 0.0

        max_steps = int(env_cfg.episode_len if max_steps_per_ep is None else max_steps_per_ep)

        for t in range(max_steps):
            print("\n" + "-" * 110)
            print(f"[STEP] t={t+1}/{max_steps}")

            # =========================
            # baseline: sample 1 action
            # =========================
            obs_b_flat = flattener.flatten(obs_b)
            gt_b = np.asarray(obs_b["gt_H"], dtype=np.float32)

            base_act_info = baseline_sample_action(
                model=model,
                normalizer=normalizer,
                obs_flat_np=obs_b_flat,
                gt_H_np=gt_b,
                device=device,
            )
            a_b = int(base_act_info["action"])
            print(f"[BASE] policy top-8 probs: {base_act_info['topk']}")
            print(f"[BASE] sampled action={a_b} logp={base_act_info['logp']:.6f} entropy={base_act_info['entropy']:.6f}")

            # compute baseline r_hat (for printing only)
            tot_hat_b, risk_hat_b, rhat_b = sr_predict_cost_and_rhat(
                model=model,
                normalizer=normalizer,
                obs_flat_np=obs_b_flat,
                gt_H_np=gt_b,
                action=a_b,
                device=device,
                lambda_risk=lambda_risk,
            )
            print(f"[BASE] SR-pred @root: total_hat={tot_hat_b:.6e} risk_hat={risk_hat_b:.6e} r_hat={rhat_b:.6e}")

            obs_b2, r_b, term_b, trunc_b, info_b2 = env_base.step(a_b)
            done_b = bool(term_b or trunc_b)
            Rb += float(r_b)

            # =========================
            # SR: plan with look-ahead
            # =========================
            obs_s_flat = flattener.flatten(obs_s)
            gt_s = np.asarray(obs_s["gt_H"], dtype=np.float32)

            plan = plan_action_sr_lookahead_beam(
                env_real=env_sr,
                model=model,
                flattener=flattener,
                normalizer=normalizer,
                obs_flat_np=obs_s_flat,
                gt_H_np=gt_s,
                device=device,
                depth=lookahead_depth,
                gamma=gamma,
                k_candidates=k_candidates,
                beam_width=beam_width,
                lambda_risk=lambda_risk,
                verbose=True,
            )
            a_s = int(plan["chosen_action"])
            print(f"[SR] chosen_action={a_s} best_seq={plan['best_seq']} best_return_hat={plan['best_return_hat']:.6e}")

            # Also compute SR immediate r_hat at chosen action for printing
            tot_hat_s, risk_hat_s, rhat_s = sr_predict_cost_and_rhat(
                model=model,
                normalizer=normalizer,
                obs_flat_np=obs_s_flat,
                gt_H_np=gt_s,
                action=a_s,
                device=device,
                lambda_risk=lambda_risk,
            )
            print(f"[SR] SR-pred @root(chosen): total_hat={tot_hat_s:.6e} risk_hat={risk_hat_s:.6e} r_hat={rhat_s:.6e}")

            obs_s2, r_s, term_s, trunc_s, info_s2 = env_sr.step(a_s)
            done_s = bool(term_s or trunc_s)
            Rs += float(r_s)

            # =========================
            # Print real reward + diff
            # =========================
            # env real reward is negative total_cost_yuan (dimensioned mode). Print both if available
            real_cost_b = safe_float(info_b2.get("total_cost_yuan", float("nan")), float("nan"))
            real_cost_s = safe_float(info_s2.get("total_cost_yuan", float("nan")), float("nan"))

            print("[REAL] baseline: reward={:.6e} total_cost_yuan={:.6e} risk_cost_yuan={:.6e}".format(
                float(r_b),
                real_cost_b,
                safe_float(info_b2.get("risk_cost_yuan", info_b2.get("risk_term", 0.0)), 0.0),
            ))
            print("[REAL] SR      : reward={:.6e} total_cost_yuan={:.6e} risk_cost_yuan={:.6e}".format(
                float(r_s),
                real_cost_s,
                safe_float(info_s2.get("risk_cost_yuan", info_s2.get("risk_term", 0.0)), 0.0),
            ))

            # difference: true reward difference and in-mind difference
            diff_real = float(r_s) - float(r_b)
            diff_rhat = float(rhat_s) - float(rhat_b)
            print(f"[DIFF] (SR - BASE) real_reward_diff={diff_real:.6e}  in_mind_r_hat_diff={diff_rhat:.6e}")
            print(f"[DIFF] (SR - BASE) |real - in_mind| baseline={abs(float(r_b) - float(rhat_b)):.6e} SR={abs(float(r_s) - float(rhat_s)):.6e}")

            per_step_records.append({
                "episode": ep,
                "t": t,
                "a_base": a_b,
                "a_sr": a_s,
                "r_base": float(r_b),
                "r_sr": float(r_s),
                "rhat_base": float(rhat_b),
                "rhat_sr": float(rhat_s),
                "diff_real": diff_real,
                "diff_rhat": diff_rhat,
                "base_total_cost_yuan": real_cost_b,
                "sr_total_cost_yuan": real_cost_s,
            })

            obs_b, obs_s = obs_b2, obs_s2

            # terminate if either ends (should be same episode_len typically)
            if done_b or done_s:
                print(f"[DONE] done_b={done_b} done_s={done_s} at t={t+1}")
                break

        print("\n" + "=" * 120)
        print(f"[EP SUMMARY] baseline_return={Rb:.6e}  SR_return={Rs:.6e}  (SR-BASE)={Rs-Rb:.6e}")
        ep_returns_base.append(float(Rb))
        ep_returns_sr.append(float(Rs))

    # Final summary
    print("\n" + "#" * 120)
    print("[FINAL SUMMARY]")
    print(f"episodes={n_episodes}")
    print(f"baseline: mean={np.mean(ep_returns_base):.6e} std={np.std(ep_returns_base):.6e}")
    print(f"SR      : mean={np.mean(ep_returns_sr):.6e} std={np.std(ep_returns_sr):.6e}")
    print(f"diff(SR-BASE): mean={np.mean(np.asarray(ep_returns_sr)-np.asarray(ep_returns_base)):.6e}")

    # Step-level diff stats
    diffs_real = np.array([r["diff_real"] for r in per_step_records], dtype=np.float64)
    diffs_rhat = np.array([r["diff_rhat"] for r in per_step_records], dtype=np.float64)
    print("\n[STEP DIFF STATS]")
    print(f"real_reward_diff (SR-BASE): mean={np.mean(diffs_real):.6e} std={np.std(diffs_real):.6e} min={np.min(diffs_real):.6e} max={np.max(diffs_real):.6e}")
    print(f"in_mind_r_hat_diff(SR-BASE): mean={np.mean(diffs_rhat):.6e} std={np.std(diffs_rhat):.6e} min={np.min(diffs_rhat):.6e} max={np.max(diffs_rhat):.6e}")

    # Print a few largest absolute step diffs for inspection
    idx_sorted = np.argsort(-np.abs(diffs_real))
    print("\n[TOP |real_reward_diff| steps]")
    for j in idx_sorted[: min(10, len(idx_sorted))]:
        rec = per_step_records[int(j)]
        print(f"  ep={rec['episode']} t={rec['t']} a_base={rec['a_base']} a_sr={rec['a_sr']} "
              f"r_base={rec['r_base']:.6e} r_sr={rec['r_sr']:.6e} diff_real={rec['diff_real']:.6e} "
              f"rhat_base={rec['rhat_base']:.6e} rhat_sr={rec['rhat_sr']:.6e} diff_rhat={rec['diff_rhat']:.6e}")


# ============================================================
# Entry: edit parameters here
# ============================================================

if __name__ == "__main__":
    # -----------------------------
    # Your trained artifacts
    # -----------------------------
    RUN_DIR = "./runs_ppo_gt_sr_warmup_joint"
    CKPT_PATH = os.path.join(RUN_DIR, "ppo_gt_sr_ckpt_update0120.pt")  # choose your checkpoint
    META_JSON = os.path.join(RUN_DIR, "train_meta_sr_warmup.json")     # your meta (as you described)

    # -----------------------------
    # Device
    # -----------------------------
    DEVICE = "cuda"  # "cpu" or "cuda"

    # -----------------------------
    # Evaluation setup
    # -----------------------------
    N_EPISODES = 2
    BASE_SEED = 0

    # -----------------------------
    # SR look-ahead planning params
    # -----------------------------
    LOOKAHEAD_DEPTH = 3      # required by your request
    GAMMA = 0.99
    K_CANDIDATES = 12        # number of candidates sampled each expansion
    BEAM_WIDTH = 6           # number of beams kept

    # SR reward definition: r_hat = -(total_hat + lambda_risk * risk_hat)
    LAMBDA_RISK = 0.0        # set as you wish; your training uses 0.0

    # limit steps per episode for quick debug (None -> full episode_len)
    MAX_STEPS_PER_EP = None  # e.g. 5 for quick test

    evaluate_compare(
        ckpt_path=CKPT_PATH,
        meta_json_path=META_JSON,
        device_str=DEVICE,
        n_episodes=N_EPISODES,
        base_seed=BASE_SEED,
        lookahead_depth=LOOKAHEAD_DEPTH,
        gamma=GAMMA,
        k_candidates=K_CANDIDATES,
        beam_width=BEAM_WIDTH,
        lambda_risk=LAMBDA_RISK,
        max_steps_per_ep=MAX_STEPS_PER_EP,
    )

# %%
