# -*- coding: utf-8 -*-
"""
power_dispatch_env_withGT.py

PowerDispatchEnv: 配电系统运行方式/重构决策环境（含 GTransformer 所需 gt_H 节点特征）

本版本针对你当前整合 GT 的实现做了“工程级稳健化修复”，重点修复：
1) bus 向量顺序一致性：bus_vm_pu / bus_va_deg / gt_H 统一按 sorted(bus.index) 输出；
2) switch_cost 解析：支持 feasible_switch_states={'0':[ACline...],'1':[...]} 的真实结构；
3) node_order_file 多进程竞争：增加跨进程锁 + 原子写，避免 JSON 被并发写坏；
4) PF 失败观测 shape 固定，避免上层崩溃。

不使用 argparse；参数见 EnvConfig。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Callable, List, Set, Iterable
import os
import copy
import json
import hashlib
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd

import gym
from gym import spaces

import pandapower as pp  # type: ignore
import config746sys as cfg  # type: ignore

from Utls.yantian_sys_746sys import (  # type: ignore
    CimEParser,
    PandaPowerFlowCalculator,
    load_feasible_feeders_switch_states,
    set_fc_state_with_acts,
)

# -----------------------------
# Inter-process lock + atomic json
# -----------------------------
@contextmanager
def _interprocess_file_lock(lock_path: str, timeout: float = 60.0, poll: float = 0.05):
    """
    Best-effort cross-process file lock.
    - On POSIX: fcntl.flock
    - On Windows: msvcrt.locking
    Fallback: no-op (still safe-ish because we use atomic replace, but read may race with write).
    """
    os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
    fp = open(lock_path, "a+", encoding="utf-8")
    start = time.time()
    locked = False
    try:
        while time.time() - start < timeout:
            try:
                if os.name == "posix":
                    import fcntl  # type: ignore
                    fcntl.flock(fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                else:
                    import msvcrt  # type: ignore
                    msvcrt.locking(fp.fileno(), msvcrt.LK_NBLCK, 1)
                locked = True
                break
            except Exception:
                time.sleep(poll)
        yield locked
    finally:
        try:
            if locked:
                if os.name == "posix":
                    import fcntl  # type: ignore
                    fcntl.flock(fp.fileno(), fcntl.LOCK_UN)
                else:
                    import msvcrt  # type: ignore
                    try:
                        fp.seek(0)
                        msvcrt.locking(fp.fileno(), msvcrt.LK_UNLCK, 1)
                    except Exception:
                        pass
        finally:
            try:
                fp.close()
            except Exception:
                pass


def _atomic_write_json(path: str, obj: Any) -> None:
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


def _read_json_with_retries(path: str, retries: int = 20, sleep: float = 0.05) -> Any:
    last_err = None
    for _ in range(int(retries)):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            last_err = e
            time.sleep(float(sleep))
    raise RuntimeError(f"Failed to read json {path}: {repr(last_err)}")


def enforce_node_order(pp_pf_calculator: Any, order_file: str) -> None:
    """
    Ensure internal ordering of ele_nodes is stable across runs.
    - Uses file lock to avoid concurrent init races
    - Uses atomic replace to avoid truncated JSON
    """
    lock_path = order_file + ".lock"
    with _interprocess_file_lock(lock_path, timeout=120.0) as locked:
        if os.path.exists(order_file):
            try:
                saved_order = _read_json_with_retries(order_file)
                order_map = {nid: idx for idx, nid in enumerate(saved_order)}
                pp_pf_calculator.ele_nodes.sort(key=lambda n: order_map.get(n.nd, len(saved_order)))
                return
            except Exception:
                # fall through to regenerate
                pass

        saved_order = [n.nd for n in pp_pf_calculator.ele_nodes]
        _atomic_write_json(order_file, saved_order)
        order_map = {nid: idx for idx, nid in enumerate(saved_order)}
        pp_pf_calculator.ele_nodes.sort(key=lambda n: order_map.get(n.nd, len(saved_order)))


def built_ppnet_for_pfcal(ori_ppnet: Any, r_switch: float, x_switch: float) -> Any:
    """
    Convert "switch-lines" (r=r_switch, x=x_switch) into pandapower bus-bus switches.
    """
    base_net_replacement = copy.deepcopy(ori_ppnet)
    indices_to_drop: List[int] = []

    if hasattr(base_net_replacement, "line") and len(base_net_replacement.line) > 0:
        for i in list(base_net_replacement.line.index):
            r = base_net_replacement.line.at[i, "r_ohm_per_km"]
            x = base_net_replacement.line.at[i, "x_ohm_per_km"]
            if r == r_switch and x == x_switch:
                pp.create_switch(
                    base_net_replacement,
                    bus=int(base_net_replacement.line.at[i, "from_bus"]),
                    element=int(base_net_replacement.line.at[i, "to_bus"]),
                    et="b",
                    closed=True,
                )
                indices_to_drop.append(i)

    if indices_to_drop:
        base_net_replacement.line.drop(indices_to_drop, inplace=True)
    return base_net_replacement


# -----------------------------
# Timeseries adapter
# -----------------------------
@dataclass
class TimeseriesSchema:
    p_col: str = "P"
    q_col: str = "Q"
    id_col: str = "GISID"
    time_col: str = "time"
    p_unit: str = "MW"    # {"MW","kW"}
    q_unit: str = "MVar"  # {"MVar","kVar"}
    timezone: Optional[str] = None


class TimeseriesDataAdapter:
    def __init__(self, df: pd.DataFrame, schema: TimeseriesSchema, *, freq: Optional[str] = None):
        self.schema = schema
        if isinstance(df.columns, pd.MultiIndex):
            self._init_from_wide_multiindex(df, freq=freq)
        else:
            self._init_from_long(df, freq=freq)

    def _init_from_wide_multiindex(self, df: pd.DataFrame, freq: Optional[str]) -> None:
        if df.index.name is None:
            df.index.name = "_ts"

        ts = pd.to_datetime(df.index)
        if self.schema.timezone:
            if ts.tz is None:
                ts = ts.tz_localize(self.schema.timezone)
            else:
                ts = ts.tz_convert(self.schema.timezone)

        df = df.copy()
        df.index = ts
        df.sort_index(inplace=True)

        lvl0 = set([str(x) for x in df.columns.get_level_values(0)])
        if not (("P" in lvl0) and ("Q" in lvl0)):
            raise ValueError("Wide format requires MultiIndex columns with level0 containing 'P' and 'Q'.")

        P = df.xs("P", axis=1, level=0, drop_level=True)
        Q = df.xs("Q", axis=1, level=0, drop_level=True)

        P.columns = P.columns.astype(str)
        Q.columns = Q.columns.astype(str)

        if self.schema.p_unit.lower() == "kw":
            P = P.astype(np.float32) / 1000.0
        elif self.schema.p_unit.lower() == "mw":
            P = P.astype(np.float32)
        else:
            raise ValueError(f"Unsupported p_unit={self.schema.p_unit}")

        if self.schema.q_unit.lower() == "kvar":
            Q = Q.astype(np.float32) / 1000.0
        elif self.schema.q_unit.lower() == "mvar":
            Q = Q.astype(np.float32)
        else:
            raise ValueError(f"Unsupported q_unit={self.schema.q_unit}")

        if freq is not None:
            P = P.resample(freq).mean().ffill()
            Q = Q.resample(freq).mean().ffill()

        self.index = P.index
        self.gisids = list(P.columns)
        self._P = P
        self._Q = Q

    def _init_from_long(self, df: pd.DataFrame, freq: Optional[str]) -> None:
        schema = self.schema
        tcol = schema.time_col
        idcol = schema.id_col
        pcol = schema.p_col
        qcol = schema.q_col

        df = df.copy()
        df[tcol] = pd.to_datetime(df[tcol])

        if schema.timezone:
            if df[tcol].dt.tz is None:
                df[tcol] = df[tcol].dt.tz_localize(schema.timezone)
            else:
                df[tcol] = df[tcol].dt.tz_convert(schema.timezone)

        df.sort_values([tcol, idcol], inplace=True)

        P = df.pivot(index=tcol, columns=idcol, values=pcol)
        Q = df.pivot(index=tcol, columns=idcol, values=qcol)

        P.columns = P.columns.astype(str)
        Q.columns = Q.columns.astype(str)

        if schema.p_unit.lower() == "kw":
            P = P.astype(np.float32) / 1000.0
        elif schema.p_unit.lower() == "mw":
            P = P.astype(np.float32)
        else:
            raise ValueError(f"Unsupported p_unit={schema.p_unit}")

        if schema.q_unit.lower() == "kvar":
            Q = Q.astype(np.float32) / 1000.0
        elif schema.q_unit.lower() == "mvar":
            Q = Q.astype(np.float32)
        else:
            raise ValueError(f"Unsupported q_unit={schema.q_unit}")

        if freq is not None:
            P = P.resample(freq).mean().ffill()
            Q = Q.resample(freq).mean().ffill()

        self.index = P.index
        self.gisids = list(P.columns)
        self._P = P
        self._Q = Q

    def __len__(self) -> int:
        return len(self.index)

    def get_pq(self, t: int, gisids: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        row_p = self._P.iloc[t]
        row_q = self._Q.iloc[t]
        p = row_p.reindex(gisids).fillna(0.0).to_numpy(dtype=np.float32)
        q = row_q.reindex(gisids).fillna(0.0).to_numpy(dtype=np.float32)
        return p, q

    def get_forecast_pq(self, t: int, horizon: int, gisids: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        horizon = int(max(horizon, 0))
        if horizon == 0:
            return (
                np.zeros((0, len(gisids)), dtype=np.float32),
                np.zeros((0, len(gisids)), dtype=np.float32),
            )

        end = min(t + horizon, len(self.index) - 1)
        steps = list(range(t + 1, end + 1))
        if len(steps) == 0:
            p = np.zeros((horizon, len(gisids)), dtype=np.float32)
            q = np.zeros_like(p)
            return p, q

        P = self._P.iloc[steps].reindex(columns=gisids).fillna(0.0).to_numpy(dtype=np.float32)
        Q = self._Q.iloc[steps].reindex(columns=gisids).fillna(0.0).to_numpy(dtype=np.float32)

        if P.shape[0] < horizon:
            pad_n = horizon - P.shape[0]
            P = np.vstack([P, np.repeat(P[-1:, :], pad_n, axis=0)])
            Q = np.vstack([Q, np.repeat(Q[-1:, :], pad_n, axis=0)])

        return P, Q


# -----------------------------
# Env config
# -----------------------------
@dataclass
class EnvConfig:
    # --- system / files ---
    pf_data_path: str = cfg.PfDataPath
    dist_tf_path: str = cfg.DistTfPath
    feasible_switch_state_path: str = os.path.join(cfg.WORKPATH, "system_file/746sys/")
    node_order_file: str = os.path.join(cfg.WORKPATH, "system_file/746sys/746sys_node_order.json")

    # --- timeseries ---
    timeseries_mode: str = "generated_parquet"
    generated_timeseries_path: str = os.path.join(cfg.WORKPATH, "system_file/virtual_timeseries_hourly.parquet")
    prefiltered_timeseries_csv: str = os.path.join(cfg.WORKPATH, "system_file/402system_timeseries_data.csv")
    timeseries_schema: TimeseriesSchema = TimeseriesSchema(p_unit="MW", q_unit="MVar")
    timeseries_freq: Optional[str] = None

    # --- episode / horizon ---
    episode_len: int = 24
    forecast_horizon: int = int(getattr(cfg, "T", 6))

    # --- PF / security limits ---
    vmin: float = 0.95
    vmax: float = 1.05
    line_loading_limit: float = 100.0
    pf_max_iter: int = 20
    pf_tol: float = 1e-6

    # --- reward weights ---
    w_loss: float = 1.0
    w_v: float = 10.0
    w_line: float = 2.0
    w_switch: float = float(getattr(cfg, "k_act_swit", 0.0))
    w_trafo_balance: float = 5.0
    fail_penalty: float = 1e4

    # --- aggregation settings ---
    voltage_viol_agg: str = "mean"  # {"mean","max","ratio"}
    line_viol_agg: str = "mean"     # {"mean","max","ratio"}
    trafo_balance_metric: str = "cv"  # {"cv","std","range"}

    # --- reset / randomness ---
    random_reset: bool = bool(getattr(cfg, "random_reset", True))
    seed: Optional[int] = 0

    # --- switching cost ---
    switch_cost_mode: str = "hamming"  # {"hamming","hamming_norm"}

    # --- GTransformer graph-state (work1) ---
    include_gt_H: bool = True
    gt_node_feat_dim: int = 6

    # If you want va_degree to match pretraining, enable voltage angles in pp.runpp.
    calculate_voltage_angles: bool = True


# -----------------------------
# Main environment
# -----------------------------
class PowerDispatchEnv(gym.Env):
    """
    Action: Discrete(a): choose a feasible switch-state index.

    Observation (dict):
        - "bus_vm_pu": (n_bus,) float32
        - "bus_va_deg": (n_bus,) float32
        - "load_p_mw": (n_load,) float32
        - "load_q_mvar": (n_load,) float32
        - "forecast_p_mw": (H, n_load) float32
        - "forecast_q_mvar": (H, n_load) float32
        - "time_index": (1,) int32
        - "topology_id": (1,) int32
        - "gt_H": (n_bus, 6) float32

    Reward:
        r = - w_loss*losses - w_v*V_viol - w_line*Line_over - w_switch*Switch_cost - w_trafo_balance*Trafo_balance
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        env_cfg: Optional[EnvConfig] = None,
        *,
        system_builder: Optional[Callable[[EnvConfig], Tuple[Any, Any, Any]]] = None,
        timeseries_df: Optional[pd.DataFrame] = None,
    ):
        super().__init__()
        self.cfg = env_cfg or EnvConfig()
        self._rng = np.random.default_rng(self.cfg.seed)

        # --- Build power system model ---
        if system_builder is None:
            system_builder = self._default_system_builder
        self.pp_pf_calculator, self.feeder_cluster, self.base_net = system_builder(self.cfg)

        # --- bus ordering (single source of truth) ---
        self.gt_node_feat_dim = int(getattr(self.cfg, "gt_node_feat_dim", 6))
        self._bus_ids: List[int] = [int(b) for b in sorted(self.base_net.bus.index)]
        self._bus_map: Dict[int, int] = {bid: i for i, bid in enumerate(self._bus_ids)}
        self._n_bus: int = int(len(self._bus_ids))

        # --- precompute total number of switch-lines for normalized cost ---
        self._n_switch_total: Optional[int] = None
        try:
            sw_lines = getattr(self.feeder_cluster, "switch_lines", None)
            if sw_lines is not None:
                self._n_switch_total = int(len(sw_lines))
        except Exception:
            self._n_switch_total = None

        # --- Timeseries ---
        if timeseries_df is None:
            timeseries_df = self._default_timeseries_loader(self.cfg)
        self.ts = TimeseriesDataAdapter(timeseries_df, self.cfg.timeseries_schema, freq=self.cfg.timeseries_freq)

        # --- Map pandapower loads to GISIDs ---
        self._load_gisids = self._infer_load_gisids()
        self._n_load = len(self._load_gisids)

        # --- Action space ---
        self.n_actions = int(len(self.feeder_cluster.feasible_switch_states))
        self.action_space = spaces.Discrete(self.n_actions)

        # --- Observation space ---
        H = int(self.cfg.forecast_horizon)
        self.observation_space = spaces.Dict(
            {
                "bus_vm_pu": spaces.Box(low=0.0, high=2.0, shape=(self._n_bus,), dtype=np.float32),
                "bus_va_deg": spaces.Box(low=-360.0, high=360.0, shape=(self._n_bus,), dtype=np.float32),
                "load_p_mw": spaces.Box(low=-1e3, high=1e3, shape=(self._n_load,), dtype=np.float32),
                "load_q_mvar": spaces.Box(low=-1e3, high=1e3, shape=(self._n_load,), dtype=np.float32),
                "forecast_p_mw": spaces.Box(low=-1e3, high=1e3, shape=(H, self._n_load), dtype=np.float32),
                "forecast_q_mvar": spaces.Box(low=-1e3, high=1e3, shape=(H, self._n_load), dtype=np.float32),
                "time_index": spaces.Box(low=0, high=10**9, shape=(1,), dtype=np.int32),
                "topology_id": spaces.Box(low=0, high=max(self.n_actions - 1, 0), shape=(1,), dtype=np.int32),
                "gt_H": spaces.Box(low=-1e6, high=1e6, shape=(self._n_bus, self.gt_node_feat_dim), dtype=np.float32),
            }
        )

        # Episode state
        self._t0: int = 0
        self._t: int = 0
        self._step_in_episode: int = 0
        self._prev_action: Optional[int] = None

        # Cached nets
        self._net_cal: Any = None
        self._trafo_sel_idx: Optional[np.ndarray] = None

    # -----------------------------
    # switch-cost utils (robust)
    # -----------------------------
    @staticmethod
    def _undirected_pair(a: Any, b: Any) -> Tuple[str, str]:
        sa = str(a)
        sb = str(b)
        return (sa, sb) if sa <= sb else (sb, sa)

    def _line_token(self, obj: Any) -> Any:
        """
        Convert a switch-line object (ACline) into a stable token for set comparison.
        Priority:
            (min_bus,max_bus) if available -> (min_nd,max_nd) -> name string -> repr hash
        """
        try:
            I = getattr(obj, "I_nd", None)
            J = getattr(obj, "J_nd", None)
            if I is not None and J is not None:
                # prefer bus ids
                ib = getattr(I, "bus", None)
                jb = getattr(J, "bus", None)
                if ib is not None and jb is not None:
                    a = int(ib); b = int(jb)
                    return ("bus", min(a, b), max(a, b))
                ind = getattr(I, "nd", None)
                jnd = getattr(J, "nd", None)
                if ind is not None and jnd is not None:
                    a, b = self._undirected_pair(ind, jnd)
                    return ("nd", a, b)
        except Exception:
            pass

        try:
            nm = getattr(obj, "name", None)
            tp = getattr(obj, "device_type", None)
            if nm is not None:
                return ("name", str(tp), str(nm))
        except Exception:
            pass

        # fallback: stable-ish hash of repr
        s = repr(obj)
        h = hashlib.md5(s.encode("utf-8")).hexdigest()[:16]
        return ("repr", h)

    def _state_to_open_tokens(self, st: Any) -> Tuple[Set[Any], Optional[int]]:
        """
        Convert feasible_switch_state into:
        - open_set: set of stable tokens for OPEN switch-lines
        - n_total: total number of switch-lines (if known)
        """
        open_set: Set[Any] = set()
        n_total: Optional[int] = self._n_switch_total

        if isinstance(st, dict):
            # canonical yantian: {'0':[open_lines], '1':[closed_lines]}
            if "0" in st:
                try:
                    for x in list(st["0"]):
                        open_set.add(self._line_token(x))
                except Exception:
                    pass
            else:
                # alternative keys
                for k in ("open_switches", "open_switch", "open", "opened", "open_sws", "open_sw"):
                    if k in st:
                        try:
                            for x in list(st[k]):
                                open_set.add(self._line_token(x))
                        except Exception:
                            pass
                        break

            if n_total is None:
                try:
                    # yantian uses switch_lines for total
                    n_total = int(len(getattr(self.feeder_cluster, "switch_lines", [])))
                except Exception:
                    n_total = None
            return open_set, n_total

        # numeric vector: interpret 0 as open
        if isinstance(st, (list, tuple, np.ndarray)):
            arr = np.asarray(st)
            try:
                arr_int = arr.astype(int).reshape(-1)
                for i, v in enumerate(arr_int.tolist()):
                    if int(v) == 0:
                        open_set.add(("idx", int(i)))
                n_total = int(len(arr_int))
                return open_set, n_total
            except Exception:
                return open_set, n_total

        return open_set, n_total

    def _calc_switch_cost(self, prev_action: Optional[int], action: int) -> Tuple[float, Dict[str, Any]]:
        if prev_action is None:
            return 0.0, {"switch_cost": 0.0, "mode": self.cfg.switch_cost_mode}

        st_prev = self.feeder_cluster.feasible_switch_states[int(prev_action)]
        st_curr = self.feeder_cluster.feasible_switch_states[int(action)]

        open_prev, n_total_prev = self._state_to_open_tokens(st_prev)
        open_curr, n_total_curr = self._state_to_open_tokens(st_curr)

        diff = len(open_prev.symmetric_difference(open_curr))
        n_total = n_total_prev if n_total_prev is not None else n_total_curr

        if self.cfg.switch_cost_mode == "hamming_norm":
            denom = float(n_total) if (n_total is not None and n_total > 0) else 1.0
            cost = float(diff) / denom
        else:
            cost = float(diff)

        return cost, {
            "switch_cost": float(cost),
            "diff": int(diff),
            "n_total": None if n_total is None else int(n_total),
            "mode": self.cfg.switch_cost_mode,
        }

    # -----------------------------
    # Transformer balance penalty
    # -----------------------------
    def _select_10_110_trafos(self, net_cal: Any) -> np.ndarray:
        if self._trafo_sel_idx is not None:
            return self._trafo_sel_idx

        if (not hasattr(net_cal, "trafo")) or (len(net_cal.trafo) == 0):
            self._trafo_sel_idx = np.array([], dtype=int)
            return self._trafo_sel_idx

        trafo = net_cal.trafo
        if ("vn_hv_kv" in trafo.columns) and ("vn_lv_kv" in trafo.columns):
            hv = trafo["vn_hv_kv"].to_numpy(dtype=float)
            lv = trafo["vn_lv_kv"].to_numpy(dtype=float)
            sel = np.where((hv >= 35.0) & (hv <= 220.0) & (lv >= 1.0) & (lv <= 35.0))[0]
            self._trafo_sel_idx = sel.astype(int)
        else:
            self._trafo_sel_idx = np.arange(len(trafo), dtype=int)

        return self._trafo_sel_idx

    def _calc_trafo_balance_penalty(self, net_cal: Any, metric: str = "cv") -> Tuple[float, Dict[str, Any]]:
        if (not hasattr(net_cal, "res_trafo")) or (len(net_cal.res_trafo) == 0):
            return 0.0, {"metric": metric, "n": 0}

        sel = self._select_10_110_trafos(net_cal)
        if sel.size == 0:
            return 0.0, {"metric": metric, "n": 0}

        loading = net_cal.res_trafo.iloc[sel]["loading_percent"].to_numpy(dtype=float)
        if loading.size == 0:
            return 0.0, {"metric": metric, "n": 0}

        mu = float(np.mean(loading))
        sd = float(np.std(loading))

        if metric == "std":
            pen = sd
        elif metric == "range":
            pen = float(np.max(loading) - np.min(loading))
        else:
            pen = sd / (mu + 1e-6)

        return float(pen), {"metric": metric, "n": int(loading.size), "mean": mu, "std": sd}

    # -----------------------------
    # Builders / loaders
    # -----------------------------
    def _default_system_builder(self, cfg_: EnvConfig) -> Tuple[Any, Any, Any]:
        parsed_cim = CimEParser(cfg_.pf_data_path)
        pp_pf_calculator = PandaPowerFlowCalculator(parsed_cim, slack_nd="703002137")
        load_feasible_feeders_switch_states(pp_pf_calculator, path=cfg_.feasible_switch_state_path)

        dist_tf = pd.read_csv(cfg_.dist_tf_path, sep="\t")
        for i, node_bus in dist_tf["Node"].items():
            node = pp_pf_calculator.bus2node[int(node_bus)]
            node.pl = float(dist_tf["P"][i])
            node.ql = float(dist_tf["Q"][i])

        pp_pf_calculator.ele_nodes = [node for node in pp_pf_calculator.ele_nodes]
        enforce_node_order(pp_pf_calculator, order_file=cfg_.node_order_file)

        node_ids = [node.nd for node in pp_pf_calculator.ele_nodes]

        base_net = pp_pf_calculator.create_pandapower_net_from_node_ids(node_ids)
        slack_bus = pp_pf_calculator.nodeID2node[pp_pf_calculator.slack_nd].bus
        pp.create_ext_grid(base_net, bus=slack_bus, vm_pu=1.0, va_degree=0.0)

        feeder_cluster = pp_pf_calculator.feeder_clusters[0]
        return pp_pf_calculator, feeder_cluster, base_net

    def _default_timeseries_loader(self, cfg_: EnvConfig) -> pd.DataFrame:
        if cfg_.timeseries_mode == "generated_parquet":
            if not os.path.exists(cfg_.generated_timeseries_path):
                raise FileNotFoundError(cfg_.generated_timeseries_path)
            return pd.read_parquet(cfg_.generated_timeseries_path)
        elif cfg_.timeseries_mode == "prefiltered_csv":
            if not os.path.exists(cfg_.prefiltered_timeseries_csv):
                raise FileNotFoundError(cfg_.prefiltered_timeseries_csv)
            return pd.read_csv(cfg_.prefiltered_timeseries_csv)
        else:
            raise ValueError(f"Unknown timeseries_mode: {cfg_.timeseries_mode}")

    def _infer_load_gisids(self) -> List[str]:
        nd2gisid = getattr(cfg, "nd2gisid", {})
        if not isinstance(nd2gisid, dict) or len(nd2gisid) == 0:
            raise RuntimeError("cfg.nd2gisid is missing/empty. Please check config746sys.py and related npz files.")

        load_gisids: List[str] = []
        for load_idx in self.base_net.load.index:
            bus = int(self.base_net.load.at[load_idx, "bus"])
            node = self.pp_pf_calculator.bus2node[bus]
            nd = str(node.nd)
            gisid = nd2gisid.get(nd, None)
            if gisid is None:
                gisid = f"__missing__{nd}"
            load_gisids.append(str(gisid))
        return load_gisids

    # -----------------------------
    # Gym API
    # -----------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if self.cfg.random_reset:
            self._t0 = int(self._rng.integers(0, max(len(self.ts) - self.cfg.episode_len - 1, 1)))
        else:
            self._t0 = 0

        self._t = self._t0
        self._step_in_episode = 0

        init_action = int(self._rng.integers(0, self.n_actions))
        self._prev_action = init_action

        net_line_repr = set_fc_state_with_acts(self.feeder_cluster, self.base_net, [init_action])
        obs, info = self._solve_and_build_obs(net_line_repr, self._t, prev_action=None, action=init_action)
        return obs, info

    def step(self, action: int):
        action = int(action)
        self._t += 1
        self._step_in_episode += 1

        net_line_repr = set_fc_state_with_acts(self.feeder_cluster, self.base_net, [action])
        obs, info = self._solve_and_build_obs(net_line_repr, self._t, prev_action=self._prev_action, action=action)

        reward = float(info.get("reward", 0.0))
        terminated = False
        truncated = self._step_in_episode >= self.cfg.episode_len

        self._prev_action = action
        return obs, reward, terminated, truncated, info

    # -----------------------------
    # Core transition
    # -----------------------------
    def _apply_timeseries_loads(self, net_cal: Any, t: int) -> None:
        p, q = self.ts.get_pq(t, self._load_gisids)
        if len(net_cal.load) != len(p):
            raise RuntimeError(f"load length mismatch: net_cal.load={len(net_cal.load)} vs timeseries={len(p)}")
        net_cal.load.loc[:, "p_mw"] = p
        net_cal.load.loc[:, "q_mvar"] = q

    def _solve_and_build_obs(
        self,
        net_line_repr: Any,
        t: int,
        *,
        prev_action: Optional[int],
        action: int,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        info: Dict[str, Any] = {
            "t": int(t),
            "action": int(action),
            "prev_action": None if prev_action is None else int(prev_action),
        }

        net_cal = built_ppnet_for_pfcal(net_line_repr, r_switch=cfg.r_switch, x_switch=cfg.x_switch)
        self._apply_timeseries_loads(net_cal, t)

        pf_failed = False
        try:
            pp.runpp(
                net_cal,
                max_iteration=self.cfg.pf_max_iter,
                tolerance_mva=self.cfg.pf_tol,
                init="auto",
                calculate_voltage_angles=bool(getattr(self.cfg, "calculate_voltage_angles", False)),
            )
        except Exception as e:
            pf_failed = True
            info["pf_error"] = repr(e)

        info["pf_failed"] = pf_failed
        self._net_cal = net_cal

        if pf_failed:
            obs = self._zero_obs(t, topology_id=action)
            switch_cost, sw_stats = self._calc_switch_cost(prev_action, action)
            info.update(sw_stats)
            info.update(
                {
                    "reward": -float(self.cfg.fail_penalty),
                    "loss_term": np.nan,
                    "v_term": np.nan,
                    "line_term": np.nan,
                    "switch_term": float(self.cfg.w_switch * switch_cost),
                    "trafo_term": np.nan,
                }
            )
            return obs, info

        loss_mw = self._calc_losses_mw(net_cal)
        v_viol = self._calc_voltage_violation(net_cal, vmin=self.cfg.vmin, vmax=self.cfg.vmax, agg=self.cfg.voltage_viol_agg)
        line_viol = self._calc_line_overload(net_cal, limit_pct=self.cfg.line_loading_limit, agg=self.cfg.line_viol_agg)
        trafo_bal, trafo_stats = self._calc_trafo_balance_penalty(net_cal, metric=self.cfg.trafo_balance_metric)
        switch_cost, sw_stats = self._calc_switch_cost(prev_action, action)

        loss_term = float(self.cfg.w_loss * loss_mw)
        v_term = float(self.cfg.w_v * v_viol)
        line_term = float(self.cfg.w_line * line_viol)
        switch_term = float(self.cfg.w_switch * switch_cost)
        trafo_term = float(self.cfg.w_trafo_balance * trafo_bal)

        reward = -(loss_term + v_term + line_term + switch_term + trafo_term)

        obs = self._build_obs(net_cal, t, topology_id=action)

        info.update(sw_stats)
        info.update(trafo_stats)
        info.update(
            {
                "reward": float(reward),
                "loss_mw": float(loss_mw),
                "v_viol": float(v_viol),
                "line_viol": float(line_viol),
                "trafo_bal": float(trafo_bal),
                "loss_term": float(loss_term),
                "v_term": float(v_term),
                "line_term": float(line_term),
                "switch_term": float(switch_term),
                "trafo_term": float(trafo_term),
            }
        )
        return obs, info

    # -----------------------------
    # Observations
    # -----------------------------
    def _zero_obs(self, t: int = 0, topology_id: int = 0) -> Dict[str, np.ndarray]:
        H = int(self.cfg.forecast_horizon)
        return {
            "bus_vm_pu": np.zeros((self._n_bus,), dtype=np.float32),
            "bus_va_deg": np.zeros((self._n_bus,), dtype=np.float32),
            "load_p_mw": np.zeros((self._n_load,), dtype=np.float32),
            "load_q_mvar": np.zeros((self._n_load,), dtype=np.float32),
            "forecast_p_mw": np.zeros((H, self._n_load), dtype=np.float32),
            "forecast_q_mvar": np.zeros((H, self._n_load), dtype=np.float32),
            "time_index": np.array([int(t)], dtype=np.int32),
            "topology_id": np.array([int(topology_id)], dtype=np.int32),
            "gt_H": np.zeros((self._n_bus, self.gt_node_feat_dim), dtype=np.float32),
        }

    def _build_obs(self, net_cal: Any, t: int, topology_id: int) -> Dict[str, np.ndarray]:
        Hh = int(self.cfg.forecast_horizon)

        # 强制按固定 bus 顺序取 vm/va（修复 silent bug）
        res_bus = net_cal.res_bus.reindex(self._bus_ids)
        vm = res_bus["vm_pu"].to_numpy(dtype=np.float32)
        va = res_bus["va_degree"].to_numpy(dtype=np.float32)

        load_p = net_cal.load["p_mw"].to_numpy(dtype=np.float32)
        load_q = net_cal.load["q_mvar"].to_numpy(dtype=np.float32)

        fp, fq = self.ts.get_forecast_pq(t, Hh, self._load_gisids)

        gt_H = (
            self._build_gt_H(net_cal)
            if bool(getattr(self.cfg, "include_gt_H", True))
            else np.zeros((self._n_bus, self.gt_node_feat_dim), dtype=np.float32)
        )

        return {
            "bus_vm_pu": vm,
            "bus_va_deg": va,
            "load_p_mw": load_p,
            "load_q_mvar": load_q,
            "forecast_p_mw": fp,
            "forecast_q_mvar": fq,
            "time_index": np.array([int(t)], dtype=np.int32),
            "topology_id": np.array([int(topology_id)], dtype=np.int32),
            "gt_H": gt_H,
        }

    def _build_gt_H(self, net_cal: Any) -> np.ndarray:
        """
        H features: [load_p, load_q, gen_p, gen_q, vm, va]
        Bus order: sorted(net_cal.bus.index) == self._bus_ids
        """
        H = np.zeros((self._n_bus, self.gt_node_feat_dim), dtype=np.float32)

        # loads
        if hasattr(net_cal, "load") and len(net_cal.load) > 0:
            for _, load in net_cal.load.iterrows():
                try:
                    b = int(load.bus)
                except Exception:
                    continue
                i = self._bus_map.get(b, None)
                if i is None:
                    continue
                H[i, 0] += float(load.get("p_mw", 0.0))
                H[i, 1] += float(load.get("q_mvar", 0.0))

        # gens (exclude ext_grid by design)
        if hasattr(net_cal, "gen") and len(net_cal.gen) > 0:
            for _, gen in net_cal.gen.iterrows():
                try:
                    b = int(gen.bus)
                except Exception:
                    continue
                i = self._bus_map.get(b, None)
                if i is None:
                    continue
                H[i, 2] += float(gen.get("p_mw", 0.0))
                H[i, 3] += float(gen.get("q_mvar", 0.0))

        # vm/va (ordered)
        if hasattr(net_cal, "res_bus") and len(net_cal.res_bus) > 0:
            res_bus = net_cal.res_bus.reindex(self._bus_ids)
            H[:, 4] = res_bus["vm_pu"].to_numpy(dtype=np.float32)
            H[:, 5] = res_bus["va_degree"].to_numpy(dtype=np.float32)

        return H

    # -----------------------------
    # Penalty terms
    # -----------------------------
    @staticmethod
    def _calc_losses_mw(net_cal: Any) -> float:
        loss_mw = 0.0
        if hasattr(net_cal, "res_line") and (len(net_cal.res_line) > 0):
            loss_mw += float(net_cal.res_line["pl_mw"].sum())
        if hasattr(net_cal, "res_trafo") and (len(net_cal.res_trafo) > 0):
            loss_mw += float(net_cal.res_trafo["pl_mw"].sum())
        return float(loss_mw)

    @staticmethod
    def _calc_voltage_violation(net_cal: Any, vmin: float, vmax: float, agg: str = "mean") -> float:
        if (not hasattr(net_cal, "res_bus")) or (len(net_cal.res_bus) == 0):
            return 0.0
        vm = net_cal.res_bus["vm_pu"].to_numpy(dtype=float)
        below = np.maximum(0.0, vmin - vm)
        above = np.maximum(0.0, vm - vmax)
        viol = below + above
        if agg == "max":
            return float(np.max(viol))
        if agg == "ratio":
            return float(np.mean(viol > 0.0))
        return float(np.mean(viol))

    @staticmethod
    def _calc_line_overload(net_cal: Any, limit_pct: float, agg: str = "mean") -> float:
        if (not hasattr(net_cal, "res_line")) or (len(net_cal.res_line) == 0):
            return 0.0
        loading = net_cal.res_line["loading_percent"].to_numpy(dtype=float)
        viol = np.maximum(0.0, loading - float(limit_pct))
        if agg == "max":
            return float(np.max(viol))
        if agg == "ratio":
            return float(np.mean(viol > 0.0))
        return float(np.mean(viol))
