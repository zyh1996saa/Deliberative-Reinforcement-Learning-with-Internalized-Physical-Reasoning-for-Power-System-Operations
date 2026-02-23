# In[]
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Callable, List, Set

import os
import copy
import json
import hashlib

import numpy as np
import pandas as pd

import gym
from gym import spaces

from Utls.yantian_sys_746sys import (  # type: ignore
    CimEParser,
    PandaPowerFlowCalculator,
    load_feasible_feeders_switch_states,
    set_fc_state_with_acts,
)
import pandapower as pp  # type: ignore
import config746sys as cfg  # type: ignore


# -----------------------------
# Helper: node order consistency
# -----------------------------
def enforce_node_order(pp_pf_calculator: Any, order_file: str) -> None:
    """
    Ensure consistent node ordering across runs.

    - If order_file exists: sort pp_pf_calculator.ele_nodes according to stored order.
    - If not: write current order to order_file.
    """
    if os.path.exists(order_file):
        with open(order_file, "r", encoding="utf-8") as f:
            saved_order = json.load(f)
        order_map = {nid: idx for idx, nid in enumerate(saved_order)}
        pp_pf_calculator.ele_nodes.sort(key=lambda n: order_map.get(n.nd, len(saved_order)))
    else:
        saved_order = [n.nd for n in pp_pf_calculator.ele_nodes]
        os.makedirs(os.path.dirname(order_file), exist_ok=True)
        with open(order_file, "w", encoding="utf-8") as f:
            json.dump(saved_order, f, ensure_ascii=False, indent=2)


def built_ppnet_for_pfcal(ori_ppnet: Any, r_switch: float, x_switch: float) -> Any:
    """
    Convert "switch-lines" (r=r_switch, x=x_switch) into pandapower bus-bus switches.
    """
    base_net_replacement = copy.deepcopy(ori_ppnet)
    indices_to_drop: List[int] = []

    for i in base_net_replacement.line.index:
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

    base_net_replacement.line.drop(indices_to_drop, inplace=True)
    return base_net_replacement


# -----------------------------
# Timeseries adapter
# -----------------------------
@dataclass
class TimeseriesSchema:
    """
    Define how to interpret timeseries data.

    Supported input formats:
      A) long format: columns [GISID, TIME, P, Q]
      B) wide format: DataFrame indexed by time, MultiIndex columns with level0 in {"P","Q"} and level1=GISID
         e.g. df.columns = MultiIndex([("P","7000..."), ("Q","7000..."), ...])
    """
    gisid_col: str = "GISID"
    time_col: str = "TIME"
    p_col: str = "P"
    q_col: str = "Q"
    p_unit: str = "MW"   # "kW" or "MW"
    q_unit: str = "MVar" # "kVar" or "MVar"
    timezone: Optional[str] = None  # e.g., "Asia/Shanghai"


class TimeseriesDataAdapter:
    """
    Load and serve (P,Q) time series by GISID.

    Internally stores:
      - self._P: DataFrame index=time, columns=GISID, values in MW
      - self._Q: DataFrame index=time, columns=GISID, values in MVar
    """
    def __init__(self, df: pd.DataFrame, schema: TimeseriesSchema, freq: Optional[str] = None):
        self.schema = schema
        df = df.copy()

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
        df.index = ts

        lvl0 = [str(x) for x in df.columns.get_level_values(0)]
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

        if self.schema.q_unit.lower() in ("kvar", "kvarh", "kvar "):
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
        s = self.schema

        if s.gisid_col not in df.columns:
            raise ValueError(f"Timeseries df missing GISID column '{s.gisid_col}'")
        if s.time_col not in df.columns:
            raise ValueError(f"Timeseries df missing time column '{s.time_col}'")
        if s.p_col not in df.columns or s.q_col not in df.columns:
            raise ValueError(f"Timeseries df missing P/Q columns '{s.p_col}', '{s.q_col}'")

        df[s.gisid_col] = df[s.gisid_col].astype(str)

        ts = pd.to_datetime(df[s.time_col])
        if s.timezone:
            if ts.dt.tz is None:
                ts = ts.dt.tz_localize(s.timezone)
            else:
                ts = ts.dt.tz_convert(s.timezone)
        df["_ts"] = ts

        p = df[s.p_col].astype(float)
        q = df[s.q_col].astype(float)
        if s.p_unit.lower() == "kw":
            p = p / 1000.0
        elif s.p_unit.lower() == "mw":
            pass
        else:
            raise ValueError(f"Unsupported p_unit={s.p_unit}")

        if s.q_unit.lower() in ("kvar", "kvarh", "kvar "):
            q = q / 1000.0
        elif s.q_unit.lower() == "mvar":
            pass
        else:
            raise ValueError(f"Unsupported q_unit={s.q_unit}")

        df["_p_mw"] = p
        df["_q_mvar"] = q

        P = df.pivot_table(index="_ts", columns=s.gisid_col, values="_p_mw", aggfunc="mean").sort_index()
        Q = df.pivot_table(index="_ts", columns=s.gisid_col, values="_q_mvar", aggfunc="mean").sort_index()

        if freq is not None:
            P = P.resample(freq).mean().ffill()
            Q = Q.resample(freq).mean().ffill()

        self.index = P.index
        self.gisids = list(P.columns)
        self._P = P.astype(np.float32)
        self._Q = Q.astype(np.float32)

    def __len__(self) -> int:
        return len(self.index)

    def get_pq(self, t: int, gisids: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        row_p = self._P.iloc[t]
        row_q = self._Q.iloc[t]
        p = row_p.reindex(gisids).fillna(0.0).to_numpy(dtype=np.float32)
        q = row_q.reindex(gisids).fillna(0.0).to_numpy(dtype=np.float32)
        return p, q

    def get_forecast_pq(self, t: int, horizon: int, gisids: List[str]) -> Tuple[np.ndarray, np.ndarray]:
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
# Environment configuration
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

    # --- reward weights (建议的更“温和”默认值；你也可按需要调整) ---
    w_loss: float = 1.0
    w_v: float = 10.0
    w_line: float = 2.0
    w_switch: float = float(getattr(cfg, "k_act_swit", 0.0))
    w_trafo_balance: float = 5.0
    fail_penalty: float = 1e4

    # --- how to aggregate violations (避免 sum 爆炸) ---
    voltage_viol_agg: str = "mean"   # {"sum","mean","max","ratio"}
    line_viol_agg: str = "mean"      # {"sum","mean","max","ratio"}

    # --- trafo balance ---
    trafo_balance_metric: str = "cv"  # {"cv","std","range"}

    # --- reset / randomness ---
    random_reset: bool = bool(getattr(cfg, "random_reset", True))
    seed: Optional[int] = 0

    # --- switching cost ---
    switch_cost_mode: str = "hamming"  # {"hamming","hamming_norm"}

    # --- (optional) stable hashing for switch tokens across runs ---
    use_stable_hash: bool = True


# -----------------------------
# Main environment
# -----------------------------
class PowerDispatchEnv(gym.Env):
    """
    Sequential dispatch / reconfiguration environment.

    Action:
        Discrete(a): choose one feasible switch-state index in feeder_cluster.feasible_switch_states

    Observation (dict):
        - "bus_vm_pu": (n_bus,) float32
        - "bus_va_deg": (n_bus,) float32
        - "load_p_mw": (n_load,) float32
        - "load_q_mvar": (n_load,) float32
        - "forecast_p_mw": (H, n_load) float32
        - "forecast_q_mvar": (H, n_load) float32
        - "time_index": (1,) int32
        - "topology_id": (1,) int32   # 当前拓扑对应的动作编号（作为“开关状态”的离散表示）
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

        # --- Timeseries ---
        if timeseries_df is None:
            timeseries_df = self._default_timeseries_loader(self.cfg)
        self.ts = TimeseriesDataAdapter(timeseries_df, self.cfg.timeseries_schema, freq=self.cfg.timeseries_freq)

        # --- Map pandapower loads to GISIDs ---
        self._load_gisids = self._infer_load_gisids()
        self._n_load = len(self._load_gisids)

        # --- Action space ---
        self.n_actions = len(self.feeder_cluster.feasible_switch_states)
        self.action_space = spaces.Discrete(self.n_actions)

        # --- Observation space (已去掉 line_loading_pct) ---
        n_bus = int(self.base_net.bus.shape[0])
        H = int(self.cfg.forecast_horizon)
        self.observation_space = spaces.Dict(
            {
                "bus_vm_pu": spaces.Box(low=0.0, high=2.0, shape=(n_bus,), dtype=np.float32),
                "bus_va_deg": spaces.Box(low=-360.0, high=360.0, shape=(n_bus,), dtype=np.float32),
                "load_p_mw": spaces.Box(low=-1e3, high=1e3, shape=(self._n_load,), dtype=np.float32),
                "load_q_mvar": spaces.Box(low=-1e3, high=1e3, shape=(self._n_load,), dtype=np.float32),
                "forecast_p_mw": spaces.Box(low=-1e3, high=1e3, shape=(H, self._n_load), dtype=np.float32),
                "forecast_q_mvar": spaces.Box(low=-1e3, high=1e3, shape=(H, self._n_load), dtype=np.float32),
                "time_index": spaces.Box(low=0, high=10**9, shape=(1,), dtype=np.int32),
                "topology_id": spaces.Box(low=0, high=max(self.n_actions - 1, 0), shape=(1,), dtype=np.int32),
            }
        )

        # Episode state
        self._t0: int = 0
        self._t: int = 0
        self._step_in_episode: int = 0
        self._prev_action: Optional[int] = None

        # Keep latest solved network
        self._net_line_repr: Any = None
        self._net_cal: Any = None

    # -----------------------------
    # Switch state parsing / cost
    # -----------------------------
    @staticmethod
    def _stable_hash_u32(s: str) -> int:
        """Deterministic 32-bit int hash from string."""
        h = hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
        return int(h, 16)

    def _state_to_open_set(self, st: Any) -> Tuple[Set[int], Optional[int]]:
        """
        Convert feasible_switch_state into:
        - open_set: set of switch identifiers (or positions) that are OPEN (0)
        - n_total: total number of switches if inferable

        Robust against dict values containing non-numeric objects (e.g., ACline).
        Supports dict encodings like {"0":[...], "1":[...]}.
        """
        def _safe_int_array(a: np.ndarray) -> Optional[np.ndarray]:
            try:
                return a.astype(int)
            except Exception:
                return None

        if isinstance(st, dict):
            lk = {str(k).lower(): k for k in st.keys()}

            # 1A) open/close lists
            open_key_candidates = ["open_switches", "open_switch", "open", "opened", "open_sws", "open_sw"]
            close_key_candidates = ["close_switches", "closed_switches", "close", "closed", "close_sws", "closed_sws"]

            open_key = next((lk[k] for k in open_key_candidates if k in lk), None)
            close_key = next((lk[k] for k in close_key_candidates if k in lk), None)

            if open_key is not None:
                open_ids = np.array(st[open_key], dtype=object).ravel().tolist()
                open_set = set(int(x) for x in open_ids if x is not None)

                n_total = None
                if close_key is not None:
                    close_ids = np.array(st[close_key], dtype=object).ravel().tolist()
                    close_set = set(int(x) for x in close_ids if x is not None)
                    n_total = len(open_set | close_set)
                return open_set, n_total

            # 1B) ids + states arrays
            id_key_candidates = ["sw_id", "switch_id", "switch_ids", "ids", "id"]
            state_key_candidates = ["state", "states", "switch_state", "switch_states", "status", "closed", "is_closed"]

            id_key = next((lk[k] for k in id_key_candidates if k in lk), None)
            state_key = next((lk[k] for k in state_key_candidates if k in lk), None)

            if id_key is not None and state_key is not None:
                ids = np.array(st[id_key], dtype=object).ravel()
                states = np.array(st[state_key], dtype=object).ravel()
                states_i = _safe_int_array(states)
                if states_i is not None and set(np.unique(states_i)).issubset({0, 1}):
                    open_set = set(int(i) for i, s in zip(ids, states_i) if int(s) == 0)
                    return open_set, int(ids.size)

            # 1BB) dict with binary keys: {"0":[...], "1":[...]}  (0=open list, 1=closed list)
            keyset = {str(k).lower() for k in st.keys()}
            if keyset.issubset({"0", "1"}):

                def _token(x: Any) -> int:
                    # Priority: int-like -> digit string -> id-like attrs -> string hash
                    if x is None:
                        return -1
                    if isinstance(x, (int, np.integer)):
                        return int(x)
                    if isinstance(x, str):
                        s = x.strip()
                        if s.isdigit():
                            return int(s)
                        return self._stable_hash_u32(s) if self.cfg.use_stable_hash else hash(s)
                    for attr in ("sw_id", "switch_id", "id", "idx", "index", "name"):
                        if hasattr(x, attr):
                            v = getattr(x, attr)
                            try:
                                return int(v)
                            except Exception:
                                s = str(v)
                                return self._stable_hash_u32(s) if self.cfg.use_stable_hash else hash(s)
                    s = str(x)
                    return self._stable_hash_u32(s) if self.cfg.use_stable_hash else hash(s)

                open_list = st.get("0", st.get(0, st.get(False, [])))
                close_list = st.get("1", st.get(1, st.get(True, [])))

                open_items = np.array(open_list, dtype=object).ravel().tolist()
                close_items = np.array(close_list, dtype=object).ravel().tolist()

                open_set = set(_token(x) for x in open_items if x is not None)
                n_total = len(open_items) + len(close_items)
                return open_set, int(n_total)

            # 1C) dict mapping switch_id -> scalar 0/1, e.g. {"12":0, "13":1}
            scalar_mapping = True
            for v in st.values():
                if isinstance(v, (list, tuple, np.ndarray, dict)):
                    scalar_mapping = False
                    break

            if scalar_mapping:
                open_set: Set[int] = set()
                for k, v in st.items():
                    try:
                        iv = int(v)
                    except Exception:
                        continue
                    if iv == 0:
                        try:
                            open_set.add(int(k))
                        except Exception:
                            s = str(k)
                            open_set.add(self._stable_hash_u32(s) if self.cfg.use_stable_hash else hash(s))
                return open_set, len(st)

            # 1D) fallback: find a 0/1 vector-like field
            for _, v in st.items():
                arr = np.array(v)
                if arr.ndim != 1 or arr.size == 0:
                    continue
                ai = _safe_int_array(arr)
                if ai is None:
                    continue
                if set(np.unique(ai)).issubset({0, 1}):
                    open_pos = set(int(i) for i in np.where(ai == 0)[0])
                    return open_pos, int(ai.size)

            raise ValueError(
                "Unsupported feasible_switch_state dict structure. "
                f"keys={list(st.keys())}. "
                "Could not find open/close lists, (id,state) arrays, {'0','1'} encoding, scalar mapping, or binary vector."
            )

        # vector-like
        arr = np.array(st)
        if arr.ndim == 1 and arr.size > 0:
            ai = _safe_int_array(arr)
            if ai is not None and set(np.unique(ai)).issubset({0, 1}):
                open_pos = set(int(i) for i in np.where(ai == 0)[0])
                return open_pos, int(ai.size)

        raise ValueError(f"Unsupported feasible_switch_state type={type(st)}")

    def _calc_switch_cost(self, prev_action: Optional[int], action: int) -> Tuple[float, Dict[str, float]]:
        """
        Switching operations cost as Hamming distance:
          number of switches whose open/close status changed between prev and current.

        Computed via symmetric difference of OPEN switch sets.
        """
        if prev_action is None:
            return 0.0, {"hamming": 0.0, "hamming_norm": 0.0, "n_switch": 0.0}

        st0 = self.feeder_cluster.feasible_switch_states[int(prev_action)]
        st1 = self.feeder_cluster.feasible_switch_states[int(action)]

        open0, n0 = self._state_to_open_set(st0)
        open1, n1 = self._state_to_open_set(st1)

        hamming = float(len(open0.symmetric_difference(open1)))

        if isinstance(n0, int) and n0 > 0:
            n_total = n0
        elif isinstance(n1, int) and n1 > 0:
            n_total = n1
        else:
            n_total = int(len(open0 | open1))

        h_norm = float(hamming / max(n_total, 1))

        mode = (getattr(self.cfg, "switch_cost_mode", "hamming") or "hamming").lower()
        cost = h_norm if mode == "hamming_norm" else hamming

        return float(cost), {"hamming": float(hamming), "hamming_norm": float(h_norm), "n_switch": float(n_total)}

    # -----------------------------
    # Trafo balance
    # -----------------------------
    @staticmethod
    def _select_10_110_trafos(net_cal: Any, hv_kv: float = 110.0, lv_kv: float = 10.0, tol: float = 1e-3) -> List[int]:
        if not hasattr(net_cal, "trafo") or len(net_cal.trafo) == 0:
            return []
        bus_vn = net_cal.bus["vn_kv"]

        idxs: List[int] = []
        for tidx in net_cal.trafo.index:
            hv_bus = int(net_cal.trafo.at[tidx, "hv_bus"])
            lv_bus = int(net_cal.trafo.at[tidx, "lv_bus"])
            hv_vn = float(bus_vn.at[hv_bus])
            lv_vn = float(bus_vn.at[lv_bus])

            cond1 = (abs(hv_vn - hv_kv) <= tol) and (abs(lv_vn - lv_kv) <= tol)
            cond2 = (abs(hv_vn - lv_kv) <= tol) and (abs(lv_vn - hv_kv) <= tol)
            if cond1 or cond2:
                idxs.append(int(tidx))
        return idxs

    @staticmethod
    def _calc_trafo_balance_penalty(net_cal: Any, metric: str = "cv") -> Tuple[float, Dict[str, float]]:
        if (not hasattr(net_cal, "res_trafo")) or (len(net_cal.res_trafo) == 0):
            return 0.0, {"mean": 0.0, "std": 0.0, "max": 0.0, "min": 0.0, "n": 0.0}

        trafo_idxs = PowerDispatchEnv._select_10_110_trafos(net_cal)
        if len(trafo_idxs) <= 1:
            return 0.0, {"mean": 0.0, "std": 0.0, "max": 0.0, "min": 0.0, "n": float(len(trafo_idxs))}

        loading = net_cal.res_trafo.loc[trafo_idxs, "loading_percent"].to_numpy(dtype=float)
        loading = loading[np.isfinite(loading)]
        if loading.size <= 1:
            return 0.0, {"mean": 0.0, "std": 0.0, "max": 0.0, "min": 0.0, "n": float(loading.size)}

        mean = float(np.mean(loading))
        std = float(np.std(loading))
        mx = float(np.max(loading))
        mn = float(np.min(loading))

        metric = (metric or "cv").lower()
        if metric == "std":
            penalty = std
        elif metric == "range":
            penalty = mx - mn
        else:
            denom = max(mean, 1e-3)
            penalty = std / denom

        stats = {"mean": mean, "std": std, "max": mx, "min": mn, "n": float(loading.size)}
        return float(penalty), stats

    # -----------------------------
    # Builders / loaders
    # -----------------------------
    @staticmethod
    def _default_system_builder(env_cfg: EnvConfig) -> Tuple[Any, Any, Any]:
        parsed_cim = CimEParser(env_cfg.pf_data_path)

        pp_pf_calculator = PandaPowerFlowCalculator(parsed_cim, slack_nd="703002137")
        load_feasible_feeders_switch_states(pp_pf_calculator, path=env_cfg.feasible_switch_state_path)

        dist_tf = pd.read_csv(env_cfg.dist_tf_path, sep="\t")
        for i, node_bus in dist_tf["Node"].items():
            node = pp_pf_calculator.bus2node[int(node_bus)]
            node.pl = float(dist_tf["P"][i])
            node.ql = float(dist_tf["Q"][i])

        pp_pf_calculator.ele_nodes = [node for node in pp_pf_calculator.ele_nodes]
        enforce_node_order(pp_pf_calculator, order_file=env_cfg.node_order_file)
        node_ids = [node.nd for node in pp_pf_calculator.ele_nodes]

        base_net = pp_pf_calculator.create_pandapower_net_from_node_ids(node_ids)
        slack_bus = pp_pf_calculator.nodeID2node[pp_pf_calculator.slack_nd].bus
        pp.create_ext_grid(base_net, bus=slack_bus, vm_pu=1.0, va_degree=0.0)

        feeder_cluster = pp_pf_calculator.feeder_clusters[0]
        return pp_pf_calculator, feeder_cluster, base_net

    @staticmethod
    def _default_timeseries_loader(env_cfg: EnvConfig) -> pd.DataFrame:
        if env_cfg.timeseries_mode == "generated_parquet":
            p = env_cfg.generated_timeseries_path
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"Generated timeseries file not found: {p}. "
                    "Please run generate_virtual_timeseries_file.py to create it."
                )
            if p.lower().endswith(".parquet"):
                return pd.read_parquet(p)
            if p.lower().endswith(".csv"):
                return pd.read_csv(p, encoding="utf-8")
            raise ValueError(f"Unsupported generated_timeseries_path extension: {p}")

        if env_cfg.timeseries_mode == "prefiltered_csv":
            p = env_cfg.prefiltered_timeseries_csv
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"Prefiltered timeseries file not found: {p}. "
                    "Either generate it or switch timeseries_mode to 'generated_parquet'."
                )
            return pd.read_csv(p, encoding="utf-8")

        raise ValueError(f"Unknown timeseries_mode={env_cfg.timeseries_mode}")

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
    # Core gym API
    # -----------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if options is not None and "start_index" in options:
            self._t0 = int(options["start_index"])
        else:
            if self.cfg.random_reset:
                max_start = max(0, len(self.ts) - self.cfg.episode_len - self.cfg.forecast_horizon - 2)
                self._t0 = int(self._rng.integers(0, max_start + 1))
            else:
                self._t0 = 0

        self._t = self._t0
        self._step_in_episode = 0
        self._prev_action = None

        init_action = 0
        self._net_line_repr = set_fc_state_with_acts(self.feeder_cluster, self.base_net, [init_action])
        self._apply_timeseries_loads(self._net_line_repr, self._t)

        obs, info = self._solve_and_build_obs(self._net_line_repr, self._t, prev_action=None, action=init_action)
        self._prev_action = init_action
        return obs, info

    def step(self, action: int):
        action = int(action)

        self._net_line_repr = set_fc_state_with_acts(self.feeder_cluster, self.base_net, [action])
        self._apply_timeseries_loads(self._net_line_repr, self._t)

        obs, info = self._solve_and_build_obs(self._net_line_repr, self._t, prev_action=self._prev_action, action=action)

        reward = float(info["reward"])
        terminated = bool(info.get("pf_failed", False))
        truncated = False

        self._prev_action = action
        self._t += 1
        self._step_in_episode += 1

        if self._step_in_episode >= self.cfg.episode_len:
            truncated = True
        if self._t >= len(self.ts) - 1:
            truncated = True

        return obs, reward, terminated, truncated, info

    # -----------------------------
    # Internal helpers
    # -----------------------------
    def _apply_timeseries_loads(self, net: Any, t: int) -> None:
        p, q = self.ts.get_pq(t, self._load_gisids)
        net.load.loc[:, "p_mw"] = p
        net.load.loc[:, "q_mvar"] = q

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

        pf_failed = False
        try:
            pp.runpp(
                net_cal,
                max_iteration=self.cfg.pf_max_iter,
                tolerance_mva=self.cfg.pf_tol,
                init="auto",
            )
        except Exception as e:
            pf_failed = True
            info["pf_error"] = repr(e)

        info["pf_failed"] = pf_failed
        self._net_cal = net_cal

        if pf_failed:
            obs = self._zero_obs(t=t, topology_id=action)

            try:
                switch_cost, sw_stats = self._calc_switch_cost(prev_action, action)
            except Exception:
                switch_cost, sw_stats = 0.0, {"hamming": 0.0, "hamming_norm": 0.0, "n_switch": 0.0}

            info.update(
                {
                    "loss_mw": np.nan,
                    "v_viol": np.nan,
                    "line_viol": np.nan,
                    "switch_cost": float(switch_cost),
                    "switch_hamming": float(sw_stats["hamming"]),
                    "switch_hamming_norm": float(sw_stats["hamming_norm"]),
                    "switch_n": int(sw_stats["n_switch"]),
                    "trafo_balance": np.nan,
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

        loss_term = self.cfg.w_loss * loss_mw
        v_term = self.cfg.w_v * v_viol
        line_term = self.cfg.w_line * line_viol
        switch_term = self.cfg.w_switch * switch_cost
        trafo_term = self.cfg.w_trafo_balance * trafo_bal

        reward = -(loss_term + v_term + line_term + switch_term + trafo_term)

        info.update(
            {
                "loss_mw": float(loss_mw),
                "v_viol": float(v_viol),
                "line_viol": float(line_viol),
                "switch_cost": float(switch_cost),
                "switch_hamming": float(sw_stats["hamming"]),
                "switch_hamming_norm": float(sw_stats["hamming_norm"]),
                "switch_n": int(sw_stats["n_switch"]),
                "trafo_balance": float(trafo_bal),
                "trafo_loading_mean": float(trafo_stats["mean"]),
                "trafo_loading_std": float(trafo_stats["std"]),
                "trafo_loading_max": float(trafo_stats["max"]),
                "trafo_loading_min": float(trafo_stats["min"]),
                "trafo_loading_n": int(trafo_stats["n"]),
                "loss_term": float(loss_term),
                "v_term": float(v_term),
                "line_term": float(line_term),
                "switch_term": float(switch_term),
                "trafo_term": float(trafo_term),
                "reward": float(reward),
            }
        )

        obs = self._build_obs(net_cal, t, topology_id=action)
        return obs, info

    def _zero_obs(self, t: int = 0, topology_id: int = 0) -> Dict[str, np.ndarray]:
        """PF 失败时返回固定 shape 的零观测（同时保留 topology_id / time_index）。"""
        H = int(self.cfg.forecast_horizon)
        n_bus = int(self.base_net.bus.shape[0])
        return {
            "bus_vm_pu": np.zeros((n_bus,), dtype=np.float32),
            "bus_va_deg": np.zeros((n_bus,), dtype=np.float32),
            "load_p_mw": np.zeros((self._n_load,), dtype=np.float32),
            "load_q_mvar": np.zeros((self._n_load,), dtype=np.float32),
            "forecast_p_mw": np.zeros((H, self._n_load), dtype=np.float32),
            "forecast_q_mvar": np.zeros((H, self._n_load), dtype=np.float32),
            "time_index": np.array([int(t)], dtype=np.int32),
            "topology_id": np.array([int(topology_id)], dtype=np.int32),
        }

    def _build_obs(self, net_cal: Any, t: int, topology_id: int) -> Dict[str, np.ndarray]:
        H = int(self.cfg.forecast_horizon)

        vm = net_cal.res_bus["vm_pu"].to_numpy(dtype=np.float32)
        va = net_cal.res_bus["va_degree"].to_numpy(dtype=np.float32)

        load_p = net_cal.load["p_mw"].to_numpy(dtype=np.float32)
        load_q = net_cal.load["q_mvar"].to_numpy(dtype=np.float32)

        fp, fq = self.ts.get_forecast_pq(t, H, self._load_gisids)

        return {
            "bus_vm_pu": vm,
            "bus_va_deg": va,
            "load_p_mw": load_p,
            "load_q_mvar": load_q,
            "forecast_p_mw": fp,
            "forecast_q_mvar": fq,
            "time_index": np.array([int(t)], dtype=np.int32),
            "topology_id": np.array([int(topology_id)], dtype=np.int32),
        }

    # -----------------------------
    # Penalty terms (support agg=mean/max/ratio; default mean to avoid sum explosion)
    # -----------------------------
    @staticmethod
    def _calc_losses_mw(net_cal: Any) -> float:
        loss = 0.0
        if hasattr(net_cal, "res_line") and len(net_cal.res_line) > 0:
            loss += float(net_cal.res_line["pl_mw"].sum())
        if hasattr(net_cal, "res_trafo") and len(getattr(net_cal, "res_trafo", [])) > 0:
            loss += float(net_cal.res_trafo["pl_mw"].sum())
        return max(loss, 0.0)

    @staticmethod
    def _calc_voltage_violation(net_cal: Any, vmin: float, vmax: float, agg: str = "mean") -> float:
        vm = net_cal.res_bus["vm_pu"].to_numpy(dtype=float)
        viol = np.maximum(vm - vmax, 0.0) + np.maximum(vmin - vm, 0.0)

        agg = (agg or "mean").lower()
        if agg == "sum":
            return float(viol.sum())
        if agg == "max":
            return float(viol.max()) if viol.size > 0 else 0.0
        if agg == "ratio":
            return float(((vm < vmin) | (vm > vmax)).mean()) if vm.size > 0 else 0.0
        return float(viol.mean()) if viol.size > 0 else 0.0

    @staticmethod
    def _calc_line_overload(net_cal: Any, limit_pct: float, agg: str = "mean") -> float:
        if not hasattr(net_cal, "res_line") or len(net_cal.res_line) == 0:
            return 0.0
        loading = net_cal.res_line["loading_percent"].to_numpy(dtype=float)
        over = np.maximum(loading - limit_pct, 0.0)

        agg = (agg or "mean").lower()
        if agg == "sum":
            return float(over.sum())
        if agg == "max":
            return float(over.max()) if over.size > 0 else 0.0
        if agg == "ratio":
            return float((loading > limit_pct).mean()) if loading.size > 0 else 0.0
        return float(over.mean()) if over.size > 0 else 0.0


# In[]
if __name__ == "__main__":
    env = PowerDispatchEnv(EnvConfig(seed=0))
    obs, info = env.reset(seed=0, options={"start_index": 0})
    print("[reset] t=", info["t"], "action=", info["action"], "pf_failed=", info.get("pf_failed"))

    total_reward = 0.0
    for step_i in range(env.cfg.episode_len):
        a = env.action_space.sample()
        obs, r, terminated, truncated, info = env.step(a)
        total_reward += r

        # 打印 reward 分量（含加权项）
        parts = {
            "t": info["t"],
            "a": a,
            "reward": info.get("reward", r),
            "loss_mw": info.get("loss_mw"),
            "v_viol": info.get("v_viol"),
            "line_viol": info.get("line_viol"),
            "switch_cost": info.get("switch_cost"),
            "trafo_balance": info.get("trafo_balance"),
            # 加权项
            "loss_term": info.get("loss_term"),
            "v_term": info.get("v_term"),
            "line_term": info.get("line_term"),
            "switch_term": info.get("switch_term"),
            "trafo_term": info.get("trafo_term"),
            "pf_failed": info.get("pf_failed"),
        }
        print(parts)

        if terminated or truncated:
            break

    print("total_reward =", total_reward)

# %%
