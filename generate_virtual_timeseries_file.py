
# In[]
"""
generate_virtual_timeseries_file.py

Pre-generate multi-year hourly load timeseries for RL training and save to a single parquet file.

Output format (wide):
- DataFrame indexed by TIME (DatetimeIndex)
- Columns are MultiIndex: level0 in {"P","Q"}, level1 is GISID string
- Values are in MW / MVar (float32)

Why wide parquet?
- It is much smaller than long CSV for multi-year, multi-GISID data.
- It loads fast and avoids pivot_table overhead.

Typical usage
-------------
python generate_virtual_timeseries_file.py

Then in EnvConfig:
    timeseries_mode="generated_parquet"
    generated_timeseries_path="<the parquet path>"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import os
import numpy as np
import pandas as pd

from Utls.yantian_sys_746sys import (  # type: ignore
    CimEParser,
    PandaPowerFlowCalculator,
    load_feasible_feeders_switch_states,
)
import pandapower as pp  # type: ignore
import config746sys as cfg  # type: ignore


@dataclass
class GenConfig:
    # RL requirement
    num_episodes: int = 3000
    episode_len: int = 24          # 1h interval => 24 steps/day
    forecast_horizon: int = int(getattr(cfg, "T", 4))
    margin_steps: int = 48         # extra buffer for random reset + forecast

    # time axis
    freq: str = "1H"
    start_time: str = "2020-01-01 00:00:00"
    seed: int = 0

    # per-load magnitude & power factor
    base_mw_range: Tuple[float, float] = (0.02, 0.50)  # MW, adjust if too large/small
    pf_range: Tuple[float, float] = (0.90, 0.99)

    # output
    out_path: str = os.path.join(cfg.WORKPATH, "system_file/virtual_timeseries_hourly.parquet")

    # system files
    pf_data_path: str = cfg.PfDataPath
    dist_tf_path: str = cfg.DistTfPath
    feasible_switch_state_path: str = os.path.join(cfg.WORKPATH, "system_file/746sys/")
    node_order_file: str = os.path.join(cfg.WORKPATH, "system_file/746sys/746sys_node_order.json")


def enforce_node_order(pp_pf_calculator: Any, order_file: str) -> None:
    import json
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


def build_system_and_get_load_gisids(gc: GenConfig) -> List[str]:
    """
    Build the base pandapower net and infer GISIDs required by net.load.
    Only those GISIDs will be generated.
    """
    parsed_cim = CimEParser(gc.pf_data_path)

    pp_pf_calculator = PandaPowerFlowCalculator(parsed_cim, slack_nd="703002137")
    load_feasible_feeders_switch_states(pp_pf_calculator, path=gc.feasible_switch_state_path)

    dist_tf = pd.read_csv(gc.dist_tf_path, sep="\t")
    for i, node_bus in dist_tf["Node"].items():
        node = pp_pf_calculator.bus2node[int(node_bus)]
        node.pl = float(dist_tf["P"][i])
        node.ql = float(dist_tf["Q"][i])

    pp_pf_calculator.ele_nodes = [node for node in pp_pf_calculator.ele_nodes]
    enforce_node_order(pp_pf_calculator, order_file=gc.node_order_file)
    node_ids = [node.nd for node in pp_pf_calculator.ele_nodes]

    base_net = pp_pf_calculator.create_pandapower_net_from_node_ids(node_ids)
    slack_bus = pp_pf_calculator.nodeID2node[pp_pf_calculator.slack_nd].bus
    pp.create_ext_grid(base_net, bus=slack_bus, vm_pu=1.0, va_degree=0.0)

    nd2gisid = getattr(cfg, "nd2gisid", {})
    if not isinstance(nd2gisid, dict) or len(nd2gisid) == 0:
        raise RuntimeError("cfg.nd2gisid is missing/empty. Please check config746sys.py and related npz files.")

    load_gisids: List[str] = []
    for load_idx in base_net.load.index:
        bus = int(base_net.load.at[load_idx, "bus"])
        node = pp_pf_calculator.bus2node[bus]
        nd = str(node.nd)
        gisid = nd2gisid.get(nd, None)
        if gisid is None:
            # missing => env will fill 0, so we do not generate it
            continue
        load_gisids.append(str(gisid))

    # unique + stable order
    load_gisids = sorted(set(load_gisids))
    return load_gisids


def required_steps(gc: GenConfig) -> int:
    """
    Ensure enough steps for >=num_episodes episodes, plus forecast & margin.
    """
    return int(gc.num_episodes * gc.episode_len + gc.forecast_horizon + gc.margin_steps)


def generate_virtual_profiles_wide(
    gisids: List[str],
    *,
    n_steps: int,
    freq: str,
    start_time: str,
    seed: int,
    base_mw_range: Tuple[float, float],
    pf_range: Tuple[float, float],
) -> pd.DataFrame:
    """
    Generate hourly MW/MVar loads with:
      - daily cycle (sin + 2nd harmonic)
      - weekly cycle (weekday/weekend)
      - annual seasonality (summer/winter)
      - global factor (shared)
      - local factor (per GISID)
      - AR(1) noise for temporal correlation

    Returns wide MultiIndex columns DataFrame: columns=(P/Q, GISID)
    """
    if len(gisids) == 0:
        raise ValueError("gisids is empty.")

    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start_time, periods=n_steps, freq=freq)

    n_g = len(gisids)

    # Base magnitude & pf per GISID
    base_mw = rng.uniform(base_mw_range[0], base_mw_range[1], size=n_g).astype(np.float64)
    pf = rng.uniform(pf_range[0], pf_range[1], size=n_g).astype(np.float64)
    q_over_p = np.tan(np.arccos(np.clip(pf, 1e-4, 0.9999)))

    # Time features
    hour = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()  # 0=Mon
    doy = idx.dayofyear.to_numpy()

    # Daily profile: two-harmonic (0.6~1.4 roughly)
    w1 = 2.0 * np.pi / 24.0
    daily = 0.95 + 0.25 * np.sin(w1 * (hour - 7)) + 0.12 * np.sin(2 * w1 * (hour - 6))
    daily = np.clip(daily, 0.55, 1.45)

    # Weekly factor: weekends slightly lower (adjust as needed)
    weekly = np.where(dow >= 5, 0.92, 1.00).astype(np.float64)

    # Annual seasonality: peak in summer (around day 200), mild winter bump
    wY = 2.0 * np.pi / 365.25
    annual = 1.00 + 0.12 * np.sin(wY * (doy - 200)) + 0.05 * np.sin(2 * wY * (doy - 15))
    annual = np.clip(annual, 0.85, 1.20)

    # Global factor: shared AR(1)
    g = np.zeros(n_steps, dtype=np.float64)
    eps = rng.standard_normal(n_steps) * 0.03
    phi = 0.97
    for t in range(1, n_steps):
        g[t] = phi * g[t - 1] + eps[t]
    global_factor = np.clip(1.0 + g, 0.85, 1.20)

    # Local AR(1) noise per GISID
    local = np.zeros((n_steps, n_g), dtype=np.float64)
    eps_l = rng.standard_normal((n_steps, n_g)) * 0.04
    phi_l = 0.95
    for t in range(1, n_steps):
        local[t, :] = phi_l * local[t - 1, :] + eps_l[t, :]
    local_factor = np.clip(1.0 + local, 0.80, 1.25)

    scale = (daily * weekly * annual * global_factor).astype(np.float64)  # (n_steps,)
    P = (scale[:, None] * base_mw[None, :] * local_factor).astype(np.float32)
    P = np.maximum(P, 0.0)

    Q = (P * q_over_p[None, :]).astype(np.float32)

    # Build wide DF with MultiIndex columns
    P_df = pd.DataFrame(P, index=idx, columns=[str(x) for x in gisids])
    Q_df = pd.DataFrame(Q, index=idx, columns=[str(x) for x in gisids])

    wide = pd.concat({"P": P_df, "Q": Q_df}, axis=1)
    wide.index.name = "TIME"
    return wide


def main():
    gc = GenConfig()

    gisids = build_system_and_get_load_gisids(gc)
    n_steps = required_steps(gc)

    print(f"GISID count used by net.load: {len(gisids)}")
    print(f"Generating {n_steps} steps @ {gc.freq} starting {gc.start_time}")

    wide = generate_virtual_profiles_wide(
        gisids,
        n_steps=n_steps,
        freq=gc.freq,
        start_time=gc.start_time,
        seed=gc.seed,
        base_mw_range=gc.base_mw_range,
        pf_range=gc.pf_range,
    )

    os.makedirs(os.path.dirname(gc.out_path), exist_ok=True)
    wide.to_parquet(gc.out_path, engine="pyarrow")
    print(f"Saved to: {gc.out_path}")
    print("DataFrame shape:", wide.shape)
    print("Columns levels:", wide.columns.names)


if __name__ == "__main__":
    main()

# %%
