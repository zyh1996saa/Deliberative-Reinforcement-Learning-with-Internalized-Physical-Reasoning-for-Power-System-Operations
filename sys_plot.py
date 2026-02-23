# In[]
import os
from config746sys import (
    WORKPATH 
)  
import pandas as pd
import pandapower as pp
import numpy as np
from new746_system_v0713 import *

# ================== partition_220_110.py ==================
from typing import Dict, List, Set, Tuple, Any, Optional
import numpy as np
from plotly.colors import qualitative as qual
# ---------- 基础工具 ----------
def _vol_eq(v: Optional[float], target: float, tol: float = 1.0) -> bool:
    try:
        return abs(float(v) - float(target)) <= tol
    except Exception:
        return False

def _collect_layer_nodes(pp_pf_calculator, volt_kv: float, tol: float=1.0):
    """返回该电压层的节点列表(对象)与其ID列表"""
    layer_nodes = [nd for nd in pp_pf_calculator.ele_nodes if _vol_eq(getattr(nd, "volt", None), volt_kv, tol)]
    node_ids = [nd.ID for nd in layer_nodes]
    id2node = {nd.ID: nd for nd in layer_nodes}
    return layer_nodes, node_ids, id2node

# ---------- 线路/开关采样与权重 ----------
def _line_length_factor(br) -> float:
    """
    若 r/x 为每公里参数，则需要乘以线路长度。
    这里尝试读取常见字段: length_km / length / L；若不存在则按 1.0 处理。
    """
    for attr in ("length_km", "length", "L"):
        if hasattr(br, attr):
            try:
                val = float(getattr(br, attr))
                if val > 0:
                    return val
            except Exception:
                pass
    return 1.0

def _iter_closed_aclines_same_layer(pp_pf_calculator, id2node: Dict[str, Any], volt_kv: float, tol: float=1.0,
                                    tie_line_alpha: float = 0.2):
    """
    遍历同层闭合的交流线路，返回 (i_id, j_id, weight) 迭代器。
    权重 = 1 / ( sqrt((rL)^2+(xL)^2) + eps ) ，对联络线（concat_switch=True）降权乘以 alpha。
    """
    eps = 1e-6
    for br in getattr(pp_pf_calculator, "all_lines", []):
        if getattr(br, "device_type", "") != "acline":
            continue
        if str(getattr(br, "closed", "1")) != "1":
            continue
        i_obj = getattr(br, "I_nd", None)
        j_obj = getattr(br, "J_nd", None)
        if i_obj is None or j_obj is None:
            continue
        if (i_obj.ID in id2node) and (j_obj.ID in id2node) and _vol_eq(getattr(br, "volt", volt_kv), volt_kv, tol):
            try:
                r = float(getattr(br, "r", 0.0))
                x = float(getattr(br, "x", 0.0))
            except Exception:
                r, x = 0.0, 0.0
            Lfac = _line_length_factor(br)
            z = np.hypot(r * Lfac, x * Lfac)  # sqrt((rL)^2 + (xL)^2)
            w = 1.0 / (z + eps)
            if bool(getattr(br, "concat_switch", False)):
                w = w * tie_line_alpha
            yield i_obj.ID, j_obj.ID, w

def _iter_closed_switches_same_layer(pp_pf_calculator, id2node: Dict[str, Any], volt_kv: float, tol: float=1.0):
    """
    遍历同层闭合的断路器/隔离开关，返回 (i_id, j_id, huge_weight) 迭代器。
    这些元件可视作近零阻抗，强制同簇的强耦合边。
    """
    HUGE_W = 1e6
    for sw in getattr(pp_pf_calculator, "all_lines", []):
        if getattr(sw, "device_type", "") not in ("breaker", "disconnector", "disconnectorwithfeeders"):
            continue
        if str(getattr(sw, "closed", "1")) != "1":
            continue
        i_obj = getattr(sw, "I_nd", None) or getattr(sw, "sub_node", None)
        j_obj = getattr(sw, "J_nd", None) or getattr(sw, "feeder_node", None)
        if i_obj is None or j_obj is None:
            continue
        if (i_obj.ID in id2node) and (j_obj.ID in id2node) and (
            not hasattr(sw, "volt") or _vol_eq(getattr(sw, "volt", volt_kv), volt_kv, tol)
        ):
            yield i_obj.ID, j_obj.ID, HUGE_W

# ---------- 邻接与图工具 ----------
def _build_adjacency(node_ids: List[str],
                     edges: List[Tuple[str, str, float]]) -> Tuple[np.ndarray, Dict[str,int]]:
    """由边(含权重)构建加权邻接矩阵与 id->idx 映射"""
    n = len(node_ids)
    id2idx = {nid:i for i, nid in enumerate(node_ids)}
    A = np.zeros((n, n), dtype=float)
    for u, v, w in edges:
        iu, iv = id2idx[u], id2idx[v]
        if iu == iv: 
            continue
        A[iu, iv] += w
        A[iv, iu] += w
    return A, id2idx

def _connected_components_from_A(A: np.ndarray) -> List[np.ndarray]:
    """基于稠密邻接的简单连通分量（阈值>0即连边），返回索引数组列表"""
    n = A.shape[0]
    seen = np.zeros(n, dtype=bool)
    comps = []
    for s in range(n):
        if seen[s]:
            continue
        stack = [s]
        seen[s] = True
        comp = [s]
        while stack:
            u = stack.pop()
            nbrs = np.where(A[u] > 0)[0]
            for v in nbrs:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
                    comp.append(v)
        comps.append(np.array(comp, dtype=int))
    return comps

# ---------- 谱划分 ----------
def _normalized_cut(A: np.ndarray, labels: np.ndarray) -> float:
    """计算简单的归一化割近似，用于是否继续划分的判断"""
    idx0 = np.where(labels == 0)[0]
    idx1 = np.where(labels == 1)[0]
    if len(idx0)==0 or len(idx1)==0:
        return 1.0
    vol0 = A[idx0][:, :].sum() + A[idx0][:, idx1].sum()
    vol1 = A[idx1][:, :].sum() + A[idx1][:, idx0].sum()
    cut = A[idx0][:, idx1].sum()
    denom = max(1e-9, (vol0 + vol1))
    return float(cut / denom)

def _spectral_bipartition(A_sub: np.ndarray, min_side_ratio: float = 0.1) -> np.ndarray:
    """用Fiedler向量进行二分；返回 0/1 标签。加入最小占比保护。"""
    d = A_sub.sum(axis=1)
    L = np.diag(d) - A_sub
    vals, vecs = np.linalg.eigh(L)
    if len(vals) < 2:
        return np.zeros((A_sub.shape[0],), dtype=int)
    fiedler = vecs[:, 1]
    labels = (fiedler > 0.0).astype(int)
    ratio = labels.mean()
    if ratio < min_side_ratio or ratio > (1.0 - min_side_ratio):
        median = np.median(fiedler)
        labels = (fiedler > median).astype(int)
        ratio = labels.mean()
        if ratio < min_side_ratio or ratio > (1.0 - min_side_ratio):
            labels = (d > np.median(d)).astype(int)
            if labels.sum() == 0 or labels.sum() == len(labels):
                k = max(1, int(round(len(labels)*0.5)))
                labels = np.zeros_like(labels)
                labels[:k] = 1
    return labels

def _try_split_cluster(A: np.ndarray,
                       idx: np.ndarray,
                       min_cut_improvement: float,
                       max_cluster_size: int,
                       min_side_ratio: float = 0.1) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """尝试二分；成功则返回 (left_idx, right_idx)，否则 None。"""
    if len(idx) <= max_cluster_size:
        return None
    labels = _spectral_bipartition(A[np.ix_(idx, idx)], min_side_ratio=min_side_ratio)
    cut_ratio = _normalized_cut(A[np.ix_(idx, idx)], labels)
    if cut_ratio > min_cut_improvement:
        return None
    left = idx[labels == 0]
    right = idx[labels == 1]
    if len(left) == 0 or len(right) == 0:
        return None
    return left, right

def _absorb_small_clusters(A: np.ndarray, clusters_idx: List[np.ndarray], tau: int = 3) -> List[np.ndarray]:
    """将 size < tau 的小簇并入跨簇权重最大的邻簇。"""
    if not clusters_idx:
        return clusters_idx
    big = [c for c in clusters_idx if len(c) >= tau]
    small = [c for c in clusters_idx if len(c) <  tau]
    if not small:
        return clusters_idx
    if not big:
        return [np.concatenate(clusters_idx)]
    new_big = [c.copy() for c in big]
    for s in small:
        best_c, best_w = None, -1.0
        for bi, b in enumerate(new_big):
            w = A[np.ix_(s, b)].sum()
            if w > best_w:
                best_w, best_c = w, bi
        if best_c is None:
            best_c = int(np.argmin([len(b) for b in new_big]))
        new_big[best_c] = np.concatenate([new_big[best_c], s])
    return new_big

def _partition_driver(A: np.ndarray,
                      target_k: int,
                      max_cluster_size: int,
                      min_cut_improvement: float,
                      min_side_ratio: float = 0.1,
                      small_absorb_tau: int = 3) -> List[np.ndarray]:
    """优先切分最大簇的 driver，直到达到目标或无法继续改进。"""
    clusters = _connected_components_from_A(A)
    locked = [False] * len(clusters)
    while len(clusters) < target_k:
        sizes = [len(c) if not locked[i] else -1 for i, c in enumerate(clusters)]
        pick = int(np.argmax(sizes))
        if sizes[pick] <= 0:
            break
        split = _try_split_cluster(A, clusters[pick], min_cut_improvement, max_cluster_size, min_side_ratio)
        if split is None:
            locked[pick] = True
            if all(locked):
                break
            continue
        left, right = split
        clusters[pick] = left
        locked[pick] = False
        clusters.append(right)
        locked.append(False)
    clusters = _absorb_small_clusters(A, clusters, tau=small_absorb_tau)
    return clusters

# ---------- 220/110 kV：谱聚类入口 ----------
def partition_transmission_layers(
    pp_pf_calculator,
    layers: List[float] = [220.0, 110.0],
    tol_kv: float = 1.0,
    target_clusters_per_layer: Dict[float, int] = None,
    max_cluster_size: int = 40,
    min_cut_improvement: float = 0.05,
    min_side_ratio: float = 0.1,
    small_absorb_tau: int = 3,
    tie_line_alpha: float = 0.2,
) -> Dict[float, Dict[str, Any]]:
    """
    返回:
    {
      220.0: {
        "nodes": [node_id,...],
        "clusters": [ [node_id,...], ...],
        "node2cluster": {node_id: cluster_index},
        "adjacency": A,
      },
      110.0: {...}
    }
    """
    results: Dict[float, Dict[str, Any]] = {}

    for volt in layers:
        layer_nodes, node_ids, id2node = _collect_layer_nodes(pp_pf_calculator, volt, tol_kv)

        # 构造加权边：acline + 闭合开关
        edges_w = []
        for i_id, j_id, w in _iter_closed_aclines_same_layer(
            pp_pf_calculator, id2node, volt, tol_kv, tie_line_alpha=tie_line_alpha
        ):
            edges_w.append((i_id, j_id, w))
        for i_id, j_id, huge_w in _iter_closed_switches_same_layer(pp_pf_calculator, id2node, volt, tol_kv):
            edges_w.append((i_id, j_id, huge_w))

        # 邻接矩阵
        A, id2idx = _build_adjacency(node_ids, edges_w)
        n = len(node_ids)

        # 目标簇数（默认启发式：每~35个点一簇，再 +1，且不少于3）
        if target_clusters_per_layer and volt in target_clusters_per_layer:
            B_target = max(1, int(target_clusters_per_layer[volt]))
        else:
            B_target = max(3, int(np.ceil(n / 35.0)) + 1)

        # driver：优先切分最大簇
        clusters_idx = _partition_driver(
            A=A,
            target_k=B_target,
            max_cluster_size=max_cluster_size,
            min_cut_improvement=min_cut_improvement,
            min_side_ratio=min_side_ratio,
            small_absorb_tau=small_absorb_tau
        )

        # 组织输出
        clusters_ids: List[List[str]] = [[node_ids[i] for i in idx_list] for idx_list in clusters_idx]
        node2cluster = {nid: ci for ci, ids in enumerate(clusters_ids) for nid in ids}

        results[volt] = {
            "nodes": node_ids,
            "clusters": clusters_ids,
            "node2cluster": node2cluster,
            "adjacency": A,
        }

    return results

# ---------- 10 kV：按馈线分块 ----------
def partition_10kv_by_feeders(
    pp_pf_calculator,
    volt_kv: float = 10.0,
    tol_kv: float = 1.0,
    include_ungrouped: bool = True,
) -> Dict[str, Any]:
    """
    按馈线（FeederGroup）直接成簇。
    返回结构与上面一致，另外增加 'cluster_names' 以保存馈线名。
    """
    # 全部 10kV 节点
    layer_nodes, node_ids, id2node = _collect_layer_nodes(pp_pf_calculator, volt_kv, tol_kv)
    tenkv_set: Set[str] = set(node_ids)

    clusters_ids: List[List[str]] = []
    cluster_names: List[str] = []

    # 逐馈线取 10kV 节点
    feeder_groups = getattr(pp_pf_calculator, "feeder_groups", []) or []
    used: Set[str] = set()

    for fg in feeder_groups:
        fg_name = getattr(fg, "name", "Feeder")
        fg_nodes = getattr(fg, "nodes", []) or []
        # 仅保留电压≈10kV 的节点
        ids = []
        for nd in fg_nodes:
            try:
                if _vol_eq(getattr(nd, "volt", None), volt_kv, tol_kv):
                    ids.append(nd.ID)
            except Exception:
                continue
        # 去重并与 10kV全集取交集（安全）
        ids = list((set(ids) & tenkv_set) - used)
        if len(ids) == 0:
            continue
        clusters_ids.append(ids)
        cluster_names.append(str(fg_name))
        used.update(ids)

    # 未被任何馈线覆盖但电压≈10kV 的节点
    if include_ungrouped:
        rest = list(tenkv_set - used)
        if len(rest) > 0:
            clusters_ids.append(rest)
            cluster_names.append("10kV-UNGROUPED")

    # node2cluster
    node2cluster = {nid: ci for ci, ids in enumerate(clusters_ids) for nid in ids}

    # 构造一个“弱邻接”（可选）：仅用于接口兼容；这里不做 10kV 谱聚类，所以给全零矩阵
    A = np.zeros((len(node_ids), len(node_ids)), dtype=float)

    return {
        "nodes": node_ids,
        "clusters": clusters_ids,
        "cluster_names": cluster_names,  # 与 clusters 同长度
        "node2cluster": node2cluster,
        "adjacency": A,
    }

# ---------- 便捷封装：三层合并 ----------
def build_voltage_partitions(pp_pf_calculator):
    """
    返回 {220.0, 110.0, 10.0} 三层分区。
    其中 220/110 采用谱聚类；10kV 以馈线为簇。
    """
    # 先做 220/110
    part_hi = partition_transmission_layers(
        pp_pf_calculator,
        layers=[220.0, 110.0],
        tol_kv=1.0,
        target_clusters_per_layer=None,   # 启发式
        max_cluster_size=40,
        min_cut_improvement=0.05,
        min_side_ratio=0.1,
        small_absorb_tau=3,
        tie_line_alpha=0.2,
    )
    # 再做 10kV by feeders
    part_10 = partition_10kv_by_feeders(pp_pf_calculator, volt_kv=10.0, tol_kv=1.0, include_ungrouped=True)

    # 合并
    part = dict(part_hi)
    part[10.0] = part_10

    # 组装 summary（10kV 带 label）
    partition_summary: Dict[float, Dict[str, Any]] = {}
    for volt, obj in part.items():
        cluster_names = obj.get("cluster_names", None)
        clusters_out = []
        for ci, ids in enumerate(obj["clusters"]):
            label = None
            if cluster_names and ci < len(cluster_names):
                label = cluster_names[ci]
            clusters_out.append({
                "id": f"{int(volt)}kV-C{ci+1}",
                "label": label,            # 10kV 显示馈线名；220/110 为 None
                "size": len(ids),
                "nodes": ids
            })
        partition_summary[volt] = {
            "num_nodes": len(obj["nodes"]),
            "num_clusters": len(obj["clusters"]),
            "clusters": clusters_out,
            "node2cluster": obj["node2cluster"],
        }
    return partition_summary

# ==========================================================
def set_fc_state_with_acts(feeder_cluster,fc_base_net,actions):
    new_net = copy.deepcopy(fc_base_net)
    #if not self.pp_pf_cal_obj.is_scan_feasible_switch_states:
    #    self.pp_pf_cal_obj.scan_feasible_switch_states
    
    closed_switches = [] #断开的开关组
    open_switches = [] #打开的开关组
    changed_switches = 0 #开关动作数
    for act_num,act in enumerate(actions):
        switch_condition_in_group = feeder_cluster.feasible_switch_states[act]
        closed_switches += switch_condition_in_group['1']
        open_switches += switch_condition_in_group['0']
        
    for line in open_switches:
        line_to_delete = new_net.line[(new_net.line['from_bus'] == line.I_nd.bus) & (new_net.line['to_bus'] == line.J_nd.bus)].index
        line.closed = '0'
        if not line_to_delete.empty:
            pp.drop_lines(new_net, line_to_delete)
            changed_switches += 1
            
    for line in closed_switches:
        line.closed = '1'
        line_exists = not new_net.line[(new_net.line['from_bus'] == line.I_nd.bus) & (new_net.line['to_bus'] == line.J_nd.bus)].empty
        if not line_exists:
            pp.create_line_from_parameters(new_net,
                from_bus=line.I_nd.bus,
                to_bus=line.J_nd.bus,
                length_km=1,
                r_ohm_per_km=float(line.r),
                x_ohm_per_km=float(line.x),
                c_nf_per_km=float(line.b) ,
                max_i_ka=float(line.max_i_ka) ,
                name=line.device_type + line.name)
            changed_switches += 1
    return new_net
# In[]
if __name__ == "__main__":
    parsed_cim = CimEParser(PfDataPath) # load the CIM data
    pp_pf_calculator = PandaPowerFlowCalculator(parsed_cim,slack_nd='703002137')    # set the slack bus   
    feeder_cluster, fc_base_net = init_feeder_net(pp_pf_calculator) # initialize the feeder network
    fc_base_net_cal = built_ppnet_for_pfcal(fc_base_net) # build the pandapower network for power flow calculation
    
    pp.runpp(fc_base_net_cal) # run power flow calculation
    
    pp.to_excel(fc_base_net, WORKPATH + "/fc_base_net.xlsx")
    fc_base_net = pp.from_excel(WORKPATH + "/fc_base_net.xlsx")
    #fc_base_net = set_fc_state_with_acts(feeder_cluster,fc_base_net,[0])
    vt_partitions = build_voltage_partitions(pp_pf_calculator)
        # =============================================
    # 基于附件2的画图方法，为附件1中的 temp_fcnet 画图
    # =============================================

    # 1) 只需要做一次分区（220 / 110 谱聚类 + 10kV 按馈线分簇）
    vt_partitions = build_voltage_partitions(pp_pf_calculator)

    # 打印结果
    for volt, info in vt_partitions.items():
        print(f"=== {int(volt)} kV ===")
        print(f"  节点数: {info['num_nodes']}, 簇数: {info['num_clusters']}")
        for c in info["clusters"]:
            if c.get("label"):
                print(f"    - {c['id']} ({c['label']}): size={c['size']}")
            else:
                print(f"    - {c['id']}: size={c['size']}")



# In[]
# === 起始两行（保持不变） ===
feeder_cluster, fc_base_net = init_feeder_net(pp_pf_calculator)
fc_base_net = set_fc_state_with_acts(feeder_cluster, fc_base_net, [98])

# === 依赖与工具 ===
import json, re
from typing import Dict, Tuple, List, Optional, Any
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
from plotly.colors import qualitative as qual


# ---------- 解析/补齐母线地理坐标 ----------
def _parse_geo_value(val) -> Optional[Tuple[float, float]]:
    if val is None or (isinstance(val, float) and np.isnan(val)): return None
    if isinstance(val, dict):
        coords = val.get("coordinates")
        if isinstance(coords, (list, tuple)) and len(coords) >= 2:
            try: return float(coords[0]), float(coords[1])
            except Exception: return None
        return None
    if isinstance(val, str):
        s = val.strip()
        if s.startswith("{"):
            try:
                obj = json.loads(s)
                coords = obj.get("coordinates")
                if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                    return float(coords[0]), float(coords[1])
            except Exception:
                return None
        m = re.match(r"POINT\s*\(\s*([+-]?\d+(\.\d+)?)\s+([+-]?\d+(\.\d+)?)\s*\)", s, flags=re.I)
        if m:
            try: return float(m.group(1)), float(m.group(3))
            except Exception: return None
    return None


def _graph_from_net(net, respect_switches: bool = True) -> nx.Graph:
    G = nx.Graph()
    buses = list(map(int, net.bus.index.tolist()))
    G.add_nodes_from(buses)

    # 线路
    for lid, row in net.line.iterrows():
        if not bool(row.get("in_service", True)): continue
        if respect_switches and hasattr(net, "switch") and not net.switch.empty:
            sw = net.switch[(net.switch["et"] == "l") & (net.switch["element"] == lid)]
            if (len(sw) > 0) and (sw["closed"].astype(bool) == False).any(): continue
        G.add_edge(int(row["from_bus"]), int(row["to_bus"]), kind="line", idx=int(lid))

    # 变压器
    if hasattr(net, "trafo") and not net.trafo.empty:
        for tid, row in net.trafo.iterrows():
            if not bool(row.get("in_service", True)): continue
            if respect_switches and hasattr(net, "switch") and not net.switch.empty:
                sw = net.switch[(net.switch["et"] == "t") & (net.switch["element"] == tid)]
                if (len(sw) > 0) and (sw["closed"].astype(bool) == False).any(): continue
            G.add_edge(int(row["hv_bus"]), int(row["lv_bus"]), kind="trafo", idx=int(tid))

    # 闭合母线-母线开关
    if hasattr(net, "switch") and not net.switch.empty:
        for sid, sw in net.switch.iterrows():
            if sw["et"] == "b" and bool(sw["closed"]):
                G.add_edge(int(sw["bus"]), int(sw["element"]), kind="bus_switch", idx=int(sid))
    return G


# ---------- 去重叠：jitter + repel ----------
def _jitter_duplicate_positions(pos: Dict[int, Tuple[float, float]],
                                radius: float = 0.015) -> Dict[int, Tuple[float, float]]:
    buckets = defaultdict(list)
    for n, (x, y) in pos.items():
        buckets[(round(x, 9), round(y, 9))].append(n)
    for _, nodes in buckets.items():
        m = len(nodes)
        if m <= 1: continue
        angles = np.linspace(0, 2*np.pi, m, endpoint=False)
        cx, cy = pos[nodes[0]]
        for node, a in zip(nodes, angles):
            pos[node] = (cx + radius*np.cos(a), cy + radius*np.sin(a))
    return pos


def _repel_overlaps(pos: Dict[int, Tuple[float, float]],
                    min_dist: float,
                    step: float = 0.8,
                    max_iter: int = 200) -> Dict[int, Tuple[float, float]]:
    nodes = list(pos.keys())
    X = np.array([pos[n] for n in nodes], dtype=float)

    def _cell_key(xy, cell):
        return (int(np.floor(xy[0]/cell)), int(np.floor(xy[1]/cell)))

    for _ in range(max_iter):
        cell = float(min_dist)
        grid: Dict[Tuple[int,int], List[int]] = {}
        for i, xy in enumerate(X):
            grid.setdefault(_cell_key(xy, cell), []).append(i)

        moved = False
        disp = np.zeros_like(X)
        for (cx, cy), idxs in grid.items():
            neigh = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    neigh += grid.get((cx+dx, cy+dy), [])
            for i in idxs:
                for j in neigh:
                    if j <= i: continue
                    dvec = X[i] - X[j]
                    d = np.hypot(dvec[0], dvec[1])
                    if d < 1e-9:
                        dirv = np.random.uniform(-1, 1, size=2)
                        dirv /= (np.linalg.norm(dirv) + 1e-12)
                    else:
                        dirv = dvec / max(d, 1e-12)
                    if d < min_dist:
                        overlap = (min_dist - d)
                        push = dirv * (overlap * 0.5)
                        disp[i] += push; disp[j] -= push
                        moved = True
        X += step * disp
        if not moved: break

    for n, xy in zip(nodes, X):
        pos[n] = (float(xy[0]), float(xy[1]))
    return pos


# ---------- 位置生成：支持去重叠 ----------
def ensure_bus_positions(
    net,
    respect_switches: bool = True,
    spring_k: Optional[float] = None,
    spread_factor: float = 4.5,   # 更松散
    iterations: int = 900,        # 更多迭代
    seed: int = 42,
    jitter_frac: float = 0.03,   # 抖开半径（相对边长）
    repel_min_frac: float = 0.05, # 最小间距（相对边长）
    repel_max_iter: int = 200,
) -> Dict[int, Tuple[float, float]]:
    pos: Dict[int, Tuple[float, float]] = {}

    # 1) 读取 geo
    if "geo" in net.bus.columns:
        for bid, val in net.bus["geo"].items():
            p = _parse_geo_value(val)
            if p is not None:
                pos[int(bid)] = (float(p[0]), float(p[1]))

    # 2) spring 补齐
    missing = [int(b) for b in net.bus.index if int(b) not in pos]
    if missing:
        G = _graph_from_net(net, respect_switches=respect_switches)
        n = max(1, G.number_of_nodes())
        k = spring_k if spring_k is not None else (float(spread_factor) / np.sqrt(n))
        fixed_nodes = list(pos.keys())
        spring_pos = nx.spring_layout(
            G,
            pos=pos if fixed_nodes else None,
            fixed=fixed_nodes if fixed_nodes else None,
            k=k,
            iterations=iterations,
            seed=seed,
            center=(0.0, 0.0),
        )
        for n_ in G.nodes:
            if n_ not in pos:
                xy = spring_pos[n_]
                pos[n_] = (float(xy[0]), float(xy[1]))

    # 3) 去重叠：按当前边长比例抖开 + 斥力
    xs = np.array([p[0] for p in pos.values()], dtype=float)
    ys = np.array([p[1] for p in pos.values()], dtype=float)
    side = max(float(xs.max() - xs.min()), float(ys.max() - ys.min()), 1.0)
    pos = _jitter_duplicate_positions(pos, radius=jitter_frac * side)
    pos = _repel_overlaps(pos, min_dist=repel_min_frac * side, step=0.8, max_iter=repel_max_iter)

    return pos


# ---------- 绘图辅助 ----------
def _segments_from_edges(edges: List[Tuple[int, int]], pos: Dict[int, Tuple[float, float]]):
    xs, ys = [], []
    for a, b in edges:
        if (a in pos) and (b in pos):
            xa, ya = pos[a]; xb, yb = pos[b]
            xs += [xa, xb, None]; ys += [ya, yb, None]
    return xs, ys


def _tight_square_ranges(pos: Dict[int, Tuple[float, float]], pad: float = 0.03) -> Tuple[List[float], List[float]]:
    xs = np.array([p[0] for p in pos.values()], dtype=float)
    ys = np.array([p[1] for p in pos.values()], dtype=float)
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    w, h = (xmax - xmin), (ymax - ymin)
    side = max(w, h) if max(w, h) > 0 else 1.0
    pad_len = side * pad
    xr = [cx - side / 2 - pad_len, cx + side / 2 + pad_len]
    yr = [cy - side / 2 - pad_len, cy + side / 2 + pad_len]
    return xr, yr


# ---------- 分区映射 & 调色 ----------
def _index_id_to_bus_and_volt(pp_pf_calculator):
    id2bus, id2volt = {}, {}
    for nd in getattr(pp_pf_calculator, "ele_nodes", []):
        nid = getattr(nd, "ID", None)
        bus = getattr(nd, "bus", None)
        volt = getattr(nd, "volt", None)
        if nid is not None and bus is not None:
            id2bus[str(nid)]  = int(bus)
            id2volt[str(nid)] = float(volt) if volt is not None else None
    return id2bus, id2volt


def build_bus_partition_map(pp_pf_calculator, partitions):
    """返回 {bus_idx: (volt_kv, cluster_idx, label)}；10kV 的 label 为馈线名，其它为 None。"""
    id2bus, _ = _index_id_to_bus_and_volt(pp_pf_calculator)
    bus_map = {}
    for volt, info in partitions.items():
        node2cluster = info.get("node2cluster", {})
        clusters_meta = info.get("clusters", [])
        for nid, ci in node2cluster.items():
            bus = id2bus.get(str(nid))
            if bus is None: continue
            label = None
            if 0 <= int(ci) < len(clusters_meta):
                label = clusters_meta[int(ci)].get("label")
            bus_map[bus] = (float(volt), int(ci), label)
    return bus_map


def _volt_base_hue(volt: float) -> float:
    # 同电压同色系：220kV=红、110kV=蓝、10kV=绿（10kV 实际用定性色板，这里备用）
    if abs(volt - 220.0) <= 2: return 10
    if abs(volt - 110.0) <= 2: return 210
    if abs(volt - 10.0)  <= 2: return 130
    return 0


def _make_palette(n, base_hue, hue_span=40, sat=70, l_center=52, l_span=12):
    """修复版：色相取模到 [0, 360)。"""
    if n <= 0: return []
    if n == 1:
        base = float(base_hue) % 360.0
        return [f"hsl({base:.1f}, {sat}%, {l_center}%)"]
    colors = []
    for i in range(n):
        t = i / (n - 1)
        raw_hue = base_hue + hue_span * (t - 0.5)
        hue = float(raw_hue) % 360.0
        light = l_center + l_span * (t - 0.5) * 2
        light = max(30, min(72, light))
        colors.append(f"hsl({hue:.1f}, {sat}%, {light:.0f}%)")
    return colors

# 电压等级配色（可以按自己喜好调整）
VOLT_LEVEL_COLORS = {
    220: "rgba(228, 26, 28, 0.9)",   # 220kV：红
    110: "rgba(55, 126, 184, 0.9)",  # 110kV：蓝
    10:  "rgba(77, 175, 74, 0.9)",   # 10kV：绿（如果需要）
}

# ---------- 主绘图（节点符号与大小统一，仅用颜色区分） ----------
def plot_fc_base_net(
    net,
    respect_switches: bool = True,
    show_bus_labels: bool = False,
    label_field: str = "name",
    width: int = 1200,
    height: int = 720,
    marker_size: float = 10.0,     # 统一大小
    spread_factor: float = 4.5,
    partitions: Optional[Dict[float, Dict[str, Any]]] = None,
    pp_pf_calculator: Optional[Any] = None,
):
    pos = ensure_bus_positions(
        net,
        respect_switches=respect_switches,
        spread_factor=spread_factor,
        iterations=900,
        jitter_frac=0.012,
        repel_min_frac=0.022,
        repel_max_iter=200,
    )

    # === 组装边 ===
    line_edges, trafo_edges, bus_sw_edges = [], [], []
    for lid, row in net.line.iterrows():
        if not bool(row.get("in_service", True)): continue
        if respect_switches and hasattr(net, "switch") and not net.switch.empty:
            sw = net.switch[(net.switch["et"] == "l") & (net.switch["element"] == lid)]
            if (len(sw) > 0) and (sw["closed"].astype(bool) == False).any(): continue
        line_edges.append((int(row["from_bus"]), int(row["to_bus"])))

    if hasattr(net, "trafo") and not net.trafo.empty:
        for tid, row in net.trafo.iterrows():
            if not bool(row.get("in_service", True)): continue
            if respect_switches and hasattr(net, "switch") and not net.switch.empty:
                sw = net.switch[(net.switch["et"] == "t") & (net.switch["element"] == tid)]
                if (len(sw) > 0) and (sw["closed"].astype(bool) == False).any(): continue
            trafo_edges.append((int(row["hv_bus"]), int(row["lv_bus"])))

    if hasattr(net, "switch") and not net.switch.empty:
        for _, sw in net.switch.iterrows():
            if sw["et"] == "b" and bool(sw["closed"]):
                bus_sw_edges.append((int(sw["bus"]), int(sw["element"])))

    lx, ly = _segments_from_edges(line_edges, pos)
    tx, ty = _segments_from_edges(trafo_edges, pos)
    bx, by = _segments_from_edges(bus_sw_edges, pos)

    # === 母线散点数据（统一大小与符号） ===
    bus_df = net.bus.copy()
    bus_df["x"] = [pos[int(b)][0] for b in bus_df.index]
    bus_df["y"] = [pos[int(b)][1] for b in bus_df.index]

    hover_text = []
    for bid, row in bus_df.iterrows():
        nm = str(row.get(label_field, f"bus {bid}"))
        vk = row["vn_kv"] if "vn_kv" in row else None
        hover_text.append(f"<b>{nm}</b><br>bus: {bid}<br>vn_kv: {vk}")

    fig = go.Figure()
    if len(lx) > 0:
        fig.add_trace(go.Scatter(x=lx, y=ly, mode="lines",
                                 line=dict(width=1.2, color="rgba(120,120,120,0.85)"),
                                 name="Lines", hoverinfo="skip"))
    if len(tx) > 0:
        fig.add_trace(go.Scatter(x=tx, y=ty, mode="lines",
                                 line=dict(width=1.4, dash="dot", color="rgba(80,80,80,0.9)"),
                                 name="Transformers", hoverinfo="skip"))
    if len(bx) > 0:
        fig.add_trace(go.Scatter(x=bx, y=by, mode="lines",
                                 line=dict(width=1.0, dash="dash", color="rgba(150,150,150,0.6)"),
                                 name="Bus-Bus Switch (closed)", hoverinfo="skip", visible="legendonly"))

    # === 分区着色（仅颜色区分；统一 symbol/size） ===
    if partitions is not None and pp_pf_calculator is not None:
        # bus -> (volt, ci, label)
        bus_partition = build_bus_partition_map(pp_pf_calculator, partitions)

        # 为每个电压层生成调色板（10kV 用离散定性色板）
        volt_palettes: Dict[float, List[str]] = {}
        for volt, info in partitions.items():
            k = int(info.get("num_clusters", len(info.get("clusters", []))))
            v = float(volt)
            if abs(v - 10.0) <= 2:
                pool = list(qual.Dark24)
                if k > len(pool):
                    pool = (pool * int(np.ceil(k / len(pool))))[:k]
                volt_palettes[v] = pool[:k]
            else:
                base_hue = _volt_base_hue(v)
                volt_palettes[v] = _make_palette(k, base_hue)

        # 分组
         # 分组：按电压等级分组（颜色由电压决定）
        groups: Dict[Any, Dict[str, List[float]]] = {}
        meta_names: Dict[Any, str] = {}
        for i, bid in enumerate(bus_df.index):
            bus = int(bid)
            x, y = bus_df.loc[bid, "x"], bus_df.loc[bid, "y"]
            t = hover_text[i]
            if bus in bus_partition:
                volt, ci, label = bus_partition[bus]
                v_round = int(round(volt))  # 10 / 110 / 220
                key = v_round                  # ← 关键：按电压分组
                if key not in groups:
                    groups[key] = {"x": [], "y": [], "text": []}
                    # 图例名称：只显示电压等级
                    meta_names[key] = f"{v_round}kV"
                groups[key]["x"].append(x)
                groups[key]["y"].append(y)
                groups[key]["text"].append(t)
            else:
                key = "Unassigned"
                if key not in groups:
                    groups[key] = {"x": [], "y": [], "text": []}
                    meta_names[key] = "Unassigned"
                groups[key]["x"].append(x)
                groups[key]["y"].append(y)
                groups[key]["text"].append(t)

        # 输出 trace：按电压等级配色
        for key, pts in groups.items():
            if key == "Unassigned":
                color = "rgba(180,180,180,0.9)"
                legendgroup = "Other"
                name = meta_names[key]
            else:
                v_round = int(key)
                color = VOLT_LEVEL_COLORS.get(v_round, "#808080")
                legendgroup = f"{v_round}kV"
                name = meta_names[key]

            fig.add_trace(go.Scatter(
                x=pts["x"], y=pts["y"], mode="markers",
                marker=dict(
                    size=marker_size,
                    color=color,
                    symbol="circle",
                    line=dict(width=0.8, color="rgba(20,20,20,0.7)")
                ),
                name=name,
                legendgroup=legendgroup,
                hoverinfo="text",
                text=pts["text"]
            ))


            fig.add_trace(go.Scatter(
                x=pts["x"], y=pts["y"], mode="markers",
                marker=dict(
                    size=marker_size,           # 统一大小
                    color=color,                # 仅颜色区分
                    symbol="circle",            # 统一符号
                    line=dict(width=0.8, color="rgba(20,20,20,0.7)")
                ),
                name=name, legendgroup=legendgroup,
                hoverinfo="text", text=pts["text"]
            ))
    else:
        # 兼容：无分区时也统一大小/符号，仅按电压数值着色
        vn = bus_df["vn_kv"] if "vn_kv" in bus_df.columns else pd.Series(1.0, index=bus_df.index)
        fig.add_trace(go.Scatter(
            x=bus_df["x"], y=bus_df["y"], mode="markers",
            marker=dict(size=marker_size, symbol="circle", color=vn, colorscale="Viridis", showscale=True,
                        line=dict(width=0.8, color="rgba(20,20,20,0.7)")),
            text=hover_text, hoverinfo="text", name="Buses"
        ))

    # 外部电源（非分区节点，保留星形以便识别）
    if hasattr(net, "ext_grid") and not net.ext_grid.empty:
        xs, ys, text = [], [], []
        for _, eg in net.ext_grid.iterrows():
            b = int(eg["bus"])
            if b in pos:
                xs.append(pos[b][0]); ys.append(pos[b][1]); text.append(f"ext_grid @ bus {b}")
        if xs:
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers",
                marker=dict(symbol="star", size=16, line=dict(width=1, color="black")),
                name="Ext Grid", hoverinfo="text", text=text, legendgroup="Other"
            ))

    if show_bus_labels:
        labels = [str(bus_df.loc[bid, label_field]) if pd.notna(bus_df.loc[bid, label_field]) else str(bid)
                  for bid in bus_df.index]
        fig.add_trace(go.Scatter(
            x=bus_df["x"], y=bus_df["y"], mode="text",
            text=labels, textposition="top center",
            textfont=dict(size=10), name="Bus Labels",
            hoverinfo="skip", showlegend=False
        ))

    # 紧致/等比坐标范围
    xr, yr = _tight_square_ranges(pos, pad=0.02)
    fig.update_xaxes(range=xr); fig.update_yaxes(range=yr, scaleanchor="x", scaleratio=1)

    fig.update_layout(
        width=width, height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="white", plot_bgcolor="white",
        #legend=dict(title="Partitions", orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        dragmode="pan"
    )
    return fig


# ====================== 使用示例 ======================
vt_partitions = build_voltage_partitions(pp_pf_calculator)
fig = plot_fc_base_net(
    fc_base_net,
    respect_switches=True,
    show_bus_labels=False,
    spread_factor=5,     # 越大越分散
    marker_size=6.0,      # 统一节点大小（按需调整）
    partitions=vt_partitions,
    pp_pf_calculator=pp_pf_calculator,
)
plt.savefig('ori_topo.svg',  bbox_inches='tight', pad_inches=0)
fig.show()
# In[]
# In[]
# ============================
# 目标：
# 1) 构造动作98前/后的net
# 2) 找到动作98导致“断开(从原始拓扑消失)”的支路（并兼容“闭合(新增)”支路）
# 3) 在动作98拓扑图上，高亮被动作支路的两端节点（同一支路两端同色），可选高亮支路本身
# ============================

import numpy as np
import plotly.graph_objects as go
from plotly.colors import qualitative as qual

ACT_ID = 98

# ---------- 1) 构造动作98前/后的网络 ----------
parsed_cim = CimEParser(PfDataPath) # load the CIM data
pp_pf_calculator = PandaPowerFlowCalculator(parsed_cim,slack_nd='703002137')    # set the slack bus   
    
feeder_cluster, fc_base_net0 = init_feeder_net(pp_pf_calculator)                      # 原始拓扑
fc_base_net98 = set_fc_state_with_acts(feeder_cluster, fc_base_net0, [ACT_ID])        # 动作98拓扑

# ---------- 2) 提取“支路集合”工具 ----------
def _net_line_pairs(net):
    """从 pandapower net.line 取无向支路集合 {(min_bus,max_bus),...}，只看 in_service=True 的线路。"""
    pairs = set()
    for _, row in net.line.iterrows():
        if not bool(row.get("in_service", True)):
            continue
        a = int(row["from_bus"]); b = int(row["to_bus"])
        pairs.add((min(a, b), max(a, b)))
    return pairs

pairs0  = _net_line_pairs(fc_base_net0)
pairs98 = _net_line_pairs(fc_base_net98)

# ---------- 3) 从 feasible_switch_states 取“动作涉及的支路(候选)” ----------
sw_state = feeder_cluster.feasible_switch_states[ACT_ID]
act_open_lines  = sw_state.get("0", [])  # 动作要求“打开”的支路对象列表
act_close_lines = sw_state.get("1", [])  # 动作要求“闭合”的支路对象列表

def _obj_pairs(lines):
    """从你的 line 对象列表提取无向(bus_i,bus_j)"""
    out = []
    for ln in lines:
        try:
            a = int(ln.I_nd.bus); b = int(ln.J_nd.bus)
            out.append((min(a, b), max(a, b)))
        except Exception:
            continue
    return out

open_pairs_candidate  = _obj_pairs(act_open_lines)
close_pairs_candidate = _obj_pairs(act_close_lines)

# ---------- 4) 识别“确实发生变化”的支路 ----------
# 4.1 原始存在但动作后消失：断开（你最关心的）
opened_effective = [p for p in open_pairs_candidate if (p in pairs0) and (p not in pairs98)]

# 4.2 原始不存在但动作后出现：闭合新增（可选，也一并标注）
closed_effective = [p for p in close_pairs_candidate if (p not in pairs0) and (p in pairs98)]

# 4.3 兜底：若 feasible_switch_states 与 net.line 表达存在不一致，则用对称差补充
symdiff = sorted(list(pairs0 ^ pairs98))  # 所有发生变化的支路（无论原因）
already = set(opened_effective) | set(closed_effective)
other_changed = [p for p in symdiff if p not in already]

# 汇总（按类型）
branches = []
for p in opened_effective:
    branches.append({"pair": p, "mode": "OPENED"})   # 原始有 -> 动作后无
for p in closed_effective:
    branches.append({"pair": p, "mode": "CLOSED"})   # 原始无 -> 动作后有
for p in other_changed:
    branches.append({"pair": p, "mode": "CHANGED"})  # 兜底

print(f"[ACT {ACT_ID}] opened_effective={len(opened_effective)}, closed_effective={len(closed_effective)}, other_changed={len(other_changed)}")
if len(branches) == 0:
    print("未检测到动作导致的支路变化（可能：动作98未改变net.line，或需要检查 feasible_switch_states / set_fc_state_with_acts 的一致性）。")

# ---------- 5) 画“动作98拓扑”的底图 ----------
# 分区（可选，用于电压等级着色）。注意：分区来自 pp_pf_calculator；其线路 closed 状态若在 set_fc_state_with_acts 中被同步修改，则会反映动作影响。
vt_partitions = build_voltage_partitions(pp_pf_calculator)

# 为确保叠加高亮点的位置与底图一致，固定随机种子并用同一套参数调用 ensure_bus_positions
np.random.seed(42)
fig = plot_fc_base_net(
    fc_base_net98,
    respect_switches=True,
    show_bus_labels=False,
    spread_factor=5,
    marker_size=6.0,
    partitions=vt_partitions,
    pp_pf_calculator=pp_pf_calculator,
)

np.random.seed(42)
pos = ensure_bus_positions(
    fc_base_net98,
    respect_switches=True,
    spread_factor=5,
    iterations=900,
    seed=42,
    jitter_frac=0.012,
    repel_min_frac=0.022,
    repel_max_iter=200,
)

# ---------- 6) 高亮：对每条“被动作支路”的两端节点同色（可选加粗该支路） ----------
palette = list(qual.Dark24)
if len(branches) > len(palette):
    palette = (palette * int(np.ceil(len(branches) / len(palette))))[:len(branches)]

HIGHLIGHT_LINE = True   # True：同时把支路本体也加粗同色；False：只染两端节点
NODE_SIZE_BOOST = 5.0   # 高亮节点比底图大多少

for i, br in enumerate(branches):
    (a, b) = br["pair"]
    mode = br["mode"]
    color = palette[i]

    if (a not in pos) or (b not in pos):
        continue

    xa, ya = pos[a]; xb, yb = pos[b]
    hover = f"<b>Act {ACT_ID} - {mode}</b><br>branch: {a} - {b}<br>idx: {i+1}"

    # 可选：高亮支路线段
    if HIGHLIGHT_LINE:
        fig.add_trace(go.Scatter(
            x=[xa, xb], y=[ya, yb],
            mode="lines",
            line=dict(width=4.0, color=color),
            hoverinfo="text",
            text=[hover, hover],
            name=f"Act{ACT_ID} branches",
            legendgroup=f"Act{ACT_ID}",
            showlegend=(i == 0),  # 图例只出现一次，避免过长
        ))

    # 高亮两端节点（同色）
    fig.add_trace(go.Scatter(
        x=[xa, xb], y=[ya, yb],
        mode="markers+text",
        text=[str(i+1), str(i+1)],          # 同一支路两端用同编号，便于对应
        textposition="top center",
        marker=dict(
            size=6.0 + NODE_SIZE_BOOST,
            color=color,
            symbol="circle",
            line=dict(width=1.2, color="rgba(0,0,0,0.9)")
        ),
        hoverinfo="text",
        textfont=dict(size=10),
        hovertext=[hover, hover],
        name=f"Act{ACT_ID} endpoints",
        legendgroup=f"Act{ACT_ID}",
        showlegend=False
    ))

# 标题（可选）
fig.update_layout(
    title=f"Topology after Action {ACT_ID} (highlight affected branch endpoints)"
)

fig.show()
# In[]