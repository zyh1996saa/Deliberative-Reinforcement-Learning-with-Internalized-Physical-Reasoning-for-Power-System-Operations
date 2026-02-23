# In[]
import os
from config746sys import (
    WORKPATH,
    DATAPATH,
    nd_load_stat,
    nd_withload,
    TimeSeriesDataPath,
    year_range,
    month_range,
    T_riskassess,
    nd2gisid,
    gisid2nd,
    DistTfPath,
    r_switch,
    x_switch,
    PfDataPath 
)   
import pandas as pd
DistTF = pd.read_csv(DistTfPath,sep='\t')
sub_746_bus_system_node_order_file = WORKPATH + "/system_file/746sys/746sys_node_order.json"
from Utls.yantian_sys_746sys import *
from Utls.utls import get_network_matrices
import pandapower as pp
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns


def enforce_node_order(fc, order_file):
    """
    第一次运行：把当前节点顺序写入 order_file
    之后运行：读取 order_file 并按记录对 fc.nodes 排序
    """
    if os.path.exists(order_file):
        # --- ② 读取并重排 ---
        with open(order_file, "r", encoding="utf-8") as f:
            saved_order = json.load(f)                 # e.g. ["107005153", ...]
        order_map = {nid: idx for idx, nid in enumerate(saved_order)}

        # 排序：已记录节点按原顺序，新节点自然排在后面
        fc.ele_nodes.sort(key=lambda n: order_map.get(n.nd, len(saved_order)))
    else:
        # --- ① 第一次执行，保存顺序 ---
        saved_order = [n.nd for n in fc.ele_nodes]
        with open(order_file, "w", encoding="utf-8") as f:
            json.dump(saved_order, f, ensure_ascii=False, indent=2)
            
def built_ppnet_for_pfcal(ori_ppnet):
    base_net_replacement = copy.deepcopy(ori_ppnet)
    
    # 收集需要删除的线路索引
    indices_to_drop = []
    
    # 安全遍历：使用索引列表而非range
    for i in base_net_replacement.line.index:
        r = base_net_replacement.line.at[i, 'r_ohm_per_km']
        x = base_net_replacement.line.at[i, 'x_ohm_per_km']
        
        if r == r_switch and x == x_switch:
            # 创建开关
            pp.create_switch(
                base_net_replacement,
                bus=int(base_net_replacement.line.at[i, 'from_bus']),
                element=int(base_net_replacement.line.at[i, 'to_bus']),
                et="b",  # 母线-母线开关
                closed=True
            )
            # 记录待删除索引
            indices_to_drop.append(i)
    
    # 批量删除标记的线路
    base_net_replacement.line.drop(indices_to_drop, inplace=True)
    
    return base_net_replacement

# 定义初始化配电网模型的函数
def init_feeder_net(pp_pf_calculator):
    np.random.seed(42)
    # 加载可行馈线的开关状态
    load_feasible_feeders_switch_states(pp_pf_calculator, path=WORKPATH + '/system_file/746sys/')
 
    for i, node_bus in DistTF['Node'].items():
        node = pp_pf_calculator.bus2node[node_bus]
        node.pl = DistTF['P'][i] 
        node.ql = DistTF['Q'][i] 
    # 选择第一个馈线簇
    #feeder_cluster = pp_pf_calculator.feeder_clusters[0]
    #ignored_buses = [793,900,852,854,270] # 忽略变压器中间端节点
    # 从pp_pf_calculator.ele_nodes中删去忽略的节点
    #pp_pf_calculator.ele_nodes = [node for node in pp_pf_calculator.ele_nodes if node.bus not in ignored_buses]
    pp_pf_calculator.ele_nodes = [node for node in pp_pf_calculator.ele_nodes ]
    enforce_node_order(pp_pf_calculator,order_file=sub_746_bus_system_node_order_file) # 确保节点顺序一致
    node_ids = [node.nd for node in pp_pf_calculator.ele_nodes]
    
    # pp_pf_calculator.node_buses2line['2024_2014'].closed = '0'
    # pp_pf_calculator.node_buses2line['13453_13474'].closed = '0'
    # pp_pf_calculator.node_buses2line['3671_3610'].closed = '0'
    
    # 根据节点列表创建pandapower网络
    base_net = pp_pf_calculator.create_pandapower_net_from_node_ids(node_ids)
    # 添加外部电源（平衡节点）
    slack_nd_bus = pp_pf_calculator.nodeID2node[pp_pf_calculator.slack_nd].bus
    
    pp.create_ext_grid(base_net, bus=slack_nd_bus, vm_pu=1, va_degree=0)
 
    

    return pp_pf_calculator.feeder_clusters[0], base_net

def select_402system_timeseres_data(feeder_cluster):
    """
    从原始时序数据文件中筛选402节点系统的时序数据，并保存到指定路径。
    """
    # 判断数据文件是否已经存在
    if os.path.exists(os.path.join(WORKPATH, 'system_file', '402system_timeseries_data.csv')):
        print("402系统时序数据已存在，直接读取。")
        return pd.read_csv(os.path.join(WORKPATH, 'system_file', '402system_timeseries_data.csv'))
    else:
        print("开始筛选402系统时序数据...")
        # 读取所有月份的时序数据文件
        df_list = [pd.read_csv(os.path.join(TimeSeriesDataPath, '2024/2024-%s.csv'%(str(month).zfill(2))), encoding='utf-8') for month in range(1,13)]
        combined_df = pd.concat(df_list, ignore_index=True)
        del df_list
        fc_node_nds = [node.nd for node in feeder_cluster.nodes]
        fc_node_gisids = [nd2gisid[nd] for nd in fc_node_nds if nd in nd2gisid] 
        # 将combined_df['GISID']的数据格式转为字符串
        combined_df['GISID'] = combined_df['GISID'].astype(str)
        # 选择GISID在fc_node_gisids中的行
        selected_df = combined_df[combined_df['GISID'].isin(fc_node_gisids)] 
        # 保存筛选后的数据到指定路径
        selected_df.to_csv(os.path.join(WORKPATH, 'system_file', '402system_timeseries_data.csv'), index=False, encoding='utf-8')
        print(f"已筛选并保存402节点系统的时序数据，共{len(selected_df)}条记录。")
        print("402系统时序数据已创建。")
        return selected_df
    


if __name__ == "__main__":
    parsed_cim = CimEParser(PfDataPath)

    pp_pf_calculator = PandaPowerFlowCalculator(parsed_cim,slack_nd='703002137')    
    #pp_pf_calculator.scan_feasible_feeders_switch_states()
    #save_feasible_feeders_switch_states(pp_pf_calculator, path=WORKPATH + '/system_file/')
    
    feeder_cluster, fc_base_net = init_feeder_net(pp_pf_calculator)
    fc_base_net_cal = built_ppnet_for_pfcal(fc_base_net)
    
    
    pp.runpp(fc_base_net_cal)
   
# In[]
if __name__ == "__main__":
    Hin, Ybus, bus_map = get_network_matrices(fc_base_net,fc_base_net_cal)
    for i in range(fc_base_net.line.shape[0]):
        line = fc_base_net.line.loc[i,:]
        from_bus = line['from_bus']
        to_bus = line['to_bus']
        mapped_from = bus_map[from_bus]
        mapped_to = bus_map[to_bus]
        Y_from_to = Ybus[mapped_from, mapped_to]
        print(f"Line from {from_bus} to {to_bus} has Ybus value: {Y_from_to}")
    sparse_Yi = csr_matrix(Ybus)

    np.save('H_ori.npy', Hin)
    save_npz('Y_ori', sparse_Yi)
    # 保持bus_map到json文件
    with open(os.path.join(WORKPATH, 'system_file', 'bus_map.json'), 'w', encoding='utf-8') as f:  
        json.dump(bus_map, f, ensure_ascii=False, indent=2)

