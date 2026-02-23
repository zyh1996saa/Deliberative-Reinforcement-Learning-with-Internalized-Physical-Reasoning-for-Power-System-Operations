# In[]
import pickle
import pandas as pd
#from glob import glob
import os
os.environ["PYTHONHASHSEED"]="0"
import sys
from config746sys import (
    WORKPATH,
    p_lower,
    p_upper,
    q_lower,
    q_upper,
    PfDataPath, 
    nd_withload, 
    nd_load_stat,
    BaseMVA,
    BaseVoltage,
    BaseZ,
    r_switch,
    x_switch,
    FeederFiles
    )
cwd = WORKPATH
#sys.path.append(cwd+'/实验/Utls')
from Utls.utls import case2AandH,refresh_busnum
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz
import numpy as np
import pandapower as pp
import random
from tqdm import tqdm
from collections import defaultdict
import pandapower.plotting as plot
import copy
import collections
from collections import deque
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandapower.plotting.plotly as ppplotly
from itertools import product
#import json

from pandapower.converter import to_ppc
import re
import pypower.runpf as runpf


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

def sample_a_new_net(feeder_cluster, fc_base_net):
    actspace = len(feeder_cluster.feasible_switch_states)
    random_act = np.random.randint(actspace)
    new_net = set_fc_state_with_acts(feeder_cluster,fc_base_net,[random_act])
    new_net.load['p_mw'] *= np.random.uniform(p_lower,p_upper,size=(fc_base_net.load['p_mw'].shape))
    new_net.load['q_mvar'] *= np.random.uniform(q_lower,q_upper,size=(fc_base_net.load['q_mvar'].shape))
    return new_net


# 初始化并查集，处理固定闭合的线路和电源节点
class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def add(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 1

    def find(self, x):
        if x not in self.parent:  # 添加检查，确保节点存在
            self.add(x)  # 如果节点不存在，先添加它
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        x_root = self.find(x)
        y_root = self.find(y)
        if x_root == y_root:
            return False  # 已连通，未合并
        if self.rank[x_root] < self.rank[y_root]:
            self.parent[x_root] = y_root
        else:
            self.parent[y_root] = x_root
            if self.rank[x_root] == self.rank[y_root]:
                self.rank[x_root] += 1
        return True

    def copy(self):
        uf_copy = UnionFind()
        uf_copy.parent = self.parent.copy()
        uf_copy.rank = self.rank.copy()
        return uf_copy

class CimEParser:
    def __init__(self, file_path: str):
        xls = pd.ExcelFile(file_path)
        
        for sheet_name in required_sheet_name:
            if sheet_name not in xls.sheet_names:
                raise ValueError("必要的sheet %s 不存在于文件 %s 中"%(sheet_name,file_path))
        for sheet_name in xls.sheet_names:
            exec(f"self.{sheet_name} = pd.read_excel(xls, sheet_name=sheet_name, dtype=str)")

        
class FeederCluster:
    def __init__(self,cluster_name,feeder_group_list):
        self.name = cluster_name
        self.nodes = []
        self.feeder_start_nodes = []
        self.main_buses_10kV = []
        self.lines = []
        self.gateways = []
        self.concat_switches = []
        self.feeder_group_list = feeder_group_list
        for fg in self.feeder_group_list:
            fg.belong_feeder_cluster = self
        self.feasible_switch_states = []
        
        for fg in self.feeder_group_list:
            for node in fg.nodes:
                if node:
                    if node not in self.nodes:
                        self.nodes.append(node)

            for line in fg.lines:
                if line:
                    if line not in self.lines:
                        self.lines.append(line)
                        
            for gateway in fg.gateways:
                if gateway:
                    if gateway not in self.gateways:
                        self.gateways.append(gateway)
                        
            for concat_line in fg.concat_lines_with_other_feeder_group:
                if concat_line:
                    if concat_line not in self.concat_switches:
                        self.concat_switches.append(concat_line)
            
            
            self.feeder_start_nodes.append(fg.feeder_start_node)
            if fg.main_bus_10kV not in self.main_buses_10kV:
                self.main_buses_10kV.append(fg.main_bus_10kV)
                
        self.switch_lines = list(set(self.gateways + self.concat_switches))
    
    def scan_feasible_switch_states(self):
        """
        使用遍历所有开关组合的方法，搜索可行的开关状态。
        考虑辐射状网络结构和供电连续性要求。
        """
        switch_lines = self.switch_lines
        n_switch = len(switch_lines)
        self.feasible_switch_states = []

        if n_switch == 0:
            return 

        # 完整的并查集实现
        class UnionFind:
            def __init__(self):
                self.parent = {}
                self.rank = {}
            
            def add(self, node):
                if node not in self.parent:
                    self.parent[node] = node
                    self.rank[node] = 0
            
            def find(self, node):
                if self.parent[node] != node:
                    self.parent[node] = self.find(self.parent[node])
                return self.parent[node]
            
            def union(self, node1, node2):
                self.add(node1)
                self.add(node2)
                root1 = self.find(node1)
                root2 = self.find(node2)
                
                if root1 == root2:
                    return False  # 已连通，合并会形成环网
                
                # 按秩合并
                if self.rank[root1] < self.rank[root2]:
                    self.parent[root1] = root2
                elif self.rank[root1] > self.rank[root2]:
                    self.parent[root2] = root1
                else:
                    self.parent[root2] = root1
                    self.rank[root1] += 1
                return True
        
        # 创建并初始化并查集，只考虑10kV节点
        def build_initial_uf():
            uf = UnionFind()
            
            # 添加所有10kV节点
            for node in self.nodes:
                if node.volt <= 10.0:  # 只考虑10kV及以下节点
                    uf.add(node.bus)
            
            # 添加虚拟主网节点并连接所有主母线
            uf.add("VIRTUAL_MAIN_GRID")
            for bus in hv_buses:
                uf.union(bus, "VIRTUAL_MAIN_GRID")
            
            # 合并固定闭合的非开关线路（只考虑10kV线路）
            for line in self.lines:
                # 只处理10kV线路
                if line.I_nd.volt <= 10.0 and line.J_nd.volt <= 10.0:
                    if line not in switch_lines and line.closed == '1':
                        if not uf.union(line.I_nd.bus, line.J_nd.bus):
                            # 固定线路形成环网
                            return None
            
            return uf

        # 获取主母线节点
        hv_buses = [nd.bus for nd in self.main_buses_10kV]
        if not hv_buses:
            return  # 无主母线，无法供电
        
        # 必须供电的节点集合（10kV及以下电压等级节点）
        must_power = {nd.bus for nd in self.nodes if nd.volt <= 10.0}
        
        # 检查初始网络是否有环
        initial_uf = build_initial_uf()
        if initial_uf is None:
            print("警告：初始网络存在环网（固定闭合线路形成环路）")
            return  # 初始网络已存在环网，无可行解
        
        # 枚举所有可能的开关组合
        total_combinations = 1 << n_switch
        solutions = []
        
        print(f"扫描 {total_combinations} 种开关组合...")
        
        # 遍历所有组合
        for combo_index in range(total_combinations):
            current_uf = build_initial_uf()  # 使用全新的并查集实例
            closed_switches = []
            valid_combo = True
            
            # 处理当前组合中的每个开关
            for switch_idx in range(n_switch):
                sw = switch_lines[switch_idx]
                # 只处理10kV开关
                if sw.I_nd.volt > 10.0 or sw.J_nd.volt > 10.0:
                    continue
                    
                if combo_index & (1 << switch_idx):  # 开关闭合
                    if not current_uf.union(sw.I_nd.bus, sw.J_nd.bus):
                        # 合并失败说明已连通，闭合会形成环网
                        valid_combo = False
                        break
                    closed_switches.append(sw)
            
            if not valid_combo:
                continue
            
            # 获取虚拟主网的根节点
            virtual_root = current_uf.find("VIRTUAL_MAIN_GRID")
            
            # 验证负荷节点供电
            for bus in must_power:
                # 只验证10kV节点
                if current_uf.find(bus) != virtual_root:
                    valid_combo = False
                    break
            
            # 保存有效组合
            if valid_combo:
                open_switches = [sw for sw in switch_lines if sw not in closed_switches]
                solutions.append({'1': closed_switches, '0': open_switches})
        
        # 保存所有可行解
        self.feasible_switch_states = solutions
        print(f"找到 {len(solutions)} 个可行开关状态")
        
    # def scan_feasible_switch_states(self):
    #     """
    #     优化后的方法，使用回溯和并查集剪枝，高效搜索可行的开关状态。
    #     """
    #     switch_lines = self.switch_lines
    #     n_switch = len(switch_lines)
    #     self.feasible_switch_states = []

    #     if n_switch == 0:
    #         return 

    #     # 初始化并查集
    #     uf = UnionFind()
    #     # 添加所有节点
    #     for node in self.nodes:
    #         uf.add(node.bus)
    #     # 合并固定闭合的线路
    #     for line in self.lines:
    #         if line not in switch_lines and line.closed == '1':
    #             uf.union(line.I_nd.bus, line.J_nd.bus)
    #     # 合并主母线节点
    #     hv_buses = [nd.bus for nd in self.main_buses_10kV]
    #     if hv_buses:
    #         root = hv_buses[0]
    #         for bus in hv_buses[1:]:
    #             uf.union(root, bus)
    #         main_root = uf.find(root)
    #     else:
    #         main_root = None

    #     # 必须供电的节点集合
    #     #must_power = {nd.bus for nd in self.nodes if getattr(nd, 'withload', 0) == 1}
    #     must_power = {nd.bus for nd in self.nodes if nd.volt <= 10.}
    #     # 回溯搜索
    #     solutions = []

    #     def backtrack(index, current_uf, closed_switches):
    #         if index == n_switch:
    #             # 检查所有必须节点是否连通到主母线
    #             if not main_root:
    #                 return
    #             for bus in must_power:
    #                 if current_uf.find(bus) != main_root:
    #                     print(f"节点 {bus} 未连通到主母线 {main_root}，跳过当前解。")
    #                     return
    #             # 记录可行解
    #             open_switches = [sw for sw in switch_lines if sw not in closed_switches]
    #             solutions.append({'1': closed_switches.copy(), '0': open_switches})
    #             return

    #         sw = switch_lines[index]
    #         i_bus = sw.I_nd.bus
    #         j_bus = sw.J_nd.bus

    #         # 剪枝1：剩余开关是否可能满足条件？
    #         remaining = n_switch - index
    #         current_closed = len(closed_switches)
    #         max_possible = current_closed + remaining
    #         if len(solutions) >= 2**7:
    #             return

    #         # 尝试闭合当前开关
    #         uf_closed = current_uf.copy()
    #         merged = uf_closed.union(i_bus, j_bus)
    #         if merged or (uf_closed.find(i_bus) != uf_closed.find(j_bus)):
    #             # 仅在未形成环时继续
    #             backtrack(index + 1, uf_closed, closed_switches + [sw])

    #         # 尝试断开当前开关（直接传递当前状态）
    #         backtrack(index + 1, current_uf.copy(), closed_switches)

    #     # 初始调用
    #     backtrack(0, uf, [])

    #     # 限制最大解数量
    #     self.feasible_switch_states = solutions#[:100]

    #     # 如果没有可行解，确保返回空列表
    #     if not self.feasible_switch_states:
    #         self.feasible_switch_states = []        
            
            
    def __str__(self):
        return f"feeder_cluster(feeders={self.name})"
        
    def __repr__(self):
        return f"feeder_cluster(feeders={self.name})"

class FeederGroup:
    def __init__(self,node_in_feeder_group,gateways,\
        lines_in_feeder_group,feeder_start_node,feed_group_name,\
            concat_lines_with_other_feeder_group,main_bus_10kV):
        self.nodes = list(node_in_feeder_group)
        for node in self.nodes:
            node.belong_feeder_group = self
        self.gateways = gateways
        self.lines = lines_in_feeder_group
        self.feeder_start_node = feeder_start_node
        self.name = feed_group_name
        self.concat_lines_with_other_feeder_group = concat_lines_with_other_feeder_group
        self.concat_feeder_group = []
        self.main_bus_10kV = main_bus_10kV
        self.belong_feeder_cluster = None
    
    def __getstate__(self):
        state = {}
        keys = ['nodes','gateways','lines','feeder_start_node','name',
                'concat_lines_with_other_feeder_group','main_bus_10kV',]
        state = {key: getattr(self, key) for key in keys}
        return state
        
    def __str__(self):
        return f"FeederGroup(name={self.name},node_num={len(self.nodes)})"

    def __repr__(self):
        return f"FeederGroup(name={self.name},node_num={len(self.nodes)})"
      
      
class EleNode:
    def __init__(self,init_data):
        ID,name,nd,volt,bus = init_data
        self.ID = ID
        self.name = str(name)
        self.nd = str(nd)
        self.volt = float(volt)
        self.bus = int(bus)
        self.device_type = 'elenode'
        self.belong_device = []
        self.possible_neighbor_nodes = []
        self.act_neighbor_nodes = []
        
        self.closest_110_node = None
        self.closest_mainbus_node = None
        self.closest_start_feeder_node = None
        
        self.mainbus_10kV = False # 是否是10kV母线
        self.start_feeder_10kV = False # 是否是10kV馈线的开始节点（与母线直接相连）
        self.in_feeder_10kV = False # 是否是10kV馈线中的节点
        self.concat_node_feeder = False #是否是馈线组间的联络线
        self.belong_feeder_group = None
        self.is_scaned_fg = False # 是否已经扫描过馈线组(每个节点只能属于一个馈线组)
        
        self.gateway_node_10kV = False #是否是馈入10kV网络的入口节点，如"107005153"
        self.concat_node_10kV = False #是否是联络线节点
        self.gateway_node_10kV_otherside = False #是否是与馈入10kV网络的入口节点相连的另一端节点,"107005171"
        
        
        # 源荷数据
        self.pl = 0 #有功负荷
        self.ql = 0 #无功负荷
        self.pg = 0 #有功发电
        self.qg = 0 #无功发电
        self.withload = 0
        # self.withload = self.check_if_with_load()
        # if self.withload :
        #     # self.pl = 0.14
        #     # self.ql = 0.005
        #     self.pl = nd_load_stat[nd_load_stat['nd']==self.nd]['max_p'].values[0]
        #     #self.ql = nd_load_stat[nd_load_stat['nd']==self.nd]['max_q'].values[0]
    
    def __getstate__(self):
        state = {}
        keys = ['ID','name','nd','volt','bus','device_type','mainbus_10kV','start_feeder_10kV',
            'in_feeder_10kV','concat_node_feeder','gateway_node_10kV','concat_node_10kV',
            'gateway_node_10kV_otherside','pl','ql','pg','qg','withload']
        state = {key: getattr(self, key) for key in keys}
        return state
            
        
    def check_if_with_load(self):
        # if '柜' in self.name or '房' in self.name:
        #     return 1
        # else:
        #     return 0
        if self.nd in nd_withload:
            return 1
        else:
            return 0    

    def __str__(self):
        return f"EleNode(nd={self.nd},volt={self.volt},bus={self.bus})"

    def __repr__(self):
        return f"EleNode(Ind={self.nd},volt={self.volt},bus={self.bus})"
        
class TransformerType2:
    def __init__(self,row):
        self.ID = row['ID']
        self.name = row['name']
        #self.I_node = row['I_node']
        #self.J_node = row['J_node']
        self.K_node = None
        self.I_nd_ID = row['I_nd']
        self.J_nd_ID = row['J_nd']
        self.K_nd_ID = None
        self.I_off = row['I_off']
        self.J_off = row['J_off']
        self.K_off = None
        if self.I_off == '0' and self.J_off == '0':
            self.closed = 1
        else:
            self.closed = 0
        #self.I_leakagelmpedence = row['I_leakagelmpedence']
        self.I_loadLoss = row['I_loadLoss']
        self.I_S = row['I_S']
        self.I_r = row['I_r']
        try:
            self.I_rPU = row['I_rPU']
        except:
            self.I_rPU = 0
        self.I_Volt = row['I_vol']
        self.I_x = row['I_x']
        try:
            self.I_xPU = row['I_xPU']
        except:
            self.I_xPU = 0
        #self.J_leakagelmpedence = row['J_leakagelmpedence']
        self.J_loadLoss = row['J_loadLoss']
        self.J_S = row['J_S']
        self.J_r = row['J_r']
        try:
            self.J_rPU = row['J_rPU']
        except:
            self.J_rPU = 0
        self.J_Volt = row['J_vol']
        self.J_x = row['J_x']
        try:
            self.J_xPU = row['J_xPU']
        except:
            self.J_xPU = 0
        self.I_nd = None
        self.J_nd = None
        self.device_type = 'transformer-type2'		
        
        self.gateway_transformer = False 
        
    def __str__(self):
        return f"Transformer_type2(nodes=[{self.I_nd_ID},{self.J_nd_ID}],volt=[{self.I_Volt},{self.J_Volt}])"
    
    def __repr__(self):
        return f"Transformer_type2(nodes=[{self.I_nd_ID},{self.J_nd_ID}],volt=[{self.I_Volt},{self.J_Volt}])" 			
    
class TransformerType3:
    def __init__(self,row):
        self.ID = row['ID']
        self.name = row['name']
        #self.I_node = row['I_node']
        #self.J_node = row['J_node']
        #self.K_node = row['K_node']
        self.I_nd_ID = row['I_nd']
        self.J_nd_ID = row['J_nd']
        self.K_nd_ID = row['K_nd']
        self.I_off = row['I_off']
        self.J_off = row['J_off']
        self.K_off = row['K_off']
        if (self.K_off=='0') and (self.I_off=='0') == (self.J_off=='0'):
            self.closed = 1
        else:
            self.closed = 0
        #self.I_leakagelmpedence = row['I_leakagelmpedence']
        self.I_loadLoss = row['I_loadLoss']
        self.I_S = row['I_S']
        self.I_r = row['I_r']
        self.I_rPU = row['I_rPU']
        self.I_Volt = row['I_vol']
        self.I_x = row['I_x']
        self.I_xPU = row['I_xPU']
        #self.J_leakagelmpedence = row['J_leakagelmpedence']
        self.J_loadLoss = row['J_loadLoss']
        self.J_S = row['J_S']
        self.J_r = row['J_r']
        self.J_rPU = row['J_rPU']
        #self.J_Volt = row['J_Volt']
        self.J_Volt = row['J_vol']
        self.J_x = row['J_x']
        self.J_xPU = row['J_xPU']
        #self.K_leakagelmpedence = row['K_leakagelmpedence']
        self.K_loadLoss = row['K_loadLoss']
        self.K_S = row['K_S']
        self.K_r = row['K_r']
        self.K_rPU = row['K_rPU']
        self.K_Volt = row['K_vol']
        self.K_x = row['K_x']
        self.K_xPU = row['K_xPU']
        self.I_nd = None
        self.J_nd = None
        self.K_nd = None
        self.device_type = 'transformer-type3'
        
        self.gateway_transformer = False
    
    def __str__(self):
        return f"Transformer_type3(nodes=[{self.I_nd_ID},{self.J_nd_ID},{self.K_nd_ID}],\
        volt=[{self.I_Volt},{self.J_Volt},{self.K_Volt}])"
    
    def __repr__(self):
        return f"Transformer_type3(nodes=[{self.I_nd_ID},{self.J_nd_ID},{self.K_nd_ID}],\
        volt=[{self.I_Volt},{self.J_Volt},{self.K_Volt}])"
        
class Load:
    def __init__(self,row):
        self.ID = row['ID']
        self.name = row['name']
        self.volt = row['volt']
        #self.node = row['node']
        self.nd_ID = row['nd']
        self.P = float(row['P'])
        self.Q = float(row['Q'])
        #self.off = row['off']
        self.off  = '0'
        self.device_type = 'load'
        self.nd = None
        self.gateway = False # 是否是关口支路
        
    def __str__(self):
        return f"Load(nodes={self.nd_ID},P={self.P},Q={self.Q})"
    
    def __repr__(self):
        return f"Load(nodes={self.nd_ID},P={self.P},Q={self.Q})"

class ACline:
    def __init__(self,row):
        self.ID = row['ID']
        self.name = row['name']
        self.volt = float(row['volt'])
        self.r = float(row['r'])
        #self.rPU = row['rPU']
        #self.ratedA = row['ratedA']
        #self.ratedCurrent = row['ratedCurrent']
        if self.volt == 110. :
            self.ratedCurrent = 800
        elif self.volt == 220. :
            self.ratedCurrent = 2000    
        elif self.volt == 10. :
            self.ratedCurrent = 400  
        else:
            self.ratedCurrent = 9999
            
        self.x = float(row['x'])
        self.b = float(row['b'])
        #self.xPU = row['xPU']
        #self.bch = row['bch']
        #self.I_node = row['I_node']
        #self.J_node = row['J_node']
        self.I_nd_ID = row['I_nd']
        self.J_nd_ID = row['J_nd']
        self.I_off = row['I_off']    
        self.J_off = row['J_off']
        self.I_nd = None
        self.J_nd = None
        if (row['I_off']=='0' and row['J_off']=='0'):
            self.closed = '1'  
        else:
            self.closed = '0'
        self.device_type = 'acline'
        self.concat_switch = False # 是否是联络线
        self.gateway = False # 是否是关口支路
        self.closest_110_node = None # 所属的最近的110kV
        self.closest_110_nodes = None # 所属的最近(2个)的110kV
        
        self.max_i_ka = float(self.ratedCurrent) / 1000
        #self.max_i_ka = 9999 #缺省值（默认值）
    
    def __str__(self):
        return f"acline(nodes=[{self.I_nd.bus},{self.J_nd.bus}],volt={self.volt}),closed={self.closed}"

    def __repr__(self):
        return f"acline(nodes=[{self.I_nd.bus},{self.J_nd.bus}],volt={self.volt}),closed={self.closed}"
    
class Breaker:
    def __init__(self,row):
        self.ID = row['ID']
        self.name = row['name']
        self.volt = row['volt']
        #self.I_node = row['I_node']
        #self.J_node = row['J_node']
        self.I_nd_ID = row['I_nd']
        self.J_nd_ID = row['J_nd']
        self.closed = row['closed']
        self.device_type = 'breaker'
        
        self.I_nd = None
        self.J_nd = None
        self.gateway = False # 是否是关口支路
        self.closest_110_node = None # 所属的最近的110kV
        self.closest_110_nodes = None # 所属的最近(2个)的110kV
        self.concat_switch = False # 是否是联络线
        
        self.r = r_switch
        self.x = x_switch
        self.b = 0
        self.max_i_ka = 9999
        
    def __str__(self):
        return f"breaker(nodes=[{self.I_nd.bus},{self.J_nd.bus}],volt={self.volt}),closed={self.closed}"
        
    def __repr__(self):
        return f"breaker(nodes=[{self.I_nd.bus},{self.J_nd.bus}],volt={self.volt}),closed={self.closed}"
        
# class DisconnectorWithFeeders:
#     def __init__(self,row):
#         self.f_bus = int(row['F_Bus'])
#         self.t_bus = int(row['T_Bus'])
#         self.closed = '1'
#         try:
#             self.name = row['F_feeder'] + '-' + row['T_feeder']
#         except:
#             self.name = row['F_feeder'] + '-' + row['T_sub']

#         self.device_type = 'disconnector'
#         self.I_nd = None
#         self.J_nd = None
#         self.volt = None
#         self.I_nd_ID = None
#         self.J_nd_ID = None
#         self.gateway = False # 是否是关口支路
#         self.closest_110_node = None # 所属的最近的110kV
#         self.closest_110_nodes = None # 所属的最近(2个)的110kV
#         self.concat_switch = False # 是否是联络线
#         self.concat_feeder = False # 馈线间联络线
#         self.feeder2sub = False #馈线与母线间的连接线
        
#         self.r = r_switch
#         self.x = x_switch
#         self.b = 0
#         self.max_i_ka = 9999
#     def __str__(self):
#         return f"disconnector(nodes=[{self.I_nd.bus},{self.J_nd.bus}],volt={self.volt}),closed={self.closed}"
        
#     def __repr__(self):
#         return f"disconnector(nodes=[{self.I_nd.bus},{self.J_nd.bus}],volt={self.volt}),closed={self.closed}"
class DisconnectorWithFeeders:
    def __init__(self,row):
        #print(row)
        if 'I_nd' in row and 'J_nd' in row:
            self.I_nd_ID = str(row['I_nd'])
            self.J_nd_ID = str(row['J_nd'])
            self.f_bus = None
            self.t_bus = None
        elif 'F_Bus' in row and 'T_Bus' in row:
            self.f_bus = int(row['F_Bus'])
            self.t_bus = int(row['T_Bus'])
            self.I_nd_ID = None
            self.J_nd_ID = None
        else:
            raise ValueError("DisconnectorWithFeeders row does not have expected columns for I_nd_ID and J_nd_ID.")
        
        self.closed = '1'
        if 'I_feeder' in row and 'J_feeder' in row:
            self.name = row['I_feeder'] + '-' + row['J_feeder']
        elif 'I_sub' in row and 'J_feeder' in row:
            self.name = row['I_sub'] + '-' + row['J_feeder']
        elif 'F_feeder' in row and 'T_feeder' in row:
            self.name = row['F_feeder'] + '-' + row['T_feeder']
        elif 'F_feeder' in row and 'T_sub' in row:
            self.name = row['F_feeder'] + '-' + row['T_sub']
        else:
            raise ValueError("DisconnectorWithFeeders row does not have expected columns for name.")


        self.device_type = 'disconnector'
        self.I_nd = None
        self.J_nd = None
        self.volt = None

        self.gateway = False # 是否是关口支路
        self.closest_110_node = None # 所属的最近的110kV
        self.closest_110_nodes = None # 所属的最近(2个)的110kV
        self.concat_switch = False # 是否是联络线
        self.concat_feeder = False # 馈线间联络线
        self.feeder2sub = False #馈线与母线间的连接线
        
        self.r = r_switch
        self.x = x_switch
        self.b = 0
        self.max_i_ka = 9999
    def __str__(self):
        return f"disconnector(nodes=[{self.I_nd.bus},{self.J_nd.bus}],volt={self.volt}),closed={self.closed}"
        
    def __repr__(self):
        return f"disconnector(nodes=[{self.I_nd.bus},{self.J_nd.bus}],volt={self.volt}),closed={self.closed}"
              
        
class Disconnector:
    def __init__(self,row):
        self.ID = row['ID']
        self.name = row['name']
        self.volt = row['volt']
        #self.I_node = row['I_node']
        #self.J_node = row['J_node']
        self.I_nd_ID = row['I_nd']
        self.J_nd_ID = row['J_nd']
        self.closed = row['closed']
        self.device_type = 'disconnector'
        self.I_nd = None
        self.J_nd = None
        self.gateway = False # 是否是关口支路
        self.closest_110_node = None # 所属的最近的110kV
        self.closest_110_nodes = None # 所属的最近(2个)的110kV
        self.concat_switch = False # 是否是联络线
        
        self.r = r_switch
        self.x = x_switch
        self.b = 0
        self.max_i_ka = 9999
        
    def __str__(self):
        return f"disconnector(nodes=[{self.I_nd.bus},{self.J_nd.bus}],volt={self.volt}),closed={self.closed}"
        
    def __repr__(self):
        return f"disconnector(nodes=[{self.I_nd.bus},{self.J_nd.bus}],volt={self.volt}),closed={self.closed}"
        
class NodeCluster:
    def __init__(self):
        self.nodes = []
        self.closest_110_node = None
        self.closest_110_node_ID = None
        self.breakers = []
        self.disconnectors = []
        self.aclines = []
        self.gateways = []
        self.concat_switches = []
        self.concated_node_cluster = []
        
class NodeGroup:
    def __init__(self,group_name,node_cluster_list):
        self.name = group_name
        self.nodes = []
        self.closest_110_nodes = []
        self.closest_110_node_IDs = []
        self.all_lines = []
        self.breakers = []
        self.disconnectors = []
        self.aclines = []
        self.gateways = []
        self.concat_switches = []
        self.node_cluster_list = node_cluster_list
        self.feasible_switch_states = []
        
        for node_cluster in self.node_cluster_list:
            for node in node_cluster.nodes:
                if node:
                    if node not in self.nodes:
                        self.nodes.append(node)

            for breaker in node_cluster.breakers:
                if breaker:
                    if breaker not in self.breakers:
                        self.breakers.append(breaker)
                        self.all_lines.append(breaker)
            for disconnector in node_cluster.disconnectors:
                if disconnector:
                    if disconnector not in self.disconnectors:
                        self.disconnectors.append(disconnector)
                        self.all_lines.append(disconnector)
            for acline in node_cluster.aclines:
                if acline:
                    if acline not in self.aclines:
                        self.aclines.append(acline)
                        self.all_lines.append(acline)
            for gateway in node_cluster.gateways:
                if gateway:
                    if gateway not in self.gateways:
                        self.gateways.append(gateway)
            for concat_switch in node_cluster.concat_switches:
                if concat_switch:
                    if concat_switch not in self.concat_switches:
                        self.concat_switches.append(concat_switch)
            
            
            self.closest_110_nodes.append(node_cluster.closest_110_node)
            self.closest_110_node_IDs.append(node_cluster.closest_110_node_ID)
            
    def __str__(self):
        return f"node-group(110_nodes={self.name})"
        
    def __repr__(self):
        return f"node-group(110_nodes={self.name})"
            
    def find_feasible_switch_states(self):
        """
        # 将gateway和concat_switch中的开关称为开关组，对开关组中的状态组合
        # 进行扫描，以找到所有可行的开关组状态。可行的开关组状态需满足以下几个
        # 条件：1）除了gateway和concat节点以外，group内所有节点都需有供电
        # （group内的可供电电源节点为group的closest_110_nodes）
        # 2） group内不存在环网（注意，认为group的closest_110_nodes之间已经
        # 互相连接了）
        # 将所有可行的开关组状态加入group.feasible_switch_states中，
        # 列表group.feasible_switch_states中的每一个元素state应该是一个字典，
        # state['1']表示闭合的支路（或开关），state['0']表示断开的支路
        """
        # 1) 收集本组中所有“可切换支路”(即 gateway 和 concat_switches) 并去重
        switch_lines = list(set(self.gateways + self.concat_switches))
        n_switch = len(switch_lines)
        if n_switch == 0:
            # 如果本组没有关口或联络开关，则认为其没有可变的开关组合
            self.feasible_switch_states = []
            return self.feasible_switch_states
        # 2) 为了后面快速构造子图，先把“保持闭合”的线路放入一个初始 adjacency
        #    （仅限于 group.all_lines 中，那些不在 switch_lines 内且本身就 closed=1 的线路）
        #    我们只关心 group 内部节点的连通性，所以 adjacency 仅针对本 group 的节点
        adjacency_always_on = {nd.nd: set() for nd in self.nodes}
        for line in self.all_lines :
            # 不在可切换集合里，且自身处于闭合态
            if (line not in switch_lines) and (str(line.closed) == '1'):
                i_id = line.I_nd_ID
                j_id = line.J_nd_ID
                adjacency_always_on[i_id].add(j_id)
                adjacency_always_on[j_id].add(i_id)
        # 3) 收集本组内所有“110kV”节点 ID (可能不止一个，如 110kV-1 和 110kV-2)
        #    注意：“假定组内所有 110kV 节点间是等效连接的”，
        #    在 BFS 时相当于把与110kV节点相连的gateway_node视为同一个“超级电源节点”。
        #    hv_node节点之间，视为两两相连
        hv_node_ids = set()
        for nd in self.nodes:
            if nd.gateway_node_10kV:
                hv_node_ids.add(nd.nd)
        hv_node_ids = list(hv_node_ids)
        for i in range(len(hv_node_ids)-1):
            adjacency_always_on[hv_node_ids[i+1]].add(hv_node_ids[i])
            adjacency_always_on[hv_node_ids[i]].add(hv_node_ids[i+1])
        # 4) 收集本组内“必须保证有电”的普通 10kV 节点的 ID
        #    只有非 gateway 且非 concat 的节点需要被供电
        must_power_ids = set()
        for nd in self.nodes:
            # 如果电压是 10kV，且它既不是 gateway 节点也不是 concat 节点
            try:
                if float(nd.volt) == 10.0 and (not nd.gateway_node_10kV) and (not nd.concat_node_10kV):
                    must_power_ids.add(nd.nd)
            except:
                pass

        feasible_combinations = []
        # 5) 穷举所有 2^n_switch 种组合
        #    对于第 i 个开关，0 表示断开，1 表示闭合
        for comb in product([0, 1], repeat=n_switch):
            # (a) 先复制一份 adjacency_always_on 作为基础
            #     adjacency_on 表示本次组合下处于闭合的所有线路形成的邻接表
            adjacency_on = {nid: set(adj) for nid, adj in adjacency_always_on.items()}

            # 用于记录本组合下每个可切换支路的开断状态
            open_list = []
            close_list = []

            # (b) 根据组合 comb，决定 switch_lines 里的支路是断开还是闭合
            for i, sw_line in enumerate(switch_lines):
                if comb[i] == 1:
                    # 该支路闭合
                    adjacency_on[sw_line.I_nd_ID].add(sw_line.J_nd_ID)
                    adjacency_on[sw_line.J_nd_ID].add(sw_line.I_nd_ID)
                    close_list.append(sw_line)
                else:
                    # 该支路断开
                    open_list.append(sw_line)

            # (c) 判断是否有环路 + 是否能覆盖 must_power_ids
            #     这里用一个多源 BFS (源是所有 hv_node_ids)，同时做无环网检测
            visited = set()
            parent = dict()  # parent[u] = v 表示在 BFS 树里，u 的父节点是 v

            has_cycle = False
            queue = collections.deque()

            # 如果组内没有 hv_nodes 节点，按题意可以理解为：该组合必然无法为 10kV 节点供电
            # 也可以根据实际情况进行约定，这里选择直接判定“不可行”。
            if len(hv_node_ids) == 0:
                # 没有电源节点 => must_power_ids 都没法供电
                continue

            # 多源 BFS：把所有 hv_id 节点当做“初始队列”
            for hv_id in hv_node_ids:
                visited.add(hv_id)
                parent[hv_id] = None
                queue.append(hv_id)

            # BFS 遍历，用于连通性和环检测（经典 undirected graph 检测环的手段）
            while queue and (not has_cycle):
                curr = queue.popleft()
                for nb in adjacency_on[curr]:
                    #print(nb)
                    if nb not in visited:
                        visited.add(nb)
                        parent[nb] = curr
                        queue.append(nb)
                    else:
                        # 在无向图里，若 nb 已访问且 nb != parent[curr]，则说明存在环
                        if nb != parent.get(curr, None) and (nb not in hv_node_ids):
                            # 发现环路
                            has_cycle = True
                            break

            if has_cycle:
                # 出现环路 => 本组合不可行
                continue

            # (d) 判断是否所有 must_power_ids 都已在 visited 里
            #     如果必须供电的节点还有没访问到，说明它没电源 => 不可行
            if must_power_ids.issubset(visited):
                # 该组合满足无环网 & 覆盖所有必供节点 => 可行
                state_dict = {
                    '1': close_list,  # 本组合下闭合的可切换支路 ID 列表
                    '0': open_list    # 本组合下断开的可切换支路 ID 列表
                }
                feasible_combinations.append(state_dict)

        # 6) 将该组内所有可行组合保存
        self.feasible_switch_states = feasible_combinations   
        #print(group.feasible_switch_states)
        #return  group.feasible_switch_states
        
class PandaPowerFlowCalculator:
    def __init__(self, parsed_cim: CimEParser,slack_nd='769002015'):
        global transformer
        self.slack_nd = slack_nd
        self.all_lines = []
        self.nodeID2node = {} # 节点ID映射到实例化的节点类
        self.nodename2node = {} # 节点名称映射到实例化的节点类
        self.bus2node = {} # 母线编号映射到实例化的节点类
        self.node_clusters_10kV = {} # 节点聚类（归属于同一110kV节点的节点）
        self.node_groups_10kV = {} # 可以通过10kV联络线接通的节点组
        self.feeder_groups = [] # 馈线组
        self.feeder_clusters = [] # 多个相连的馈线组构成馈线簇
        self.fgname2fg = {} # 根据馈线组的名字找到馈线组
        self.fcname2fc = {} # 根据馈线组的名字找到馈线组
        self.start_feeder_nodes = [] # 馈线开始节点列表
        self.mainbus_10kV = [] # 十千伏母线节点列表
        self.node_buses2line = {} # 根据提供两个节点的bus编号找到对应的线路（广义）
        
        self.is_scan_feasible_switch_states = False
        self.is_scan_feasible_feeders_switch_states = False
        
        
        # 加载节点
        self.load_nodes()
        # 加载ACLines
        self.load_aclines()
        # # 加载transformers
        self.load_transformers()
        # # 加载loads
        #self.load_loads()
        # # 加载breakers
        self.load_breakers()
        # # 加载disconnectors
        self.load_disconnectors()
        # # 找到所有电气岛
        self.islands = self.find_islands()
        # # 找到主岛
        self.main_island = self.find_main_island()
        # # 扫描邻居节点
        self.scan_neighbor_nodes()
        # # 10kV母线分区
        self.scan_owned_nodes()
        # # 扫描馈线组相关
        #self.scan_feeders()
        # 从指定馈线文件中导入馈线对应的节点、关口支路、连接线路等
        self.load_feeders()
        # # 扫描关口开关
        #self.scan_gateway_switches()
        # # 扫描节点簇(cluster)和节点组(group)
        #self.scan_clusters_groups()
        
                
        # 创建潮流文件
        #self.create_pf()
        #self.caculate_flow()
    
    def load_feeders(self):
        nodes_in_feeder_groups = []
        for feederfile in FeederFiles:    
            feederbranches = pd.read_csv(feederfile,sep='\t')
            node_in_feeder_group = set()
            for index, row in feederbranches.iterrows():
                f_node = self.bus2node[row['F-Node']]
                t_node = self.bus2node[row['T-Node']]
                
                if not f_node.is_scaned_fg:
                    node_in_feeder_group.add(f_node)
                    f_node.is_scaned_fg = True
                if not t_node.is_scaned_fg:
                    node_in_feeder_group.add(t_node)
                    t_node.is_scaned_fg = True
                
                if not t_node:
                    node_in_feeder_group.add(t_node)
                if f_node.start_feeder_10kV:
                    feeder_start_node = f_node
                elif t_node.start_feeder_10kV:
                    feeder_start_node = t_node
                    
            for node in node_in_feeder_group:
                node.closest_start_feeder_node = feeder_start_node
            
            for nbr in feeder_start_node.act_neighbor_nodes:
                if nbr.mainbus_10kV==True:
                    node_in_feeder_group.add(nbr)
                    
            nodes_in_feeder_groups.append(node_in_feeder_group)
            
            
        
        for i in range(len(nodes_in_feeder_groups)):
            node_in_feeder_group = nodes_in_feeder_groups[i]
            gateways = []
            lines_in_feeder_group = set()
            concat_lines_with_other_feeder_group = set()
            for node in node_in_feeder_group :
                if node.mainbus_10kV:
                    continue
                elif node.start_feeder_10kV:
                    feeder_start_node = node
                for nbr in node.act_neighbor_nodes:
                    if nbr.volt >=110.:
                        continue
                    elif nbr.start_feeder_10kV:
                        continue
                    if nbr.mainbus_10kV==True:
                        #node_in_feeder_group.add(nbr)
                        gateway = self.node_buses2line[self._nodebus2str([node.bus,nbr.bus])]
                        gateways.append(gateway)
                        lines_in_feeder_group.add(gateway)
                        main_bus_10kV = nbr
                    elif nbr in node_in_feeder_group:
                        line = self.node_buses2line[self._nodebus2str([node.bus,nbr.bus])]
                        lines_in_feeder_group.add(line)
                    elif nbr not in node_in_feeder_group and (nbr.closest_start_feeder_node!= node.closest_start_feeder_node) :
                        #print(nbr.bus,node.bus)
                        nbr.concat_node_feeder = True
                        node.concat_node_feeder = True
                        concat_line = self.node_buses2line[self._nodebus2str([node.bus,nbr.bus])]
                        lines_in_feeder_group.add(concat_line)
                        concat_lines_with_other_feeder_group.add(concat_line)
                        
                        
            if len(list(node_in_feeder_group)) >= 2:
                for nbr in feeder_start_node.act_neighbor_nodes:
                    #print(nbr,nbr.mainbus_10kV)
                    if nbr.mainbus_10kV==True:
                        brk = self.node_buses2line[self._nodebus2str([nbr.bus,feeder_start_node.bus])]
                        feed_group_name = brk.name
                feeder_group_obj = FeederGroup(node_in_feeder_group,gateways,lines_in_feeder_group,feeder_start_node,feed_group_name,concat_lines_with_other_feeder_group,main_bus_10kV)
                self.feeder_groups.append(feeder_group_obj)
                self.fgname2fg[feeder_group_obj.name] = feeder_group_obj
                
        for feeder_group in self.feeder_groups:
            for concat_line in feeder_group.concat_lines_with_other_feeder_group:
                if concat_line.I_nd in feeder_group.nodes and concat_line.J_nd.belong_feeder_group:   
                    feeder_group.concat_feeder_group.append(concat_line.J_nd.belong_feeder_group)
                elif concat_line.J_nd in feeder_group.nodes and concat_line.I_nd.belong_feeder_group:   
                    feeder_group.concat_feeder_group.append(concat_line.I_nd.belong_feeder_group)
                    
        print('正在扫描馈线簇')
        visited_start_feeder_node_buses = set()  # set存储已访问的馈线开始节点
        for start_feeder_node in tqdm(list(self.start_feeder_nodes)):
            # 如果这个 start_feeder_node 已经被归入了某个 feeder_group，就跳过
            if start_feeder_node in visited_start_feeder_node_buses:
                continue
            feeder_group = start_feeder_node.belong_feeder_group
            # BFS队列初始化
            queue = [feeder_group]
            group_clusters = []  # 本次 BFS 找到的所有节点簇
            while queue:
                current_fg = queue.pop(0)
                if current_fg.feeder_start_node not in visited_start_feeder_node_buses:
                    visited_start_feeder_node_buses.add(current_fg.feeder_start_node)
                    group_clusters.append(current_fg)
                    # 将相连的馈线也加入 queue
                    for neigh_fg in current_fg.concat_feeder_group:
                        neigh_fg_start_feeder_node = neigh_fg.feeder_start_node
                        if neigh_fg_start_feeder_node not in visited_start_feeder_node_buses:
                            queue.append(neigh_fg)

            feeder_cluster_name = '_'.join(sorted([fg.name for fg in group_clusters]))
            fc = FeederCluster(feeder_cluster_name,group_clusters ) 
            self.fcname2fc[fc.name] = fc
            self.feeder_clusters.append(fc)   
            
    def scan_feeders(self):
        print('正在扫描馈线组')
        visited = set()
        # 扫描馈线组
        for disconnector in tqdm(self.disc_feeder2sub):
            if disconnector.feeder_node.bus in except_feeder_bus:
                continue
            feeder_start_node = disconnector.feeder_node
            #visited = set([feeder_start_node])
            visited.add(feeder_start_node)
            queue = deque([feeder_start_node])
            node_in_feeder_group = set()
            node_in_feeder_group.add(feeder_start_node)
            gateways = []
            lines_in_feeder_group = []
            concat_lines_with_other_feeder_group = []
            while queue:
                current = queue.popleft()
                # if current.bus == 1161:
                #     print(current.bus)
                for nbr in current.act_neighbor_nodes:
                    # if nbr.bus == 1158:
                    #     print(nbr.bus)
                    # if current.bus == 1161 and nbr.bus == 1158:
                    #     print(nbr.volt >=110.,nbr.start_feeder_10kV,nbr.mainbus_10kV==True)
                    #     print(current.concat_node_feeder and nbr.concat_node_feeder)
                    #     print((nbr.closest_start_feeder_node != current.closest_start_feeder_node)\
                    #     and (nbr.closest_start_feeder_node) and (current.closest_start_feeder_node))
                    if nbr.volt >=110.:
                        continue
                    elif nbr.start_feeder_10kV:
                        continue
                    elif nbr.mainbus_10kV==True:
                        node_in_feeder_group.add(nbr)
                        gateway = self.node_buses2line[self._nodebus2str([current.bus,nbr.bus])]
                        gateways.append(gateway)
                        lines_in_feeder_group.append(gateway)
                        main_bus_10kV = nbr
                        continue
                    elif current.concat_node_feeder and nbr.concat_node_feeder:
                        concat_line = self.node_buses2line[self._nodebus2str([current.bus,nbr.bus])]
                        concat_lines_with_other_feeder_group.append(concat_line) 
                        line = self.node_buses2line[self._nodebus2str([current.bus,nbr.bus])]
                        lines_in_feeder_group.append(line)
                        continue
                    elif (nbr.closest_start_feeder_node != current.closest_start_feeder_node)\
                        and (nbr.closest_start_feeder_node) and (current.closest_start_feeder_node):
                        current.concat_node_feeder = True
                        nbr.concat_node_feeder = True
                        concat_line = self.node_buses2line[self._nodebus2str([current.bus,nbr.bus])]
                        concat_lines_with_other_feeder_group.append(concat_line)
                        lines_in_feeder_group.append(concat_line)
                        continue
                    elif (nbr not in visited) and (nbr.mainbus_10kV==False) and (nbr.start_feeder_10kV==False)\
                        and (not (current.concat_node_feeder and nbr.concat_node_feeder)):
                        visited.add(nbr)
                        queue.append(nbr)
                        node_in_feeder_group.add(nbr)
                        line = self.node_buses2line[self._nodebus2str([current.bus,nbr.bus])]
                        lines_in_feeder_group.append(line)
                        continue
                
                
            #print(len(list(node_in_feeder_group)))
            if len(list(node_in_feeder_group)) >= 2:
                feed_group_name = disconnector.name
                feeder_group_obj = FeederGroup(node_in_feeder_group,gateways,lines_in_feeder_group,feeder_start_node,feed_group_name,concat_lines_with_other_feeder_group,main_bus_10kV)
                self.feeder_groups.append(feeder_group_obj)
                self.fgname2fg[feeder_group_obj.name] = feeder_group_obj
                
        for feeder_group in self.feeder_groups:
            for concat_line in feeder_group.concat_lines_with_other_feeder_group:
                if concat_line.I_nd in feeder_group.nodes and concat_line.J_nd.belong_feeder_group:   
                    feeder_group.concat_feeder_group.append(concat_line.J_nd.belong_feeder_group)
                elif concat_line.J_nd in feeder_group.nodes and concat_line.I_nd.belong_feeder_group:   
                    feeder_group.concat_feeder_group.append(concat_line.I_nd.belong_feeder_group)
        print('正在扫描馈线簇')
        visited_start_feeder_node_buses = set()  # set存储已访问的馈线开始节点
        for start_feeder_node in tqdm(list(self.start_feeder_nodes)):
            # 如果这个 start_feeder_node 已经被归入了某个 feeder_group，就跳过
            if start_feeder_node in visited_start_feeder_node_buses:
                continue
            feeder_group = start_feeder_node.belong_feeder_group
            # BFS队列初始化
            queue = [feeder_group]
            group_clusters = []  # 本次 BFS 找到的所有节点簇
            while queue:
                current_fg = queue.pop(0)
                if current_fg.feeder_start_node not in visited_start_feeder_node_buses:
                    visited_start_feeder_node_buses.add(current_fg.feeder_start_node)
                    group_clusters.append(current_fg)
                    # 将相连的馈线也加入 queue
                    for neigh_fg in current_fg.concat_feeder_group:
                        neigh_fg_start_feeder_node = neigh_fg.feeder_start_node
                        if neigh_fg_start_feeder_node not in visited_start_feeder_node_buses:
                            queue.append(neigh_fg)

            feeder_cluster_name = '_'.join(sorted([fg.name for fg in group_clusters]))
            fc = FeederCluster(feeder_cluster_name,group_clusters ) 
            self.fcname2fc[fc.name] = fc
            self.feeder_clusters.append(fc)
    
    def _nodebus2str(self,node_buses=[]):
        return '_'.join([str(node_bus) for node_bus in node_buses])
    
    def change_opr_mod(self,temp_net,random_load=True):
        new_net = copy.deepcopy(temp_net)
        #if not self.is_scan_feasible_feeders_switch_states:
        #    self.scan_feasible_feeders_switch_states()
            
        closed_switches = []
        open_switches = []
        for fc in self.feeder_clusters : 
            if len(fc.feasible_switch_states) > 0:
                random_condition = random.choice(fc.feasible_switch_states)
                closed_switches += random_condition['1']
                open_switches += random_condition['0']
        
        for line in open_switches:
            line_to_delete = new_net.line[(new_net.line['from_bus'] == line.I_nd.bus) & (new_net.line['to_bus'] == line.J_nd.bus)].index
            if not line_to_delete.empty:
                pp.drop_lines(new_net, line_to_delete)
        
        for line in closed_switches:
            line_exists = not new_net.line[(new_net.line['from_bus'] == line.I_nd.bus) & (new_net.line['to_bus'] == line.J_nd.bus)].empty
            if line.I_nd.bus not in new_net.bus.index:
                node = line.I_nd
                pp.create_bus(new_net, vn_kv=float(node.volt), index=int(node.bus),name=node.closest_110_node.nd + '/' + node.nd)
            if line.J_nd.bus not in new_net.bus.index:
                node = line.J_nd
                pp.create_bus(new_net, vn_kv=float(node.volt), index=int(node.bus),name=node.closest_110_node.nd + '/' + node.nd)
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
        
        if random_load:
            new_net.load["p_mw"] *= np.random.uniform(0.5,1.2,size=new_net.load["p_mw"].shape)
            new_net.load["q_mvar"] *= np.random.uniform(-1,1,size=new_net.load["q_mvar"].shape)
             
        return new_net
    
    def sample_a_pf_case(self,random_load=False):
        #if not self.is_scan_feasible_feeders_switch_states:
        #    self.scan_feasible_feeders_switch_states()
            
        closed_switches = []
        open_switches = []
        # for fc in self.feeder_clusters : 
        #     if len(fc.feasible_switch_states) > 0:
        #         random_condition = random.choice(fc.feasible_switch_states)
        #         closed_switches += random_condition['1']
        #         open_switches += random_condition['0']
            
        '''
        步骤一，寻找主岛
        '''
            
        # 初始化邻接表
        adjacency = defaultdict(set)  # 使用节点对象作为键
        
        # 处理线路连接
        for line in self.all_lines:
            if (line.closed == '1' and line not in open_switches) or (line in closed_switches):
                if line.I_nd and line.J_nd:  # 确保节点存在
                    adjacency[line.I_nd].add(line.J_nd)
                    adjacency[line.J_nd].add(line.I_nd)

        # 处理两绕组变压器 (需要两端都闭合)
        for transformer in self.transformer_type2:
            if transformer.I_off == '0' and transformer.J_off == '0':
                if transformer.I_nd and transformer.J_nd:
                    adjacency[transformer.I_nd].add(transformer.J_nd)
                    adjacency[transformer.J_nd].add(transformer.I_nd)

        # 处理三绕组变压器 (需要所有端闭合)
        for transformer in self.transformer_type3:
            if (transformer.I_off == '0' and 
                transformer.J_off == '0' and 
                transformer.K_off == '0'):
                nodes = [n for n in [transformer.I_nd, transformer.J_nd, transformer.K_nd] if n]
                # 全连接拓扑（三个节点两两互联）
                for i in range(len(nodes)):
                    for j in range(i+1, len(nodes)):
                        adjacency[nodes[i]].add(nodes[j])
                        adjacency[nodes[j]].add(nodes[i])

        # 执行DFS搜索连通分量
        visited = set()
        islands = []
        all_nodes = set(self.ele_nodes)  # 获取所有节点对象
        
        for node in all_nodes:
            if node not in visited:
                stack = [node]
                visited.add(node)
                current_island = []
                
                while stack:
                    current_node = stack.pop()
                    current_island.append(current_node)
                    
                    # 遍历邻接节点
                    for neighbor in adjacency[current_node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            stack.append(neighbor)
                
                islands.append(set(current_island))  # 使用集合去重
        
        island_node_num = []
        for island in islands:
            island_node_num.append(len(island))
        island_node_num = np.array(island_node_num)
        main_island = islands[island_node_num.argmax()]
        
        '''
        步骤二，创建潮流文件
        '''
        # 创建空net
        temp_net = pp.create_empty_network()
        # 创建母线、支路
        
        #print('创建母线&负荷&发电机...')
        for node in tqdm(self.ele_nodes):
            if node in main_island:
                if node.closest_110_node:
                    pp.create_bus(temp_net, vn_kv=float(node.volt), index=int(node.bus),name=node.closest_110_node.nd + '/' + node.nd)
                else:
                    pp.create_bus(temp_net, vn_kv=float(node.volt), index=int(node.bus),name=node.nd)
                if float(node.pl) > 0 or float(node.ql) > 0:
                    if random_load:
                        pp.create_load(temp_net,
                                    bus=node.bus,
                                    p_mw=node.pl * random.uniform(0.5,1),
                                    q_mvar=node.ql * random.uniform(-1,1),
                                    name=node.name)
                    else:
                        pp.create_load(temp_net,
                                        bus=node.bus,
                                        p_mw=node.pl,
                                        q_mvar=node.ql,
                                        name=node.name)
                if float(node.pg) > 0:
                    pp.create_gen(temp_net, 
                                  bus=node.bus, 
                                  p_mw=node.pg, 
                                  vn_kv=float(node.volt), 
                                  vm_pu=1.02, 
                                  sn_mva=node.pg, 
                                  name=node.nd+r'/Gen',
                                  rdss_ohm=0.1,
                                  xdss_pu=0.1)
                                    
        #print('创建平衡节点...')
        pp.create_ext_grid(temp_net, bus=self.nodeID2node[self.slack_nd].bus, vm_pu=1, va_degree=0)
        
        for transformer in tqdm(self.transformer_type2):
            if transformer.I_off == '0' and transformer.J_off == '0':
                if transformer.I_nd in main_island and transformer.J_nd in main_island:                   
                    pp.create_transformer_from_parameters(temp_net,
                                                            hv_bus=transformer.I_nd.bus,
                                                            lv_bus=transformer.J_nd.bus,
                                                            sn_mva=float(transformer.I_S),
                                                            vn_hv_kv=float(transformer.I_Volt),
                                                            vn_lv_kv=float(transformer.J_Volt),
                                                            vkr_percent=float(transformer.I_rPU),
                                                            vk_percent=random.uniform(5,7),
                                                            pfe_kw=(float(transformer.I_loadLoss) + float(transformer.J_loadLoss)) / 1000,
                                                            i0_percent=0,
                                                            name="transformer2/" + transformer.name,
                                                            )
        
        #print('创建transformer-type3...')
        for transformer in tqdm(self.transformer_type3):
            if transformer.I_off == '0' and transformer.J_off == '0' and transformer.K_off == '0':
                if transformer.I_nd in main_island and transformer.J_nd in main_island and transformer.K_nd in main_island:
                    #print('创建transformer%s-%s'%(transformer.I_nd.bus,transformer.J_nd.bus))
                    pp.create_transformer3w_from_parameters(temp_net,
                                                            hv_bus=transformer.I_nd.bus,
                                                            mv_bus=transformer.J_nd.bus,
                                                            lv_bus=transformer.K_nd.bus,
                                                            vn_hv_kv=float(transformer.I_Volt),
                                                            vn_mv_kv=float(transformer.J_Volt),
                                                            vn_lv_kv=float(transformer.K_Volt),
                                                            sn_hv_mva=float(transformer.I_S),
                                                            sn_mv_mva=float(transformer.K_S),
                                                            sn_lv_mva=float(transformer.J_S),
                                                            vk_hv_percent = 5.5,
                                                            vk_mv_percent = 6,
                                                            vk_lv_percent = 6.5,
                                                            #vk_hv_percent=random.uniform(5,7),
                                                            #vk_mv_percent=random.uniform(5,7),
                                                            #vk_lv_percent=random.uniform(5,7),
                                                            vkr_hv_percent=float(transformer.I_rPU),
                                                            vkr_mv_percent=float(transformer.K_rPU),
                                                            vkr_lv_percent=float(transformer.J_rPU),
                                                            pfe_kw=(float(transformer.I_loadLoss) + float(transformer.J_loadLoss) + float(transformer.K_loadLoss)) / 1000,
                                                            i0_percent=0,
                                                            name="transformer3/" + transformer.name) 
        #print('创建支路...')
        
        for line in tqdm(self.all_lines):
            if (line.closed == '1' and line not in open_switches) or (line in closed_switches):
                if line.I_nd in main_island and line.J_nd in main_island:
                    #print('创建acline%s-%s'%(acline.I_nd.bus,acline.J_nd.bus))
                    pp.create_line_from_parameters(temp_net,
                                                   from_bus=line.I_nd.bus,
                                                   to_bus=line.J_nd.bus,
                                                   length_km=1,
                                                   r_ohm_per_km=float(line.r),
                                                   x_ohm_per_km=float(line.x),
                                                   c_nf_per_km=float(line.b) ,
                                                   max_i_ka=float(line.max_i_ka) ,
                                                   name=line.device_type + line.name)
    
        return temp_net
    
    def scan_feasible_feeders_switch_states(self):
        print('扫描可行馈线组开关状态......')
        for fc in tqdm(self.feeder_clusters):
            #print(f'扫描{fc}的可行开关组状态')
            fc.scan_feasible_switch_states()
            
        self.is_scan_feasible_feeders_switch_states = True
        
    def scan_feasible_switch_states(self):
        print('扫描可行开关组状态......')
        for group_name,group in tqdm(self.node_groups_10kV.items()):
            #print(f'扫描{group}的可行开关组状态')
            group.find_feasible_switch_states()
            #print(group.feasible_switch_states)
        self.is_scan_feasible_switch_states = True
                        
    def scan_clusters_groups(self):
        print('扫描节点簇与节点组')
        for node in self.ele_nodes:
            if node.closest_110_node:
                if node.closest_110_node.nd not in self.node_clusters_10kV.keys():
                    self.node_clusters_10kV[node.closest_110_node.nd] = NodeCluster()
                    self.node_clusters_10kV[node.closest_110_node.nd].nodes.append(node)
                    self.node_clusters_10kV[node.closest_110_node.nd].closest_110_node = node.closest_110_node
                    self.node_clusters_10kV[node.closest_110_node.nd].closest_110_node_ID = node.closest_110_node.nd
                else:
                    self.node_clusters_10kV[node.closest_110_node.nd].nodes.append(node)
        for breaker in self.breakers: 
            if breaker.I_nd and breaker.J_nd:
                if (breaker.I_nd.closest_110_node and breaker.J_nd.closest_110_node)\
                and (breaker.closed == '1'):
                    if breaker.I_nd.closest_110_node == breaker.J_nd.closest_110_node:
                        self.node_clusters_10kV[breaker.I_nd.closest_110_node.nd].breakers.append(breaker)
                        if breaker.gateway:
                            self.node_clusters_10kV[breaker.I_nd.closest_110_node.nd].gateways.append(breaker)
                    else:
                        self.node_clusters_10kV[breaker.I_nd.closest_110_node.nd].breakers.append(breaker)
                        self.node_clusters_10kV[breaker.J_nd.closest_110_node.nd].breakers.append(breaker)
                        self.node_clusters_10kV[breaker.I_nd.closest_110_node.nd].concat_switches.append(breaker)
                        self.node_clusters_10kV[breaker.J_nd.closest_110_node.nd].concat_switches.append(breaker)
                        self.node_clusters_10kV[breaker.J_nd.closest_110_node.nd].concated_node_cluster.append(self.node_clusters_10kV[breaker.I_nd.closest_110_node.nd])
                        self.node_clusters_10kV[breaker.I_nd.closest_110_node.nd].concated_node_cluster.append(self.node_clusters_10kV[breaker.J_nd.closest_110_node.nd])

        for disconnector in self.disconnectors: 
            if disconnector.I_nd and disconnector.J_nd:
                if (disconnector.I_nd.closest_110_node and disconnector.J_nd.closest_110_node)\
                and (disconnector.closed == '1'):
                    if disconnector.I_nd.closest_110_node == disconnector.J_nd.closest_110_node:
                        self.node_clusters_10kV[disconnector.I_nd.closest_110_node.nd].disconnectors.append(disconnector)
                        if disconnector.gateway:
                            self.node_clusters_10kV[disconnector.I_nd.closest_110_node.nd].gateways.append(disconnector)
                    else:
                        self.node_clusters_10kV[disconnector.I_nd.closest_110_node.nd].breakers.append(disconnector)
                        self.node_clusters_10kV[disconnector.J_nd.closest_110_node.nd].breakers.append(disconnector)
                        self.node_clusters_10kV[disconnector.I_nd.closest_110_node.nd].concat_switches.append(disconnector)
                        self.node_clusters_10kV[disconnector.J_nd.closest_110_node.nd].concat_switches.append(disconnector)
                        self.node_clusters_10kV[disconnector.J_nd.closest_110_node.nd].concated_node_cluster.append(self.node_clusters_10kV[disconnector.I_nd.closest_110_node.nd])
                        self.node_clusters_10kV[disconnector.I_nd.closest_110_node.nd].concated_node_cluster.append(self.node_clusters_10kV[disconnector.J_nd.closest_110_node.nd])
        
        for acline in self.ac_lines: 
            if acline.I_nd and acline.J_nd:
                if acline.I_nd.closest_110_node and acline.J_nd.closest_110_node:
                    if acline.I_nd.closest_110_node == acline.J_nd.closest_110_node:
                        self.node_clusters_10kV[acline.I_nd.closest_110_node.nd].aclines.append(acline)
                        if acline.gateway:
                            self.node_clusters_10kV[acline.I_nd.closest_110_node.nd].gateways.append(acline)
                    else:
                        self.node_clusters_10kV[acline.I_nd.closest_110_node.nd].concat_switches.append(acline)
                        self.node_clusters_10kV[acline.J_nd.closest_110_node.nd].concat_switches.append(acline)
                        self.node_clusters_10kV[acline.J_nd.closest_110_node.nd].concated_node_cluster.append(self.node_clusters_10kV[acline.I_nd.closest_110_node.nd])
                        self.node_clusters_10kV[acline.I_nd.closest_110_node.nd].concated_node_cluster.append(self.node_clusters_10kV[acline.J_nd.closest_110_node.nd])
    
        visited_110kV_node_IDs = set()  # 改用set来存储已访问的110kV节点ID
        for closest_110_node_id in list(self.node_clusters_10kV.keys()):
            # 如果这个 110kV 节点ID 已经被归入了某个 group，就跳过
            if closest_110_node_id in visited_110kV_node_IDs:
                continue

            # BFS队列初始化
            queue = [closest_110_node_id]
            group_clusters = []  # 本次 BFS 找到的所有节点簇
            while queue:
                current_id = queue.pop(0)
                if current_id not in visited_110kV_node_IDs:
                    visited_110kV_node_IDs.add(current_id)
                    # 取出这个节点簇对象
                    current_cluster = self.node_clusters_10kV[current_id]
                    group_clusters.append(current_cluster)
                    # 将相连的“节点簇”也加入 queue
                    for neigh_cluster in current_cluster.concated_node_cluster:
                        neigh_id = neigh_cluster.closest_110_node.nd
                        if neigh_id not in visited_110kV_node_IDs:
                            queue.append(neigh_id)

            # 根据这个 group_clusters 里的“所有 110kV 节点ID”创建 group_name
            # 例如把每个closest_110_node.ID 连起来
            group_name = '-'.join(sorted([cl.closest_110_node.nd for cl in group_clusters]))
            self.node_groups_10kV[group_name] = NodeGroup(group_name,group_clusters )   
            

    
    def scan_gateway_switches(self):       
        for acline in self.ac_lines:
            if acline.I_nd and acline.J_nd:
                if (float(acline.I_nd.volt) ==  float(acline.J_nd.volt) == 10.)\
                    and (acline.I_off == '0' and acline.J_off == '0'):
                    if acline.I_nd.gateway_node_10kV:
                        acline.gateway = True
                        acline.J_nd.gateway_node_10kV_otherside = True
                    if acline.J_nd.gateway_node_10kV:
                        acline.gateway = True
                        acline.I_nd.gateway_node_10kV_otherside = True
                    if acline.I_nd.closest_110_node == acline.J_nd.closest_110_node:
                        acline.closest_110_node = acline.I_nd.closest_110_node
                    else:
                        acline.concat_switch = True
                        acline.I_nd.concat_node_10kV = True
                        acline.J_nd.concat_node_10kV = True
                        acline.closest_110_nodes = [acline.I_nd.closest_110_node,acline.J_nd.closest_110_node]
                        
        for breaker in self.breakers:
            if breaker.I_nd and breaker.J_nd:
                if float(breaker.I_nd.volt) ==  float(breaker.J_nd.volt) == 10.\
                    and (breaker.closed == '1'):
                    if breaker.I_nd.gateway_node_10kV:
                        breaker.gateway = True
                        breaker.J_nd.gateway_node_10kV_otherside = True
                    if breaker.J_nd.gateway_node_10kV:
                        breaker.gateway = True
                        breaker.I_nd.gateway_node_10kV_otherside = True
                    if breaker.I_nd.closest_110_node == breaker.J_nd.closest_110_node:
                        breaker.closest_110_node = breaker.I_nd.closest_110_node
                    else:
                        breaker.concat_switch = True
                        breaker.I_nd.concat_node_10kV = True
                        breaker.J_nd.concat_node_10kV = True
                        breaker.closest_110_nodes = [breaker.I_nd.closest_110_node,breaker.J_nd.closest_110_node]
                
        for disconnector in self.disconnectors:
            if disconnector.I_nd and disconnector.J_nd:
                if float(disconnector.I_nd.volt) ==  float(disconnector.J_nd.volt) == 10.\
                    and (disconnector.closed == '1'):
                    if disconnector.I_nd.gateway_node_10kV:
                        disconnector.gateway = True
                        disconnector.J_nd.gateway_node_10kV_otherside = True
                    if disconnector.J_nd.gateway_node_10kV:
                        disconnector.gateway = True
                        disconnector.I_nd.gateway_node_10kV_otherside = True
                    if disconnector.I_nd.closest_110_node == disconnector.J_nd.closest_110_node:
                        disconnector.closest_110_node = disconnector.I_nd.closest_110_node
                    else:
                        disconnector.concat_switch = True
                        disconnector.I_nd.concat_node_10kV = True
                        disconnector.J_nd.concat_node_10kV = True
                        disconnector.closest_110_nodes = [disconnector.I_nd.closest_110_node,disconnector.J_nd.closest_110_node]
    
    def caculate_flow(self, save_path: str = None, diagnostic: bool = True):
        if diagnostic:
            pp.diagnostic(self.net, report_style='detailed', warnings_only=True, return_result_dict=False)
        pp.runpp(self.net, max_iteration=100, tolerance_mva=1e-5)
        pp.to_excel(self.net, save_path)
        return self.net
        
        
    def create_pf(self):
        # pp主网络初始化
        self.net = pp.create_empty_network()
        self.create_buses()
        self.create_ext_grid()
        self.create_lines()
        self.create_transformers()
        self.create_load()
        return self.net
        
    def create_load(self):
        print('创建负荷...')
        # for load in tqdm(self.loads):
        #     if load.off == '0':
        #         if load.nd in self.main_island:
        #             pp.create_load(self.net,
        #                            bus=load.nd.bus,
        #                            p_mw=float(2.18),
        #                            q_mvar=float(0.13),
        #                            name=load.name)
        
        
    def create_ext_grid(self):
        pp.create_ext_grid(self.net, bus=self.nodeID2node[self.slack_nd].bus, vm_pu=1, va_degree=0)
    
    def create_transformers(self):
        print('创建transformer-type2...')
        for transformer in tqdm(self.transformer_type2):
            if transformer.I_off == '0' and transformer.J_off == '0':
                    if transformer.I_nd in self.main_island and transformer.J_nd in self.main_island:
                        #print('创建transformer%s-%s'%(transformer.I_nd.bus,transformer.J_nd.bus))
                        pp.create_transformer_from_parameters(self.net,
                                                              hv_bus=transformer.I_nd.bus,
                                                              lv_bus=transformer.J_nd.bus,
                                                              sn_mva=float(transformer.I_S),
                                                              vn_hv_kv=float(transformer.I_Volt),
                                                              vn_lv_kv=float(transformer.J_Volt),
                                                              vkr_percent=float(transformer.I_rPU),
                                                              vk_percent=random.uniform(5,7),
                                                              pfe_kw=(float(transformer.I_loadLoss) + float(transformer.J_loadLoss)) / 1000,
                                                              i0_percent=0,
                                                              name="transformer2/" + transformer.name)
        
        print('创建transformer-type3...')
        for transformer in tqdm(self.transformer_type3):
            if transformer.I_off == '0' and transformer.J_off == '0' and transformer.K_off == '0':
                    if transformer.I_nd in self.main_island and transformer.J_nd in self.main_island and transformer.K_nd in self.main_island:
                        #print('创建transformer%s-%s'%(transformer.I_nd.bus,transformer.J_nd.bus))
                        pp.create_transformer3w_from_parameters(self.net,
                                                              hv_bus=transformer.I_nd.bus,
                                                              mv_bus=transformer.J_nd.bus,
                                                              lv_bus=transformer.K_nd.bus,
                                                              vn_hv_kv=float(transformer.I_Volt),
                                                              vn_mv_kv=float(transformer.J_Volt),
                                                              vn_lv_kv=float(transformer.K_Volt),
                                                              sn_hv_mva=float(transformer.I_S),
                                                              sn_mv_mva=float(transformer.K_S),
                                                              sn_lv_mva=float(transformer.J_S),
                                                              vk_hv_percent=random.uniform(5,7),
                                                              vk_mv_percent=random.uniform(5,7),
                                                              vk_lv_percent=random.uniform(5,7),
                                                              vkr_hv_percent=float(transformer.I_rPU),
                                                              vkr_mv_percent=float(transformer.K_rPU),
                                                              vkr_lv_percent=float(transformer.J_rPU),
                                                              pfe_kw=(float(transformer.I_loadLoss) + float(transformer.J_loadLoss) + float(transformer.K_loadLoss)) / 1000,
                                                              i0_percent=0,
                                                              name="transformer3/" + transformer.name) 
        
        
    def create_lines(self):
        print('创建acline...')
        for acline in tqdm(self.ac_lines):
            if acline.I_off == '0' and acline.J_off == '0':
                if acline.I_nd in self.main_island and acline.J_nd in self.main_island:
                    #print('创建acline%s-%s'%(acline.I_nd.bus,acline.J_nd.bus))
                    pp.create_line_from_parameters(self.net,
                                                   from_bus=acline.I_nd.bus,
                                                   to_bus=acline.J_nd.bus,
                                                   length_km=1,
                                                   r_ohm_per_km=float(acline.r),
                                                   x_ohm_per_km=float(acline.x),
                                                   c_nf_per_km=float(acline.b) ,
                                                   max_i_ka=float(acline.ratedCurrent) / 1000,
                                                   name="acline/" + acline.name)
        print('创建breaker...')
        for breaker in tqdm(self.breakers):
            if breaker.closed == '1':
                if breaker.I_nd in self.main_island and breaker.J_nd in self.main_island:
                    
                    #print('创建breaker%s-%s'%(breaker.I_nd.bus,breaker.J_nd.bus))
                    pp.create_line_from_parameters(self.net,
                                                   from_bus=breaker.I_nd.bus,
                                                   to_bus=breaker.J_nd.bus,
                                                   length_km=1,
                                                   r_ohm_per_km=float(breaker.r),
                                                   x_ohm_per_km=float(breaker.x),
                                                   c_nf_per_km=float(breaker.b) ,
                                                   max_i_ka=float(breaker.max_i_ka),
                                                   name="breaker/" + breaker.name)
        print('创建disconnector...')
        for disconnector in tqdm(self.disconnectors):
            if disconnector.closed == '1':
                if disconnector.I_nd in self.main_island and disconnector.J_nd in self.main_island:
                    
                    #print('创建disconnector%s-%s'%(disconnector.I_nd.bus,disconnector.J_nd.bus))
                    pp.create_line_from_parameters(self.net,
                                                   from_bus=disconnector.I_nd.bus,
                                                   to_bus=disconnector.J_nd.bus,
                                                   length_km=1,
                                                   r_ohm_per_km=float(disconnector.r),
                                                   x_ohm_per_km=float(disconnector.x),
                                                   c_nf_per_km=float(disconnector.b) ,
                                                   max_i_ka=float(disconnector.max_i_ka),
                                                   name="disconnector/" + disconnector.name)
    
    def create_buses(self):
        print('创建母线...')
        for node in tqdm(self.ele_nodes):
            if node in self.main_island:
                #print('创建节点%s'%int(node.bus))
                if node.closest_110_node:
                    pp.create_bus(self.net, vn_kv=float(node.volt), index=int(node.bus),name=node.closest_110_node.nd + '/' + node.nd)
                else:
                    pp.create_bus(self.net, vn_kv=float(node.volt), index=int(node.bus),name=node.nd)
    
    def load_nodes(self):
        self.ele_nodes = []
        print('加载节点...')
        for idx, row in tqdm(parsed_cim.node.iterrows()):

            ID,name,nd,volt,bus = row['ID'],row['name'],\
                row['nd'],float(row['volt']),int(row['Bus'])
            _tempnode = EleNode([ID,name,nd,volt,bus])
            self.ele_nodes.append(_tempnode)
            self.nodeID2node[nd] = _tempnode
            self.nodename2node[name] = _tempnode
            self.bus2node[bus] = _tempnode

    
    def load_aclines(self):
        # 加载ACLine
        print('加载ACLine...')
        self.ac_lines = []
        for idx, row in tqdm(parsed_cim.ACLine.iterrows()):
            acline_obj = ACline(row)
            #if acline_obj.closed=='1':
            self.ac_lines.append(acline_obj)
            self.all_lines.append(acline_obj)
        # acline和node互相链接
        for ac_line in self.ac_lines:
            if type(ac_line.I_nd_ID) == str and type(ac_line.J_nd_ID) == str and ac_line.closed=='1':
                ac_line.I_nd = self.nodeID2node[ac_line.I_nd_ID]
                ac_line.J_nd = self.nodeID2node[ac_line.J_nd_ID]
                self.nodeID2node[ac_line.I_nd_ID].belong_device.append(ac_line)
                self.nodeID2node[ac_line.J_nd_ID].belong_device.append(ac_line)
                self.node_buses2line[self._nodebus2str([ac_line.I_nd.bus,ac_line.J_nd.bus])]\
                    = ac_line
                self.node_buses2line[self._nodebus2str([ac_line.J_nd.bus,ac_line.I_nd.bus])]\
                    = ac_line
                
                
                
    def load_transformers(self):
        print('加载Transformer...')   
        self.transformer_type2 = []
        self.transformer_type3 = [] 
        for idx, row in tqdm(parsed_cim.Transformer.iterrows()):
            transformer_type = row['type']  
            #print(row)
            if transformer_type == '2':
                self.transformer_type2.append(TransformerType2(row))
            elif transformer_type == '3':
                self.transformer_type3.append(TransformerType3(row))
        for transformer in self.transformer_type2:
            if type(transformer.I_nd_ID) == str and type(transformer.J_nd_ID) == str:
                transformer.I_nd = self.nodeID2node[transformer.I_nd_ID]
                self.nodeID2node[transformer.I_nd_ID].belong_device.append(transformer)
                transformer.J_nd = self.nodeID2node[transformer.J_nd_ID]
                self.nodeID2node[transformer.J_nd_ID].belong_device.append(transformer)
                self.node_buses2line[self._nodebus2str([transformer.I_nd.bus,transformer.J_nd.bus])]\
                        = transformer
                self.node_buses2line[self._nodebus2str([transformer.J_nd.bus,transformer.I_nd.bus])]\
                    = transformer
        
        
        for transformer in self.transformer_type3:
            if type(transformer.I_nd_ID) == str:
                transformer.I_nd = self.nodeID2node[transformer.I_nd_ID]
                self.nodeID2node[transformer.I_nd_ID].belong_device.append(transformer)
            if type(transformer.J_nd_ID) == str:
                transformer.J_nd = self.nodeID2node[transformer.J_nd_ID]
                self.nodeID2node[transformer.J_nd_ID].belong_device.append(transformer)
            if type(transformer.K_nd_ID) == str:
                transformer.K_nd = self.nodeID2node[transformer.K_nd_ID]
                self.nodeID2node[transformer.K_nd_ID].belong_device.append(transformer)
                
    def load_loads(self):
        print('加载负荷数据...') 
        self.loads = [] 
        for idx, row in tqdm(parsed_cim.Load.iterrows()):
            self.loads.append(Load(row))
        for load in self.loads:
            if type(load.nd_ID) == str:
                load.nd = self.nodeID2node[load.nd_ID]
                self.nodeID2node[load.nd_ID].belong_device.append(load) 
                self.nodeID2node[load.nd_ID].pl = float(load.P) 
                self.nodeID2node[load.nd_ID].ql = float(load.Q)
                
    def load_breakers(self):  
        print('加载breaker...') 
        self.breakers = [] 
        for idx, row in tqdm(parsed_cim.Breaker.iterrows()):
            breaker_obj = Breaker(row)
            #if breaker_obj.closed == '1':
            self.breakers.append(breaker_obj)
            self.all_lines.append(breaker_obj)
        for breaker in self.breakers:
            if type(breaker.I_nd_ID) == str and type(breaker.J_nd_ID) == str and breaker.closed == '1':
                breaker.I_nd = self.nodeID2node[breaker.I_nd_ID]
                self.nodeID2node[breaker.I_nd_ID].belong_device.append(breaker) 
                breaker.J_nd = self.nodeID2node[breaker.J_nd_ID]
                self.nodeID2node[breaker.J_nd_ID].belong_device.append(breaker) 
                self.node_buses2line[self._nodebus2str([breaker.I_nd.bus,breaker.J_nd.bus])]\
                        = breaker
                self.node_buses2line[self._nodebus2str([breaker.J_nd.bus,breaker.I_nd.bus])]\
                    = breaker
                
    def load_disconnectors(self):
        print('加载disconnector...') 
        self.disconnectors = [] 
        self.disc_feeders = []
        self.disc_feeder2sub = []
        for idx, row in tqdm(parsed_cim.Disconnector.iterrows()):
            disconnector_obj = Disconnector(row)
            #if disconnector_obj.closed == '1':
            self.disconnectors.append(disconnector_obj)
            self.all_lines.append(disconnector_obj)
        for disconnector in self.disconnectors:
            if type(disconnector.I_nd_ID) == str and type(disconnector.J_nd_ID) == str and disconnector.closed=='1':
                disconnector.I_nd = self.nodeID2node[disconnector.I_nd_ID]
                self.nodeID2node[disconnector.I_nd_ID].belong_device.append(disconnector) 
                disconnector.J_nd = self.nodeID2node[disconnector.J_nd_ID]
                self.nodeID2node[disconnector.J_nd_ID].belong_device.append(disconnector) 
                self.node_buses2line[self._nodebus2str([disconnector.I_nd.bus,disconnector.J_nd.bus])]\
                        = disconnector
                self.node_buses2line[self._nodebus2str([disconnector.J_nd.bus,disconnector.I_nd.bus])]\
                    = disconnector
        print('加载disconnector with feeders...')    
        for idx, row in tqdm(parsed_cim.Disconnector包含馈线.iterrows()):
            disconnector_obj = DisconnectorWithFeeders(row)
            #disconnector_obj.device_type += '-sub2feeder'
            disconnector_obj.feeder2sub = True
            if not disconnector_obj.f_bus and not disconnector_obj.t_bus:
                I_nd,J_nd  = self.nodeID2node[disconnector_obj.I_nd_ID],\
                            self.nodeID2node[disconnector_obj.J_nd_ID]
            elif not disconnector_obj.I_nd_ID and not disconnector_obj.J_nd_ID:
                I_nd,J_nd = self.bus2node[disconnector_obj.f_bus],\
                            self.bus2node[disconnector_obj.t_bus]
            
            disconnector_obj.I_nd = I_nd
            disconnector_obj.J_nd = J_nd
            disconnector_obj.volt = I_nd.volt
            disconnector_obj.I_nd_ID = I_nd.nd
            disconnector_obj.J_nd_ID = J_nd.nd
            
            if self._nodebus2str([I_nd.bus,J_nd.bus]) in self.node_buses2line.keys():
                #print(self._nodebus2str([I_nd.bus,J_nd.bus]))
                disconnector_obj.closed = self.node_buses2line[self._nodebus2str([I_nd.bus,J_nd.bus])].closed
                #self.disconnectors.append(disconnector_obj)
                #self.all_lines.append(disconnector_obj)  
                self.disc_feeder2sub.append(disconnector_obj) 
            else:
                self.node_buses2line[self._nodebus2str([J_nd.bus,I_nd.bus])] = disconnector_obj
                self.node_buses2line[self._nodebus2str([I_nd.bus,J_nd.bus])] = disconnector_obj
                self.disc_feeder2sub.append(disconnector_obj)
                self.disconnectors.append(disconnector_obj)
                self.all_lines.append(disconnector_obj)
                
        for idx, row in tqdm(parsed_cim.Disconnector包含馈线支路.iterrows()):
            disconnector_obj = DisconnectorWithFeeders(row)
            #disconnector_obj.device_type += '-feeder2feeder'
            disconnector_obj.concat_feeder = True
            # I_nd,J_nd = self.bus2node[disconnector_obj.f_bus],\
            #             self.bus2node[disconnector_obj.t_bus]
            if not disconnector_obj.f_bus and not disconnector_obj.t_bus:
                I_nd,J_nd  = self.nodeID2node[disconnector_obj.I_nd_ID],\
                            self.nodeID2node[disconnector_obj.J_nd_ID]
            elif not disconnector_obj.I_nd_ID and not disconnector_obj.J_nd_ID:
                I_nd,J_nd = self.bus2node[disconnector_obj.f_bus],\
                            self.bus2node[disconnector_obj.t_bus]
            
            disconnector_obj.I_nd = I_nd
            disconnector_obj.J_nd = J_nd
            disconnector_obj.volt = I_nd.volt
            disconnector_obj.I_nd_ID = I_nd.nd
            disconnector_obj.J_nd_ID = J_nd.nd
            if self._nodebus2str([I_nd.bus,J_nd.bus]) in self.node_buses2line.keys():
                disconnector_obj.closed = self.node_buses2line[self._nodebus2str([I_nd.bus,J_nd.bus])].closed
                #self.disconnectors.append(disconnector_obj)
                #self.all_lines.append(disconnector_obj)  
                self.disc_feeders.append(disconnector_obj) 
            else:
                self.node_buses2line[self._nodebus2str([I_nd.bus,J_nd.bus])] = disconnector_obj
                self.node_buses2line[self._nodebus2str([J_nd.bus,I_nd.bus])] = disconnector_obj
                #print(disconnector_obj.I_nd.bus,disconnector_obj.J_nd.bus)
                self.disc_feeders.append(disconnector_obj)
                self.disconnectors.append(disconnector_obj)
                self.all_lines.append(disconnector_obj)
        for disconnector in self.disc_feeder2sub:
            # f_bus,t_bus = disconnector.f_bus,disconnector.t_bus
            #I_nd, J_nd = self.bus2node[f_bus],self.bus2node[t_bus]
            I_nd, J_nd = disconnector.I_nd, disconnector.J_nd
            I_nd_neighbor_num,J_nd_neighbor_num = len(I_nd.act_neighbor_nodes),\
                len(J_nd.act_neighbor_nodes),
            # 根据邻居节点个数多少判断谁是母线端谁是馈线端
            if I_nd_neighbor_num > J_nd_neighbor_num:
                I_nd.mainbus_10kV = True
                self.mainbus_10kV.append(I_nd)
                J_nd.start_feeder_10kV = True
                self.start_feeder_nodes.append(J_nd)
                disconnector.sub_node = I_nd
                disconnector.feeder_node = J_nd
                J_nd.closest_mainbus_node = I_nd
            elif I_nd_neighbor_num > J_nd_neighbor_num:
                J_nd.mainbus_10kV = True
                self.mainbus_10kV.append(J_nd)
                I_nd.start_feeder_10kV = True
                self.start_feeder_nodes.append(I_nd)
                disconnector.sub_node = J_nd
                disconnector.feeder_node = I_nd
                I_nd.closest_mainbus_node = J_nd
                #print('顺序相反')
            else:
                I_nd.mainbus_10kV = True
                self.mainbus_10kV.append(I_nd)
                J_nd.start_feeder_10kV = True
                self.start_feeder_nodes.append(J_nd)
                disconnector.sub_node = I_nd
                disconnector.feeder_node = J_nd
                J_nd.closest_mainbus_node = I_nd
                #print('节点%s和%s的邻接节点数相等'%(f_bus,t_bus))
        for idx, row in parsed_cim.Disconnector包含馈线支路.iterrows():        
            #f_bus,t_bus = int(row['F_Bus']),int(row['T_Bus'])
            if 'I_nd' in row and 'J_nd' in row:
                I_nd_ID,J_nd_ID = str(row['I_nd']),str(row['J_nd'])
                I_nd, J_nd = self.nodeID2node[I_nd_ID],self.nodeID2node[J_nd_ID]
            elif 'F_Bus' in row and 'T_Bus' in row:
                f_bus,t_bus = int(row['F_Bus']),int(row['T_Bus'])
                I_nd, J_nd = self.bus2node[f_bus],self.bus2node[t_bus]
            else:
                raise ValueError("Disconnector row must contain either I_nd and J_nd or F_Bus and T_Bus")
            #I_nd.concat_node_feeder = True
            #J_nd.concat_node_feeder = True
        
                
    def report_node_act_neighbors_by_nodeID(self,node_ID):
        """根据节点编号报告实际邻居节点的编号"""
        selected_node = self.nodeID2node[node_ID]
        neighbor_node_IDs = [act_neighbor_node.nd for act_neighbor_node in selected_node.act_neighbor_nodes]
        return neighbor_node_IDs

    
    def scan_owned_nodes(self):
        print('扫描10kV母线对应的最近110kV母线')
        for node in self.ele_nodes:
            if float(node.volt) == 10.:
                cloest_110_node_ID = self.find_closest_setVolt_node_id_by_node_id(node.nd, setVolt=110.,return_path=False)
                if cloest_110_node_ID:
                    node.closest_110_node = self.nodeID2node[cloest_110_node_ID]
        print('扫描10kV馈线节点对应的最近10kV母线')
        for node in self.ele_nodes:
            if float(node.volt) == 10. :
                closest_mainbus_10kV_ID = self.find_closest_mainbus_node_id_by_node_id(node.nd, setVolt=110.,return_path=False)
                if closest_mainbus_10kV_ID:
                    node.closest_mainbus_node = self.nodeID2node[closest_mainbus_10kV_ID]
        print('扫描10kV馈线节点对应的最近10kV馈线起始节点')
        for node in self.ele_nodes:
            if float(node.volt) == 10. :
                closest_start_feeder_10kV_ID = self.find_closest_start_feeder_node_id_by_node_id(node.nd, setVolt=110.,return_path=False)
                if closest_start_feeder_10kV_ID:
                    node.closest_start_feeder_node = self.nodeID2node[closest_start_feeder_10kV_ID]
 
    
    def scan_neighbor_nodes(self):
        """扫描所有的连通性支路，建立所有节点的实际邻居、潜在邻居节点表"""
        print('正在扫描节点连通支路')
        for ac_line in self.ac_lines:
            if ac_line.I_off == '0' and ac_line.J_off == '0':
                if ac_line.I_nd and ac_line.J_nd:  # 确保节点存在
                    ac_line.I_nd.act_neighbor_nodes.append(ac_line.J_nd)
                    ac_line.J_nd.act_neighbor_nodes.append(ac_line.I_nd)
                    ac_line.I_nd.possible_neighbor_nodes.append(ac_line.J_nd)
                    ac_line.J_nd.possible_neighbor_nodes.append(ac_line.I_nd)
                    
            else:
                if ac_line.I_nd and ac_line.J_nd:  # 确保节点存在
                    ac_line.I_nd.possible_neighbor_nodes.append(ac_line.J_nd)
                    ac_line.J_nd.possible_neighbor_nodes.append(ac_line.I_nd)
        
        for breaker in self.breakers:
            if breaker.closed == '1':
                if breaker.I_nd and breaker.J_nd:  # 确保节点存在
                    breaker.I_nd.act_neighbor_nodes.append(breaker.J_nd)
                    breaker.J_nd.act_neighbor_nodes.append(breaker.I_nd)
                    breaker.I_nd.possible_neighbor_nodes.append(breaker.J_nd)
                    breaker.J_nd.possible_neighbor_nodes.append(breaker.I_nd)
                    
            else:
                if breaker.I_nd and breaker.J_nd:  # 确保节点存在
                    breaker.I_nd.possible_neighbor_nodes.append(breaker.J_nd)
                    breaker.J_nd.possible_neighbor_nodes.append(breaker.I_nd)
        
        for disconnector in self.disconnectors:
            if disconnector.closed == '1':
                if disconnector.I_nd and disconnector.J_nd:  # 确保节点存在
                    disconnector.I_nd.act_neighbor_nodes.append(disconnector.J_nd)
                    disconnector.J_nd.act_neighbor_nodes.append(disconnector.I_nd)
                    disconnector.I_nd.possible_neighbor_nodes.append(disconnector.J_nd)
                    disconnector.J_nd.possible_neighbor_nodes.append(disconnector.I_nd)
            else:
                if disconnector.I_nd and disconnector.J_nd:  # 确保节点存在
                    disconnector.I_nd.possible_neighbor_nodes.append(disconnector.J_nd)
                    disconnector.J_nd.possible_neighbor_nodes.append(disconnector.I_nd)
        
        for transformer in self.transformer_type2:
            if transformer.I_off == '0' and transformer.J_off == '0':
                if transformer.I_nd and transformer.J_nd:
                    transformer.I_nd.act_neighbor_nodes.append(transformer.J_nd)
                    transformer.J_nd.act_neighbor_nodes.append(transformer.I_nd)
                    transformer.I_nd.possible_neighbor_nodes.append(transformer.J_nd)
                    transformer.J_nd.possible_neighbor_nodes.append(transformer.I_nd)
                    # 判断是否是关口变压器
                    if (float(transformer.I_nd.volt)==10.) and (float(transformer.J_nd.volt) > 10.):
                        transformer.gateway_transformer = True
                        transformer.I_nd.gateway_node_10kV = True
                    elif (float(transformer.J_nd.volt)==10.) and (float(transformer.I_nd.volt) > 10.):
                        transformer.gateway_transformer = True
                        transformer.J_nd.gateway_node_10kV = True
            else:
                if transformer.I_nd and transformer.J_nd:
                    transformer.I_nd.possible_neighbor_nodes.append(transformer.J_nd)
                    transformer.J_nd.possible_neighbor_nodes.append(transformer.I_nd)

        # 处理三绕组变压器 (需要所有端闭合)
        for transformer in self.transformer_type3:
            if (transformer.I_off == '0' and 
                transformer.J_off == '0' and 
                transformer.K_off == '0'):
                if transformer.I_nd and transformer.J_nd and transformer.K_nd:
                    transformer.I_nd.act_neighbor_nodes.append(transformer.J_nd)
                    transformer.I_nd.act_neighbor_nodes.append(transformer.K_nd)
                    transformer.J_nd.act_neighbor_nodes.append(transformer.I_nd)
                    transformer.J_nd.act_neighbor_nodes.append(transformer.K_nd)
                    transformer.K_nd.act_neighbor_nodes.append(transformer.I_nd)
                    transformer.K_nd.act_neighbor_nodes.append(transformer.J_nd)
                    transformer.I_nd.possible_neighbor_nodes.append(transformer.J_nd)
                    transformer.I_nd.possible_neighbor_nodes.append(transformer.K_nd)
                    transformer.J_nd.possible_neighbor_nodes.append(transformer.I_nd)
                    transformer.J_nd.possible_neighbor_nodes.append(transformer.K_nd)
                    transformer.K_nd.possible_neighbor_nodes.append(transformer.I_nd)
                    transformer.K_nd.possible_neighbor_nodes.append(transformer.J_nd)
                    # 判断是否是关口变压器
                    if (float(transformer.I_nd.volt)==10.):
                        transformer.gateway_transformer = True
                        transformer.I_nd.gateway_node_10kV = True
                    elif (float(transformer.J_nd.volt)==10.) :
                        transformer.gateway_transformer = True
                        transformer.J_nd.gateway_node_10kV = True
                    elif (float(transformer.K_nd.volt)==10.) :
                        transformer.gateway_transformer = True
                        transformer.K_nd.gateway_node_10kV = True
            else:
                if transformer.I_nd and transformer.J_nd and transformer.K_nd:
                    transformer.I_nd.possible_neighbor_nodes.append(transformer.J_nd)
                    transformer.I_nd.possible_neighbor_nodes.append(transformer.K_nd)
                    transformer.J_nd.possible_neighbor_nodes.append(transformer.I_nd)
                    transformer.J_nd.possible_neighbor_nodes.append(transformer.K_nd)
                    transformer.K_nd.possible_neighbor_nodes.append(transformer.I_nd)
                    transformer.K_nd.possible_neighbor_nodes.append(transformer.J_nd)
                    
    

    
    def find_islands(self):
        """识别电网中的电气孤岛，返回包含节点集合的列表"""
        # 初始化邻接表
        adjacency = defaultdict(set)  # 使用节点对象作为键
        
        # 处理AC线路连接 (需要两端都闭合)
        for ac_line in self.ac_lines:
            if ac_line.I_off == '0' and ac_line.J_off == '0':
                if ac_line.I_nd and ac_line.J_nd:  # 确保节点存在
                    adjacency[ac_line.I_nd].add(ac_line.J_nd)
                    adjacency[ac_line.J_nd].add(ac_line.I_nd)

        # 处理断路器连接 (闭合状态)
        for breaker in self.breakers:
            if breaker.closed == '1':
                if breaker.I_nd and breaker.J_nd:
                    adjacency[breaker.I_nd].add(breaker.J_nd)
                    adjacency[breaker.J_nd].add(breaker.I_nd)

        # 处理隔离开关连接 (闭合状态)
        for disconnector in self.disconnectors:
            if disconnector.closed == '1':
                if disconnector.I_nd and disconnector.J_nd:
                    adjacency[disconnector.I_nd].add(disconnector.J_nd)
                    adjacency[disconnector.J_nd].add(disconnector.I_nd)

        # 处理两绕组变压器 (需要两端都闭合)
        for transformer in self.transformer_type2:
            if transformer.I_off == '0' and transformer.J_off == '0':
                if transformer.I_nd and transformer.J_nd:
                    adjacency[transformer.I_nd].add(transformer.J_nd)
                    adjacency[transformer.J_nd].add(transformer.I_nd)

        # 处理三绕组变压器 (需要所有端闭合)
        for transformer in self.transformer_type3:
            if (transformer.I_off == '0' and 
                transformer.J_off == '0' and 
                transformer.K_off == '0'):
                nodes = [n for n in [transformer.I_nd, transformer.J_nd, transformer.K_nd] if n]
                # 全连接拓扑（三个节点两两互联）
                for i in range(len(nodes)):
                    for j in range(i+1, len(nodes)):
                        adjacency[nodes[i]].add(nodes[j])
                        adjacency[nodes[j]].add(nodes[i])

        # 执行DFS搜索连通分量
        visited = set()
        islands = []
        all_nodes = set(self.ele_nodes)  # 获取所有节点对象
        
        for node in all_nodes:
            if node not in visited:
                stack = [node]
                visited.add(node)
                current_island = []
                
                while stack:
                    current_node = stack.pop()
                    current_island.append(current_node)
                    
                    # 遍历邻接节点
                    for neighbor in adjacency[current_node]:
                        if neighbor not in visited:
                            #print(neighbor,current_node)
                            visited.add(neighbor)
                            stack.append(neighbor)
                
                islands.append(set(current_island))  # 使用集合去重

        return islands
    
    def find_main_island(self):
        #island_nums = len(self.islands)
        island_node_num = []
        for island in self.islands:
            island_node_num.append(len(island))
        island_node_num = np.array(island_node_num)
        return self.islands[island_node_num.argmax()]
    
    def find_all_reachable_setVolt_nodes_and_paths(self, node_id: str, setVolt: float):
        """
        从给定 node_id 出发，基于当前“实际连通关系”(act_neighbor_nodes) 进行 BFS 搜索，
        找出所有电压 == setVolt 的可达节点，并为每个目标节点回溯一条最短路径。

        返回:
        ----
        paths_dict : dict
            { 目标节点ID: [node_id(起点), ..., 目标节点ID] }
            如果从起点所在的连通分量里没有任何电压为 setVolt 的节点，则返回 {}。

        注意:
        ----
        1. 请先调用 self.scan_neighbor_nodes() 以填充所有节点的 act_neighbor_nodes。
        2. 如果起点 node_id 本身就是电压 == setVolt，则会直接包含在结果中。
        3. 如果同一目标节点有多条等长的最短路径，下面的代码只会回溯并返回**一条**路径；
        如果需要“所有最短路径”，需在 `parents[...]` 中存储列表，额外处理。
        """

        if node_id not in self.nodeID2node:
            # 如果给定起点ID无效，直接返回空
            return {}

        # 1) 初始化
        start_node = self.nodeID2node[node_id]
        visited = set([start_node])
        parent = {start_node: None}  # parent[x] = y 记录 x 的父节点是 y
        queue = deque([start_node])

        # 用于记录“所有电压等于 setVolt 的节点”
        found_setVolt_nodes = set()

        # 2) BFS 遍历
        while queue:
            current_node = queue.popleft()

            # 判断当前节点是否满足目标电压
            try:
                if float(current_node.volt) == setVolt:
                    found_setVolt_nodes.add(current_node)
            except:
                pass

            # 遍历所有实际连通的邻居
            for neighbor in current_node.act_neighbor_nodes:
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = current_node
                    queue.append(neighbor)

        # 3) 若没有找到任何电压为 setVolt 的节点，返回 {}
        if not found_setVolt_nodes:
            return {}

        # 4) 回溯路径: 对每个目标节点，重构一条从起点到该节点的最短路径
        def backtrack_path(end_node):
            """
            从 end_node 沿着 parent[...] 一直回溯到 start_node，得到反向路径，
            然后再 reverse() 成为正向路径。
            """
            path = []
            cur = end_node
            while cur is not None:     # 回溯到起点时 parent[start_node] = None
                path.append(cur.nd)
                cur = parent[cur]
            path.reverse()
            return path

        # 结果字典: 目标节点ID -> 最短路径ID列表
        result = {}
        for target_node in found_setVolt_nodes:
            result[target_node.nd] = backtrack_path(target_node)

        return result
    
    def find_closest_mainbus_node_id_by_node_id(self, node_id: str, setVolt: float,return_path=True):
        # 起始节点对象
        start_node = self.nodeID2node[node_id]
        # 用于标记已访问节点的集合
        visited = set([start_node])
        # parent 字典，用来记录搜索过程中的「父节点」
        parent = {start_node: None}

        # BFS 队列
        queue = deque([start_node])

        # 宽度优先搜索（BFS）
        while queue:
            current_node = queue.popleft()

            # 检查电压是否符合要求
            try:
                if float(current_node.mainbus_10kV) == True:
                    # 找到目标电压等级的节点，开始回溯以获取最短路径
                    path = []
                    temp = current_node
                    # 一直回溯到起始节点
                    while temp is not None:
                        path.append(temp.nd)        # 将节点的 ID 加到 path
                        temp = parent[temp]         # 继续回溯
                    path.reverse()                  # 将回溯序列反转，得到从起点到终点的路径
                    if return_path:
                        return current_node.nd, path
                    else:
                        return current_node.nd
            except:
                pass

            # 继续将邻居节点入队
            for neighbor in current_node.act_neighbor_nodes:
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = current_node  # 记录邻居节点的父节点
                    queue.append(neighbor)

        # 如果搜索结束还没有找到，则返回 None 和空路径
        if return_path:
            return None, []
        else:
            return None
        
    def find_closest_start_feeder_node_id_by_node_id(self, node_id: str, setVolt: float,return_path=True):
        # 起始节点对象
        start_node = self.nodeID2node[node_id]
        # 用于标记已访问节点的集合
        visited = set([start_node])
        # parent 字典，用来记录搜索过程中的「父节点」
        parent = {start_node: None}

        # BFS 队列
        queue = deque([start_node])

        # 宽度优先搜索（BFS）
        while queue:
            current_node = queue.popleft()

            # 检查电压是否符合要求
            try:
                if float(current_node.start_feeder_10kV) == True:
                    # 找到目标电压等级的节点，开始回溯以获取最短路径
                    path = []
                    temp = current_node
                    # 一直回溯到起始节点
                    while temp is not None:
                        path.append(temp.nd)        # 将节点的 ID 加到 path
                        temp = parent[temp]         # 继续回溯
                    path.reverse()                  # 将回溯序列反转，得到从起点到终点的路径
                    if return_path:
                        return current_node.nd, path
                    else:
                        return current_node.nd
            except:
                pass

            # 继续将邻居节点入队
            for neighbor in current_node.act_neighbor_nodes:
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = current_node  # 记录邻居节点的父节点
                    queue.append(neighbor)

        # 如果搜索结束还没有找到，则返回 None 和空路径
        if return_path:
            return None, []
        else:
            return None

    def find_closest_setVolt_node_id_by_node_id(self, node_id: str, setVolt: float,return_path=True):
        """
        从给定 node_id 出发，在当前实际连通关系下 (act_neighbor_nodes)，
        搜索距离最近的电压等级为 setVolt 的节点，并返回两项结果：
        1) 该节点的 ID
        2) 从 node_id 到这个目标节点的最短路径（按顺序排列的节点ID列表）

        如果在连通分量中没有任何电压等级为 setVolt 的节点，则返回 (None, [])。

        注意：在调用本函数之前，请先调用 self.scan_neighbor_nodes()，以保证节点的
            act_neighbor_nodes 列表已填充完毕。
        """

        # 起始节点对象
        start_node = self.nodeID2node[node_id]
        # 用于标记已访问节点的集合
        visited = set([start_node])
        # parent 字典，用来记录搜索过程中的「父节点」
        parent = {start_node: None}

        # BFS 队列
        queue = deque([start_node])

        # 宽度优先搜索（BFS）
        while queue:
            current_node = queue.popleft()

            # 检查电压是否符合要求
            try:
                if float(current_node.volt) == setVolt:
                    # 找到目标电压等级的节点，开始回溯以获取最短路径
                    path = []
                    temp = current_node
                    # 一直回溯到起始节点
                    while temp is not None:
                        path.append(temp.nd)        # 将节点的 ID 加到 path
                        temp = parent[temp]         # 继续回溯
                    path.reverse()                  # 将回溯序列反转，得到从起点到终点的路径
                    if return_path:
                        return current_node.nd, path
                    else:
                        return current_node.nd
            except:
                pass

            # 继续将邻居节点入队
            for neighbor in current_node.act_neighbor_nodes:
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = current_node  # 记录邻居节点的父节点
                    queue.append(neighbor)

        # 如果搜索结束还没有找到，则返回 None 和空路径
        if return_path:
            return None, []
        else:
            return None
    
    def find_all_shortest_paths_to_closest_setVolt_nodes(self, start_node_id: str, setVolt: float):
        """
        从给定 start_node_id 出发，基于 possible_neighbor_nodes 进行 BFS，
        找到“最近”的电压等级 == setVolt 的所有节点，并返回它们的所有最短路径。

        返回值:
        -------
        result_dict : dict
            字典形式返回, 其中 key 为“找到的目标节点ID”，
            value 为“从 start_node_id 到该目标节点ID 的所有最短路径(列表)”，
            每条路径是一个从起点到目标节点的 node_id 顺序列表。

            若在与起点同一连通分量内未找到任何电压为 setVolt 的节点，返回空字典 {}。
        """

        if start_node_id not in self.nodeID2node:
            # 如果起点ID无效，直接返回空
            print("start_node_id不存在于self.nodeID2node")
            return {}

        start_node = self.nodeID2node[start_node_id]

        # 如果起点本身的电压就等于 setVolt，最短路径长度为0，直接返回
        try:
            if float(start_node.volt) == setVolt:
                return {
                    start_node_id: [[start_node_id]]
                }
        except:
            pass

        # ------------------------------
        # 1) BFS 准备阶段
        # ------------------------------
        visited = set()
        visited.add(start_node)
        distance = {start_node: 0}  # 每个节点到起点的最短距离
        parents = {start_node: []}  # 回溯路径用的父节点列表

        queue = deque([start_node])
        found_distance = None       # 首次找到目标电压节点的 BFS 层级(最短距离)
        found_nodes = []            # 收集所有满足 setVolt 且距离 == found_distance 的节点

        # ------------------------------
        # 2) 开始 BFS
        # ------------------------------
        while queue:
            current = queue.popleft()
            curr_dist = distance[current]

            # 如果已经找到了目标电压节点，且当前节点层级 > found_distance，
            # 说明已超出最短路径层级，无需再搜索更深
            if found_distance is not None and curr_dist > found_distance:
                break

            # 检查当前节点电压是否符合要求
            try:
                if float(current.volt) == setVolt:
                    # 这是一个目标电压节点
                    if found_distance is None:
                        found_distance = curr_dist  # 记录首次找到的层级
                    if curr_dist == found_distance:
                        found_nodes.append(current)
            except:
                pass

            # 如果尚未到达 found_distance 层，继续拓展下一层
            # 或者还没发现任何目标节点
            if found_distance is None or curr_dist < found_distance:
                for neighbor in current.possible_neighbor_nodes:
                    if neighbor not in distance:
                        # neighbor 第一次被发现
                        distance[neighbor] = curr_dist + 1
                        parents[neighbor] = [current]
                        visited.add(neighbor)
                        queue.append(neighbor)
                    else:
                        # 已访问过 neighbor，但可能是同一距离(说明存在多条最短路径)
                        if distance[neighbor] == curr_dist + 1:
                            parents[neighbor].append(current)

        # 如果在同一连通分量内压根没有找到目标电压节点，直接返回 {}
        if not found_nodes:
            return {}

        # ------------------------------
        # 3) 回溯所有最短路径
        # ------------------------------
        def backtrack_all_paths(end_node):
            """
            基于 BFS 生成的 parents 字典，回溯 end_node 的所有最短路径。
            返回：二维列表，每个元素是一条最短路径(按顺序存放节点ID)。
            """
            if end_node == start_node:
                return [[start_node_id]]  # 只有起点本身
            
            all_paths = []
            for p in parents[end_node]:
                sub_paths = backtrack_all_paths(p)
                for sp in sub_paths:
                    all_paths.append(sp + [end_node.nd])
            return all_paths

        result_dict = {}
        for tgt in found_nodes:
            # 对找到的目标节点逐一回溯出所有最短路径
            paths = backtrack_all_paths(tgt)
            # 存入结果字典：键是目标节点ID，值是路径列表
            result_dict[tgt.nd] = paths

        return result_dict
    
    
    
    def find_subnodes_for_lower_voltage_nodes(self, root_node_id: str):
        """
        从给定 root_node_id 出发，在“实际连通关系”(act_neighbor_nodes) 上做遍历，
        找到所有可达节点，并筛选出电压等级比根节点更低的节点 (可选地包含根节点)，
        最后将这些节点及它们之间的线路、变压器、负荷等打包到一个新的 pandapower 网络里。

        返回: sub_net (pandapowerNet)
        
        使用示例:
        -------
            # 1. 初始化并解析Excel
            calc = PandaPowerFlowCalculator(parsed_cim)
            # 2. 扫描实际邻居关系
            calc.scan_neighbor_nodes()
            # 3. 构建子网络
            sub_net = calc.create_subnetwork_for_lower_voltage_nodes("769002015")
            # sub_net 中就包含了根节点 + 所有电压等级低于根节点且可达的节点以及它们的互连元件
        """
        # 如果根节点无效，直接返回一个空网络
        if root_node_id not in self.nodeID2node:
            return pp.create_empty_network()

        # 1) 获取根节点及其电压
        root_node = self.nodeID2node[root_node_id]
        try:
            root_volt = float(root_node.volt)
        except:
            # 如果根节点电压为空或无法转换，视为 110
            root_volt = 110.

        # 2) BFS 找到与根节点在同一连通分量的所有节点
        visited = set([root_node])
        queue = deque([root_node])
        sub_nodes_ID = set()
        sub_nodes_ID.add(root_node.nd)
        while queue:
            current = queue.popleft()
            for nbr in current.act_neighbor_nodes:
                if nbr not in visited and float(nbr.volt) < root_volt:
                    visited.add(nbr)
                    queue.append(nbr)
                    sub_nodes_ID.add(nbr.nd)
        return sub_nodes_ID
    
    def create_pandapower_net_from_node_ids(self, node_ids: set[str]):
        """
        给定一批节点 ID (node_ids)，在 self.nodeID2node, self.ac_lines, self.breakers,
        self.disconnectors, self.transformer_type2, self.transformer_type3, self.loads 的基础上，
        创建一个仅包含这些节点及其互连元件的子网（pandapowerNet）并返回。

        如果 node_ids 为空，将返回一个空的 pandapowerNet。
        """
        # 1) 创建空的 pandapower 网络
        sub_net = pp.create_empty_network()

        # 若给定的节点集合为空，直接返回空网
        if not node_ids:
            return sub_net
        
        # 2) 创建母线 (bus) 并构建一个“节点ID -> bus索引”的映射
        nodeID_to_bus = {}
        for nid in node_ids:
            # 在 sub_net 中创建 bus
            node = self.nodeID2node[nid]
            bus_index = pp.create_bus(sub_net, vn_kv=float(node.volt), name=node.name,index=node.bus)
            nodeID_to_bus[nid] = bus_index
            #if node.withload:
            # if node.pl != 0. or node.ql != 0.:
            #     # 如果节点有负荷，创建负荷
            #     pp.create_load(sub_net,
            #                                 bus=node.bus,
            #                                 p_mw=node.pl,
            #                                 q_mvar=node.ql,
            #                                 name=node.name)
        #3) 根据子节点集合，筛选并创建线路 / 断路器 / 隔离开关 / 变压器 / 负荷 等元件
        #    只要其所有端点都在 node_ids 中，且处于“有效”或“闭合”状态，就复制到子网

        # 3.1) 线路 (ACLine)
        for line in self.ac_lines:
            # 判断此线路两端是否在子节点集合中，且都未断开
            # if (line.I_nd_ID in node_ids and 
            #     line.J_nd_ID in node_ids and 
            #     line.I_off == '0' and 
            #     line.J_off == '0'):
            if (line.I_nd_ID in node_ids and 
                line.J_nd_ID in node_ids and 
                line.closed == '1'):

                pp.create_line_from_parameters(
                    sub_net,
                    from_bus=nodeID_to_bus[line.I_nd_ID],
                    to_bus=nodeID_to_bus[line.J_nd_ID],
                    length_km=1.0,
                    r_ohm_per_km=line.r,
                    x_ohm_per_km=line.x,
                    #c_nf_per_km=(b_ch / 314.0) if b_ch else 0.0,
                    c_nf_per_km = line.b,
                    max_i_ka=line.max_i_ka,
                    name=line.name
                )
            else:
                print(f"线路 {line.name} 的端点不在子节点集合中或已断开，跳过")
                continue

        # 3.2) 断路器 (Breaker) - 用线的方式建模（或你也可以用开关开关建模）
        for brk in self.breakers:
            if (brk.I_nd_ID in node_ids and 
                brk.J_nd_ID in node_ids and 
                brk.closed == '1'):
                # 用一个几乎“理想”的线路来表示断路器
                pp.create_line_from_parameters(
                    sub_net,
                    from_bus=nodeID_to_bus[brk.I_nd_ID],
                    to_bus=nodeID_to_bus[brk.J_nd_ID],
                    length_km=1,
                    r_ohm_per_km=brk.r,
                    x_ohm_per_km=brk.x,
                    c_nf_per_km=brk.b,
                    max_i_ka=brk.max_i_ka,
                    name=brk.name
                )
            else:
                print(brk.I_nd_ID, brk.J_nd_ID,brk.closed)
                print(f"断路器 {brk.name} 的端点不在子节点集合中或已断开，跳过")
                continue

        # 3.3) 隔离开关 (Disconnector) - 同理，用线(或开关)方式建模
        for dsc in self.disconnectors:
            if (dsc.I_nd_ID in node_ids and 
                dsc.J_nd_ID in node_ids and 
                dsc.closed == '1'):
                pp.create_line_from_parameters(
                    sub_net,
                    from_bus=nodeID_to_bus[dsc.I_nd_ID],
                    to_bus=nodeID_to_bus[dsc.J_nd_ID],
                    length_km=1,
                    r_ohm_per_km=dsc.r,
                    x_ohm_per_km=dsc.x,
                    c_nf_per_km=dsc.b,
                    max_i_ka=dsc.max_i_ka,
                    name=dsc.name
                )
            else:
                print(dsc.I_nd_ID, dsc.J_nd_ID,dsc.closed)
                print(f"隔离开关 {dsc.name} 的端点不在子节点集合中或已断开，跳过")
                continue

        # 3.4) 两绕组变压器
        for trafo in self.transformer_type2:
            if (trafo.I_nd_ID in node_ids and 
                trafo.J_nd_ID in node_ids and 
                trafo.I_off == '0' and 
                trafo.J_off == '0'):
                try:
                    sn_mva = float(trafo.I_S)  # 视在容量
                    vn_hv = float(trafo.I_Volt)
                    vn_lv = float(trafo.J_Volt)
                    vkr_percent = float(trafo.I_rPU)
                    # 负载损耗(kW)，pfe_kw
                    load_losses_kw = (float(trafo.I_loadLoss) + float(trafo.J_loadLoss)) / 1000.0
                except:
                    sn_mva, vn_hv, vn_lv, vkr_percent, load_losses_kw = 10.0, 110.0, 10.0, 1.0, 0.0

                pp.create_transformer_from_parameters(
                    sub_net,
                    hv_bus=nodeID_to_bus[trafo.I_nd_ID],
                    lv_bus=nodeID_to_bus[trafo.J_nd_ID],
                    sn_mva=sn_mva,
                    vn_hv_kv=vn_hv,
                    vn_lv_kv=vn_lv,
                    vkr_percent=vkr_percent,
                    vk_percent=6.0,        # 先随意给个值
                    pfe_kw=load_losses_kw,
                    i0_percent=0.0,
                    name=trafo.name
                )

        # 3.5) 三绕组变压器
        for trafo3 in self.transformer_type3:
            if (trafo3.I_nd_ID in node_ids and 
                trafo3.J_nd_ID in node_ids and 
                trafo3.K_nd_ID in node_ids and
                trafo3.I_off == '0' and 
                trafo3.J_off == '0' and
                trafo3.K_off == '0'):
                try:
                    vn_hv = float(trafo3.I_Volt)
                    vn_mv = float(trafo3.J_Volt)
                    vn_lv = float(trafo3.K_Volt)
                    sn_hv = float(trafo3.I_S)
                    sn_mv = float(trafo3.J_S)
                    sn_lv = float(trafo3.K_S)
                    vkr_hv = float(trafo3.I_rPU)
                    vkr_mv = float(trafo3.J_rPU)
                    vkr_lv = float(trafo3.K_rPU)
                    loadloss_kw = ((float(trafo3.I_loadLoss) + 
                                    float(trafo3.J_loadLoss) + 
                                    float(trafo3.K_loadLoss)) / 1000.0)
                except:
                    # 若解析失败，给一些默认值
                    vn_hv, vn_mv, vn_lv = 110.0, 35.0, 10.0
                    sn_hv, sn_mv, sn_lv = 100.0, 50.0, 30.0
                    vkr_hv, vkr_mv, vkr_lv, loadloss_kw = 1.0, 1.0, 1.0, 0.0

                pp.create_transformer3w_from_parameters(
                    sub_net,
                    hv_bus=nodeID_to_bus[trafo3.I_nd_ID],
                    mv_bus=nodeID_to_bus[trafo3.J_nd_ID],
                    lv_bus=nodeID_to_bus[trafo3.K_nd_ID],
                    vn_hv_kv=vn_hv,
                    vn_mv_kv=vn_mv,
                    vn_lv_kv=vn_lv,
                    sn_hv_mva=sn_hv,
                    sn_mv_mva=sn_mv,
                    sn_lv_mva=sn_lv,
                    vk_hv_percent=6.0,   # 简化处理
                    vk_mv_percent=6.0,
                    vk_lv_percent=6.0,
                    vkr_hv_percent=vkr_hv,
                    vkr_mv_percent=vkr_mv,
                    vkr_lv_percent=vkr_lv,
                    pfe_kw=loadloss_kw,
                    i0_percent=0.0,
                    name=trafo3.name
                )

        #3.6) 负荷
        for node in self.ele_nodes:
            if node.nd in node_ids and (node.pl != 0. or node.ql != 0.):
                # 只添加那些在 node_ids 中的节点，并且有负荷的节点
                pp.create_load(
                    sub_net,
                    bus=nodeID_to_bus[node.nd],
                    p_mw=node.pl,
                    q_mvar=node.ql,
                    name=node.name
                )
        # for ld in self.loads:
        #     if (ld.nd_ID in node_ids) and (ld.off == '0'):

        #         pp.create_load(
        #             sub_net,
        #             bus=nodeID_to_bus[ld.nd_ID],
        #             p_mw=ld.P,
        #             q_mvar=ld.Q,
        #             name=ld.name
        #         )

        # 4) 返回构建好的子网
        return sub_net
    
    def create_pandapower_net_from_node(self, node_id: str):
        sub_nodes = self.find_subnodes_for_lower_voltage_nodes(node_id)
        sub_net = self.create_pandapower_net_from_node_ids(sub_nodes)
        return sub_net
    def plotly_colored_by_vlevel_and_load(self,net,
                                      bus_size=10,
                                      line_width=2,
                                      respect_switches=True,
                                      auto_open=False,
                                      fig_size=(1600, 900),
                                      purple_node_buses=[]):
        """
        使用 Plotly 方式绘制 pandapower 网络 net，并根据电压等级和是否带负荷来区分母线颜色：
        - 若母线上有负荷，则标为绿色
        - 否则按电压等级区分: 220->红, 110->橙, 10->蓝, 其余->灰
        线路统一用灰色，支持指定线路宽度、母线点大小等参数。
        
        参数：
        ------
        net : pandapowerNet
            需要绘图的 pandapower 网络。
        bus_size : int, default=10
            母线在图中的大小。
        line_width : int, default=2
            线路在图中的线宽。
        respect_switches : bool, default=True
            是否在绘图时考虑开关的断开/闭合状态。
        auto_open : bool, default=False
            是否在生成图后自动在浏览器中打开。
        
        返回：
        ------
        fig : plotly.graph_objs._figure.Figure
            可交互的 Plotly 图对象，可使用 fig.show() 或 fig.write_html(...) 等操作。
        """

        # 1) 如果 net.bus_geodata 为空，可以由 simple_plotly 内部自动生成，或者你手动先调用:
        #    plot.create_generic_coordinates(net, respect_switches=respect_switches)
        # 2) 构造与每个母线 index 对应的颜色列表
        bus_color = []
        
        for bus_idx in net.bus.index:
            try:
                node = self.bus2node[bus_idx]
                v_level = net.bus.at[bus_idx, "vn_kv"]  # 母线额定电压
            except:
                bus_color.append("red")
                continue
            # 判断此母线上是否有负荷
            # （如果有多处负荷，只要其中一个负荷在该母线上就视为“有负荷”）
            if len(purple_node_buses) == 0:

                # 按电压等级分颜色
                if v_level == 220:
                    bus_color.append("red")
                elif v_level == 110:
                    bus_color.append("orange")
                elif v_level == 10:
                    if node. concat_node_feeder  :
                        bus_color.append("yellow")
                    elif node. start_feeder_10kV  :
                        bus_color.append("cyan")
                    #elif node.withload == 1:
                    elif node.pl != 0 or node.ql != 0:
                        bus_color.append("green")
                    else:
                        bus_color.append("blue")
                else:
                    bus_color.append("gray")
            else:
                for purple_bus in purple_node_buses:
                    if purple_bus == bus_idx:
                        print(1)
                        bus_color.append("purple")
                    else:
                        if bus_idx in net.load.bus.values:
                            # 标记绿色
                            bus_color.append("green")
                        else:
                            # 按电压等级分颜色
                            if v_level == 220:
                                bus_color.append("red")
                            elif v_level == 110:
                                bus_color.append("orange")
                            elif v_level == 10:
                                if node. concat_node_feeder  :
                                    bus_color.append("yellow")
                                else:
                                    bus_color.append("blue")
                            else:
                                bus_color.append("gray")

        # 3) 线路统一用灰色
        line_color = "gray"
        
        # 4) 调用 simple_plotly，传入自定义的颜色数组
        fig = ppplotly.simple_plotly(
            net=net,
            respect_switches=respect_switches,
            bus_size=bus_size,
            line_width=line_width,
            bus_color=bus_color,
            line_color=line_color,
            on_map=False,     # 若有地理坐标且想叠加到地图上，可设 True
            auto_open=auto_open  
            )
        fig.update_layout(width=int(fig_size[0]), height=int(fig_size[1]))
        return fig
    
    def to_excel(self):
        self.node_df = pd.DataFrame({
            'ID':[node.nd for node in self.ele_nodes],
            'name':[node.name for node in self.ele_nodes],
            'volt':[float(node.volt) for node in self.ele_nodes],
            'closest_110_node_ID':[node.closest_110_node.nd if node.closest_110_node else 'N\A' for node in self.ele_nodes ],
            'gateway_node':[node.gateway_node_10kV for node in self.ele_nodes],
            'concat_node':[node.concat_node_10kV for node in self.ele_nodes],
            'gateway_node_otherside':[node.gateway_node_10kV_otherside for node in self.ele_nodes],
            'pl':[node.pl for node in self.ele_nodes],
            'ql':[node.ql for node in self.ele_nodes],
            'pg':[node.pg for node in self.ele_nodes],
            'qg':[node.qg for node in self.ele_nodes],
        })
        
        self.all_lines_df = pd.DataFrame({
            'ID':[line.ID for line in self.all_lines],
            'name':[line.name for line in self.all_lines],
            'volt':[line.volt for line in self.all_lines],
            'I_node':[line.I_node for line in self.all_lines],
            'J_node':[line.J_node for line in self.all_lines],
            'I_nd_ID':[line.I_nd_ID for line in self.all_lines],
            'J_nd_ID':[line.J_nd_ID for line in self.all_lines],
            'closed':[int(line.closed) for line in self.all_lines],
            'device_type':[line.device_type for line in self.all_lines],
            'gateway':[line.gateway for line in self.all_lines],
            'concat_switch':[line.concat_switch for line in self.all_lines],
            'r':[float(line.r) for line in self.all_lines],
            'x':[float(line.x) for line in self.all_lines],
            #'bch':[float(line.bch) for line in self.all_lines],
            'max_i_ka':[float(line.ratedCurrent) / 1000  if line.device_type == 'acline' else float(line.max_i_ka) for line in self.all_lines],
            'closest_110_node_ID':[line.closest_110_node.nd if line.closest_110_node else 'N\A' for line in self.all_lines],
            'closest_110_nodes_IDs':[(line.closest_110_nodes[0].nd + '-' + line.closest_110_nodes[1].nd) if line.closest_110_nodes else 'N\A' for line in self.all_lines],
        })
        
        self.transformer_2_df = pd.DataFrame({
            'ID':[transformer.nd for transformer in self.transformer_type2],
            'name':[transformer.name for transformer in self.transformer_type2],
            'I_node':[transformer.I_node for transformer in self.transformer_type2],
            'J_node':[transformer.J_node for transformer in self.transformer_type2],
            'I_nd_ID':[transformer.I_nd_ID for transformer in self.transformer_type2],
            'J_nd_ID':[transformer.J_nd_ID for transformer in self.transformer_type2],
            'closed':[transformer.closed for transformer in self.transformer_type2],
            'I_leakagelmpedence':[transformer.I_leakagelmpedence for transformer in self.transformer_type2],
            'I_loadLoss':[transformer.I_loadLoss for transformer in self.transformer_type2],
            'I_S':[transformer.I_S for transformer in self.transformer_type2],
            'I_r':[transformer.I_r for transformer in self.transformer_type2],      
            'I_rPU':[transformer.I_rPU for transformer in self.transformer_type2],
            'I_Volt':[transformer.I_Volt for transformer in self.transformer_type2],
            'I_x':[transformer.I_x for transformer in self.transformer_type2],
            'I_xPU':[transformer.I_xPU for transformer in self.transformer_type2],
            'J_leakagelmpedence':[transformer.J_leakagelmpedence for transformer in self.transformer_type2],
            'J_loadLoss':[transformer.J_loadLoss for transformer in self.transformer_type2],
            'J_S':[transformer.J_S for transformer in self.transformer_type2],
            'J_r':[transformer.J_r for transformer in self.transformer_type2],      
            'J_rPU':[transformer.J_rPU for transformer in self.transformer_type2],
            'J_Volt':[transformer.J_Volt for transformer in self.transformer_type2],
            'J_x':[transformer.J_x for transformer in self.transformer_type2],
            'J_xPU':[transformer.J_xPU for transformer in self.transformer_type2],
            'device_type':[transformer.device_type for transformer in self.transformer_type2],
            'gateway_transformer':[transformer.gateway_transformer for transformer in self.transformer_type2],
                    })

        self.transformer_3_df = pd.DataFrame({
            'ID':[transformer.nd for transformer in self.transformer_type3],
            'name':[transformer.name for transformer in self.transformer_type3],
            'I_node':[transformer.I_node for transformer in self.transformer_type3],
            'J_node':[transformer.J_node for transformer in self.transformer_type3],
            'K_node':[transformer.K_node for transformer in self.transformer_type3],
            'I_nd_ID':[transformer.I_nd_ID for transformer in self.transformer_type3],
            'J_nd_ID':[transformer.J_nd_ID for transformer in self.transformer_type3],
            'K_nd_ID':[transformer.K_nd_ID for transformer in self.transformer_type3],
            'closed':[transformer.closed for transformer in self.transformer_type3],
            'I_leakagelmpedence':[transformer.I_leakagelmpedence for transformer in self.transformer_type3],
            'I_loadLoss':[transformer.I_loadLoss for transformer in self.transformer_type3],
            'I_S':[transformer.I_S for transformer in self.transformer_type3],
            'I_r':[transformer.I_r for transformer in self.transformer_type3],      
            'I_rPU':[transformer.I_rPU for transformer in self.transformer_type3],
            'I_Volt':[transformer.I_Volt for transformer in self.transformer_type3],
            'I_x':[transformer.I_x for transformer in self.transformer_type3],
            'I_xPU':[transformer.I_xPU for transformer in self.transformer_type3],
            'J_leakagelmpedence':[transformer.J_leakagelmpedence for transformer in self.transformer_type3],
            'J_loadLoss':[transformer.J_loadLoss for transformer in self.transformer_type3],
            'J_S':[transformer.J_S for transformer in self.transformer_type3],
            'J_r':[transformer.J_r for transformer in self.transformer_type3],      
            'J_rPU':[transformer.J_rPU for transformer in self.transformer_type3],
            'J_Volt':[transformer.J_Volt for transformer in self.transformer_type3],
            'J_x':[transformer.J_x for transformer in self.transformer_type3],
            'J_xPU':[transformer.J_xPU for transformer in self.transformer_type3],
            'K_leakagelmpedence':[transformer.K_leakagelmpedence for transformer in self.transformer_type3],
            'K_loadLoss':[transformer.K_loadLoss for transformer in self.transformer_type3],
            'K_S':[transformer.K_S for transformer in self.transformer_type3],
            'K_r':[transformer.K_r for transformer in self.transformer_type3],      
            'K_rPU':[transformer.K_rPU for transformer in self.transformer_type3],
            'K_Volt':[transformer.K_Volt for transformer in self.transformer_type3],
            'K_x':[transformer.K_x for transformer in self.transformer_type3],
            'K_xPU':[transformer.K_xPU for transformer in self.transformer_type3],
            'device_type':[transformer.device_type for transformer in self.transformer_type3],
            'gateway_transformer':[transformer.gateway_transformer for transformer in self.transformer_type3],
                    })
        
        # oriPandaPowerFile_name = cwd+'/实验/YanTian/oriPandaPowerFile.xlsx'
        # with pd.ExcelWriter(oriPandaPowerFile_name, engine='xlsxwriter') as writer:
        #     self.node_df.to_excel(writer, sheet_name='nodes', index=False)
        #     self.all_lines_df.to_excel(writer, sheet_name='all_lines', index=False)
        #     self.transformer_2_df.to_excel(writer, sheet_name='transformer_type2', index=False)
        #     self.transformer_3_df.to_excel(writer, sheet_name='transformer_type3', index=False)
        # print(f"数据已成功写入 {oriPandaPowerFile_name}")
        
        if not self.is_scan_feasible_switch_states:
            self.scan_feasible_switch_states()
      
        
        for group_name,group in self.node_groups_10kV.items():
            group_node_df = pd.DataFrame({
            'ID':[node.nd for node in group.nodes],
            'name':[node.name for node in group.nodes],
            'volt':[float(node.volt) for node in group.nodes],
            'closest_110_node_ID':[node.closest_110_node.nd if node.closest_110_node else 'N\A' for node in group.nodes ],
            'gateway_node':[node.gateway_node_10kV for node in group.nodes],
            'concat_node':[node.concat_node_10kV for node in group.nodes],
            'gateway_node_otherside':[node.gateway_node_10kV_otherside for node in group.nodes],
            'pl':[node.pl for node in group.nodes],
            'ql':[node.ql for node in group.nodes],
            'pg':[node.pg for node in group.nodes],
            'qg':[node.qg for node in group.nodes],
            })
        
            group_all_lines_df = pd.DataFrame({
            'ID':[line.ID for line in group.all_lines],
            'name':[line.name for line in group.all_lines],
            'volt':[line.volt for line in group.all_lines],
            'I_node':[line.I_node for line in group.all_lines],
            'J_node':[line.J_node for line in group.all_lines],
            'I_nd_ID':[line.I_nd_ID for line in group.all_lines],
            'J_nd_ID':[line.J_nd_ID for line in group.all_lines],
            'closed':[int(line.closed) for line in group.all_lines],
            'device_type':[line.device_type for line in group.all_lines],
            'gateway':[line.gateway for line in group.all_lines],
            'concat_switch':[line.concat_switch for line in group.all_lines],
            'r':[float(line.r) for line in group.all_lines],
            'x':[float(line.x) for line in group.all_lines],
            #'bch':[float(line.bch) for line in group.all_lines],
            'max_i_ka':[float(line.ratedCurrent) / 1000  if line.device_type == 'acline' else float(line.max_i_ka) for line in group.all_lines],
            'closest_110_node_ID':[line.closest_110_node.nd if line.closest_110_node else 'N\A' for line in group.all_lines],
            'closest_110_nodes_IDs':[(line.closest_110_nodes[0].nd + '-' + line.closest_110_nodes[1].nd) if line.closest_110_nodes else 'N\A' for line in group.all_lines],
            })
            
            if len(group.feasible_switch_states) > 0:
                switch_group_states = pd.DataFrame({
                '1':[','.join(state['1'][i].nd for i in range(len(state['1']))) for state in group.feasible_switch_states ],
                '0':[','.join(state['0'][i].nd for i in range(len(state['0']))) for state in group.feasible_switch_states ],
                })
                
            else:
                switch_group_states = pd.DataFrame({
                '1':[],
                '0':[]
                })

            # group_file_name = cwd+f'/实验/YanTian/开关组模型/{group_name}.xlsx'
            # with pd.ExcelWriter(group_file_name, engine='xlsxwriter') as writer:
            #     group_node_df.to_excel(writer, sheet_name='nodes', index=False)
            #     group_all_lines_df.to_excel(writer, sheet_name='all_lines', index=False)
            #     switch_group_states.to_excel(writer, sheet_name='switch_group_states', index=False)
            #     print(f"数据已成功写入 {group_file_name}")

def plot_pandapower_net_compat(net, show_bus_names: bool = False, show_bus_indices: bool = False, re_create_geo=True,figsize=(16, 9)):
    """
    - 可根据参数显示母线名称或索引

    返回 fig, ax
    """
    if re_create_geo:
        plot.create_generic_coordinates(net)  # respect_switches=True 可加可不加
    """
    # 解析geo坐标到临时x,y列
    net.bus['x'] = np.nan
    net.bus['y'] = np.nan
    for idx in net.bus.index:
        try:
            geo = json.loads(net.bus.at[idx, 'geo'])
            net.bus.at[idx, 'x'] = geo['coordinates'][0]
            net.bus.at[idx, 'y'] = geo['coordinates'][1]
        except:
            net.bus.at[idx, 'x'] = idx  # 默认x坐标
            net.bus.at[idx, 'y'] = 0     # 默认y坐标
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 准备电压等级颜色映射
    voltage_colors = {
        220: 'red',
        110: 'orange',
        10: 'blue'
    }
    
    # 获取所有有负荷的节点
    load_buses = set(net.load.bus.values) if len(net.load) > 0 else set()

    # 按电压等级分组绘制母线
    for vn_kv, color in voltage_colors.items():
        buses = net.bus[net.bus.vn_kv == vn_kv].index
        if len(buses) == 0:
            continue
            
        # 检查是否有负荷的特殊处理
        load_mask = [bus in load_buses for bus in buses]
        normal_buses = buses[~np.array(load_mask)]
        load_buses_subset = buses[load_mask]

        # 绘制普通节点
        if len(normal_buses) > 0:
            bc = plot.create_bus_collection(
                net,
                buses=normal_buses,
                color=color,
                size=0.15,
                zorder=2
            )
            plot.draw_collections([bc], ax=ax)

        # 绘制带负荷节点（绿色覆盖）
        if len(load_buses_subset) > 0:
            bc_load = plot.create_bus_collection(
                net,
                buses=load_buses_subset,
                color='green',
                size=0.2,
                zorder=3
            )
            plot.draw_collections([bc_load], ax=ax)

    # 绘制线路和变压器等连接设备
    lc = plot.create_line_collection(net, net.line.index, color='gray', linewidths=1, zorder=1)
    trafo_col = plot.create_trafo_collection(net, color='purple', linewidth=1.5, zorder=1)
    plot.draw_collections([lc, trafo_col], ax=ax)
    """
    # 添加节点标签
    for idx, row in net.bus.iterrows():
        x = net.bus.at[idx, 'x']
        y = net.bus.at[idx, 'y']
        
        # 生成标签文本
        label = []
        if show_bus_indices:
            label.append(str(idx))
        if show_bus_names:
            label.append(str(row['name']))
        label_text = "\n".join(label)
        
        if label_text:
            ax.text(x, y+0.02, label_text, 
                   fontsize=8, 
                   ha='center', va='bottom',
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    """
    # 设置坐标轴
    #ax.set_xlim(net.bus.x.min()-0.1, net.bus.x.max()+0.1)
    #ax.set_ylim(net.bus.y.min()-0.1, net.bus.y.max()+0.1)
    ax.axis('off')

    return fig, ax



def copy_bus_geodata_from_main_to_sub(main_net, sub_net):
    """
    将 main_net 中 bus.name -> bus.geo 的对应关系，复制到 sub_net。
    如果 sub_net 的母线 bus.name 能在 main_net 找到同名母线，就把 geo 坐标复制过来。
    否则 sub_net 的该母线 geo 留空（或设为 None）。
    """
    # 1) 先从 main_net 中构造一个字典: {bus_name: geo_json_string}
    name_to_geo = dict(zip(main_net.bus["name"], main_net.bus["geo"]))

    # 2) 遍历 sub_net 的母线
    for b_idx in sub_net.bus.index:
        bname = sub_net.bus.at[b_idx, "name"]
        if bname in name_to_geo and pd.notnull(name_to_geo[bname]):
            sub_net.bus.at[b_idx, "geo"] = name_to_geo[bname]
        else:
            # 如果找不到对应的 geo，自己定如何处理
            sub_net.bus.at[b_idx, "geo"] = None
            
def save_feasible_feeders_switch_states(pp_pf_calculator,path=cwd + "/system_file/746sys/"):
    global fcID2fcname,fcname2fcID,fcID2fs
    fc_num = 0
    fcID2fcname = {}
    fcname2fcID = {}
    fcID2fs = {}
    for fc in pp_pf_calculator.feeder_clusters:
        fc_ID = fc_num
        fcID2fcname[fc_ID] = fc.name
        fcname2fcID[fc.name] = fc_ID
        fcID2fs[fc_ID] = []
        for feasible_switch_states in fc.feasible_switch_states:
            feasible_switch_states_dict = {}
            feasible_switch_states_dict['1'] = [(line.I_nd.bus,line.J_nd.bus) for line in feasible_switch_states['1']]
            feasible_switch_states_dict['0'] = [(line.I_nd.bus,line.J_nd.bus) for line in feasible_switch_states['0']]
            fcID2fs[fc_ID].append(feasible_switch_states_dict)
        fc_num += 1
    with open(path + "fcID2fcname_746sys.pkl", "wb") as f:  # 以二进制写模式打开
        pickle.dump(fcID2fcname, f)
    with open(path + "fcname2fcID_746sys.pkl", "wb") as f:  # 以二进制写模式打开
        pickle.dump(fcname2fcID, f)
    with open(path + "fcID2fs_746sys.pkl", "wb") as f:  # 以二进制写模式打开
        pickle.dump(fcID2fs, f)    
        
def load_feasible_feeders_switch_states(pp_pf_calculator,path=cwd + "/system_file/746sys/"):
    global fcID2fcname,fcname2fcID,fcID2fs
    
    with open(path+"fcID2fcname_746sys.pkl", "rb") as f:  # 以二进制读模式打开
        fcID2fcname = pickle.load(f)
    with open(path+"fcname2fcID_746sys.pkl", "rb") as f:  # 以二进制读模式打开
        fcname2fcID = pickle.load(f)
    with open(path+"fcID2fs_746sys.pkl", "rb") as f:  # 以二进制读模式打开
        fcID2fs= pickle.load(f)
    
    for fc_num in range(len(pp_pf_calculator.feeder_clusters)):
        fc_name = fcID2fcname[fc_num]
        fs = fcID2fs[fc_num]
        fc = pp_pf_calculator.fcname2fc[fc_name]
        fc.feasible_switch_states = []
        for feasible_switch_states in fs:            
            feasible_switch_states_dict = {}
            feasible_switch_states_dict['1'] = [pp_pf_calculator.node_buses2line[pp_pf_calculator._nodebus2str(buses)] for buses in feasible_switch_states['1']]
            feasible_switch_states_dict['0'] = [pp_pf_calculator.node_buses2line[pp_pf_calculator._nodebus2str(buses)] for buses in feasible_switch_states['0']]
            fc.feasible_switch_states.append(feasible_switch_states_dict)

np.random.seed(42)
file_path =  PfDataPath
required_sheet_name = ['ACLine',
                        'Transformer',
                        'Load',
                        'Breaker',
                        'Disconnector',
                        #'Compensator',
                        'node',
                        'Disconnector包含馈线',
                        'Disconnector包含馈线支路',
                        ]

except_feeder_bus = []
parsed_cim = CimEParser(file_path)
# %%
