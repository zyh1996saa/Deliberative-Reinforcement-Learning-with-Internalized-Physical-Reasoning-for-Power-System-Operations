# In[]
import os
import sys
sys.path.append(r"/home/user/Desktop/zyh/self-refl")
from config746sys import WORKPATH,DATAPATH,sub_402_bus_system_node_order_file,DistTfPath
from Utls.yantian_sys_746sys import *
from Utls.utls import get_network_matrices
import pandapower as pp
import numpy as np
from multiprocessing import Pool
from new746_system_v0713 import init_feeder_net, enforce_node_order, built_ppnet_for_pfcal
from tqdm import tqdm
import random
import time
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



def generate_and_save(sample_idx):
    seed = int(time.time() * 1e6) % (2**32 - 1) 
    np.random.seed(seed)
    random.seed(seed)
    New_net = sample_a_new_net(feeder_cluster, fc_base_net)
    New_net_cal = built_ppnet_for_pfcal(New_net)
    # print(f'负荷扰动前的负荷功率：{New_net.load["p_mw"].sum()}')
    # New_net.load['p_mw'] *= np.random.uniform(0.4, 1.2)
    # New_net.load['q_mvar'] *= np.random.uniform(-1, 0.3)
    # print(f'负荷扰动后的负荷功率：{New_net.load["p_mw"].sum()}')
    try:
        pp.runpp(New_net_cal, tolerance_mva=1e-6, max_iteration=100, calculate_voltage_angles=True)
        Hi, Yi, bus_map = get_network_matrices( New_net, New_net_cal)
        for i in range(fc_base_net.line.shape[0]):
            line = fc_base_net.line.loc[i,:]
            from_bus = line['from_bus']
            to_bus = line['to_bus']
            mapped_from = bus_map[from_bus]
            mapped_to = bus_map[to_bus]
            Y_from_to = Yi[mapped_from, mapped_to]
            #print(f"Line from {from_bus} to {to_bus} has Ybus value: {Y_from_to}")
        sparse_Yi = csr_matrix(Yi)

        #print( "sample_idx:",sample_idx, "Yi[1,15]",Yi[1,15])
        np.save( DATAPATH + f'/yantian752_260105/H_{sample_idx}.npy', Hi)
        save_npz( DATAPATH + f'/yantian752_260105/Y_{sample_idx}', sparse_Yi)
        return Hi.shape, Yi.shape  # 不是必须，只是看情况要不要返回
    except Exception as e:
        # print(f"Error processing sample {sample_idx}: {e}")
        return None, None  # 如果运行失败，返回None
    
    
    

if __name__ == "__main__":

    parsed_cim = CimEParser(PfDataPath)

    pp_pf_calculator = PandaPowerFlowCalculator(parsed_cim,slack_nd='703002137')
    #pp_pf_calculator.scan_feasible_feeders_switch_states()
    #save_feasible_feeders_switch_states(pp_pf_calculator, path=WORKPATH + '/system_file/')
    
    feeder_cluster, fc_base_net = init_feeder_net(pp_pf_calculator)
    fc_base_net_cal = built_ppnet_for_pfcal(fc_base_net)
    # pp.to_excel(fc_base_net_cal,  r'/data/FQ/746Node/编排决策/电网模型/752node.xlsx')
    pp.runpp(fc_base_net_cal)
    
    new_net = sample_a_new_net(feeder_cluster, fc_base_net)
    new_net_cal = built_ppnet_for_pfcal(new_net)
    pp.runpp(new_net_cal)
    

# In[]
if __name__ == "__main__":
    new_net = sample_a_new_net(feeder_cluster, fc_base_net)
    #fig = pp_pf_calculator.plotly_colored_by_vlevel_and_load(new_net,fig_size=(800, 450),bus_size=3,)
    print(len(new_net.bus))
   
    start_num = 2048 * 32
    datasetSize = 2048 * 128
    #start_num = 11800
    #datasetSize =  200
    #datasetSize =  2048 * 16
    # availableDataNum = start_num
    task_args = [i for i in range(start_num, datasetSize + start_num)]
    num_workers = 128
    with Pool(processes = num_workers) as pool:
        results = []
        for result in tqdm(pool.imap_unordered(generate_and_save, task_args), 
                            total=datasetSize, 
                            desc='正在生成样本'):
            results.append(result)
  
    # for sample_idx in range(start_num, datasetSize):
    #     Hi_shape, Yi_shape = generate_and_save(sample_idx)



    
    



# %%
