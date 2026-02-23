# In[]
import numpy as np
import copy
from scipy.sparse import load_npz
import tensorflow as tf

import pandapower as pp
from pypower.loadcase import loadcase
from pypower.ext2int import ext2int
from pypower.makeYbus import makeYbus
from pypower.makeSbus import makeSbus
from pypower.makeBdc import makeBdc
from pypower.ext2int import ext2int
from pypower.makeSbus import makeSbus
#from pypower.idx_bus import PD, QD, VM, VA, GS, BUS_TYPE, PV, PQ, REF

def get_network_matrices(orinet,calnet):
    """
    从pandapower网络中提取节点特征矩阵H和导纳矩阵Y
    处理节点索引不连续的情况，使用从0开始的连续索引
    
    参数:
        net: pandapower网络对象
        
    返回:
        H: N×6矩阵，节点特征矩阵 [有功负荷, 无功负荷, 有功发电, 无功发电, 电压幅值, 电压相角]
        Y: N×N复数矩阵，系统导纳矩阵
        bus_map: 原始节点索引到连续索引的映射字典
    """
    # 确保运行潮流计算以获取最新状态
    try:
        pp.runpp(calnet)
        pp.runpp(orinet)
    except:
        pass
    
    # 创建节点映射：原始节点索引 -> 连续索引 (0, 1, 2, ..., N-1)
    original_bus_indices = sorted(calnet.bus.index)
    bus_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(original_bus_indices)}
    n_buses = len(original_bus_indices)
    
    # 创建空特征矩阵
    H = np.zeros((n_buses, 6))
    
    # 1. 处理负荷数据
    if 'load' in calnet and len(calnet.load) > 0:
        for _, load in calnet.load.iterrows():
            orig_bus_idx = load.bus
            if orig_bus_idx in bus_map:
                new_bus_idx = bus_map[orig_bus_idx]
                H[new_bus_idx, 0] += load.p_mw    # 有功负荷
                H[new_bus_idx, 1] += load.q_mvar  # 无功负荷
    
    # 2. 处理发电机数据
    if 'gen' in calnet and len(calnet.gen) > 0:
        for _, gen in calnet.gen.iterrows():
            orig_bus_idx = gen.bus
            if orig_bus_idx in bus_map:
                new_bus_idx = bus_map[orig_bus_idx]
                H[new_bus_idx, 2] += gen.p_mw    # 有功发电
                H[new_bus_idx, 3] += gen.q_mvar  # 无功发电
    
    # 3. 处理外部电网数据（视为发电机）
    if 'ext_grid' in calnet and len(calnet.ext_grid) > 0:
        for _, ext_grid in calnet.ext_grid.iterrows():
            orig_bus_idx = ext_grid.bus
            if orig_bus_idx in bus_map:
                #print(ext_grid)
                new_bus_idx = bus_map[orig_bus_idx]
                
                # 外部电网通常不指定功率，但可能影响电压，这里主要处理电压
                # 如果需要，可以添加功率处理
    
    # 4. 添加电压幅值和相角
    for orig_bus_idx, bus_data in calnet.res_bus.iterrows():
        if orig_bus_idx in bus_map:
            new_bus_idx = bus_map[orig_bus_idx]
            H[new_bus_idx, 4] = bus_data.vm_pu        # 电压幅值 (p.u.)
            H[new_bus_idx, 5] = bus_data.va_degree    # 电压相角 (度)
            
    
    orig_slack_bus_idx = calnet.ext_grid.bus[0]
    new_bus_idx = bus_map[orig_slack_bus_idx]
    H[new_bus_idx, 2] += calnet.res_ext_grid.p_mw [0]   
    H[new_bus_idx, 3] += calnet.res_ext_grid.q_mvar[0]
    
    # 5. 获取导纳矩阵Y（按原始节点顺序）
    Ybus = orinet._ppc["internal"]["Ybus"]
    Y_original = Ybus.todense()
    
    # 6. 创建按连续索引排序的导纳矩阵
    # 首先创建映射：原始索引 -> 在原始矩阵中的位置
    original_bus_positions = {bus_idx: pos for pos, bus_idx in enumerate(calnet.bus.index)}
    
    # 创建新的N×N导纳矩阵
    Y = np.zeros((n_buses, n_buses), dtype=complex)
    
    # 填充新的导纳矩阵
    for i_orig, i_new in bus_map.items():
        for j_orig, j_new in bus_map.items():
            # 找到原始矩阵中的位置
            orig_i_pos = original_bus_positions[i_orig]
            orig_j_pos = original_bus_positions[j_orig]
            # 复制导纳值
            Y[i_new, j_new] = Y_original[orig_i_pos, orig_j_pos]
    
    return H, Y, bus_map

# 定义自定义损失函数
def masked_mse_loss(y_true, y_pred):
    """
    自定义的掩码均方误差损失函数。

    参数:
    - y_true: [batch_size, max_sys_size, units]，真实值已根据掩码被置零
    - y_pred: [batch_size, max_sys_size, units]，预测值

    返回:
    - masked_mse: 标量，掩码后的均方误差
    """
    # 获取掩码：mask = 1.0 if y_true 不为0，else 0.0
    mask = tf.cast(tf.not_equal(y_true, 0.0), tf.float32)  # [batch_size, max_sys_size, units]

    # 计算平方误差
    mse = tf.square(y_true - y_pred)  # [batch_size, max_sys_size, units]

    # 计算每个节点的平均MSE
    mse_per_node = tf.reduce_mean(mse, axis=-1)  # [batch_size, max_sys_size]

    # 应用掩码
    masked_mse = mse_per_node * tf.reduce_max(mask, axis=-1)  # [batch_size, max_sys_size]
    # 使用 tf.reduce_max 来确保掩码为1的节点被正确处理

    # 计算平均损失，避免除以零
    return tf.reduce_sum(masked_mse) / (tf.reduce_sum(tf.reduce_max(mask, axis=-1)) + 1e-8)

def pad_adjacency_matrices_sparse(A_sparse, max_sys_size, start_label, sample_for_each_iter):
    """
    将批量稀疏邻接矩阵填充到指定的最大系统大小，并调整批次索引。

    参数：
    - A_sparse: tf.sparse.SparseTensor，形状为 [total_sample_num, sys_size, sys_size]
    - max_sys_size: int，填充后的系统大小
    - start_label: int，当前批次的起始索引
    - sample_for_each_iter: int，当前批次的样本数量

    返回：
    - padded_A: tf.sparse.SparseTensor，形状为 [sample_for_each_iter, max_sys_size, max_sys_size]
    """
    # 获取原始 dense_shape
    original_dense_shape = A_sparse.dense_shape.numpy()
    total_samples, sys_size_1, sys_size_2 = original_dense_shape

    # 创建新的 dense_shape
    new_dense_shape = [sample_for_each_iter, max_sys_size, max_sys_size]

    # 调整批次索引
    indices = A_sparse.indices.numpy()
    # 仅选择当前批次的索引
    mask = (indices[:, 0] >= start_label) & (indices[:, 0] < start_label + sample_for_each_iter)
    indices = indices[mask]
    values = A_sparse.values.numpy()[mask]

    # 调整批次索引为相对于当前批次
    indices[:, 0] = indices[:, 0] - start_label

    padded_A = tf.sparse.SparseTensor(
        indices=indices,
        values=values,
        dense_shape=new_dense_shape
    )
    padded_A = tf.sparse.reorder(padded_A)

    return padded_A

def pad_node_features(H, max_sys_size, units=6):
    """
    将节点特征填充到指定的最大系统大小。
    
    参数:
    - H: 原始特征，形状为 (batch_size, num_nodes, units)
    - max_sys_size: int，填充后的系统大小
    - units: int，特征维度
    
    返回:
    - padded_H: 填充后的特征，形状为 (batch_size, max_sys_size, units)
    """
    padded_H = np.zeros((H.shape[0], max_sys_size, units), dtype=np.float32)
    for i in range(H.shape[0]):
        num_nodes = H[i].shape[0]
        padded_H[i, :num_nodes, :] = H[i]
    return padded_H

def create_NodalMask(H, max_sys_size):
    NodalMask = np.zeros((H.shape[0], max_sys_size), dtype=np.float32)
    for i in range(H.shape[0]):
        num_nodes = np.count_nonzero(np.sum(H[i], axis=-1))
        NodalMask[i, :num_nodes+1] = 1.0
    return NodalMask

def mask_features_by_type(features, isPQ, isPV, isPt, mask_prob=0.15):
    """
    根据节点类型随机掩盖输入特征的一部分。

    参数:
    - features: 原始特征，维度为(batch_size, num_nodes, feature_dim)
    - isPQ: 标记每个节点是否为PQ节点的向量，维度为(num_nodes,)
    - isPV: 标记每个节点是否为PV节点的向量，维度为(num_nodes,)
    - isPt: 标记每个节点是否为Pt节点的向量，维度为(num_nodes,)
    - mask_prob: 每个特征被掩盖的概率

    返回值:
    - masked_features: 掩盖后的特征
    - masks: 实际掩盖的位置标志，用于损失计算
    """
    batch_size, num_nodes, feature_dim = features.shape
    # 初始化掩码为全0，表示不掩盖
    masks = np.zeros_like(features, dtype=bool)
    
    # 根据节点类型设置掩码
    for node_type, feature_indices in zip([isPQ, isPV, isPt], [[4, 5], [3, 5], [2, 3]]):
        for i, is_type in enumerate(node_type):
            if is_type:  # 如果节点属于当前类型
                # 根据mask_prob随机决定是否掩盖对应特征
                masks[:, i, feature_indices] = np.random.rand(batch_size, len(feature_indices)) < mask_prob
    
    # 生成掩盖后的特征，掩盖部分设为0
    masked_features = features * (1 - masks)
    return masked_features, masks

def create_self_supervised_data_multi_decoder(H_iter, isPQ, isPV, isPt, mask_prob=0.15,):
    """
    创建自监督任务的输入数据和多个输出数据。
    
    参数:
    - H_iter: 原始数据，维度为(batch_size, num_nodes, feature_dim)
    - mask_prob: 掩码概率
    
    返回值:
    - masked_H_iter: 掩盖后的输入特征
    - H_iter_splits: 分割后的原始未掩盖的特征，作为多个目标输出
    - masks: 掩码，指示哪些特征被掩盖，用于损失计算时忽略未掩盖的特征
    """
    masked_H_iter, masks = mask_features_by_type(H_iter, isPQ, isPV, isPt, mask_prob)
    
    # 分割 H_iter 为多个目标，每个目标对应一个特征列
    H_iter_splits = [H_iter[:, :, i:i+1] for i in range(H_iter.shape[-1])]
    
    return masked_H_iter, H_iter_splits, masks


def pad_node_type(is_type, max_sys_size, sys_size):
    padded = np.zeros(max_sys_size, dtype=np.float32)
    padded[:sys_size] = is_type
    return padded

def case2AandH(case):
    tempcase = copy.deepcopy(case)
    h_shape0 = tempcase['bus'].shape[0]
    h_in = np.zeros((h_shape0, 6))
    gbus = (tempcase['gen'][:,0] - 1).astype('int')
    h_in[:,0] = tempcase['bus'][:,2]
    h_in[:,1] = tempcase['bus'][:,3]
    h_in[gbus,2] = tempcase['gen'][:,1]
    h_in[gbus,3] = tempcase['gen'][:,2]
    h_in[:,4] = tempcase['bus'][:,7]
    h_in[:,5] = tempcase['bus'][:,8]
    
    ppc = loadcase(tempcase)
    ppc = ext2int(ppc)
    baseMVA, bus, gen, branch = \
        ppc["baseMVA"], ppc["bus"], ppc["gen"], ppc["branch"]
    Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)
    Sbus = makeSbus(baseMVA, bus, gen)
    return h_in,Ybus,Sbus

def PQ(ori_case):
    isPQ = np.zeros((ori_case['bus'].shape[0],))
    wherePQ = np.where(ori_case['bus'][:,1]==1)[0]
    for i in range(wherePQ.shape[0]):
        isPQ[wherePQ[i]] = 1
    return isPQ

def PV(ori_case):
    isPV = np.zeros((ori_case['bus'].shape[0],))
    wherePV = np.where(ori_case['bus'][:,1]==2)[0]
    for i in range(wherePV.shape[0]):
        isPV[wherePV[i]] = 1
    return isPV

def Pt(ori_case):
    isPt= np.zeros((ori_case['bus'].shape[0],))
    wherePt = np.where(ori_case['bus'][:,1]==3)[0]
    for i in range(wherePt.shape[0]):
        isPt[wherePt[i]] = 1
    return isPt

def load_H(start_label,end_label,path,sys_size,sample_for_each_iter=1000,sys_name='YanTian'):
    filepath = path
    H_in = np.zeros((end_label-start_label,sys_size,6))
    for _ in range(start_label,end_label):
        H_in[_%sample_for_each_iter,:,:] = np.load(filepath+r'/H_%s.npy'%_)
        print('\r加载H矩阵进度%s/%s'%(_,end_label),end='\r')
    return H_in

def load_opf_H(start_label,end_label,path,sys_size,dataset=None,datatype='input',sample_for_each_iter=1000,):
    if not dataset:
        filepath = path + r'/数据/OPF_res/%s-system/%s'%(sys_size,datatype)
    else:
        filepath = path + r'/数据/OPF_res/%s-system/%s/%s'%(sys_size,dataset,datatype)
    H_in = np.zeros((end_label-start_label,sys_size,6))
    for _ in range(start_label,end_label):
        H_in[_%sample_for_each_iter,:,:] = np.load(filepath+r'/casezj_H_%s.npy'%_)
        print('\r加载H矩阵进度%s/%s'%(_,end_label),end='\r')
    return H_in

def load_A(start_label,end_label,path,sys_size,dataset='trainingSet',datatype='input',sample_for_each_iter=1000):
    #filepath = path + r'/数据/潮流图格式/%s-system/%s/%s'%(sys_size,dataset,datatype)
    A_in = np.zeros((end_label-start_label,sys_size,sys_size))
    for _ in range(start_label,end_label):
        if datatype:
            Y = load_npz(path + r'/数据/潮流图格式/%s-system/%s/%s/casezj_Y_%s.npz'%(sys_size,dataset,datatype,_)).toarray()
        else:
            Y = load_npz(path + r'/数据/潮流图格式/%s-system/%s/casezj_Y_%s.npz'%(sys_size,dataset,_)).toarray()
        A = tf.cast(np.where(Y != 0, 1, 0), dtype=tf.float32)
        A_in[_%sample_for_each_iter,:,:] = A
        print('\r加载A矩阵进度%s/%s'%(_,end_label),end='\r')
    return A_in

def load_A_sparse(start_label, end_label, path, sys_size,  sample_for_each_iter=1000):
    
    # 使用 list 来批量加载数据
    indices_all = []
    values_all = []
    shapes_all = []
    
    for _ in range(start_label, end_label):
        Y = load_npz(path + r'/Y_%s.npz' %_)
        # 直接处理稀疏矩阵，避免将其转为稠密矩阵
        A = (Y != 0).astype(int)  # 创建稀疏矩阵的二进制版本（非零为1）

        # 获取稀疏矩阵的非零元素
        indices = np.column_stack(np.nonzero(A))  # 直接得到非零元素的索引
        values = A[indices[:, 0], indices[:, 1]]  # 获取这些非零元素的值
        values = np.ravel(values)
        shape = A.shape  # 矩阵的原始形状
        
        # 将批次索引添加到 `indices`
        batch_indices = np.full((indices.shape[0], 1), _)  # 生成一个形状为 (num_non_zero_elements, 1) 的数组，用来表示批次索引
        indices_with_batch = np.hstack((batch_indices, indices))  # 将批次索引与原始索引合并

        # 将每个矩阵的索引、值和形状存储起来
        indices_all.append(indices_with_batch)
        values_all.append(values)
        shapes_all.append(shape)

        print('\r加载A矩阵进度 %s/%s' % (_, end_label), end='\r')

    # 将所有的 indices 和 values 合并成大数组
    all_indices = np.vstack(indices_all)
    all_values = np.concatenate(values_all)
    
    # 假设所有稀疏矩阵的形状一致，取第一个矩阵的形状
    if len(shapes_all) > 0:
        dense_shape = (end_label - start_label, sys_size, sys_size)  # 更新 dense_shape 包含批次维度
    else:
        print("error: 稀疏矩阵形状不一致")
        return None
    
    # 创建最终的稀疏张量
    sparse_tensor_stack = tf.sparse.SparseTensor(indices=all_indices, values=all_values, dense_shape=dense_shape)

    return sparse_tensor_stack

def load_opf_A(start_label,end_label,path,sys_size,dataset=None,datatype='input',sample_for_each_iter=1000):
    #filepath = path + r'/数据/潮流图格式/%s-system/%s/%s'%(sys_size,dataset,datatype)
    A_in = np.zeros((end_label-start_label,sys_size,sys_size))
    for _ in range(start_label,end_label):
        if not dataset:
            Y = load_npz(path + r'/数据/OPF_res/%s-system/%s/casezj_Y_%s.npz'%(sys_size,datatype,_)).toarray()
        else:
            Y = load_npz(path + r'/数据/OPF_res/%s-system/%s/%s/casezj_Y_%s.npz'%(sys_size,dataset,datatype,_)).toarray()
        A = tf.cast(np.where(Y != 0, 1, 0), dtype=tf.float32)
        A_in[_%sample_for_each_iter,:,:] = A
        print('\r加载A矩阵进度%s/%s'%(_,end_label),end='\r')
    return A_in

# 标准化 + 归一化
def zscore_H(H_in, given_stat=False, mean_per_node=None, std_per_node=None):
    """ 直接对原始H_in进行z-score标准化 """
    if not given_stat:
        mean_per_node = np.mean(H_in, axis=0)
        std_per_node = np.std(H_in, axis=0)
    
    H_zscored = np.zeros_like(H_in)
    for i in range(H_in.shape[1]):
        for j in range(H_in.shape[2]):
            if std_per_node[i, j] != 0:
                H_zscored[:, i, j] = (H_in[:, i, j] - mean_per_node[i, j]) / std_per_node[i, j]
            else:
                H_zscored[:, i, j] = H_in[:, i, j] - mean_per_node[i, j]
    return H_zscored, mean_per_node, std_per_node

def norm_H(H_zscored, given_stat=False, max_per_node=None, min_per_node=None):
    """ 在z-score后的数据上做0-1归一化 """
    if not given_stat:
        max_per_node = np.max(H_zscored, axis=0)
        min_per_node = np.min(H_zscored, axis=0)
    
    H_normalized = np.zeros_like(H_zscored)
    for i in range(H_zscored.shape[1]):
        for j in range(H_zscored.shape[2]):
            if max_per_node[i, j] != min_per_node[i, j]:
                H_normalized[:, i, j] = (H_zscored[:, i, j] - min_per_node[i, j]) / (max_per_node[i, j] - min_per_node[i, j])
            else:
                H_normalized[:, i, j] = 0.0
    return H_normalized, max_per_node, min_per_node

# 反归一化
def de_norm_H(H_normalized, max_per_node, min_per_node):
    """ 将0-1归一化的数据恢复到z-score后的数据 """
    H_zscored_recovered = np.zeros_like(H_normalized)
    for i in range(H_normalized.shape[1]):
        for j in range(H_normalized.shape[2]):
            H_zscored_recovered[:, i, j] = H_normalized[:, i, j] * (max_per_node[i, j] - min_per_node[i, j]) + min_per_node[i, j]
    return H_zscored_recovered

# 反标准化
def de_zscore_H(H_zscored, mean_per_node, std_per_node):
    """ 将z-score后的数据恢复到原始数据 """
    H_recovered = np.zeros_like(H_zscored)
    for i in range(H_zscored.shape[1]):
        for j in range(H_zscored.shape[2]):
            H_recovered[:, i, j] = H_zscored[:, i, j] * std_per_node[i, j] + mean_per_node[i, j]
    return H_recovered

# 总的恢复函数
def recover_H(H_normalized, mean_per_node, std_per_node, max_per_node, min_per_node):
    """ 从归一化后的数据直接恢复到原始输入 """
    H_zscored = de_norm_H(H_normalized, max_per_node, min_per_node)   # 先反归一化
    H_in_recovered = de_zscore_H(H_zscored, mean_per_node, std_per_node)  # 再反z-score
    return H_in_recovered


def refresh_busnum(case):
    case_copy = copy.deepcopy(case)
    old_bus_order = case['bus'][:,0]
    new_bus_order = range(1,case['bus'].shape[0]+1)
    mapping = {int(old_bus_order[i]):new_bus_order[i] for i in range(case['bus'].shape[0])}
    
    case_copy['bus'][:,0] = [mapping[case_copy['bus'][:,0][i]] for i in range(case['bus'].shape[0])]
    case_copy['gen'][:,0] = [mapping[case_copy['gen'][:,0][i]] for i in range(case['gen'].shape[0])]
    case_copy['branch'][:,0] = [mapping[case_copy['branch'][:,0][i]] for i in range(case_copy['branch'].shape[0])]
    case_copy['branch'][:,1] = [mapping[case_copy['branch'][:,1][i]] for i in range(case_copy['branch'].shape[0])]
    
    return case_copy
# %%
