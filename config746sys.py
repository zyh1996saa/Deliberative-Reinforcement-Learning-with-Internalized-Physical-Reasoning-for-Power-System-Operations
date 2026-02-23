# In[]
import numpy as np
import pandas as pd
# 数据路径
WORKPATH = '/home/user/Desktop/zyh/self-refl'
DATAPATH = '/data2/zyh'
#TimeSeriesDataPath = DATAPATH + '/yantian_timeseries_data_v20250707'
TimeSeriesDataPath = DATAPATH + '/yantian_timeseries_data_v20250717'
#PfDataPath = WORKPATH + r'/system_file/parsed_cim_with_feeders.xlsx'
PfDataPath = WORKPATH + r"/system_file/746sys/parsed_cim_new402_v0716.xlsx"
DistTfPath = WORKPATH + r"/system_file/746sys/DistTf.csv"
FeederFiles = [
    WORKPATH + r"/system_file/746sys/FeederFiles/branch_F02合景同创线.csv",
    WORKPATH + r"/system_file/746sys/FeederFiles/branch_F09明铁线.csv",
    WORKPATH + r"/system_file/746sys/FeederFiles/branch_F14临田保合线.csv",
    WORKPATH + r"/system_file/746sys/FeederFiles/branch_F31明交线.csv",
]

# 盐田区系统设置
BaseMVA = 100 # 基准容量，单位MVA
BaseVoltage = 10 # 基准电压。单位 KV
BaseZ = (BaseVoltage*1e3)**2/(BaseMVA*1e6) # 基准阻抗，单位oumu
r_switch = 0.0 # 开关的电阻，单位oumu
x_switch = 1e-6 # 开关的电抗，单位oumu
npz1 = np.load(WORKPATH + '/system_file/746sys/nd2gisid.npz')
nd2gisid = {k: npz1[k] for k in npz1.files}
nd2gisid = {k: v.tolist() for k, v in nd2gisid.items()}
npz2 = np.load(WORKPATH + '/system_file/746sys/gisid2nd.npz')
gisid2nd = {k: npz2[k] for k in npz2.files}
gisid2nd = {k: v.tolist() for k, v in gisid2nd .items()}
nd_withload = list(nd2gisid.keys()) # 有负荷的节点ID
nd_load_stat = pd.read_csv(WORKPATH + '/system_file/746sys/nd_load_stat.csv')
year_range = [2024]
month_range = range(7,13)
sub_402_bus_system_node_order_file = WORKPATH + "/system_file/746sys/fc_node_order.json"
# 显卡设备
CUDA_VISIBLE_DEVICES = '2'
# 算法配置
# 强化学习环境配置
d_in = 6 # 状态维度
T = 4 # 预测/决策时间窗口
T_riskassess = 24 # 风险评估时间窗口
k_bal = 2 # 平衡系数
k_act_swit = 0 # 开关切换成本系数
use_encoder = False # 是否使用编码器
paras_inherit = False # 是否继承预训练编码器的权重
encoder_trainable = False # encoder是否可训练
p_lower = 0.4 # 有功负荷波动下界
p_upper = 1.2 # 有功负荷波动上界
q_lower = -1 # 有功负荷波动下界
q_upper = 0.3 # 有功负荷波动上界
single_act_punish_coef = 0.8 # 1表示无惩罚
random_reset = True # 每次reset时是否随机化


# 决策网络参数配置
units = d_in 
num_heads = 8 # 注意力头数
d_model = 48 # Transformer token的维数
blockNum = 2 # Transformer bloch数
normlayer = 'BatchNorm'
mlpNeuron = 128

# 强化学习算法配置
episode = 200 # 强化学习回合数
epoch = 5 # 每回合决策网络训练轮次数
episodes_per_batch = 4 # 经验池持续回合数
lr = 3e-4 # 决策网络学习率
entropy_coef = 0 # 决策量的熵占损失函数的系数
kl_coef = 0 # kl散度占损失函数的系数
clip_epsilon = 0.1 #剪切函数系数
group_size = 64 #GRPO算法的群数
# %%
