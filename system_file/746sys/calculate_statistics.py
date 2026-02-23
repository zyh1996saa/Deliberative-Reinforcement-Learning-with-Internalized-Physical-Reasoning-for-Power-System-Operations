# In[]
import os
from config746sys import CUDA_VISIBLE_DEVICES,WORKPATH,DATAPATH
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # 让编号与 nvidia‑smi 一致
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES  
os.chdir(WORKPATH)
import sys
sys.path.append(WORKPATH + r'/Utls')

#from Utls.utls import PQ,PV,Pt,load_A,refresh_busnum
#from Utls.utls import load_H, load_A_sparse
from Utls.GTransformerSparseNodalmasksAddAttnUtls import DyMPN,NodeGTransformer
from Utls.utls import load_H, norm_H, zscore_H, recover_H

import numpy as np
import pandas as pd
import copy


import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
from tensorflow.keras import layers, activations
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.utils import *
from scipy.sparse import load_npz
from tensorflow.keras.callbacks import CSVLogger, TensorBoard, EarlyStopping
from tensorflow.keras.layers import BatchNormalization, LayerNormalization
from tensorflow.keras.losses import MeanSquaredError

import time
import pandapower as pp



# In[]
fc_base_net = pp.from_excel(WORKPATH + "/system_file/746sys/fc_base_net.xlsx")
sys_size = fc_base_net.bus.shape[0]
total_sample_num = 2048 * 16

# Step 1 加载原始数据 
H_in = load_H(path=DATAPATH+r'/yantian752_251001', start_label=0, end_label=total_sample_num, sys_size=sys_size, 
           sample_for_each_iter=total_sample_num)

# Step 2 先做z-score
H_zscored, mean_per_node, std_per_node = zscore_H(H_in)

# Step 3 再做归一化（0-1）
H_normalized, max_per_node, min_per_node = norm_H(H_zscored)

# Step 4 保存处理参数
np.save(WORKPATH+'/system_file/746sys/mean_per_node.npy', mean_per_node)
np.save(WORKPATH+'/system_file/746sys/std_per_node.npy', std_per_node)
np.save(WORKPATH+'/system_file/746sys/max_per_node.npy', max_per_node)
np.save(WORKPATH+'/system_file/746sys/min_per_node.npy', min_per_node)

# In[]
# 恢复流程
mean_per_node = np.load(WORKPATH+'/system_file/746sys/mean_per_node.npy')
std_per_node = np.load(WORKPATH+'/system_file/746sys/std_per_node.npy')
max_per_node = np.load(WORKPATH+'/system_file/746sys/max_per_node.npy')
min_per_node = np.load(WORKPATH+'/system_file/746sys/min_per_node.npy')

# 恢复回原始数据
H_in_recovered = recover_H(H_normalized, mean_per_node, std_per_node, max_per_node, min_per_node)

# %%
