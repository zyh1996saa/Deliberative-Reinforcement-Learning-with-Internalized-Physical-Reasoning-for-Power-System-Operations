# In[]
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, activations
from tensorflow.keras.models import Model

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.utils import *
from tensorflow.keras import mixed_precision
import sys
#cwd = r"/home/user/Desktop/预训练大模型/实验"
#sys.path.append(cwd + r'/Utls')

from .utls import load_H, norm_H, load_A_sparse
import tensorflow.keras.backend as K
from tensorflow.keras.losses import MeanSquaredError

# 定义自定义层
# class RandomizedMessagePassingLayer(layers.Layer):
#     def __init__(self, units, aggregate_mode='graph_conv', **kwargs):
#         super(RandomizedMessagePassingLayer, self).__init__(**kwargs)
#         self.units = units
#         self.aggregate_mode = aggregate_mode
#         # 定义层的权重和偏置
#         self.W = self.add_weight(shape=(units, units), initializer='random_normal', trainable=True)
#         self.b = self.add_weight(shape=(units,), initializer='zeros', trainable=True)

#     def build(self, input_shape):
#         super(RandomizedMessagePassingLayer, self).build(input_shape)

#     def call(self, inputs, training=False):
#         x, adjacency_matrix = inputs
#         adjacency_matrix = tf.stop_gradient(adjacency_matrix)  # 阻止梯度流动

#         if training:
#             K_l = tf.random.uniform(shape=(), minval=1, maxval=5, dtype=tf.int32)
#         else:
#             K_l = 2

#         for _ in range(K_l):
#             x = self.message_passing(x, adjacency_matrix)
#         return x

#     def message_passing(self, x, adjacency_matrices):
#         # x: [batch_size, num_nodes, features]
#         # adjacency_matrices: SparseTensor with shape [batch_size, num_nodes, num_nodes]

#         def single_matmul(inputs):
#             x_sample, adj_sample = inputs
#             adj_sample = tf.stop_gradient(adj_sample)  # 阻止梯度流动
#             return tf.sparse.sparse_dense_matmul(adj_sample, x_sample)  # [num_nodes, features]

#         # 使用 tf.map_fn 逐批处理，使用 fn_output_signature 替代 dtype
#         x_neighbors = tf.map_fn(
#             single_matmul,
#             (x, adjacency_matrices),
#             fn_output_signature=tf.TensorSpec(shape=(None, self.units), dtype=tf.float32)
#         )  # [batch_size, num_nodes, features]
        
#         # 继续现有的聚合方式
#         if self.aggregate_mode == 'mean':
#             degree_matrices = [tf.reduce_sum(tf.sparse.maximum(adj, 0), axis=-1, keepdims=True) for adj in adjacency_matrices]
#             degree_matrices = tf.stack(degree_matrices, axis=0)  # [batch_size, num_nodes, 1]
#             x_neighbors = x_neighbors / (degree_matrices + 1e-8)  # 避免除以零

#         # 应用线性变换和激活函数
#         x_neighbors_transformed = tf.nn.swish(tf.matmul(x_neighbors, self.W) + self.b)  # [batch_size, num_nodes, units]

#         return x_neighbors_transformed

class RandomizedMessagePassingLayer(layers.Layer):
    """
    修正后的消息传递层。
    主要改动在 message_passing 方法中的 tf.map_fn 调用，
    以确保在模型保存时张量形状的一致性。
    """
    def __init__(self, units, aggregate_mode='graph_conv', **kwargs):
        super(RandomizedMessagePassingLayer, self).__init__(**kwargs)
        self.units = units
        self.aggregate_mode = aggregate_mode
        # 定义层的权重和偏置
        self.W = self.add_weight(shape=(units, units), initializer='random_normal', trainable=True, name='W')
        self.b = self.add_weight(shape=(units,), initializer='zeros', trainable=True, name='b')

    def build(self, input_shape):
        super(RandomizedMessagePassingLayer, self).build(input_shape)

    def call(self, inputs, training=False):
        x, adjacency_matrix = inputs
        adjacency_matrix = tf.stop_gradient(adjacency_matrix)  # 阻止梯度流动

        if training:
            # 在训练时，使用随机的传播步数
            K_l = tf.random.uniform(shape=(), minval=1, maxval=5, dtype=tf.int32)
        else:
            # 在推理时，使用固定的传播步数
            K_l = 2

        # K_l 是一个张量, AutoGraph 会将这个 Python for 循环转换为 tf.while_loop
        for _ in range(K_l):
            x = self.message_passing(x, adjacency_matrix)
        return x

    def message_passing(self, x, adjacency_matrix):
        # x: [batch_size, num_nodes, features]
        # adjacency_matrix: SparseTensor, 形状为 [batch_size, num_nodes, num_nodes]

        def single_matmul(inputs):
            x_sample, adj_sample = inputs
            adj_sample = tf.stop_gradient(adj_sample)
            # 对单个样本执行稀疏矩阵乘法
            return tf.sparse.sparse_dense_matmul(adj_sample, x_sample)  # [num_nodes, features]

        x_neighbors = tf.map_fn(
            single_matmul,
            (x, adjacency_matrix),
            fn_output_signature=tf.TensorSpec(shape=(x.shape[1], self.units), dtype=tf.float32)
        )

        # 继续现有的聚合方式
        if self.aggregate_mode == 'mean':
            degree_matrices = [tf.reduce_sum(tf.sparse.maximum(adj, 0), axis=-1, keepdims=True) for adj in adjacency_matrix]
            degree_matrices = tf.stack(degree_matrices, axis=0)  # [batch_size, num_nodes, 1]
            x_neighbors = x_neighbors / (degree_matrices + 1e-8)  # 避免除以零

        # 应用线性变换和激活函数
        x_neighbors_transformed = tf.nn.swish(tf.matmul(x_neighbors, self.W) + self.b)  # [batch_size, num_nodes, units]

        return x_neighbors_transformed

class DyMPN(layers.Layer):
    def __init__(self, units, num_iterations=3, **kwargs):
        super(DyMPN, self).__init__(**kwargs)
        self.num_iterations = num_iterations
        self.message_passing_layers = [RandomizedMessagePassingLayer(units) for _ in range(num_iterations)]
        # 移除 readout 层，因为我们需要返回节点级别特征

    def call(self, inputs, NodalMask=None, training=False):
        x, adjacency_matrix = inputs
        for i in range(self.num_iterations):
            x = self.message_passing_layers[i]([x, adjacency_matrix], training=training)
        # 应用掩码
        if NodalMask is not None:
            NodalMask = tf.cast(tf.expand_dims(NodalMask, axis=-1), tf.float32)  # [batch_size, num_nodes, 1]
            x = x * NodalMask  # 将填充节点的特征置零
        return x  # 返回节点级别特征

class LinearAttention(layers.Layer):
    def __init__(self, units, **kwargs):
        super(LinearAttention, self).__init__(**kwargs)
        self.units = units
        # 线性变换层，用于投影 K 和 V
        self.proj_k = layers.Dense(units, use_bias=False, name='proj_k')
        self.proj_v = layers.Dense(units, use_bias=False, name='proj_v')
        self.proj_q = layers.Dense(units, use_bias=False, name='proj_q')
        self.output_proj = layers.Dense(units, use_bias=False, name='output_proj')
    
    def call(self, query, key, value, mask=None):
        """
        实现线性注意力机制。
        
        参数:
        - query: [batch_size, seq_len, units]
        - key: [batch_size, seq_len, units]
        - value: [batch_size, seq_len, units]
        - mask: [batch_size, seq_len, 1] 或 None
        
        返回:
        - output: [batch_size, seq_len, units]
        """
        # 投影 K, V, Q
        proj_k = self.proj_k(key)  # [batch_size, seq_len, units]
        proj_v = self.proj_v(value)  # [batch_size, seq_len, units]
        proj_q = self.proj_q(query)  # [batch_size, seq_len, units]
        
        # 应用掩码（如果提供）
        if mask is not None:
            proj_k *= mask  # [batch_size, seq_len, units]
            proj_v *= mask  # [batch_size, seq_len, units]

        # 规范化 K（可选，根据需要）
        proj_k = tf.nn.swish(proj_k)
        
        # 计算 K^T V
        # 首先将 proj_k 转置为 [batch_size, units, seq_len]
        proj_k_transposed = tf.transpose(proj_k, perm=[0, 2, 1])  # [batch_size, units, seq_len]
        kv = tf.matmul(proj_k_transposed, proj_v)  # [batch_size, units, units]
        
        # 计算 Q (K^T V)
        output = tf.matmul(proj_q, kv)  # [batch_size, seq_len, units]
        
        # 最终线性变换
        output = self.output_proj(output)  # [batch_size, seq_len, units]
        
        return output

class NodeGTransformer(tf.keras.Model):
    def __init__(self, units, num_heads, **kwargs):
        super(NodeGTransformer, self).__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.prefix = self.name
        self.dympn_q = DyMPN(units, name=f'{self.prefix}_dympn_q')
        self.dympn_k = DyMPN(units, name=f'{self.prefix}_dympn_k')
        self.dympn_v = DyMPN(units, name=f'{self.prefix}_dympn_v')
        
        # 使用线性注意力替代 MultiHeadAttention
        self.linear_attention = LinearAttention(units=units, name=f'{self.prefix}_linear_attention')
        
        self.norm = layers.BatchNormalization(name=f'{self.prefix}_batch_norm')
        self.dense_proj = layers.Dense(units, activation='swish', name=f'{self.prefix}_dense_proj')
        self.residual_conn = layers.Dense(units, name=f'{self.prefix}_residual_conn')  # 长距离残差连接
        self.linear = layers.Dense(units, name=f'{self.prefix}_linear')
        
    def call(self, inputs, NodalMask=None):
        x, adjacency_matrix = inputs
        initial_x = x  # [batch_size, max_sys_size, units]
        x = self.linear(x)  # [batch_size, max_sys_size, units]
    
        # dyMPN 层提取 queries, keys, values
        q = self.dympn_q([x, adjacency_matrix], NodalMask=NodalMask)  # [batch_size, max_sys_size, units]
        k = self.dympn_k([x, adjacency_matrix], NodalMask=NodalMask)  # [batch_size, max_sys_size, units]
        v = self.dympn_v([x, adjacency_matrix], NodalMask=NodalMask)  # [batch_size, max_sys_size, units]
    
        # 注意力机制处理
        if NodalMask is not None:
            # 将 mask 转换为形状 [batch_size, seq_len, 1] 以匹配 LinearAttention 的要求
            attention_mask = tf.expand_dims(NodalMask, axis=-1)  # [batch_size, seq_len, 1]
        else:
            attention_mask = None
    
        # 使用线性注意力
        attn_output = self.linear_attention(query=q, key=k, value=v, mask=attention_mask)  # [batch_size, max_sys_size, units]
    
        # 残差连接和 LayerNormalization
        attn_output = self.norm(attn_output + initial_x)  # [batch_size, max_sys_size, units]
        output = self.dense_proj(attn_output)  # [batch_size, max_sys_size, units]
    
        # 应用长距离残差连接
        output += self.residual_conn(initial_x)  # [batch_size, max_sys_size, units]
        if NodalMask is not None:
            NodalMask_expanded = tf.cast(tf.expand_dims(NodalMask, axis=-1), tf.float32)  # [batch_size, max_sys_size, 1]
            output = output * NodalMask_expanded  # 将填充节点的特征置零
        return output  # [batch_size, max_sys_size, units]

def pad_node_features(H, max_sys_size, units):
    padded_H = np.zeros((H.shape[0], max_sys_size, units), dtype=np.float32)
    for i in range(H.shape[0]):
        num_nodes = H[i].shape[0]
        padded_H[i, :num_nodes, :] = H[i]
    return padded_H

def pad_adjacency_matrices_sparse(A_sparse, max_sys_size):
    """
    将批量稀疏邻接矩阵填充到指定的最大系统大小。
    
    参数：
    - A_sparse: tf.sparse.SparseTensor，形状为 [batch_size, sys_size, sys_size]
    - max_sys_size: int，填充后的系统大小
    
    返回：
    - padded_A: tf.sparse.SparseTensor，形状为 [batch_size, max_sys_size, max_sys_size]
    """
    # 获取原始 dense_shape
    original_dense_shape = A_sparse.dense_shape.numpy()  # 例如 [10, 1581, 1581]
    
    # 检查 max_sys_size 是否大于等于原始系统大小
    batch_size, sys_size_1, sys_size_2 = original_dense_shape
    if max_sys_size < sys_size_1 or max_sys_size < sys_size_2:
        raise ValueError("max_sys_size 必须大于等于原始的系统大小。")
    
    # 创建新的 dense_shape
    new_dense_shape = [batch_size, max_sys_size, max_sys_size]  # 例如 [10, 2000, 2000]
    
    # 创建新的 SparseTensor，保持原有的 indices 和 values
    # 需要将索引中的 sys_size_1 和 sys_size_2 进行偏移
    # 假设填充是在右下角进行的，因此不需要偏移
    
    padded_A = tf.sparse.SparseTensor(
        indices=A_sparse.indices,
        values=A_sparse.values,
        dense_shape=new_dense_shape
    )
    
    # 重新排序稀疏张量（可选，但推荐）
    padded_A = tf.sparse.reorder(padded_A)
    
    return padded_A

def create_NodalMask(H, max_sys_size):
    NodalMask = np.zeros((H.shape[0], max_sys_size), dtype=np.float32)
    for i in range(H.shape[0]):
        num_nodes = np.count_nonzero(np.sum(H[i], axis=-1))
        NodalMask[i, :num_nodes] = 1.0
    return NodalMask

def create_SSSGNN(max_sys_size, sys_size, units=6, num_heads=8, d_model=24, blockNum=4, mlpNeuron=24000):
    # 定义节点特征输入
    hin = Input(shape=(max_sys_size, units), dtype=tf.float32, name='input1')
    # 定义掩码输入
    NodalMask = Input(shape=(max_sys_size,), dtype=tf.float32, name='nodalMask')
    # 定义邻接矩阵输入
    A_sparse_input = Input(shape=(max_sys_size, max_sys_size), dtype=tf.float32, sparse=True, name='input2')
    A_sparse = layers.Lambda(lambda x: tf.stop_gradient(x))(A_sparse_input)

    h_embeded = layers.Dense(d_model, activation='swish', name='embedding_layer')(hin)

    # 使用 NodeGTransformer 处理节点特征和邻接矩阵
    h_0 = NodeGTransformer(units=d_model, num_heads=num_heads)([h_embeded, A_sparse], NodalMask=NodalMask)
    if blockNum > 1:
        for block_iter in range(blockNum-1):
            h_0 = NodeGTransformer(
                units=d_model, 
                num_heads=num_heads,
                name=f"gtr_block{block_iter}"
                )([h_0, A_sparse], NodalMask=NodalMask)

    h_deEmbeded = layers.Dense(units, activation='swish', name='deEmbedding_layer')(h_0)

    # 将初始特征和处理后的特征进行拼接
    h_mlp_in = Concatenate()([h_deEmbeded, hin])
    # 应用掩码，将填充节点的特征置零
    h_mlp_in = Lambda(lambda x: x[0] * tf.expand_dims(x[1], axis=-1))([h_mlp_in, NodalMask])

    # 展平并通过全连接层
    h_mlp_in = Reshape((max_sys_size * units * 2,))(h_mlp_in)  # [batch_size, max_sys_size * units * 2]

    h_mlp_hidden = Dense(mlpNeuron, activation='swish', name='dense1')(h_mlp_in)
    h_mlp_hidden = Dense(mlpNeuron, activation='swish', name='dense2')(h_mlp_hidden)
    h_mlp_out = Dense(max_sys_size * units, name='dense3')(h_mlp_hidden)
    h_mlp_out = Reshape((max_sys_size, units))(h_mlp_out)  # [batch_size, max_sys_size, units]

    # 创建和编译模型，仅输出预测值
    model = Model([hin, A_sparse_input, NodalMask], h_mlp_out)

    return model

def createMLPs(sys_size, units=6, num_heads=8, d_model=768, blockNum=12, mlpNeuron=128):
    hin = Input(shape=(sys_size, units), dtype=tf.float32, name='input1')
    h_mlp_in = Concatenate()([ hin])
    h_mlp_in = Reshape((sys_size * units ,))(h_mlp_in)

    h_mlp_hidden = Dense(mlpNeuron, activation='swish', name='dense1')(h_mlp_in)
    h_mlp_hidden = Dense(mlpNeuron, activation='swish', name='dense2')(h_mlp_hidden)
    h_mlp_out = Dense(sys_size * units, name='dense3')(h_mlp_hidden)
    h_mlp_out = Reshape((sys_size, units))(h_mlp_out)

    # 创建和编译模型
    model = Model([hin], h_mlp_out)
    model.compile(optimizer='adam', loss='mse', metrics=['mean_absolute_error'])

    return model

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
    # 获取掩码：mask = 1.0 如果 y_true 不为0，否则为0.0
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

# In[]
if __name__ == "__main__":
    datapath = r"/home/user/storage"
    dataset_name = 'pre-trainingDataset'

    max_sys_size = 20000 
    sys_size = 1581

    total_sample_num = 4096
    sample_for_each_iter = total_sample_num

    # 加载数据
    H = load_H(start_label=0, end_label=total_sample_num, dataset=dataset_name, path=datapath, sys_size=sys_size, datatype=None, sample_for_each_iter=sample_for_each_iter)
    H_norm, H_mean_train, H_std_train = norm_H(H)
    A_sparse = load_A_sparse(start_label=0, end_label=total_sample_num, dataset=dataset_name, path=datapath, sys_size=sys_size, datatype=None, sample_for_each_iter=sample_for_each_iter)

    print("A_sparse shape:", A_sparse.dense_shape) 
    print("H_norm shape:", H_norm.shape)          
    print("A_sparse type:", type(A_sparse))        # 应为 tf.sparse.SparseTensor

    # 填充节点特征和邻接矩阵
    padded_H = pad_node_features(H_norm, max_sys_size, units=6)
    padded_A = pad_adjacency_matrices_sparse(A_sparse, max_sys_size)

    # 创建掩码
    NodalMasks = create_NodalMask(padded_H, max_sys_size)  # [batch_size, max_sys_size]

    # 应用掩码到 y_true
    y_true_masked = padded_H * NodalMasks[..., np.newaxis]  # [batch_size, max_sys_size, units]

    # 创建模型
    SSGNN = create_SSSGNN(
        max_sys_size=max_sys_size, 
        sys_size=sys_size, 
        units=6, 
        num_heads=8, 
        d_model=24,
        blockNum=4,
        mlpNeuron=int(sys_size/6)
    )

    # 编译模型，使用自定义损失函数
    SSGNN.compile(optimizer='adam', loss=masked_mse_loss, metrics=['mean_absolute_error'])

    # 验证模型结构
    # SSGNN.summary()

    # 开始训练，传递合并后的 y_true
    history = SSGNN.fit(
         [padded_H, padded_A, NodalMasks],
         y_true_masked,
         epochs=50,
         batch_size=32
     ) 

    # mlps.fit(H_norm, H_norm, epochs=50)
    # H_pre_rebuild = mlps.predict(H_norm) * H_std_train + H_mean_train
# %%
