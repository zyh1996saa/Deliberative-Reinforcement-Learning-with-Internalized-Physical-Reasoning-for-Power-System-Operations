# In[]
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K

# --------------------------------------
# 掩码 MSE（保持不变）
# --------------------------------------
def masked_mse_loss(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, 0.0), tf.float32)
    mse = tf.square(y_true - y_pred)
    mse_per_node = tf.reduce_mean(mse, axis=-1)
    node_mask = tf.reduce_max(mask, axis=-1)
    masked_mse = mse_per_node * node_mask
    return tf.reduce_sum(masked_mse) / (tf.reduce_sum(node_mask) + 1e-8)

# --------------------------------------
# 动态消息传递层（简化版，避免tf.while_loop问题）
# --------------------------------------
class RandomizedMessagePassingLayer(layers.Layer):
    def __init__(self, units, aggregate_mode='graph_conv', **kwargs):
        super(RandomizedMessagePassingLayer, self).__init__(**kwargs)
        self.units = units
        self.aggregate_mode = aggregate_mode
        self.W = self.add_weight(shape=(units, units), initializer='random_normal', trainable=True, name='W')
        self.b = self.add_weight(shape=(units,), initializer='zeros', trainable=True, name='b')

    def call(self, inputs, training=False):
        x, adjacency_matrix = inputs
        adjacency_matrix = tf.stop_gradient(adjacency_matrix)
        
        # 简化：始终使用2步传播，避免动态循环问题
        # 将稀疏矩阵转换为稠密矩阵
        A_dense = tf.sparse.to_dense(adjacency_matrix)
        A_dense = tf.cast(A_dense, x.dtype)
        
        # 第一步传播
        neigh1 = tf.matmul(A_dense, x)
        x1 = tf.nn.swish(tf.matmul(neigh1, self.W) + self.b)
        
        # 第二步传播
        neigh2 = tf.matmul(A_dense, x1)
        x2 = tf.nn.swish(tf.matmul(neigh2, self.W) + self.b)
        
        return x2

class DyMPN(layers.Layer):
    def __init__(self, units, num_iterations=3, **kwargs):
        super(DyMPN, self).__init__(**kwargs)
        self.num_iterations = num_iterations
        self.mps = [RandomizedMessagePassingLayer(units) for _ in range(num_iterations)]

    def call(self, inputs, NodalMask=None, training=False):
        x, A = inputs
        for layer in self.mps:
            x = layer([x, A], training=training)
        if NodalMask is not None:
            mask = tf.cast(tf.expand_dims(NodalMask, -1), tf.float32)
            x = x * mask
        return x

# --------------------------------------
# 路由令牌机制 - 修正版
# --------------------------------------
class RoutingTokens(layers.Layer):
    def __init__(self, num_blocks, num_routing_tokens, d_model, **kwargs):
        super(RoutingTokens, self).__init__(**kwargs)
        self.num_blocks = num_blocks
        self.num_routing_tokens = num_routing_tokens
        self.d_model = d_model
        
        # 可学习的路由令牌初始化 - 修正维度
        self.routing_tokens = self.add_weight(
            shape=(1, num_blocks, num_routing_tokens, d_model),  # 添加批次维度
            initializer='random_normal',
            trainable=True,
            name='routing_tokens'
        )
        
    def call(self, batch_size):
        # 扩展路由令牌到批次维度 [B, G, R, D]
        routing_tokens = tf.tile(self.routing_tokens, [batch_size, 1, 1, 1])
        return routing_tokens

class CrossBlockCommunication(layers.Layer):
    def __init__(self, d_model, num_heads, num_blocks, block_connectivity=None, **kwargs):
        super(CrossBlockCommunication, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_blocks = int(num_blocks)
        self.block_connectivity = block_connectivity  # 可为 None 或 [G,G] 0/1

        # 只在“token维”做注意力
        self.comm_attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            attention_axes=(1,),                # [B, T, D] 的 T 维
            name='comm_attention'
        )
        self.read_attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            attention_axes=(2,),                # [B, G, R, D] 的 R 维
            name='read_attention'
        )
        self.write_attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            attention_axes=(2,),                # [B, G, Ng, D] 的 Ng 维
            name='write_attention'
        )

        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.norm3 = layers.LayerNormalization()

    def build(self, input_shape):
        # 若未显式给出块间连通性，默认全连通（也可以按电网物理邻接构造）
        if self.block_connectivity is None:
            G = self.num_blocks
            self.block_connectivity = tf.ones([G, G], dtype=tf.float32)
        else:
            self.block_connectivity = tf.cast(self.block_connectivity, tf.float32)
        super(CrossBlockCommunication, self).build(input_shape)

    def call(self, block_features, routing_tokens, training=False):
        """
        block_features: [B, G, Ng_max, D]
        routing_tokens: [B, G, R, D]
        return:         [B, G, Ng_max, D]
        """
        B = tf.shape(block_features)[0]
        G = tf.shape(block_features)[1]
        R = tf.shape(routing_tokens)[2]

        # === 1) Read ===
        # 让每个块的 R 个路由 token 只在该块内部（Ng 维）做注意力
        read_routing = self.read_attention(
            query=routing_tokens,
            key=block_features,
            value=block_features,
            training=training
        )  # [B, G, R, D]
        read_routing = self.norm1(read_routing + routing_tokens)

        # === 2) Communicate ===
        # 将 [B, G, R, D] 合并为 [B, G*R, D]
        comm_qkv = tf.reshape(read_routing, [B, -1, self.d_model])  # [B, GR, D]

        # 注意力 mask 形状应为 [B, Tq, Tk] 或 [Tq, Tk]
        # 这里构造 [Tq, Tk]，会对 batch 广播
        comm_mask_2d = self._create_block_mask_2d(R)  # [GR, GR] bool

        comm_out = self.comm_attention(
            query=comm_qkv, value=comm_qkv, key=comm_qkv,
            attention_mask=comm_mask_2d,  # 2D -> 广播到各 batch
            training=training
        )  # [B, GR, D]

        # 残差 + 归一化
        comm_out = self.norm2(comm_out + comm_qkv)
        comm_out = tf.reshape(comm_out, [B, G, R, self.d_model])  # [B, G, R, D]

        # === 3) Write ===
        write_updates = self.write_attention(
            query=block_features,   # [B, G, Ng_max, D]
            key=comm_out,           # [B, G, R, D]
            value=comm_out,
            training=training
        )  # [B, G, Ng_max, D]

        updated = self.norm3(write_updates + block_features)
        return updated

    def _create_block_mask_2d(self, R):
        """
        从 [G, G] 的块连通矩阵构造 [G*R, G*R] 的令牌级掩码（布尔）。
        True 表示允许注意，False 表示屏蔽。
        """
        conn = tf.cast(self.block_connectivity > 0, tf.bool)     # [G, G]
        conn = tf.repeat(conn, repeats=R, axis=0)                # [G*R, G]
        conn = tf.repeat(conn, repeats=R, axis=1)                # [G*R, G*R]
        return conn  # 2D bool，MHA 会广播到 [B, Tq, Tk]


# --------------------------------------
# 分块稀疏注意力（修正版）
# --------------------------------------
class BlockSparseSelfAttention(layers.Layer):
    def __init__(self, units, num_heads=8, **kwargs):
        super(BlockSparseSelfAttention, self).__init__(**kwargs)
        assert units % num_heads == 0, "units 必须能被 num_heads 整除"
        self.units = units
        self.num_heads = num_heads
        self.head_dim = units // num_heads
        
        self.Wq = Dense(units, use_bias=False)
        self.Wk = Dense(units, use_bias=False)
        self.Wv = Dense(units, use_bias=False)
        self.Wo = Dense(units, use_bias=False)

    def call(self, x, blockB, NodalMask=None):
        # x:[B,N,D], blockB: [N,G] 或 [B,N,G]
        Bsz = tf.shape(x)[0]
        N = tf.shape(x)[1]
        
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)
        
        def split_heads(t):
            t = tf.reshape(t, [Bsz, N, self.num_heads, self.head_dim])
            return tf.transpose(t, [0, 2, 1, 3])  # [B,H,N,d]
            
        qh, kh, vh = split_heads(q), split_heads(k), split_heads(v)
        
        scores = tf.matmul(qh, kh, transpose_b=True) / tf.math.sqrt(tf.cast(self.head_dim, tf.float32))  # [B,H,N,N]
        
        # -------- Block mask（同块=1，跨块=-inf）---------
        if tf.rank(blockB) == 2:  # [N,G]
            block_same = tf.matmul(blockB, blockB, transpose_b=True)  # [N,N]
            block_same = tf.reshape(block_same, [1, 1, N, N])
        else:  # [B,N,G]
            block_same = tf.matmul(blockB, blockB, transpose_b=True)  # [B,N,N]
            block_same = tf.expand_dims(block_same, axis=1)  # [B,1,N,N]
            
        block_bias = (1.0 - block_same) * (-1e9)  # 同块=0, 跨块=-1e9
        scores = scores + block_bias
        
        # -------- key 侧节点有效性掩码 --------
        if NodalMask is not None:
            key_mask = tf.cast(tf.reshape(NodalMask, [Bsz, 1, 1, N]), tf.float32)
            scores = scores + (1.0 - key_mask) * (-1e9)
            
        weights = tf.nn.softmax(scores, axis=-1)
        out = tf.matmul(weights, vh)  # [B,H,N,d]
        out = tf.transpose(out, [0, 2, 1, 3])  # [B,N,H,d]
        out = tf.reshape(out, [Bsz, N, self.units])  # [B,N,D]
        
        return self.Wo(out)

# --------------------------------------
# 简化的块分组逻辑
# --------------------------------------


class BlockGrouper(layers.Layer):
    def __init__(self, num_blocks, **kwargs):
        super(BlockGrouper, self).__init__(**kwargs)
        self.num_blocks = num_blocks

    def call(self, x, block_onehot):
        """
        x: [B, N, D]
        block_onehot: [B, N, G] 或 [N, G]
        return: [B, G, Ng_max, D]
        """
        batch_size = tf.shape(x)[0]
        N = tf.shape(x)[1]
        D = x.shape[-1] or tf.shape(x)[-1]  # 优先用静态D

        # —— 统一保证 block_onehot 为 rank=3: [B', N, G] ——
        def add_batch_dim():
            return tf.expand_dims(block_onehot, 0)  # [1, N, G]
        def keep_as_is():
            return block_onehot                      # [B, N, G]
        block_onehot = tf.cond(
            tf.equal(tf.rank(block_onehot), 2),
            add_batch_dim,
            keep_as_is
        )  # 现在一定是 rank=3

        # 如有需要，把 [1, N, G] tile 成 [B, N, G]
        cur_b = tf.shape(block_onehot)[0]
        def tile_to_batch():
            multiples = tf.stack([batch_size, tf.constant(1, tf.int32), tf.constant(1, tf.int32)])  # [B,1,1]
            return tf.tile(block_onehot, multiples)
        def no_tile():
            return block_onehot
        block_onehot = tf.cond(tf.equal(cur_b, 1), tile_to_batch, no_tile)  # [B, N, G]

        # 每个块的大小，并取 batch 内最大块长度
        block_sizes = tf.reduce_sum(tf.cast(block_onehot > 0, tf.int32), axis=1)  # [B, G]
        max_block_size = tf.reduce_max(block_sizes)  # 标量

        # 为每个块收集/填充特征
        block_features_list = []
        for g in range(self.num_blocks):
            block_mask = block_onehot[:, :, g]                 # [B, N]
            block_mask_exp = tf.expand_dims(block_mask, -1)    # [B, N, 1]
            masked = x * block_mask_exp                        # [B, N, D]

            def collect_one(i):
                sample_features = masked[i]        # [N, D]
                sample_mask = block_mask[i]        # [N]
                idx = tf.where(sample_mask > 0)    # [Ng, 1]
                feat = tf.gather_nd(sample_features, idx)  # [Ng, D]
                pad_len = max_block_size - tf.shape(feat)[0]
                padding = tf.zeros([pad_len, D], dtype=feat.dtype)
                return tf.concat([feat, padding], axis=0)      # [max_block_size, D]

            # 注意：fn_output_signature 的维度用 None 或静态 D，避免把张量塞到 TensorSpec 里
            block_padded = tf.map_fn(
                collect_one,
                tf.range(batch_size),
                fn_output_signature=tf.TensorSpec([None, D], dtype=x.dtype)
            )  # [B, max_block_size, D]
            # 为了让 shape 更明确（便于后续层推理），显式设置第二维：
            block_padded = tf.ensure_shape(block_padded, [None, None, D])

            block_features_list.append(block_padded)

        # 堆叠所有块 → [B, G, Ng_max, D]
        block_features = tf.stack(block_features_list, axis=1)
        return block_features

class BlockUngrouper(layers.Layer):
    def __init__(self, num_blocks, **kwargs):
        super(BlockUngrouper, self).__init__(**kwargs)
        self.num_blocks = num_blocks

    def call(self, block_features, block_onehot, output_shape):
        """
        block_features: [B, G, Ng_max, D]
        block_onehot:  [B, N, G] 或 [N, G]
        output_shape:  [B, N, D]（list/tuple/tensor，B/N/D 仅作尺寸提示）
        return: [B, N, D]
        """
        B = tf.shape(block_features)[0]
        G = tf.shape(block_features)[1]
        N = output_shape[1]
        D = block_features.shape[-1] or tf.shape(block_features)[-1]

        # 将 block_onehot 统一成 [B, N, G]
        def add_batch_dim():
            return tf.expand_dims(block_onehot, 0)  # [1, N, G]
        def keep_as_is():
            return block_onehot                      # [B, N, G]
        block_onehot = tf.cond(
            tf.equal(tf.rank(block_onehot), 2),
            add_batch_dim,
            keep_as_is
        )
        cur_b = tf.shape(block_onehot)[0]
        def tile_to_batch():
            multiples = tf.stack([B, tf.constant(1, tf.int32), tf.constant(1, tf.int32)])
            return tf.tile(block_onehot, multiples)
        def no_tile():
            return block_onehot
        block_onehot = tf.cond(tf.equal(cur_b, B), no_tile, tile_to_batch)  # [B, N, G]

        # 初始化输出
        out = tf.zeros([B, N, D], dtype=block_features.dtype)

        for g in range(self.num_blocks):
            block_mask = block_onehot[:, :, g]  # [B, N]

            def scatter_one(i):
                sample_out  = out[i]                      # [N, D]
                sample_feat = block_features[i, g]        # [Ng_max, D]
                sample_mask = block_mask[i]               # [N]
                idx = tf.where(sample_mask > 0)           # [Ng, 1]
                Ng = tf.shape(idx)[0]
                updates = sample_feat[:Ng]                # [Ng, D]
                return tf.tensor_scatter_nd_update(sample_out, idx, updates)

            out = tf.map_fn(
                scatter_one,
                tf.range(B),
                fn_output_signature=tf.TensorSpec([None, D], dtype=block_features.dtype)
            )
            out = tf.ensure_shape(out, [None, None, D])

        return out

# --------------------------------------
# 修正的结点级 Graph Transformer
# --------------------------------------
class NodeGTransformerBlocks(tf.keras.Model):
    def __init__(self, units, num_heads, num_blocks, num_routing_tokens=4, 
                 use_cross_block_comm=True, **kwargs):
        super(NodeGTransformerBlocks, self).__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.num_routing_tokens = num_routing_tokens
        self.use_cross_block_comm = use_cross_block_comm
        
        # 基础变换
        self.linear = Dense(units, name='pre_linear')
        self.dympn_q = DyMPN(units, name='dympn_q')
        self.dympn_k = DyMPN(units, name='dympn_k')
        self.dympn_v = DyMPN(units, name='dympn_v')
        
        # 块内注意力
        self.block_attn = BlockSparseSelfAttention(units, num_heads=num_heads, name='block_attn')
        
        # 跨块通讯组件
        if self.use_cross_block_comm:
            self.routing_tokens_layer = RoutingTokens(
                num_blocks=num_blocks,
                num_routing_tokens=num_routing_tokens,
                d_model=units,
                name='routing_tokens'
            )
            self.cross_block_comm = CrossBlockCommunication(
                d_model=units,
                num_heads=num_heads,
                num_blocks=self.num_blocks,    # << 新增
                name='cross_block_comm'
            )
            self.block_grouper = BlockGrouper(num_blocks=num_blocks, name='block_grouper')
            self.block_ungrouper = BlockUngrouper(num_blocks=num_blocks, name='block_ungrouper')
        
        # 归一化和投影
        self.norm1 = layers.LayerNormalization(name='post_attn_norm')
        self.norm2 = layers.LayerNormalization(name='post_comm_norm') if use_cross_block_comm else None
        self.proj = Dense(units, activation='swish', name='post_proj')
        self.res_long = Dense(units, name='long_res')
        
    def call(self, inputs, BlockOneHot=None, NodalMask=None, training=False):
        x, A = inputs  # x:[B,N,D], A: SparseTensor
        batch_size = tf.shape(x)[0]
        N = tf.shape(x)[1]
        
        x0 = self.linear(x)
        
        # === 块内注意力 ===
        q = self.dympn_q([x0, A], NodalMask=NodalMask, training=training)
        k = self.dympn_k([x0, A], NodalMask=NodalMask, training=training)
        v = self.dympn_v([x0, A], NodalMask=NodalMask, training=training)
        
        attn_out = self.block_attn(q, BlockOneHot, NodalMask=NodalMask)
        attn_out = self.norm1(attn_out + x0)
        
        # === 跨块通讯 ===
        if self.use_cross_block_comm and BlockOneHot is not None:
            # 将节点特征按块分组
            block_features = self.block_grouper(attn_out, BlockOneHot)  # [B, G, N_g, D]
            
            # 获取路由令牌
            routing_tokens = self.routing_tokens_layer(batch_size)  # [B, G, R, D]
            
            # 跨块通讯
            comm_out = self.cross_block_comm(block_features, routing_tokens, training=training)
            
            # 将块特征重新组合回节点序列
            comm_out_flat = self.block_ungrouper(comm_out, BlockOneHot, [batch_size, N, self.units])
            attn_out = self.norm2(comm_out_flat + attn_out) if self.norm2 else comm_out_flat
        
        # === 最终投影和残差 ===
        output = self.proj(attn_out) + self.res_long(x0)
        
        if NodalMask is not None:
            mask = tf.cast(tf.expand_dims(NodalMask, -1), tf.float32)
            output = output * mask
            
        return output

# --------------------------------------
# 创建分块版多解码模型
# --------------------------------------
def create_SSSGNN_multi_decoder_with_blocks(
    max_sys_size, num_blocks, units=6, num_heads=8, d_model=48, 
    blockNum=2, mlpNeuron=512, num_outputs=6, num_routing_tokens=4,
    use_cross_block_comm=True):
    
    hin = Input(shape=(max_sys_size, units), dtype=tf.float32, name='input_features')  # [B,N,6]
    NodalMask = Input(shape=(max_sys_size,), dtype=tf.float32, name='nodalMask')  # [B,N]
    A_sparse_in = Input(shape=(max_sys_size, max_sys_size), dtype=tf.float32, sparse=True, name='adj_sparse')
    A_sparse = Lambda(lambda x: tf.stop_gradient(x))(A_sparse_in)
    
    # 允许 [N,G] 或 [B,N,G] 两种输入
    BlockOneHot = Input(shape=(max_sys_size, num_blocks), dtype=tf.float32, name='blockOneHot')  # [B,N,G]
    
    # 嵌入层
    h = Dense(d_model, activation='relu', name='embedding')(hin)
    h = BatchNormalization(name='embedding_bn')(h)
    
    # Graph Transformer 块（包含跨块通讯）
    gt = NodeGTransformerBlocks(
        units=d_model, 
        num_heads=num_heads, 
        num_blocks=num_blocks,
        num_routing_tokens=num_routing_tokens,
        use_cross_block_comm=use_cross_block_comm,
        name='gtr_block_0'
    )([h, A_sparse], BlockOneHot=BlockOneHot, NodalMask=NodalMask)
    
    for bi in range(1, blockNum):
        gt = NodeGTransformerBlocks(
            units=d_model, 
            num_heads=num_heads, 
            num_blocks=num_blocks,
            num_routing_tokens=num_routing_tokens,
            use_cross_block_comm=use_cross_block_comm,
            name=f'gtr_block_{bi}'
        )([gt, A_sparse], BlockOneHot=BlockOneHot, NodalMask=NodalMask)
    
    # 解码器部分
    de = Dense(units, activation='relu', name='deEmbedding')(gt)
    de = BatchNormalization(name='deEmbedding_bn')(de)
    
    h_concat = Concatenate()([de, hin])
    h_masked = Lambda(lambda x: x[0] * tf.expand_dims(x[1], -1))([h_concat, NodalMask])
    h_flat = Flatten()(h_masked)
    
    # 多输出解码器
    outs = []
    for i in range(num_outputs):
        t = Dense(mlpNeuron, activation='relu', name=f'decoder_{i}_fc1')(h_flat)
        t = Dense(mlpNeuron, activation='relu', name=f'decoder_{i}_fc2')(t)
        t = Dense(max_sys_size * 1, name=f'decoder_{i}_out')(t)
        t = Reshape((max_sys_size, 1), name=f'dec_{i}')(t)
        outs.append(t)
    
    model = Model([hin, A_sparse_in, NodalMask, BlockOneHot], outs)
    return model