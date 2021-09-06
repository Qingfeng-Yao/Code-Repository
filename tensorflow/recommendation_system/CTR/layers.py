import math
import numpy as np

import tensorflow as tf
from tensorflow.contrib import layers
from config import *
from utils import _my_top_k, _rowwise_unsorted_segment_sum, _prob_in_top_k, _gates_to_load, cv_squared, smooth_step, _add_entropy_regularization_loss

def stack_dense_layer(inputs, hidden_units, dropout_rate, batch_norm, mode, scope='dense'):
    with tf.compat.v1.variable_scope(scope):
        for i, unit in enumerate(hidden_units):
            if i == 0:
                outputs = tf.layers.dense(inputs, units = unit, activation = 'relu',
                                        name = 'dense{}'.format(i))
            else:
                outputs = tf.layers.dense(outputs, units = unit, activation = 'relu',
                                        name = 'dense{}'.format(i))

                if batch_norm:
                    outputs = tf.layers.batch_normalization(outputs, center = True, scale = True,
                                                        trainable = True,
                                                        training = (mode == tf.estimator.ModeKeys.TRAIN))
                if dropout_rate > 0:
                    outputs = tf.layers.dropout(outputs, rate = dropout_rate,
                                            training = (mode == tf.estimator.ModeKeys.TRAIN))

        if inputs.get_shape().as_list()[-1] == outputs.get_shape().as_list()[-1]:
            outputs += inputs
    return outputs

def attention(queries, keys, params, keys_id=None, queries_id=None, values=None, scope='multihead_attention'):
    with tf.compat.v1.variable_scope(scope):
        query_len = tf.shape(queries)[1]  
        key_len = tf.shape(keys)[1] 

        queries_2d = tf.reshape(queries, [-1, queries.get_shape().as_list()[-1]])
        keys_2d = tf.reshape(keys, [-1, keys.get_shape().as_list()[-1]])
        if values is not None:
            value_len = tf.shape(values)[1] 
            values_2d = tf.reshape(values, [-1, values.get_shape().as_list()[-1]])
        Q = tf.layers.dense(queries_2d, params['attention_hidden_unit'], activation = tf.nn.relu, name = 'attention_Q')  
        Q = tf.reshape(Q, [-1, tf.shape(queries)[1], Q.get_shape().as_list()[-1]])
        K = tf.layers.dense(keys_2d, params['attention_hidden_unit'], activation = tf.nn.relu, name = 'attention_K')  
        K = tf.reshape(K, [-1, tf.shape(keys)[1], K.get_shape().as_list()[-1]])
        if values is not None:
            V = tf.layers.dense(values_2d, params['emb_dim'], activation = tf.nn.relu, name = 'attention_V')  
            V = tf.reshape(V, [-1, tf.shape(values)[1], V.get_shape().as_list()[-1]])
        else:
            V = tf.layers.dense(keys_2d, params['emb_dim'], activation = tf.nn.relu, name = 'attention_V')  
            V = tf.reshape(V, [-1, tf.shape(keys)[1], V.get_shape().as_list()[-1]])

        if params['num_heads'] > 1:
            Q_ = tf.concat(tf.split(Q, params['num_heads'], axis=2), axis=0)  
            K_ = tf.concat(tf.split(K, params['num_heads'], axis=2), axis=0) 
            V_ = tf.concat(tf.split(V, params['num_heads'], axis=2), axis=0)
        else:
            Q_ = Q
            K_ = K
            V_ = V
        if params['atten_mode'] == 'ln':
            # Layer Norm
            Q_ = layers.layer_norm(Q_, begin_norm_axis=-1, begin_params_axis=-1)
            K_ = layers.layer_norm(K_, begin_norm_axis=-1, begin_params_axis=-1)
            # Multiplication
            outputs = tf.matmul(Q_, K_, transpose_b=True)  
            # Scale
            outputs = outputs * (K_.get_shape().as_list()[-1] ** (-0.5))

        # key Masking
        if keys_id is not None:
            key_masks = tf.not_equal( keys_id, 0 )
            key_masks = tf.tile(tf.reshape(key_masks, [-1, 1, key_len]), [params['num_heads'], query_len, 1])
            paddings = tf.fill(tf.shape(outputs), tf.constant(-2 ** 32 + 1, dtype=tf.float32))
            outputs = tf.where(key_masks, outputs, paddings)

        # Activation
        outputs = tf.nn.softmax(outputs)
        
        if queries_id is not None:
            # Query Masking
            query_masks = tf.not_equal( queries_id, 0 )
            query_masks = tf.tile(tf.reshape(query_masks, [-1, query_len]), [params['num_heads'], 1])  
            outputs = tf.reshape(outputs, [-1, key_len])  
            paddings = tf.zeros_like(outputs, dtype=tf.float32)  
            outputs = tf.where(tf.reshape(query_masks, [-1]), outputs, paddings)  
            outputs = tf.reshape(outputs, [-1, query_len, key_len])  

        # Attention vector
        att_vec = outputs

        # Weighted sum
        outputs = tf.matmul(outputs, V_) 

        # Restore shape
        if params['num_heads'] > 1:
            outputs = tf.concat(tf.split(outputs, params['num_heads'], axis=0), axis=2)

    return outputs

def noisy_top_k_gating(x, params, mode, noisy_gating=True, noise_epsilon=1e-2, name='noisy_top_k_gating'):
    with tf.compat.v1.variable_scope(name):
        input_size = x.get_shape().as_list()[-1]
        w_gate = tf.compat.v1.get_variable('w_gate', [input_size, params['num_of_expert']], tf.float32, initializer=tf.zeros_initializer())
        if noisy_gating:
            w_noise = tf.compat.v1.get_variable("w_noise", [input_size, params['num_of_expert']], tf.float32, initializer=tf.zeros_initializer())
        clean_logits = tf.matmul(x, w_gate)
        if noisy_gating:
            raw_noise_stddev = tf.matmul(x, w_noise)
            if mode == tf.estimator.ModeKeys.TRAIN: # 只在训练期间加噪声
                noise_stddev = tf.nn.softplus(raw_noise_stddev) + noise_epsilon
            else:
                noise_stddev = 0
            noisy_logits = clean_logits + (tf.random_normal(tf.shape(clean_logits)) * noise_stddev)
            logits = noisy_logits
        else:
          logits = clean_logits
        top_logits, top_indices = _my_top_k(logits, min(params['k'] + 1, params['num_of_expert']))
        # top k logits has shape [batch, k]
        top_k_logits = tf.slice(top_logits, [0, 0], [-1, params['k']])
        top_k_indices = tf.slice(top_indices, [0, 0], [-1, params['k']])
        top_k_gates = tf.nn.softmax(top_k_logits)
        # This will be a `Tensor` of shape `[batch_size, n]`, with zeros in the
        # positions corresponding to all but the top k experts per example.
        gates = _rowwise_unsorted_segment_sum(top_k_gates, top_k_indices, params['num_of_expert'])

        if noisy_gating and params['k'] < params['num_of_expert']:
            load = tf.reduce_sum(_prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits, params), 0)
        else:
            load = _gates_to_load(gates)

        return gates, load

def compute_expert_weights_with_dselect(params, name='dselect_gating'):
    with tf.compat.v1.variable_scope(name):
        # z_logits is a trainable 3D tensor used for selecting the experts.
        # Axis 0: Number of non-zero experts to select.
        # Axis 1: Dummy axis of length 1 used for broadcasting.
        # Axis 2: Each num_binary-dimensional row corresponds to a "single-expert"
        # selector.
        num_binary = math.ceil(math.log2(params['num_of_expert']))
        z_logits = tf.compat.v1.get_variable('z_logits', [params['k'], 1, num_binary], initializer=tf.keras.initializers.RandomUniform(
        -1.0 / 100, 1.0 / 100), trainable=True)
        z_logits = tf.tile(z_logits, [1, params['num_of_expert'], 1])
        # w_logits is a trainable tensor used to assign weights to the
        # single-expert selectors. Each element of w_logits is a logit.
        w_logits = tf.compat.v1.get_variable('w_logits', [params['k'], 1], initializer=tf.keras.initializers.RandomUniform(), trainable=True)
        # binary_matrix is a (num_experts, num_binary)-matrix used for binary
        # encoding. The i-th row contains a num_binary-digit binary encoding of the
        # integer i.
        binary_matrix = np.array([list(np.binary_repr(val, width=num_binary))
            for val in range(params['num_of_expert'])]).astype(bool)
        # A constant tensor = binary_matrix, with an additional dimension for
        # broadcasting.
        binary_codes = tf.tile(tf.expand_dims(
            tf.constant(binary_matrix, dtype=bool), axis=0), [params['k'], 1, 1])

        # Shape = (k, num_experts, num_binary).
        smooth_step_activations = smooth_step(z_logits)
        # Shape = (k, num_experts).
        selector_outputs = tf.math.reduce_prod(
            tf.where(binary_codes, smooth_step_activations,
                    1 - smooth_step_activations), axis=2)
        # Weights for the single-expert selectors: shape = (k, 1).
        selector_weights = tf.nn.softmax(w_logits, axis=0)
        expert_weights = tf.math.reduce_sum(
            selector_weights * selector_outputs, axis=0)
    return expert_weights, selector_outputs

def compute_example_conditioned_expert_weights_with_dselect(dense, params, name='dselect_example_conditioned_gating'):
    with tf.compat.v1.variable_scope(name):
        num_binary = math.ceil(math.log2(params['num_of_expert']))
        z_logits = tf.keras.layers.Dense(params['k'] * num_binary,
            kernel_initializer=tf.keras.initializers.RandomUniform(-1.0 / 100, 1.0 / 100),
            bias_initializer=tf.keras.initializers.RandomUniform(-1.0 / 100, 1.0 / 100))
        w_logits = tf.keras.layers.Dense(params['k'],
            kernel_initializer=tf.keras.initializers.RandomUniform(),
            bias_initializer=tf.keras.initializers.RandomUniform())
        binary_matrix = np.array([list(np.binary_repr(val, width=num_binary))
            for val in range(params['num_of_expert'])]).astype(bool)
        # A constant tensor = binary_matrix, with an additional dimension for
        # broadcasting.
        binary_codes = tf.tile(tf.expand_dims(tf.expand_dims(
            tf.constant(binary_matrix, dtype=bool), axis=0), axis=0), [tf.shape(dense)[0], params['k'], 1, 1])
        
        sample_logits = tf.reshape(
            z_logits(dense),
            [-1, params['k'], 1, num_binary])
        sample_logits = tf.tile(sample_logits, [1, 1, params['num_of_expert'], 1])
        smooth_step_activations = smooth_step(sample_logits)
        # Shape = (batch_size, k, num_experts).
        selector_outputs = tf.math.reduce_prod(
            tf.where(
                binary_codes, smooth_step_activations,
                1 - smooth_step_activations), 3)
        # Weights for the single-expert selectors.
        # Shape = (batch_size, k, 1).
        selector_weights = tf.expand_dims(w_logits(dense), 2)
        selector_weights = tf.nn.softmax(selector_weights, axis=1)
        # Sum over the single-expert selectors. Shape = (batch_size, num_experts).
        expert_weights = tf.math.reduce_sum(
            selector_weights * selector_outputs, axis=1)

    return expert_weights, selector_outputs

# mean pooling, max pooling
def seq_pooling_layer(features, params, emb_dict, mode):
    for s in params['seq_names']:
        hist_name = 'hist_{}_list'.format(s)
        with tf.compat.v1.variable_scope('Seq_Pooling_Layer_{}'.format(s)):
            sequence = emb_dict['{}_hist_emb'.format(s)]
            sequence = stack_dense_layer(sequence, [sequence.get_shape().as_list()[-1], sequence.get_shape().as_list()[-1]], params['dropout_rate'], params['batch_norm'], mode)

            seq_2d = tf.reshape(sequence, [-1, tf.shape(sequence)[2]])
            sequence_mask = tf.not_equal(features[hist_name], 0)
            seq_vec = tf.reshape(tf.where(tf.reshape(sequence_mask, [-1]),
                                                  seq_2d, tf.zeros_like(seq_2d)),
                                         tf.shape(sequence))
            emb_dict['{}_max_pool_emb'.format(s)] = tf.reduce_max(seq_vec, axis=1)

            seq_length = tf.reduce_sum(tf.cast(sequence_mask, tf.float32), axis=1,
                                                   keep_dims=True)  # [batch_size, 1]
            seq_length_tile = tf.tile(seq_length, [1, seq_vec.get_shape().as_list()[-1]])  # [batch_size, emb_dim]
            seq_vec_mean = tf.multiply(tf.reduce_sum(seq_vec, axis=1), tf.pow(seq_length_tile, -1))
            emb_dict['{}_mean_pool_emb'.format(s)] = seq_vec_mean

def target_attention_layer(features, params, emb_dict):
    for s in params['seq_names']:
        hist_name = 'hist_{}_list'.format(s)
        with tf.compat.v1.variable_scope('Target_Attention_Layer_{}'.format(s)):
            atten_query = emb_dict['{}_emb'.format(s)]
            atten_query = tf.expand_dims(atten_query, 1)  
            atten_key = emb_dict['{}_hist_emb'.format(s)]
            att_emb = attention(atten_query, atten_key, params, features[hist_name]) 
            att_emb = tf.reshape(att_emb, [-1, params['emb_dim']])
            emb_dict['{}_att_emb'.format(s)] = att_emb

def group_layer(features, params, emb_dict):
    for s in params['seq_names']:
        hist_name = 'hist_{}_list'.format(s)
        with tf.compat.v1.variable_scope('Group_Layer_{}'.format(s)):
            hist_emb = emb_dict['{}_hist_emb'.format(s)]
        
            # direct mean pooling 
            seq_2d = tf.reshape(hist_emb, [-1, tf.shape(hist_emb)[2]])
            sequence_mask = tf.not_equal(features[hist_name], 0)
            seq_vec = tf.reshape(tf.where(tf.reshape(sequence_mask, [-1]),
                                                seq_2d, tf.zeros_like(seq_2d)),
                                        tf.shape(hist_emb))
            seq_length = tf.reduce_sum(tf.cast(sequence_mask, tf.float32), axis=1,
                                                keep_dims=True)  # [batch_size, 1]
            seq_length_tile = tf.tile(seq_length,
                                            [1,
                                            seq_vec.get_shape().as_list()[-1]])  # [batch_size, emb_dim]
            seq_vec_mean = tf.multiply(tf.reduce_sum(seq_vec, axis=1),
                                            tf.pow(seq_length_tile, -1))

            # self atten
            seq_att_3d = attention(hist_emb, hist_emb, params, features[hist_name], features[hist_name])
            seq_2d_att = tf.reshape(seq_att_3d, [-1, tf.shape(seq_att_3d)[2]])  
            seq_vec_att = tf.reshape(tf.where(tf.reshape(sequence_mask, [-1]),
                                                seq_2d_att, tf.zeros_like(seq_2d_att)),
                                        tf.shape(seq_att_3d)) 
            seq_vec_mean_att = tf.multiply(tf.reduce_sum(seq_vec_att, axis=1),
                                                tf.pow(seq_length_tile, -1))

            emb_dict['{}_self_att_emb'.format(s)] = seq_vec_mean_att

            seq_group_embedding_table = tf.compat.v1.get_variable(
                        name="{}_group_embedding".format(s),
                        shape=[params['num_user_groups'], params['emb_dim']],
                        initializer=tf.truncated_normal_initializer()
                    )
            group_hidden = tf.layers.dense(seq_vec_mean, units = params['num_user_groups'], activation = tf.nn.softmax, name ='group_hidden')
            if params['use_cluster_loss']:
                # target = tf.compat.v1.get_variable(
                #         name="{}_target_prob".format(s),
                #         shape=[tf.shape(group_hidden)[0], group_hidden.get_shape().as_list()[1]],
                #         initializer=tf.truncated_normal_initializer(),
                #     )
                # expect_prob = tf.random_normal(shape=[params['num_user_groups']], mean=0, stddev=1)
                expect_prob = tf.random_uniform(shape=[params['num_user_groups']], minval=0, maxval=1)
                pred_prob = tf.reduce_mean(group_hidden, axis=0)
                # loss_cluster = tf.reduce_sum(tf.multiply(target, tf.log(tf.div(target, group_hidden))))+tf.reduce_sum(tf.multiply(pred_prob, tf.log(tf.div(pred_prob, expect_prob))))
                loss_cluster = tf.reduce_sum(tf.multiply(pred_prob, tf.log(tf.div(pred_prob, expect_prob))))
                tf.add_to_collection('all_loss_cluster', loss_cluster)
            index_ids = tf.argmax(group_hidden, axis=-1)
            seq_group_embedding = tf.nn.embedding_lookup(seq_group_embedding_table, index_ids)
            weighted_group_emb = tf.matmul(group_hidden, seq_group_embedding_table)
            group_out = tf.layers.dense(weighted_group_emb, units = params['emb_dim'], activation = tf.nn.sigmoid, name ='group_out')

            loss_sim = tf.maximum(1 - tf.reduce_mean(tf.reduce_sum(
                        tf.multiply(tf.nn.l2_normalize(group_out, dim=1), tf.nn.l2_normalize(seq_vec_mean_att, dim=1)),
                        axis=-1)), 0)
            tf.add_to_collection('all_loss_sim', loss_sim)

            emb_dict['{}_group_emb'.format(s)] = seq_group_embedding

def att_weight_layer(emb_dict, params, scope):
    for s in params['seq_names']:
        with tf.compat.v1.variable_scope(scope+"_"+s):
            query = emb_dict['{}_emb'.format(s)]
            per_emb = emb_dict['{}_self_att_emb'.format(s)]
            group_emb = emb_dict['{}_group_emb'.format(s)]

            per_hidden = tf.layers.dense(per_emb, units = query.get_shape().as_list()[-1], activation = tf.nn.relu, name ='per_hidden')
            group_hidden = tf.layers.dense(group_emb, units = query.get_shape().as_list()[-1], activation = tf.nn.relu, name ='group_hidden')
            per_query_sim = tf.reduce_sum(tf.multiply(per_hidden, query), axis=1, keep_dims=True)
            group_query_sim = tf.reduce_sum(tf.multiply(group_hidden, query), axis=1, keep_dims=True)

            logit_per = tf.exp(per_query_sim)
            logit_group = tf.exp(group_query_sim)
            per_emb = tf.multiply(per_emb, tf.div(logit_per, logit_per + logit_group))
            group_emb = tf.multiply(group_emb, tf.div(logit_group, logit_per + logit_group))

            emb_dict['{}_self_att_emb'.format(s)] = per_emb     
            emb_dict['{}_group_emb'.format(s)] = group_emb

def moe_layer(dense, params, mode, scope):
    with tf.compat.v1.variable_scope(scope):
        outputs = []
        for n in range(params['num_of_expert']):
            out = stack_dense_layer(dense, params['hidden_units'], params['dropout_rate'], params['batch_norm'], mode, scope='Expert_{}'.format(n))
            outputs.append(out)
            
        out = stack_dense_layer(dense, params['hidden_units'], params['dropout_rate'], params['batch_norm'], mode, scope='Gate')
        y = tf.layers.dense(out, units=params['num_of_expert'], name = 'gate_out') 
        gate_weights = tf.nn.softmax(y, dim=1)

        expert_output = tf.reduce_sum(tf.multiply(tf.expand_dims(gate_weights, -1), tf.stack(values=outputs, axis=1)), axis=1)
        expert_output = stack_dense_layer(expert_output, params['hidden_units'], params['dropout_rate'], params['batch_norm'], mode, scope='CTR_Task_Dense')
        
    return expert_output

def virtual_moe_layer(dense, params, mode, emb_dict, scope):
    with tf.compat.v1.variable_scope(scope):
        outputs = []
        user_qs = tf.compat.v1.get_variable(
            name="user_q_embeddings",
            shape=[params['num_of_expert'], params['emb_dim']],
            initializer=tf.truncated_normal_initializer(stddev=0.001)
        )
        for n in range(params['num_of_expert']):
            index = tf.tile(tf.constant([n]), [tf.shape(dense)[0]])
            user_q = tf.nn.embedding_lookup(user_qs, index)
            user_q = tf.expand_dims(user_q, 1)  
            att_emb = attention(user_q, dense, params, scope='att_{}'.format(n)) 
            att_emb = tf.reshape(att_emb, [-1, params['emb_dim']])

            out = stack_dense_layer(att_emb, params['hidden_units'], params['dropout_rate'], params['batch_norm'], mode, scope='Expert_{}'.format(n))
            outputs.append(out)
        gate_q = emb_dict['item_emb']
        gate_q = tf.expand_dims(gate_q, 1)  
        user_qs = tf.expand_dims(user_qs, 0) 
        final_att_emb = attention(gate_q, tf.tile(user_qs, [tf.shape(dense)[0], 1, 1]), params, values=tf.stack(outputs, axis=1))
        final_att_emb = tf.reshape(final_att_emb, [-1, params['emb_dim']])

        expert_output = stack_dense_layer(final_att_emb, params['hidden_units'], params['dropout_rate'], params['batch_norm'], mode, scope='CTR_Task_Dense')
        
    return expert_output

def sparse_moe_layer(dense, params, mode, scope):
    with tf.compat.v1.variable_scope(scope):
        outputs = []
        for n in range(params['num_of_expert']):
            out = stack_dense_layer(dense, params['hidden_units'], params['dropout_rate'], params['batch_norm'], mode, scope='Expert_{}'.format(n))
            outputs.append(out)

        gates, load = noisy_top_k_gating(dense, params, mode)
        importance = tf.reduce_sum(gates, 0)
        loss = cv_squared(importance) + cv_squared(load)
        loss *= params['loss_coef']
        tf.add_to_collection('all_loss_balance', loss)

        expert_output = tf.reduce_sum(tf.multiply(tf.expand_dims(gates, -1), tf.stack(values=outputs, axis=1)), axis=1)
        expert_output = stack_dense_layer(expert_output, params['hidden_units'], params['dropout_rate'], params['batch_norm'], mode, scope='CTR_Task_Dense')
    return expert_output

def sparse_moe_layer_with_dselect(dense, params, mode, scope):
    with tf.compat.v1.variable_scope(scope):
        outputs = []
        for n in range(params['num_of_expert']):
            out = stack_dense_layer(dense, params['hidden_units'], params['dropout_rate'], params['batch_norm'], mode, scope='Expert_{}'.format(n))
            outputs.append(out)

        if params['example_conditioned']:
            expert_weights, selector_outputs = compute_example_conditioned_expert_weights_with_dselect(dense, params) # (b, num_experts), (b, k, num_experts)
            expert_output = tf.math.accumulate_n([tf.reshape(expert_weights[:, i], [-1, 1]) * outputs[i] for i in range(params['num_of_expert'])])
        else:
            expert_weights, selector_outputs = compute_expert_weights_with_dselect(params) # (num_experts, ), (k, num_experts)
            expert_output = tf.math.accumulate_n([expert_weights[i] * outputs[i] for i in range(params['num_of_expert'])])
        
        expert_output = stack_dense_layer(expert_output, params['hidden_units'], params['dropout_rate'], params['batch_norm'], mode, scope='CTR_Task_Dense')

        loss = _add_entropy_regularization_loss(params, selector_outputs)
        tf.add_to_collection('all_loss_entropy', loss)
    return expert_output

def mmoe_layer(dense, params, mode, scope):
    with tf.compat.v1.variable_scope(scope):
        outputs = []
        for n in range(params['num_of_expert']):
            out = stack_dense_layer(dense, params['hidden_units'], params['dropout_rate'], params['batch_norm'], mode, scope='Expert_{}'.format(n))
            outputs.append(out)
            
        out = stack_dense_layer(dense, params['hidden_units'], params['dropout_rate'], params['batch_norm'], mode, scope='Gate')
        y = tf.layers.dense(out, units=params['num_of_expert'], name = 'gate_out') 
        gate_weights = tf.nn.softmax(y, dim=1)
        if not params['use_one_gate']:
            out = stack_dense_layer(dense, params['hidden_units'], params['dropout_rate'], params['batch_norm'], mode, scope='Gate_plus')
            y = tf.layers.dense(out, units=params['num_of_expert'], name = 'gate_out_plus') 
            gate_weights_plus = tf.nn.softmax(y, dim=1)

        expert_output_ctr = tf.reduce_sum(tf.multiply(tf.expand_dims(gate_weights, -1), tf.stack(values=outputs, axis=1)), axis=1)
        if params['use_one_gate']:
            expert_output_recognition = tf.reduce_sum(tf.multiply(tf.expand_dims(gate_weights_plus, -1), tf.stack(values=outputs, axis=1)), axis=1)
        else:
            expert_output_recognition = tf.reduce_sum(tf.multiply(tf.expand_dims(gate_weights, -1), tf.stack(values=outputs, axis=1)), axis=1)
        expert_output_ctr = stack_dense_layer(expert_output_ctr, params['hidden_units'], params['dropout_rate'], params['batch_norm'], mode, scope='CTR_Task_Dense')
        expert_output_recognition = stack_dense_layer(expert_output_recognition, params['hidden_units'], params['dropout_rate'], params['batch_norm'], mode, scope='Recognition_Task_Dense')
        
    return expert_output_ctr, expert_output_recognition

def star_layer(dense, params, features, mode, scope):
    with tf.compat.v1.variable_scope(scope):
        if params['data_name'] == 'amazon':
            user_group_name = 'reviewer_group'
        elif params['data_name'] == 'movielens':
            user_group_name = 'user_group'
        user_level_0_mask = tf.cast(tf.reshape(tf.equal(features[user_group_name], 0), [-1, 1]), tf.float32)
        user_level_1_mask = tf.cast(tf.reshape(tf.equal(features[user_group_name], 1), [-1, 1]), tf.float32)
        user_level_2_mask = tf.cast(tf.reshape(tf.equal(features[user_group_name], 2), [-1, 1]), tf.float32)
        user_level_0_input = user_level_0_mask*dense
        user_level_1_input = user_level_1_mask*dense
        user_level_2_input = user_level_2_mask*dense

        user_level_0_output = moe_layer(user_level_0_input, params, mode, scope='user_0_moe')
        user_level_1_output = moe_layer(user_level_1_input, params, mode, scope='user_1_moe')
        user_level_2_output = moe_layer(user_level_2_input, params, mode, scope='user_2_moe')
        share_output = moe_layer(dense, params, mode, scope='share_moe')

        user_level_0_final_output = user_level_0_output*share_output/2
        user_level_1_final_output = user_level_1_output*share_output/2
        user_level_2_final_output = user_level_2_output*share_output/2

        final_output = user_level_0_final_output+user_level_1_final_output+user_level_2_final_output
    return final_output
