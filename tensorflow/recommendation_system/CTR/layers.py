import tensorflow as tf
from tensorflow.contrib import layers
from config import *

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

def attention(queries, keys, params, keys_id=None, queries_id=None, scope='multihead_attention'):
    with tf.compat.v1.variable_scope(scope):
        query_len = tf.shape(queries)[1]  
        key_len = tf.shape(keys)[1] 

        queries_2d = tf.reshape(queries, [-1, queries.get_shape().as_list()[-1]])
        keys_2d = tf.reshape(keys, [-1, keys.get_shape().as_list()[-1]])
        Q = tf.layers.dense(queries_2d, params['attention_hidden_unit'], activation = tf.nn.relu, name = 'attention_Q')  
        Q = tf.reshape(Q, [-1, tf.shape(queries)[1], Q.get_shape().as_list()[-1]])
        K = tf.layers.dense(keys_2d, params['attention_hidden_unit'], activation = tf.nn.relu, name = 'attention_K')  
        K = tf.reshape(K, [-1, tf.shape(keys)[1], K.get_shape().as_list()[-1]])
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
