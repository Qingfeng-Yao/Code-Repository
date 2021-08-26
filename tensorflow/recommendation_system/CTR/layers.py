import tensorflow as tf
from tensorflow.contrib import layers

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
            seq_length = tf.reduce_sum(tf.cast(sequence_mask, tf.float32), axis=1,
                                                   keep_dims=True)  # [batch_size, 1]
            seq_length_tile = tf.tile(seq_length, [1, seq_vec.get_shape().as_list()[-1]])  # [batch_size, emb_dim]
            seq_vec_mean = tf.multiply(tf.reduce_sum(seq_vec, axis=1), tf.pow(seq_length_tile, -1))
            emb_dict['{}_pool_emb'.format(s)] = seq_vec_mean

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

def mmoe_layer(dense, hidden_units, dropout_rate, batch_norm, mode, add_summary):
    with tf.compat.v1.variable_scope('Mmoe'):
        num_of_expert = 2
        outputs = []
        for n in range(num_of_expert):
            with tf.compat.v1.variable_scope('Expert_{}'.format(n)):
                out = stack_dense_layer(dense, hidden_units, dropout_rate, batch_norm, mode, add_summary)
                outputs.append(out)
            
        with tf.compat.v1.variable_scope('Gate'):
            out = stack_dense_layer(dense, hidden_units, dropout_rate, batch_norm, mode, add_summary)
            y = tf.layers.dense(out, units =1, name = 'out')
            
        gate_weights = tf.nn.softmax(y, dim=1)

        expert_output = tf.reduce_sum(tf.multiply(tf.expand_dims(gate_weights, -1), tf.stack(values=outputs, axis=1)), axis=1)
        with tf.compat.v1.variable_scope('CTR_Task'):
            expert_output = stack_dense_layer(expert_output, hidden_units, dropout_rate, batch_norm, mode, add_summary)
        
    return expert_output

def ubc_layer(features, params, embtb, name):
    hist_name = 'hist_{}_list'.format(name)
    with tf.compat.v1.variable_scope('UBC_Layer'):
        with tf.compat.v1.variable_scope(name):
            hist_emb = tf.nn.embedding_lookup(embtb, features[hist_name])  # batch * padded_size * emb_dim

            # direct mean pooling 
            seq_2d = tf.reshape(hist_emb, [-1, tf.shape(hist_emb)[2]])
            sequence_mask = tf.not_equal(features[hist_name], 0)
            seq_vec = tf.reshape(tf.where(tf.reshape(sequence_mask, [-1]),
                                                  seq_2d, tf.zeros_like(seq_2d)),
                                         tf.shape(hist_emb))
            seq_length = tf.reduce_sum(tf.cast(sequence_mask, tf.float32), axis=1,
                                                   keep_dims=True)  # [batch, 1]
            seq_length_tile = tf.tile(seq_length,
                                            [1,
                                            seq_vec.get_shape().as_list()[-1]])  # [batch, emb_dim]
            seq_vec_mean = tf.multiply(tf.reduce_sum(seq_vec, axis=1),
                                               tf.pow(seq_length_tile, -1))

            # self atten
            seq_att_3d, _ = attention(hist_emb, hist_emb, params, features[hist_name], features[hist_name])
            seq_2d_att = tf.reshape(seq_att_3d, [-1, tf.shape(seq_att_3d)[2]])  
            seq_vec_att = tf.reshape(tf.where(tf.reshape(sequence_mask, [-1]),
                                                seq_2d_att, tf.zeros_like(seq_2d_att)),
                                        tf.shape(seq_att_3d)) 
            seq_vec_mean_att = tf.multiply(tf.reduce_sum(seq_vec_att, axis=1),
                                                   tf.pow(seq_length_tile, -1))

            seq_group_embedding_table = tf.compat.v1.get_variable(
                        name="{}_group_embedding".format(name),
                        shape=[params['num_user_group'], params['amazon_emb_dim']],
                        initializer=tf.truncated_normal_initializer()
                    )
            ubc_hidden = tf.layers.dense(seq_vec_mean, units = params['num_user_group'], activation = tf.nn.softmax, name ='ubc_hidden')
            index_ids = tf.argmax(ubc_hidden, axis=-1)
            seq_group_embedding = tf.nn.embedding_lookup(seq_group_embedding_table, index_ids)
            ubc_hidden_group_weight = tf.matmul(ubc_hidden, seq_group_embedding_table)
            ubc_out = tf.layers.dense(ubc_hidden_group_weight, units = params['amazon_emb_dim'], activation = tf.nn.sigmoid, name ='ubc_out')

            loss_sim = tf.maximum(1 - tf.reduce_mean(tf.reduce_sum(
                        tf.multiply(tf.nn.l2_normalize(ubc_out, dim=1), tf.nn.l2_normalize(seq_vec_mean_att, dim=1)),
                        axis=-1)), 0)
            tf.add_to_collection('all_loss_sim', loss_sim)

            importance = tf.reduce_sum(ubc_hidden, axis=0)
            eps = 1e-10
            mean, variance  = tf.nn.moments(importance, axes=0)
            loss_balance = variance / (mean**2 + eps)
            tf.add_to_collection('all_loss_balance', loss_balance)

    return seq_group_embedding, seq_vec_mean_att

def att_weight_layer(att_emb, ubc_emb, query, name):
    with tf.compat.v1.variable_scope(name):
        att_hidden = tf.layers.dense(att_emb, units = query.get_shape().as_list()[-1], activation = tf.nn.relu, name ='att_hidden')
        ubc_hidden = tf.layers.dense(ubc_emb, units = query.get_shape().as_list()[-1], activation = tf.nn.relu, name ='ubc_hidden')
        att_query_sim = tf.reduce_sum(tf.multiply(att_hidden, query), axis=1, keep_dims=True)
        ubc_query_sim = tf.reduce_sum(tf.multiply(ubc_hidden, query), axis=1, keep_dims=True)

        logit_att = tf.exp(att_query_sim)
        logit_ubc = tf.exp(ubc_query_sim)
        att_emb = tf.multiply(att_emb, tf.div(logit_att, logit_att + logit_ubc))
        ubc_emb = tf.multiply(ubc_emb, tf.div(logit_ubc, logit_att + logit_ubc))
    return att_emb, ubc_emb
