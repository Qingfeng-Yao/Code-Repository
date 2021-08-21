import tensorflow as tf
from utils import add_layer_summary

def sparse_embedding(feature_size, embedding_size, field_size, feat_ids, feat_vals, add_summary):
    with tf.variable_scope('Sparse_Embedding'):
        v = tf.get_variable( shape=[feature_size, embedding_size],
                             initializer=tf.truncated_normal_initializer(),
                             name='embedding_weight' )

        embedding_matrix = tf.nn.embedding_lookup( v, feat_ids ) # batch * field_size * embedding_size
        embedding_matrix = tf.multiply( embedding_matrix, tf.reshape(feat_vals, [-1, field_size,1] ) )

        if add_summary:
            add_layer_summary( 'embedding_matrix', embedding_matrix )

    return embedding_matrix


def sparse_linear(feature_size, feat_ids, feat_vals, add_summary):
    with tf.variable_scope('Linear_output'):
        weight = tf.get_variable( shape=[feature_size],
                             initializer=tf.truncated_normal_initializer(),
                             name='linear_weight' )
        bias = tf.get_variable( shape=[1],
                             initializer=tf.glorot_uniform_initializer(),
                             name='linear_bias' )

        linear_output = tf.nn.embedding_lookup( weight, feat_ids )
        linear_output = tf.reduce_sum( tf.multiply( linear_output, feat_vals ), axis=1, keepdims=True )
        linear_output = tf.add( linear_output, bias )

        if add_summary:
            add_layer_summary('linear_output', linear_output)

    return linear_output


def stack_dense_layer(dense, hidden_units, dropout_rate, batch_norm, mode, add_summary):
    with tf.compat.v1.variable_scope('Dense'):
        for i, unit in enumerate(hidden_units):
            dense = tf.layers.dense(dense, units = unit, activation = 'relu',
                                    name = 'dense{}'.format(i))
            if batch_norm:
                dense = tf.layers.batch_normalization(dense, center = True, scale = True,
                                                      trainable = True,
                                                      training = (mode == tf.estimator.ModeKeys.TRAIN))
            if dropout_rate > 0:
                dense = tf.layers.dropout(dense, rate = dropout_rate,
                                          training = (mode == tf.estimator.ModeKeys.TRAIN))

            if add_summary:
                add_layer_summary(dense.name, dense)

    return dense

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

def ubc_layer(features, params, embtb, attemb, name):
    hist_name = 'hist_{}_list'.format(name)
    with tf.compat.v1.variable_scope('UBC_Layer'):
        with tf.compat.v1.variable_scope(name):
            hist_emb = tf.nn.embedding_lookup(embtb, features[hist_name])  # batch * padded_size * emb_dim
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

            seq_group_embedding_table = tf.compat.v1.get_variable(
                        name="{}_group_embedding".format(name),
                        shape=[params['num_user_group'], params['amazon_emb_dim']],
                        initializer=tf.truncated_normal_initializer()
                    )
            ubc_hidden = tf.layers.dense(seq_vec_mean, units = params['num_user_group'], activation = tf.nn.softmax, name ='ubc_hidden')
            index_ids = tf.argmax(ubc_hidden, axis=-1)
            seq_group_embedding = tf.nn.embedding_lookup(seq_group_embedding_table, index_ids)
            ubc_hidden_group_weight = tf.matmul(ubc_hidden, seq_group_embedding_table)
            ubc_out = tf.layers.dense(ubc_hidden_group_weight, units = params['amazon_emb_dim'], activation = tf.nn.relu, name ='ubc_out')

            loss_sim = tf.maximum(1 - tf.reduce_mean(tf.reduce_sum(
                        tf.multiply(tf.nn.l2_normalize(ubc_out, dim=1), tf.nn.l2_normalize(attemb, dim=1)),
                        axis=-1)), 0)
            tf.add_to_collection('all_loss_sim', loss_sim)

    return seq_group_embedding

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
