"""
将主网络改成多个专家网络和门网络，即引入mmoe_layer

"""


import tensorflow as tf

from const import *
from model.STAR.preprocess import build_features
from utils import build_estimator_helper, tf_estimator_model, add_layer_summary
from layers import stack_dense_layer, mmoe_layer

def attention(queries, keys, keys_id, params):
    """
    :param queries: target embedding (batch_size * emb_dim)
    :param keys: history embedding (batch * padded_size * emb_dim)
    :param keys_id: history id (batch * padded_size)
    :return: attention_emb: weighted average of history embedding (batch * emb_dim)
    """
    # Differ from paper, for computation efficiency: outer product -> hadamard product
    padded_size = tf.shape(keys)[1]
    queries = tf.tile(tf.expand_dims(queries, axis=1), [1, padded_size, 1]) # batch * emb_dim -> batch * padded_size * emb_dim
    dense = tf.concat([keys, queries, queries - keys, queries * keys], axis =2 ) # batch * padded_size * emb_dim

    for i, unit in enumerate(params['attention_hidden_units']):
        dense = tf.layers.dense(dense, units= unit, activation = tf.nn.relu, name = 'attention_{}'.format(i))
        add_layer_summary(dense.name, dense)
    weight = tf.layers.dense(dense, units=1, activation=tf.sigmoid, name ='attention_weight') # batch * padded_size * 1

    zero_mask = tf.expand_dims(tf.not_equal( keys_id, 0 ), axis=2)  # batch * padded_size * 1
    zero_weight = tf.ones_like(weight) * (-2 ** 32 + 1) # small number logits ~ 0
    weight = tf.where(zero_mask, weight, zero_weight) # apply zero-mask for padded keys

    weight = tf.nn.softmax(weight) # rescale weight to sum(weight)=1
    add_layer_summary('attention_weight' , weight )

    attention_emb = tf.reduce_mean(tf.multiply(weight, keys), axis=1) # weight average ->batch * emb_dim

    return attention_emb


@tf_estimator_model
def model_fn_varlen(features, labels, mode, params):
    f_dense, f_user_group = build_features()
    f_dense = tf.compat.v1.feature_column.input_layer(features, f_dense)
    f_user_group = tf.compat.v1.feature_column.input_layer(features, f_user_group)

    # Embedding Look up: history list item and category list
    item_embedding = tf.compat.v1.get_variable(shape = [params['amazon_item_count'], params['amazon_emb_dim']],
                                     initializer = tf.truncated_normal_initializer(),
                                     name = 'item_embedding')
    cate_embedding = tf.compat.v1.get_variable(shape = [params['amazon_cate_count'], params['amazon_emb_dim']],
                                     initializer = tf.truncated_normal_initializer(),
                                     name = 'cate_embedding')

    with tf.compat.v1.variable_scope('Attention_Layer'):
        with tf.compat.v1.variable_scope('item_attention'):
            item_hist_emb = tf.nn.embedding_lookup( item_embedding,
                                                    features['hist_item_list'] )  # batch * padded_size * emb_dim
            item_emb = tf.nn.embedding_lookup( item_embedding, features['item'] )  # batch * emb_dim
            item_att_emb = attention(item_emb, item_hist_emb, features['hist_item_list'], params) # batch * emb_dim

        with tf.compat.v1.variable_scope('category_attention'):
            cate_hist_emb = tf.nn.embedding_lookup( cate_embedding,
                                                    features['hist_category_list'] )  # batch * padded_size * emb_dim
            cate_emb = tf.nn.embedding_lookup( cate_embedding, features['item_category'] )  # batch * emd_dim
            cate_att_emb = attention(cate_emb, cate_hist_emb, features['hist_category_list'], params) # batch * emb_dim

    # Concat attention embedding and all other features
    with tf.compat.v1.variable_scope('Concat_Layer'):
        fc = tf.concat([item_att_emb, cate_att_emb, item_emb, cate_emb, f_dense], axis=1)
        fc_auxiliary = tf.concat([item_att_emb, cate_att_emb, item_emb, cate_emb, f_dense, f_user_group], axis=1)


    # whatever model you want after fc: here for simplicity use only MLP, you can try DCN/DeepFM
    dense = mmoe_layer(fc, params['hidden_units'],
                              params['dropout_rate'], params['batch_norm'],
                              mode, add_summary = True)

    with tf.compat.v1.variable_scope('Bias_Layer'):
        bias = stack_dense_layer(fc, params['hidden_units'],
                                params['dropout_rate'], params['batch_norm'],
                                mode, add_summary = True)

    with tf.compat.v1.variable_scope('Auxiliary_Layer'):
        auxiliary = stack_dense_layer(fc_auxiliary, params['hidden_units'],
                                params['dropout_rate'], params['batch_norm'],
                                mode, add_summary = True)
    

    with tf.compat.v1.variable_scope('main_output'):
        main_y = tf.layers.dense(dense, units =1)
        add_layer_summary( 'main_output', main_y )

    with tf.compat.v1.variable_scope('bias_output'):
        bias_y = tf.layers.dense(bias, units =1)
        add_layer_summary( 'bias_output', bias_y )

    with tf.compat.v1.variable_scope('auxiliary_output'):
        auxiliary_y = tf.layers.dense(auxiliary, units =1)
        add_layer_summary( 'auxiliary_output', auxiliary_y )

    return main_y+bias_y+auxiliary_y


build_estimator = build_estimator_helper(
    model_fn = {
        'amazon' :model_fn_varlen
    },
    params = {
        'amazon':{ 'dropout_rate' : 0.2,
                   'batch_norm' : True,
                   'learning_rate' : 0.01,
                   'hidden_units' : [80,40],
                   'attention_hidden_units':[80,40],
                   'amazon_item_count': AMAZON_ITEM_COUNT,
                   'amazon_cate_count': AMAZON_CATE_COUNT,
                   'amazon_emb_dim': AMAZON_EMB_DIM,
                   'model_name': 'star'
            }
    }
)
