"""
继承MMOEBIAS
添加ubc模块，即考虑半个性化表示

"""


import tensorflow as tf

from const import *
from model.UBC.preprocess import build_features
from utils import build_estimator_helper, tf_estimator_model, add_layer_summary
from layers import attention, stack_dense_layer, mmoe_layer, ubc_layer, att_weight_layer

@tf_estimator_model
def model_fn_varlen(features, labels, mode, params):
    f_dense, f_user = build_features()
    f_dense = tf.compat.v1.feature_column.input_layer(features, f_dense)
    f_user = tf.compat.v1.feature_column.input_layer(features, f_user)

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

    item_ubc_emb = ubc_layer(features, params, embtb=item_embedding, name='item')
    cate_ubc_emb = ubc_layer(features, params, embtb=cate_embedding, name='category')

    with tf.compat.v1.variable_scope('Attention_Weight_Layer'):
        item_att_emb, item_ubc_emb = att_weight_layer(item_att_emb, item_ubc_emb, f_user, name='item')
        cate_att_emb, cate_ubc_emb = att_weight_layer(cate_att_emb, cate_ubc_emb, f_user, name='cate')


    # Concat attention embedding and all other features
    with tf.compat.v1.variable_scope('Concat_Layer'):
        fc = tf.concat([item_att_emb, cate_att_emb, item_ubc_emb, cate_ubc_emb, item_emb, cate_emb, f_dense],  axis=1 )
        add_layer_summary('fc_concat', fc)

    # whatever model you want after fc: here for simplicity use only MLP, you can try DCN/DeepFM
    dense = mmoe_layer(fc, params['hidden_units'],
                              params['dropout_rate'], params['batch_norm'],
                              mode, add_summary = True)
    with tf.compat.v1.variable_scope('Bias_Layer'):
        bias = stack_dense_layer(fc, params['hidden_units'],
                                params['dropout_rate'], params['batch_norm'],
                                mode, add_summary = True)

    with tf.compat.v1.variable_scope('main_output'):
        main_y = tf.layers.dense(dense, units =1)
        add_layer_summary( 'main_output', main_y )

    with tf.compat.v1.variable_scope('bias_output'):
        bias_y = tf.layers.dense(bias, units =1)
        add_layer_summary( 'bias_output', bias_y )

    return main_y+bias_y


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
                   'num_user_group': 50,
                   'model_name': 'ubc'
            }
    }
)
