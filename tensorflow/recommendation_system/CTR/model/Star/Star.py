import tensorflow as tf

from const import *
from model.Star.preprocess import build_features
from utils import build_estimator_helper, tf_estimator_model
from layers import seq_pooling_layer, target_attention_layer, star_layer, stack_dense_layer

@tf_estimator_model
def model_fn_varlen(features, labels, mode, params):
    # ---general embedding layer---
    emb_dict = {}
    f_dense, f_user_group = build_features(params)
    f_dense = tf.compat.v1.feature_column.input_layer(features, f_dense) # 用户嵌入表示 [batch_size, f_num*emb_dim]
    f_user_group = tf.compat.v1.feature_column.input_layer(features, f_user_group)
    emb_dict['dense_emb'] = f_dense
    emb_dict['user_group_emb'] = f_user_group


    # Embedding Look up: history item list and category list
    item_embedding = tf.compat.v1.get_variable(shape = [params['item_count'], params['emb_dim']],
                                     initializer = tf.truncated_normal_initializer(),
                                     name = 'item_embedding')
    cate_embedding = tf.compat.v1.get_variable(shape = [params['cate_count'], params['emb_dim']],
                                     initializer = tf.truncated_normal_initializer(),
                                     name = 'cate_embedding')
    item_emb = tf.nn.embedding_lookup( item_embedding, features['item'] )  # [batch_size, emb_dim]
    emb_dict['item_emb'] = item_emb
    item_hist_emb = tf.nn.embedding_lookup( item_embedding, features['hist_item_list'] )  # [batch_size, padded_size, emb_dim]
    emb_dict['item_hist_emb'] = item_hist_emb
    cate_emb = tf.nn.embedding_lookup( cate_embedding, features['item_cate'] )  # [batch_size, emb_dim]
    emb_dict['cate_emb'] = cate_emb
    cate_hist_emb = tf.nn.embedding_lookup( cate_embedding, features['hist_cate_list'] )  # [batch_size, padded_size, emb_dim]
    emb_dict['cate_hist_emb'] = cate_hist_emb

    # ---sequence embedding layer---
    seq_pooling_layer(features, params, emb_dict, mode)
    target_attention_layer(features, params, emb_dict)

    # Concat features
    concat_features = []
    for f in params['input_features']:
        concat_features.append(emb_dict[f])
    fc = tf.concat(concat_features, axis=1)
    concat_auxiliary_features = []
    for f in params['auxiliary_input_features']:
        concat_auxiliary_features.append(emb_dict[f])
    auxiliary_fc = tf.concat(concat_auxiliary_features, axis=1)

    # ---dnn layer---
    main_net = star_layer(fc, params, features, mode, scope='main_moe_star')
    bias_net = stack_dense_layer(fc, params['hidden_units'], params['dropout_rate'], params['batch_norm'],
                              mode, scope='bias_dense')
    auxiliary_net = stack_dense_layer(auxiliary_fc, params['hidden_units'],
                                params['dropout_rate'], params['batch_norm'],
                                mode, scope='auxiliary_dense')

    # ---logits layer---
    main_y = tf.layers.dense(main_net, units=1, name='main_logit_net')
    bias_y = tf.layers.dense(bias_net, units=1, name='bias_logit_net')
    auxiliary_y = tf.layers.dense(auxiliary_net, units=1, name='auxiliary_logit_net')

    return main_y+bias_y+auxiliary_y


build_estimator = build_estimator_helper(
    model_fn = {
        'amazon': model_fn_varlen,
        'movielens': model_fn_varlen
    },
    params = {
        'amazon':{ 'dropout_rate' : 0.2,
                   'batch_norm' : True,
                   'learning_rate' : 0.01,
                   'hidden_units' : [80,40],
                   'attention_hidden_unit': 80,
                   'atten_mode': 'ln', 
                   'num_heads': 2,
                   'item_count': AMAZON_ITEM_COUNT,
                   'cate_count': AMAZON_CATE_COUNT,
                   'seq_names': ['item', 'cate'],
                   'num_of_expert': 2,
                   'num_user_groups': 3,
                   'sparse_emb_dim': 128,
                   'emb_dim': AMAZON_EMB_DIM,
                   'model_name': 'star',
                   'data_name': 'amazon',
                   'input_features': ['dense_emb', 'item_emb', 'cate_emb', 'item_att_emb', 'cate_att_emb'],
                   'auxiliary_input_features': ['user_group_emb', 'dense_emb', 'item_emb', 'cate_emb', 'item_att_emb', 'cate_att_emb']
            },
        'movielens':{ 'dropout_rate' : 0.2,
                   'batch_norm' : True,
                   'learning_rate' : 0.01,
                   'hidden_units' : [80,40],
                   'attention_hidden_unit': 80,
                   'atten_mode': 'ln', 
                   'num_heads': 2,
                   'item_count': ML_ITEM_COUNT,
                   'cate_count': ML_CATE_COUNT,
                   'seq_names': ['item', 'cate'],
                   'num_of_expert': 2,
                   'num_user_groups': 3,
                   'sparse_emb_dim': 128,
                   'emb_dim': ML_EMB_DIM,
                   'model_name': 'star',
                   'data_name': 'movielens',
                   'input_features': ['dense_emb', 'item_emb', 'cate_emb', 'item_att_emb', 'cate_att_emb'],
                   'auxiliary_input_features': ['user_group_emb', 'dense_emb', 'item_emb', 'cate_emb', 'item_att_emb', 'cate_att_emb']
            }
    }
)
