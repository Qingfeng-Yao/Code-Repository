from const import *
import tensorflow as tf


def build_features(params):
    if params['data_name'] == 'amazon':
        user_id_name = 'reviewer_id'
        user_group_name = 'reviewer_group'
    elif params['data_name'] == 'movielens':
        user_id_name = 'user_id'
        user_group_name = 'user_group'

    f_user = tf.feature_column.categorical_column_with_identity(
        user_id_name,
        num_buckets = AMAZON_USER_COUNT,
        default_value = 0
    )
    f_user = tf.feature_column.embedding_column(f_user, dimension = params['sparse_emb_dim'])

    f_user_group = tf.feature_column.categorical_column_with_identity(
        user_group_name,
        num_buckets = params['num_user_groups'],
        default_value = 0
    )
    f_user_group = tf.feature_column.embedding_column(f_user_group, dimension = params['sparse_emb_dim'])

    f_item_length = tf.feature_column.numeric_column('hist_length') # 不考虑序列长度信息

    # f_dense = [f_item_length, f_user]
    f_dense = [f_user]
    f_user_group = [f_user_group]

    return f_dense, f_user_group

