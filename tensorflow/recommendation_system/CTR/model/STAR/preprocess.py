from const import *
import tensorflow as tf


def build_features():
    # everything but the feature used in attention
    f_reviewer = tf.feature_column.categorical_column_with_identity(
        'reviewer_id',
        num_buckets = AMAZON_USER_COUNT,
        default_value = 0
    )
    f_reviewer = tf.feature_column.embedding_column(f_reviewer, dimension = 128)

    f_user_group = tf.feature_column.categorical_column_with_identity(
        'reviewer_group',
        num_buckets = 3,
        default_value = 0
    )
    f_user_group = tf.feature_column.embedding_column(f_user_group, dimension = 128)

    f_item_length = tf.feature_column.numeric_column('hist_length')

    f_dense = [f_item_length, f_reviewer]
    f_user_group = [f_user_group]

    return f_dense, f_user_group

