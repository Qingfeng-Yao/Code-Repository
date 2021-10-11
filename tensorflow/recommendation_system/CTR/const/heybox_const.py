import tensorflow as tf
import pickle


HEYBOX_PROTO = {
    'reviewer_id': tf.io.FixedLenFeature( [], tf.int64 ), # userid
    'hist_item_list': tf.io.VarLenFeature( tf.int64 ), # post
    'hist_cate_list': tf.io.VarLenFeature(tf.int64), # topic
    'hist_length': tf.io.FixedLenFeature([], tf.int64),
    'item': tf.io.FixedLenFeature( [], tf.int64 ), # post
    'item_cate': tf.io.FixedLenFeature([], tf.int64), # topic of post
    'target': tf.io.FixedLenFeature( [], tf.int64 )
}

HEYBOX_TARGET = 'target'

HEYBOX_VARLEN = ['hist_item_list','hist_cate_list']

with open('data/heybox/remap.pkl', 'rb') as f:
    _ = pickle.load(f)
    HEYBOX_USER_COUNT, HEYBOX_ITEM_COUNT, HEYBOX_CATE_COUNT, _ = pickle.load(f)
    print("heybox: n_user{}, n_item{}, n_cate{}".format(HEYBOX_USER_COUNT, HEYBOX_ITEM_COUNT, HEYBOX_CATE_COUNT))

HEYBOX_EMB_DIM = 64