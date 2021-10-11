import tensorflow as tf
import pickle


AMAZON_PROTO = {
    'user_id': tf.io.FixedLenFeature( [], tf.int64 ),
    'hist_item_list': tf.io.VarLenFeature( tf.int64 ),
    'hist_cate_list': tf.io.VarLenFeature(tf.int64),
    'hist_length': tf.io.FixedLenFeature([], tf.int64),
    'item': tf.io.FixedLenFeature( [], tf.int64 ),
    'item_cate': tf.io.FixedLenFeature([], tf.int64),
    'target': tf.io.FixedLenFeature( [], tf.int64 )
}

AMAZON_TARGET = 'target'

AMAZON_VARLEN = ['hist_item_list','hist_cate_list']

with open('data/amazon/remap.pkl', 'rb') as f:
    _ = pickle.load(f)
    AMAZON_CATE_LIST  = pickle.load(f)
    AMAZON_USER_COUNT, AMAZON_ITEM_COUNT, AMAZON_CATE_COUNT, _ = pickle.load(f)
    print("amazon: n_user{}, n_item{}, n_cate{}".format(AMAZON_USER_COUNT, AMAZON_ITEM_COUNT, AMAZON_CATE_COUNT))

AMAZON_EMB_DIM = 64