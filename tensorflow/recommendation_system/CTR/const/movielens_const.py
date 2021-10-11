import tensorflow as tf
import pickle


ML_PROTO = {
    'user_id': tf.io.FixedLenFeature( [], tf.int64 ),
    'hist_item_list': tf.io.VarLenFeature( tf.int64 ),
    'hist_cate_list': tf.io.VarLenFeature(tf.int64),
    'hist_length': tf.io.FixedLenFeature([], tf.int64),
    'item': tf.io.FixedLenFeature( [], tf.int64 ),
    'item_cate': tf.io.FixedLenFeature([], tf.int64),
    'target': tf.io.FixedLenFeature( [], tf.int64 ),
    # 'user_group': tf.io.FixedLenFeature( [], tf.int64 )
}

ML_TARGET = 'target'

ML_VARLEN = ['hist_item_list','hist_cate_list']

with open('data/movielens/remap.pkl', 'rb') as f:
    _ = pickle.load(f) 
    ML_CATE_LIST  = pickle.load(f)
    ML_USER_COUNT, ML_ITEM_COUNT, ML_CATE_COUNT, _ = pickle.load(f)
    print("movielens: n_user{}, n_item{}, n_cate{}".format(ML_USER_COUNT, ML_ITEM_COUNT, ML_CATE_COUNT))

ML_EMB_DIM = 64