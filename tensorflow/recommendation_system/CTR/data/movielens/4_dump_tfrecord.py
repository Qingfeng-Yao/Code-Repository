# -*- coding:utf-8 -*-
import pickle
import tensorflow as tf
import numpy as np
import os


class TFDump(object):
    def __init__(self):
        self.load_data()

    def load_data(self):
        with open('dataset.pkl', 'rb') as f:
            self.train = pickle.load(f)
            self.valid = pickle.load(f)

        with open('remap.pkl', 'rb') as f:
            _ = pickle.load(f)
            self.cate_list = pickle.load(f)

    @staticmethod
    def int_feature(value):
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature( int64_list=tf.train.Int64List( value= value ) )

    def dump(self, data, type):
        with tf.io.TFRecordWriter('movielens_{}.tfrecords'.format(type)) as writer:
            n = 0
            for record in data:
                n += 1
                example = tf.train.Example(
                    features = tf.train.Features(
                        feature = {
                            'user_id': TFDump.int_feature( record[0] ),
                            'hist_item_list': TFDump.int_feature( record[1] ),
                            'hist_cate_list': TFDump.int_feature( [self.cate_list[i] for i in record[1]] ),
                            'hist_length': TFDump.int_feature( len( record[1] ) ),
                            'item': TFDump.int_feature( record[2] ),
                            'item_cate': TFDump.int_feature( self.cate_list[record[2]] ),
                            'target': TFDump.int_feature( record[3] ),
                            # 'user_group': TFDump.int_feature( record[4] ) # 添加用户群组信息，对应修改数据集配置，包括const和config文件
                        }
                    )
                )

                writer.write(example.SerializeToString())
            writer.close()
            print(n)

    def execute(self):
        self.dump(self.train, 'train')
        self.dump(self.valid, 'valid')


if __name__ == '__main__':
    preprocess = TFDump()
    preprocess.execute()

