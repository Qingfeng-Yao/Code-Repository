# -*- coding:utf-8 -*-
import pickle
import tensorflow as tf
import numpy as np
import os


class TFDump(object):
    def __init__(self):
        self.load_data()

    def load_data(self):
        
        with open('dataset_small.pkl', 'rb') as f:
            self.train = pickle.load(f)
            self.valid = pickle.load(f)

    @staticmethod
    def int_feature(value):
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature( int64_list=tf.train.Int64List( value= value ) )

    def dump(self, data, type):
        with tf.io.TFRecordWriter('heybox_{}.tfrecords'.format(type)) as writer:
            n = 0
            for record in data:
                n += 1
                example = tf.train.Example(
                    features = tf.train.Features(
                        feature = {
                            'reviewer_id': TFDump.int_feature( record[0] ),
                            'hist_item_list': TFDump.int_feature( record[1] ),
                            'hist_cate_list': TFDump.int_feature( record[2] ),
                            'hist_length': TFDump.int_feature( len( record[1] ) ),
                            'item': TFDump.int_feature( record[3] ),
                            'item_cate': TFDump.int_feature( record[4] ),
                            'target': TFDump.int_feature( record[5] )
                        }
                    )
                )

                writer.write(example.SerializeToString())
            writer.close()
            print(n)

    def execute(self):
        self.dump(self.train, 'train_small')
        self.dump(self.valid, 'valid_small')


if __name__ == '__main__':
    preprocess = TFDump()
    preprocess.execute()

