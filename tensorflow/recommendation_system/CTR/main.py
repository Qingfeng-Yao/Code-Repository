# -*- coding:utf-8 -*-
import argparse
import importlib
import shutil
import pandas as pd
import os

from config import CONFIG
from utils import *

# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

def main(args):
    model = args.model
    config = CONFIG(model_name = model, data_name = args.dataset)

    try:
        shutil.rmtree(config.checkpoint_dir)
        print('{} model cleaned'.format(config.checkpoint_dir))
    except Exception as e:
        print('Error! {} occured at model cleaning'.format(e))

    # build estimator
    build_estimator = getattr(importlib.import_module('model.{}.{}'.format(model, model)),
                             'build_estimator')
    estimator = build_estimator(config)

    # train or predict
    if args.step == 'train':
        early_stopping = tf.estimator.experimental.stop_if_no_decrease_hook(
            estimator,
            metric_name="loss",
            max_steps_without_decrease= 20 * 100)

        train_spec = tf.estimator.TrainSpec(input_fn = input_fn(step = 'train',
                                             is_predict = 0,
                                            config = config), hooks = [early_stopping])

        eval_spec = tf.estimator.EvalSpec(input_fn = input_fn(step ='valid',
                                           is_predict = 1,
                                           config = config),
                                           steps = 200,
                                           throttle_secs = 60)

        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    if args.step =='predict':
        prediction = estimator.predict(input_fn = input_fn(step='valid',
                                        is_predict = 1,
                                        config = config))

        predict_prob = pd.DataFrame({'predict_prob': [i['prediction_prob'][1] for i in prediction]})
        predict_prob.to_csv('./result/prediction_{}_{}.csv'.format(args.dataset, model))


if __name__ =='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, help = 'which model to use: DIN | MOE | Bias | UserLoss | UserInput | Star | UserCluster', default='DIN')
    parser.add_argument('--dataset', type=str, help= 'which dataset to use: amazon | movielens', default='amazon')
    parser.add_argument('--cuda', type=str, help= 'which gpu to use', default='1')
    
    parser.add_argument('--step', type = str, help = 'Train or Predict', default='train')

    # һ��ģ�Ͳ�����config�ļ��У�����batch_size, num_epochs, buffer_size(����shuffle)�ȣ�����amazon��Ӧ��batch_sizeΪ512,
    # �ض�ģ�Ͳ����ڸ���ģ���ļ��У�����dropout_rate, batch_norm, learning_rate, hidden_units, attention_hidden_units, atten_mode, item_count, cate_count, seq_names, emb_dim, model_name, data_name, input_features��
    # ���в�����util�ļ��е�build_estimator_helper�����б����壬����save_summary_steps, log_step_count_steps, keep_checkpoint_max, save_checkpoints_steps��
    
    
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    main(args)