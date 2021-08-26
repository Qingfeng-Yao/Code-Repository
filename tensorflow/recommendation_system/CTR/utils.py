import tensorflow as tf
from const import *
from config import *

def parse_example_helper_tfreocrd_amazon(line):
    features = tf.io.parse_single_example(line, features = AMAZON_PROTO)

    for i in AMAZON_VARLEN:
        features[i] = tf.sparse.to_dense(features[i])

    target = tf.reshape(tf.cast( features.pop( AMAZON_TARGET ), tf.float32),[-1])

    return features, target

def parse_example_helper_tfreocrd_movielens(line):
    features = tf.io.parse_single_example(line, features = ML_PROTO)

    for i in ML_VARLEN:
        features[i] = tf.sparse.to_dense(features[i])

    target = tf.reshape(tf.cast( features.pop( ML_TARGET ), tf.float32),[-1])

    return features, target

def input_fn(step, is_predict, config):
    def func():
        if config.input_parser == 'tfrecord' and config.data_name == 'amazon':
            dataset = tf.data.TFRecordDataset(config.data_dir.format(step)) \
                .map(parse_example_helper_tfreocrd_amazon, num_parallel_calls=8)
        elif config.input_parser == 'tfrecord' and config.data_name == 'movielens':
            dataset = tf.data.TFRecordDataset(config.data_dir.format(step)) \
                .map(parse_example_helper_tfreocrd_movielens, num_parallel_calls=8)

        else:
            raise Exception('Only [amazon.tfrecord and movielens.tfrecord] are supported now')

        if not is_predict:
            # shuffle before repeat and batch last
            dataset = dataset \
                .shuffle(MODEL_PARAMS['buffer_size']) \
                .repeat(MODEL_PARAMS['num_epochs']) \

        if 'varlen' in config.input_type:
            dataset = dataset\
                .padded_batch(batch_size = MODEL_PARAMS['batch_size'],
                              padded_shapes = config.pad_shape)
        else:
            dataset = dataset \
                .batch(MODEL_PARAMS['batch_size'])

        return dataset
    return func


def add_layer_summary(tag, value):
  tf.compat.v1.summary.scalar('{}/fraction_of_zero_values'.format(tag), tf.math.zero_fraction(value))
  tf.compat.v1.summary.histogram('{}/activation'.format(tag), value)


def tf_estimator_model(model_fn):
    def model_fn_helper(features, labels, mode, params):

        y = model_fn(features, labels, mode, params)

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'prediction_prob': tf.sigmoid( y )
            }
            return tf.estimator.EstimatorSpec( mode=tf.estimator.ModeKeys.PREDICT,
                                               predictions=predictions )

        
        if params['model_name'] == 'ubc':
            print("ubc loss!")
            cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=y))
            # cross_entropy += (tf.reduce_sum(tf.compat.v1.get_collection('all_loss_sim'))+tf.reduce_sum(tf.compat.v1.get_collection('all_loss_balance')))
            cross_entropy += 2*tf.reduce_sum(tf.compat.v1.get_collection('all_loss_sim'))
        elif params['model_name'] == 'userloss':
            print("userloss!")
            # 'reviewer_group': <tf.Tensor 'IteratorGetNext:5' shape=(?,) dtype=int64>
            user_level_0_weight = tf.cast(tf.reshape(tf.equal(features['reviewer_group'], 0), [-1, 1]), tf.float32) * 2
            user_level_1_weight = tf.cast(tf.reshape(tf.equal(features['reviewer_group'], 1), [-1, 1]), tf.float32) * 1
            user_level_2_weight = tf.cast(tf.reshape(tf.equal(features['reviewer_group'], 2), [-1, 1]), tf.float32) * 0.8
            final_weight = user_level_0_weight+user_level_1_weight+user_level_2_weight
            cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=y)*final_weight)
        else:
            cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=y))

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.compat.v1.train.AdagradOptimizer( learning_rate=params['learning_rate'] )
            update_ops = tf.compat.v1.get_collection( tf.compat.v1.GraphKeys.UPDATE_OPS )
            with tf.control_dependencies( update_ops ):
                train_op = optimizer.minimize( cross_entropy,
                                               global_step=tf.compat.v1.train.get_global_step() )
            return tf.estimator.EstimatorSpec( mode, loss=cross_entropy, train_op=train_op )
        else:
            eval_metric_ops = {
                'accuracy': tf.compat.v1.metrics.accuracy( labels=labels,
                                                 predictions=tf.to_float(tf.greater_equal(tf.sigmoid(y),0.5))  ),
                'auc': tf.compat.v1.metrics.auc( labels=labels,
                                       predictions=tf.sigmoid( y )),
                'pr': tf.compat.v1.metrics.auc( labels=labels,
                                      predictions=tf.sigmoid( y ),
                                      curve='PR' )
            }
            return tf.estimator.EstimatorSpec( mode, loss=cross_entropy, eval_metric_ops=eval_metric_ops )

    return model_fn_helper


def build_estimator_helper(model_fn, params):
    def build_estimator(config):

        if config.data_name not in model_fn:
            raise Exception('Only [{}] are supported'.format(','.join(model_fn.keys())))

        run_config = tf.estimator.RunConfig(
            save_summary_steps=50,
            log_step_count_steps=50,
            keep_checkpoint_max = 3,
            save_checkpoints_steps =50
        )

        model_dir = config.checkpoint_dir

        estimator = tf.estimator.Estimator(
            model_fn = model_fn[config.data_name],
            config = run_config,
            params = params[config.data_name],
            model_dir = model_dir
        )

        return estimator
    return build_estimator
