import math

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

        if params['model_name'] == 'userrecognition':
            y, y_recognition = model_fn(features, labels, mode, params)
        else:
            y = model_fn(features, labels, mode, params)

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'prediction_prob': tf.sigmoid( y )
            }
            return tf.estimator.EstimatorSpec( mode=tf.estimator.ModeKeys.PREDICT,
                                               predictions=predictions )

        
        if params['model_name'] == 'usercluster':
            print("usercluster loss!")
            cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=y))
            cross_entropy += params['weight_of_loss_sim']*tf.reduce_sum(tf.compat.v1.get_collection('all_loss_sim'))
            if params['use_cluster_loss']:
                cross_entropy += tf.reduce_sum(tf.compat.v1.get_collection('all_loss_cluster'))
        elif params['model_name'] == 'userloss':
            print("userloss!")
            if params['data_name'] == 'amazon':
                user_group_name = 'reviewer_group'
            elif params['data_name'] == 'movielens':
                user_group_name = 'user_group'
            # 'reviewer_group': <tf.Tensor 'IteratorGetNext:5' shape=(?,) dtype=int64>
            user_level_0_weight = tf.cast(tf.reshape(tf.equal(features[user_group_name], 0), [-1, 1]), tf.float32) * params['weight_of_user_0']
            user_level_1_weight = tf.cast(tf.reshape(tf.equal(features[user_group_name], 1), [-1, 1]), tf.float32) * params['weight_of_user_1']
            user_level_2_weight = tf.cast(tf.reshape(tf.equal(features[user_group_name], 2), [-1, 1]), tf.float32) * params['weight_of_user_2']
            final_weight = user_level_0_weight+user_level_1_weight+user_level_2_weight
            cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=y)*final_weight)
        elif params['model_name'] == 'userrecognition':
            print("userrecognition loss!")
            cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=y))
            if params['data_name'] == 'amazon':
                user_group_name = 'reviewer_group'
            elif params['data_name'] == 'movielens':
                user_group_name = 'user_group'
            user_level = tf.reshape(features[user_group_name], [-1])
            user_level = tf.one_hot(user_level, params['num_user_group'])
            user_level = tf.cast(user_level, tf.float32)
            cross_entropy += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=user_level, logits=y_recognition))
        elif params['model_name'] == 'usersparseexpert':
            print("usersparseexpert loss!")
            cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=y))
            if params['use_dselect']:
                cross_entropy += tf.reduce_sum(tf.compat.v1.get_collection('all_loss_entropy'))
            else:
                cross_entropy += tf.reduce_sum(tf.compat.v1.get_collection('all_loss_balance'))
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

def _my_top_k(x, k):
    if k > 10:
        return tf.nn.top_k(x, k)
    values = []
    indices = []
    depth = tf.shape(x)[1]
    for i in range(k):
        values.append(tf.reduce_max(x, 1))
        argmax = tf.argmax(x, 1)
        indices.append(argmax)
        if i + 1 < k:
            x += tf.one_hot(argmax, depth, -1e9)
    return tf.stack(values, axis=1), tf.to_int32(tf.stack(indices, axis=1))

def _rowwise_unsorted_segment_sum(values, indices, n):
    batch, k = tf.unstack(tf.shape(indices), num=2)
    indices_flat = tf.reshape(indices, [-1]) + tf.div(tf.range(batch * k), k) * n
    ret_flat = tf.unsorted_segment_sum(
        tf.reshape(values, [-1]), indices_flat, batch * n)
    return tf.reshape(ret_flat, [batch, n])

def _prob_in_top_k(clean_values, noisy_values, noise_stddev, noisy_top_values, params):
    batch = tf.shape(clean_values)[0]
    m = tf.shape(noisy_top_values)[1]
    top_values_flat = tf.reshape(noisy_top_values, [-1])
    threshold_positions_if_in = tf.range(batch) * m + params['k']
    threshold_if_in = tf.expand_dims(tf.gather(top_values_flat, threshold_positions_if_in), 1)
    is_in = tf.greater(noisy_values, threshold_if_in)
    if noise_stddev is None:
        return tf.to_float(is_in)
    threshold_positions_if_out = threshold_positions_if_in - 1
    threshold_if_out = tf.expand_dims(tf.gather(top_values_flat, threshold_positions_if_out), 1)
    # is each value currently in the top k.
    prob_if_in = _normal_distribution_cdf(clean_values - threshold_if_in,
                                            noise_stddev)
    prob_if_out = _normal_distribution_cdf(clean_values - threshold_if_out,
                                            noise_stddev)
    prob = tf.where(is_in, prob_if_in, prob_if_out)
    return prob

def _normal_distribution_cdf(x, stddev):
    return 0.5 * (1.0 + tf.erf(x / (math.sqrt(2) * stddev + 1e-20)))

def _gates_to_load(gates):
    return tf.reduce_sum(tf.to_float(gates > 0), 0)

def cv_squared(x):
    epsilon = 1e-10
    float_size = tf.to_float(tf.size(x)) + epsilon
    mean = tf.reduce_sum(x) / float_size
    variance = tf.reduce_sum(tf.squared_difference(x, mean)) / float_size
    return variance / (tf.square(mean) + epsilon)

def smooth_step(inputs, gamma = 1.0):
    _lower_bound = -gamma / 2
    _upper_bound = gamma / 2
    _a3 = -2 / (gamma**3)
    _a1 = 3 / (2 * gamma)
    _a0 = 0.5
    return tf.where(
        inputs <= _lower_bound, tf.zeros_like(inputs),
        tf.where(inputs >= _upper_bound, tf.ones_like(inputs),
                 _a3 * (inputs**3) + _a1 * inputs + _a0))

def _add_entropy_regularization_loss(params, selector_outputs):
    """Adds regularization loss based on the selector outputs.
    Args:
      selector_outputs: a tensor with shape (batch_size, num_nonzero,
        num_experts) or (num_nonzero, num_experts), where the last dimension
        stores the weight vector of each single-expert selector.
    """
    num_binary = math.ceil(math.log2(params['num_of_expert']))
    _power_of_2 = (params['num_of_expert'] == 2**num_binary)

    _num_calls = tf.compat.v1.get_variable('num_calls', [], initializer=tf.zeros_initializer(), trainable=False)
    _schedule_fn = lambda x: 1e-6
    assign_op = _num_calls.assign_add(1, read_value=False)
    preconditions = [] if assign_op is None else [assign_op]
    with tf.control_dependencies(preconditions):
        reg_param = _schedule_fn(_num_calls)
        entropy = -tf.math.reduce_sum(selector_outputs * tf.math.log(selector_outputs + 1e-6))
        loss =  reg_param * entropy

    if not _power_of_2:
    # If the number of experts is not a power of 2, we add a regularization
    # term to prevent the "non-reachable" experts from getting all the nonzero
    # weights for any single-expert selector. The regularization term is equal
    # to 1/sum(weights of reachable experts) so that the reachable experts
    # cannot get zero weights.
    # In case of example conditioning, this regularizer is added per example.
    # NOTE: This regularization term has no effect once the sum of the weights
    # of the reachable experts reaches 1, which is the typical/expected case.
        loss += tf.math.reduce_sum(1 / tf.math.reduce_sum(selector_outputs, axis=-1))
    return loss