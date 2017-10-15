import tensorflow as tf
import math
from tensorflow.contrib.learn import ModeKeys





def get_train_op_fn(loss, params):
    return tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        optimizer=tf.train.AdamOptimizer,
        learning_rate=params.learning_rate
    )


def get_eval_metric_ops(labels, predictions):
    return {
        'Accuracy': tf.metrics.accuracy(
            labels=tf.argmax(labels, axis=-1),
            predictions=predictions,
            name='accuracy')
    }


def model_fn(features, labels, mode, params):
    filter_size = [512]
    filter_names = ['64', '128', '256', '512a', '512b']
    dropouts = [0.1, 0.2, 0.3, 0.5, 0.5]
    output_size = labels.shape[1]
    is_training = mode == ModeKeys.TRAIN
    logits = architecture(features, filter_size, filter_names, dropouts, output_size, is_training)
    predictions = tf.argmax(logits, axis=-1)
    loss = None
    train_op = None
    eval_metric_ops = {}
    if mode != ModeKeys.INFER:
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=tf.cast(labels, tf.int32),
            logits=logits)
        train_op = get_train_op_fn(loss, params)
        eval_metric_ops = get_eval_metric_ops(labels, predictions)
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops
    )
