import tensorflow as tf
from tensorflow.contrib.learn import ModeKeys
from math import sqrt

class PityCNN:
    def __init__(self, summary_net = [64]):
        self.summary_net = summary_net
        self.
        pass

    def block(self, inputs, filters, name, dropout, is_training):
        conv1 = tf.layers.conv2d(inputs,
                                 filters,
                                 4,
                                 padding="same",
                                 activation=tf.nn.relu,
                                 name="conv{}_1".format(name))
        conv2 = tf.layers.conv2d(conv1,
                                 filters,
                                 4,
                                 padding="same",
                                 activation=tf.nn.relu,
                                 name="conv{}_2".format(name))
        conv3 = tf.layers.conv2d(conv2,
                                 filters,
                                 4,
                                 padding="same",
                                 activation=tf.nn.relu,
                                 name="conv{}_3".format(name))
        pool = tf.layers.max_pooling2d(conv3,
                                       pool_size=2,
                                       strides=2,
                                       name="pool{}".format(name))
        summary = None
        if filters in self.summary_net:
            ix = 260
            iy = 260
            channels = filters
            V = tf.slice(conv3, (0, 0, 0, 0), (1, -1, -1, -1))
            V = tf.reshape(V, (iy, ix, channels))
            ix += 4
            iy += 4
            V = tf.image.resize_image_with_crop_or_pad(V, iy, ix)
            cy = sqrt(filters)
            cx = sqrt(filters)
            V = tf.reshape(V, (iy, ix, cy, cx))
            V = tf.transpose(V, (2, 0, 3, 1))
            V = tf.reshape(V, (1, cy * iy, cx * ix, 1))
            summary = tf.summary.image("image_conv{}_3".format(name), V)

        dropout = tf.layers.dropout(pool, dropout, training=is_training, name="dropout{}".format(name))
        return dropout, summary

    def architecture(self, inputs, filters_size, filter_names, dropouts, output_size, is_training):
        summaries = []
        with tf.variable_scope('CovNet'):

            inputs = tf.layers.batch_normalization(inputs, name="batch_normalization")

            for i, filter_size in enumerate(filters_size):
                with tf.name_scope("conv-maxpool-{}s".format(filter_names[i])):
                    inputs, summary = self.block(inputs, filter_size, filter_names[i], dropouts[i], is_training)
                    if summary:
                        summaries.append(summary)

            with tf.name_scope("conv-dense"):
                inputs = tf.reshape(inputs, [-1, 8 * 8 * 512])
                dense1 = tf.layers.dense(
                    inputs,
                    4096,
                    activation=tf.nn.relu,
                    name='dense1',
                )
                dropout = tf.layers.dropout(dense1, 0.5, training=is_training, name="dense_dropout")
                dense2 = tf.layers.dense(
                    dropout,
                    4096,
                    activation=tf.nn.relu,
                    name='dense2',
                )
                net = tf.layers.dense(
                    dense2,
                    output_size,
                    name='dense3',
                )

        return net, summaries

    def get_train_op_fn(self, loss, params):
        return tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            optimizer=tf.train.AdamOptimizer,
            learning_rate=params.learning_rate
        )

    def get_eval_metric_ops(self, labels, predictions):
        return {
            'Accuracy': tf.metrics.accuracy(
                labels=tf.argmax(labels, 0),
                predictions=predictions,
                name='accuracy')
        }

    def model_fn(self, features, labels, mode, params):
        filter_size = [64, 128, 256, 512, 512]
        filter_names = ['64', '128', '256', '512a', '512b']
        dropouts = [0.1, 0.2, 0.3, 0.5, 0.5]
        output_size = labels.shape[1]
        is_training = mode == ModeKeys.TRAIN
        logits, summaries = self.architecture(features, filter_size, filter_names, dropouts, output_size, is_training)
        predictions = tf.argmax(logits, axis=-1)
        loss = None
        train_op = None
        eval_metric_ops = {}
        if mode != ModeKeys.INFER:
            loss = tf.losses.softmax_cross_entropy(
                onehot_labels=tf.cast(labels, tf.int32),
                logits=logits)
            train_op = self.get_train_op_fn(loss, params)
            eval_metric_ops = self.get_eval_metric_ops(labels, predictions)
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops
        )
