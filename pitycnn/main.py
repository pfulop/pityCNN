from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn import RunConfig
from tensorflow.contrib.training import HParams
import tensorflow as tf
import math
import numpy as np

from pitycnn.inputs import Inputs
from pitycnn.experiment import generate_experiment_fn
from pitycnn.prep import prepare_data


class PityCnn:
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size

    def __prepare_data(self):
        train_images, train_labels, valid_images, valid_labels, n_classes = prepare_data(self.data)
        train_inputs = Inputs(train_images, train_labels, n_classes, batch_size=self.batch_size, shuffle=True)
        valid_inputs = Inputs(valid_images, valid_labels, n_classes, name="valid", batch_size=self.batch_size)
        with tf.device('/cpu:0'):
            self.iterator_train = train_inputs.generate_iterator()
            self.iterator_valid = valid_inputs.generate_iterator()

        self.train_batches_per_epoch = int(np.floor(train_inputs.size / self.batch_size))
        self.valid_batches_per_epoch = int(np.floor(train_inputs.size / self.batch_size))
        self.n_classes = n_classes

    def __create_model(self):
        filter_size = [64, 128, 256, 512, 512]
        filter_names = ['64', '128', '256', '512a', '512b']
        dropouts = [0.1, 0.2, 0.3, 0.5, 0.5]
        is_training = mode == ModeKeys.TRAIN
        logits = architecture(features, filter_size, filter_names, dropouts, self.n_classes, is_training)
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

    def __block(self, inputs, filters, name, dropout, is_training):

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

        ix = inputs.shape[2].value
        iy = ix

        channels = filters
        v = tf.slice(conv3, (0, 0, 0, 0), (1, -1, -1, -1))
        v = tf.reshape(v, (iy, ix, channels))
        ix += 4
        iy += 4
        v = tf.image.resize_image_with_crop_or_pad(v, iy, ix)
        cy = math.sqrt(channels)

        iswhole = cy.is_integer()
        if not iswhole:
            cy = (int)(math.ceil(cy))
        else:
            cy = (int)(cy)

        cx = cy

        if not iswhole:
            pad = (cy ** 2) - channels
            v = tf.pad(v, [[0, 0], [0, 0], [0, pad]])

        v = tf.reshape(v, (iy, ix, cy, cx))
        v = tf.transpose(v, (2, 0, 3, 1))
        v = tf.reshape(v, (1, cy * iy, cx * ix, 1))
        tf.summary.image("image_conv{}_3".format(name), v)

        dropout = tf.layers.dropout(pool, dropout, training=is_training, name="dropout{}".format(name))
        return dropout

    def __architecture(self, inputs, filters_size, filter_names, dropouts, output_size, is_training):
        with tf.variable_scope('CovNet'):
            inputs = tf.layers.batch_normalization(inputs, name="batch_normalization")

            ix = inputs.shape[2].value
            iy = ix

            v = tf.slice(inputs, (0, 0, 0, 0), (1, -1, -1, -1))
            v = tf.reshape(v, (1, iy, ix, 3))
            ix += 4
            iy += 4
            v = tf.image.resize_image_with_crop_or_pad(v, iy, ix)
            tf.summary.image("image_orig", v)

            for i, filter_size in enumerate(filters_size):
                with tf.name_scope("conv-maxpool-{}s".format(filter_names[i])):
                    inputs = self.__block(inputs, filter_size, filter_names[i], dropouts[i], is_training)

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

        return net


def main(files, gpu_memory_fraction=1, min_eval_frequency=500, train_steps=5000, learning_rate=0.001,
         job_dir='model', batch_size=128):
    train_images, train_labels, valid_images, valid_labels, n_classes = prepare_data(files)

    params = HParams(
        learning_rate=learning_rate,
        n_classes=n_classes,
        train_steps=train_steps,
        min_eval_frequency=min_eval_frequency
    )

    experiment_fn = generate_experiment_fn(train_images, train_labels, valid_images, valid_labels, n_classes,
                                           batch_size=batch_size)

    run_config = RunConfig(gpu_memory_fraction=gpu_memory_fraction)
    run_config = run_config.replace(model_dir=job_dir)

    learn_runner.run(experiment_fn, run_config=run_config, schedule="train_and_evaluate", hparams=params)
