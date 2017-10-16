import tensorflow as tf
import math
import numpy as np
import datetime
from os import path

from pitycnn.inputs import Inputs
from pitycnn.prep import prepare_data

image_width = 234
image_height = 234
image_depth = 3


class PityCnn:
    def __init__(self, data, batch_size, learning_rate, model_path, last_file=None, display_step=100):
        self.data = data
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model_path = model_path
        self.last_file = last_file
        self.display_step = display_step
        self.__prepare_data()
        self.__create_model()
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.config = tf.ConfigProto(gpu_options=gpu_options)
        self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

    def load_checkpoint(self, last_file):
        self.last_file = last_file

    def predict(self, data):
        with tf.Session() as sess:
            sess.close()

    def train(self, num_epochs):
        next_train = self.iterator_train.get_next()
        next_valid = self.iterator_valid.get_next()
        with tf.Session(config=self.config) as sess:

            tf.global_variables_initializer().run()
            if self.last_file:
                self.saver.restore(sess, self.last_file)

            self.writer.add_graph(sess.graph)

            for epoch in range(num_epochs):

                sess.run(self.iterator_train.initializer)

                for step in range(self.train_batches_per_epoch):
                    batch_data, batch_labels = sess.run(next_train)
                    feed_dict = {self.features: batch_data, self.labels: batch_labels}
                    _, l, predictions = sess.run([self.optimizer, self.loss, self.train_prediction],
                                                 feed_dict=feed_dict, options=self.run_options)

                    if step % self.display_step == 0:
                        s = sess.run(self.merged_summary, feed_dict={self.labels: batch_labels,
                                                                     self.logits: predictions,
                                                                     self.is_training: True})

                        self.writer.add_summary(s, epoch * self.train_batches_per_epoch + step)

                sess.run(self.iterator_valid.initializer)
                valid_acc = 0.
                valid_count = 0
                for _ in range(self.valid_batches_per_epoch):
                    batch_data, batch_labels = sess.run(next_valid)
                    acc = sess.run(self.accuracy, feed_dict={self.features: batch_data,
                                                             self.labels: batch_labels, self.is_training: True})
                    valid_acc += acc
                    valid_count += 1
                valid_acc /= valid_count
                print("{} Validation Accuracy = {:.4f}".format(datetime.now(),
                                                               valid_acc))
                self.last_file = 'model_epoch{}.ckpt'.format(epoch + 1)
                checkpoint_name = path.join(self.model_path, self.last_file)
                self.saver.save(sess, checkpoint_name)

            sess.close()

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
        self.is_training = tf.placeholder(tf.bool)
        self.features = tf.placeholder(tf.float32, shape=(None, image_width, image_height, image_depth),
                                       name="features")
        self.labels = tf.placeholder(tf.float32, shape=(None, self.n_classes), name="labels")

        self.filters_size = [64, 128, 256, 512, 512]
        self.filter_names = ['64', '128', '256', '512a', '512b']
        self.dropouts = [0.1, 0.2, 0.3, 0.5, 0.5]
        self.logits = self.__architecture()
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.predictions = tf.nn.softmax(self.logits)
        correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.features, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def __block(self, inputs, filters, name, dropout):

        conv1 = tf.layers.conv2d(inputs,
                                 filters,
                                 3,
                                 padding="same",
                                 activation=tf.nn.relu,
                                 name="conv{}_1".format(name))
        conv2 = tf.layers.conv2d(conv1,
                                 filters,
                                 3,
                                 padding="same",
                                 activation=tf.nn.relu,
                                 name="conv{}_2".format(name))
        conv3 = tf.layers.conv2d(conv2,
                                 filters,
                                 3,
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

        dropout = tf.layers.dropout(pool, dropout, training=self.is_training, name="dropout{}".format(name))
        return dropout

    def __architecture(self):
        with tf.variable_scope('CovNet'):
            inputs = tf.layers.batch_normalization(self.features, name="batch_normalization")

            ix = inputs.shape[2].value
            iy = ix

            v = tf.slice(inputs, (0, 0, 0, 0), (1, -1, -1, -1))
            v = tf.reshape(v, (1, iy, ix, 3))
            ix += 4
            iy += 4
            v = tf.image.resize_image_with_crop_or_pad(v, iy, ix)
            tf.summary.image("image_orig", v)

            for i, filter_size in enumerate(self.filters_size):
                with tf.name_scope("conv-maxpool-{}s".format(self.filter_names[i])):
                    inputs = self.__block(inputs, filter_size, self.filter_names[i], self.dropouts[i])

            with tf.name_scope("conv-dense"):
                inputs = tf.reshape(inputs, [-1, 8 * 8 * 512])
                dense1 = tf.layers.dense(
                    inputs,
                    4096,
                    activation=tf.nn.relu,
                    name='dense1',
                )
                dropout = tf.layers.dropout(dense1, 0.5, training=self.is_training, name="dense_dropout")
                dense2 = tf.layers.dense(
                    dropout,
                    4096,
                    activation=tf.nn.relu,
                    name='dense2',
                )
                net = tf.layers.dense(
                    dense2,
                    self.n_classes,
                    name='dense3',
                )

        return net

    def __init_summary(self):
        tf.summary.scalar('cross_entropy', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)
        self.merged_summary = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.model_path)
        self.saver = tf.train.Saver()

