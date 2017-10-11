from pitycnn.iteratorinitializerhook import IteratorInitializerHook
import tensorflow as tf
from tensorflow import name_scope
from tensorflow.contrib.data import Dataset, Iterator
from tensorflow import one_hot, read_file
from tensorflow import constant

class Inputs:
    def __init__(self, images_paths, labels, num_classes, batch_size=10, buffer_size=1000, name="train", shuffle=False):
        self.images_paths = images_paths
        self.labels = labels
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.name = name
        self.shuffle = shuffle
        self.iterator_initializer_hook = IteratorInitializerHook()

    def generate_input_fn(self):
        def _input_fn():
            with name_scope(self.name):
                dataset, images, labls = self.__convert()
                iterator = Iterator.from_dataset(dataset)
                next_element = iterator.get_next()

                training_init_op = iterator.make_initializer(dataset)
                self.iterator_initializer_hook.iterator_initializer_func = lambda sess: sess.run(training_init_op)

                return next_element

        # Return function and hook
        return _input_fn, self.iterator_initializer_hook

    def __input_parser(self, img_path, label):
        self.num_classes
        oh = one_hot(label, self.num_classes)
        img_file = read_file(img_path)
        img_decoded = tf.image.decode_jpeg(img_file, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [260, 260])

        return img_resized, oh

    def __convert(self):
        imgs = constant(self.images_paths)
        labels = constant(self.labels)

        data = Dataset.from_tensor_slices((imgs, labels))

        data = data.map(self.__input_parser, num_threads=8,
                        output_buffer_size=100 * self.batch_size)

        if self.shuffle:
            data = data.shuffle(self.buffer_size)

        data = data.batch(self.batch_size)
        return data, imgs, labels
