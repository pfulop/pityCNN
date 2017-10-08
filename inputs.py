import tensorflow as tf
from tensorflow.contrib.data import Dataset, Iterator

def input_parser(img_path, label):
    NUM_CLASSES=10
    one_hot = tf.one_hot(label, NUM_CLASSES)
    img_file = tf.read_file(img_path)
    img_decoded = tf.image.decode_jpeg(img_file, channels = 3)
    img_resized = tf.image.resize_images(img_decoded, [260, 260])

    return img_resized, one_hot


def convert(images,labels, mode = 'training', batch_size = 10, shuffle = True, buffer_size=1000):

    imgs = tf.constant(images)
    labels = tf.constant(labels)
    
    data = Dataset.from_tensor_slices((imgs, labels))
    
    if mode == 'training':
        data = data.map(input_parser, num_threads=8,
                    output_buffer_size=100*batch_size)
    
    elif mode == 'inference':
        data = data.map(input_parser, num_threads=8,
                        output_buffer_size=100*batch_size)
    
    if shuffle:
    	data = data.shuffle(buffer_size)
    data = data.batch(batch_size)
    return data, imgs, labels    

