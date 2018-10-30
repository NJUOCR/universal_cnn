import tensorflow as tf


class Model:

    def __init__(self, input_width, input_height, num_class, mode):
        self.images = tf.placeholder(tf.float32, shape=[None, input_height, input_width, 1], name='input')
        self.labels = tf.placeholder(tf.float32, shape=[])