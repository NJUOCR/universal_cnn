import tensorflow as tf

from args import args


class Main:
    def __init__(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.saver = None


if __name__ == '__main__':
    print(args)
