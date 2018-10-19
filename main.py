import os

import tensorflow as tf

from args import args
from data import RotationData
from model import Model


class Main:
    def __init__(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.saver = None

    def run(self, mode):
        self.sess.run(tf.global_variables_initializer())
        if mode in ('train', ):
            self.train()
        elif mode in ('infer', 'pred'):
            self.infer()
        else:
            print('%s ??' % mode)

    def train(self):
        print('start training')

        model = Model(args['input_width'], args['input_height'], args['num_class'], 'train')
        model.build()
        self.sess.run(tf.global_variables_initializer())

        val_data = RotationData(args['input_height'], args['input_width'], args['num_class']).read(args['dir_val'])
        train_data = RotationData(args['input_height'], args['input_width'], args['num_class']).read(args['dir_train'])

        if args['restore']:
            self.restore()

        # start training
        step = 0
        cost_between_val = 0
        samples_between_val = 0
        batch_size = args['batch_size']
        for itr in range(args['num_epochs']):
            train_data.shuffle_indices()
            train_batch = train_data.next_batch(batch_size)
            while train_batch is not None:
                images, labels = train_batch
                feed_dict = model.feed(images, labels)
                step, loss, _ = self.sess.run([model.step, model.loss, model.train_op],
                                              feed_dict=feed_dict)
                train_batch = train_data.next_batch(batch_size)
                cost_between_val += loss
                samples_between_val += batch_size

                if step % args['save_interval'] == 1:
                    self.save(step)

                if step % args['val_interval'] == 0:
                    print("#%d[%d]\t\t" % (step, itr), end='')

                    val_data.init_indices()
                    val_batch = val_data.next_batch(batch_size)

                    self.sess.run(tf.local_variables_initializer())
                    acc = 0.0
                    val_cost = val_samples = 0
                    while val_batch is not None:
                        val_image, val_labels = val_batch
                        val_feed_dict = model.feed(val_image, val_labels)
                        loss, _acc, acc = self.sess.run([model.loss, model.val_acc_update_op, model.val_acc],
                                                        feed_dict=val_feed_dict)
                        val_cost += loss
                        val_samples += batch_size
                        val_batch = val_data.next_batch(batch_size)
                    print("#validation: accuracy=%.6f,\t average_batch_loss:%.4f" % (acc, val_cost / val_samples))
                    cost_between_val = samples_between_val = 0
        self.save(step)

    def infer(self):
        model = Model(args['input_width'], args['input_height'], args['num_class'], 'infer')
        model.build()
        self.restore()
        print("start inferring")
        batch_size = args['batch_size']
        infer_data = RotationData(args['input_height'], args['input_width'], args['num_class'])
        infer_data.read(args['dir_infer'])
        infer_data.init_indices()
        infer_batch = infer_data.next_batch(batch_size)
        self.sess.run(tf.local_variables_initializer())
        while infer_batch is not None:
            infer_images, infer_labels = infer_batch
            infer_feed_dict = model.feed(infer_images, infer_labels)
            classes = self.sess.run([model.classes],
                                    feed_dict=infer_feed_dict)

    def restore(self):
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
        ckpt = tf.train.latest_checkpoint(args['ckpt'])
        if ckpt:
            self.saver.restore(self.sess, ckpt)
            print('successfully restored from %s' % args['ckpt'])
        else:
            print('cannot restore from %s' % args['ckpt'])

    def save(self, step):
        if self.saver is None:
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
        self.saver.save(self.sess, os.path.join(args['ckpt'], 'rotation_model'), global_step=step)
        print('ckpt saved')


def main(_):
    print('using tensorflow', tf.__version__)
    m = Main()
    if args['gpu'] == -1:
        dev = '/cpu:0'
    else:
        dev = '/gpu:%d' % args['gpu']

    # with tf.device(dev):
    #     print('---')
    #     print(dev)
    #     m.run('train')
    m.run('train')


if __name__ == '__main__':
    tf.logging.set_verbosity('INFO')
    tf.app.run()
