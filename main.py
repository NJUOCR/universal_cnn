import os

import tensorflow as tf
from args import args
from data import SingleCharData as Data
# from data import RotationData as Data
from models.single_char_model import Model


class Main:
    def __init__(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.saver = None

    def run(self, mode):
        self.sess.run(tf.global_variables_initializer())
        if mode in ('train',):
            self.train()
        elif mode in ('infer', 'pred'):
            self.infer(dump=True)
        else:
            print('%s ??' % mode)

    def train(self):

        model = Model(args['input_width'], args['input_height'], args['num_class'], 'train')
        model.build()
        self.sess.run(tf.global_variables_initializer())

        val_data = Data(args['input_height'], args['input_width'], args['num_class']) \
            .load_char_map(args['charmap_path']) \
            .read(args['dir_val'], size=args['val_size'], make_char_map=True)
        # .dump_char_map('label_maps/single_char.json')
        train_data = Data(args['input_height'], args['input_width'], args['num_class']) \
            .load_char_map(args['charmap_path']) \
            .read(args['dir_train'], size=args['train_size'], make_char_map=False) \
            .shuffle_indices()
        print('start training')

        if args['restore']:
            self.restore()

        # init tensorboard
        writer = tf.summary.FileWriter(args['tb_dir'])

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
                        loss, _acc, acc = self.sess.run(
                            [model.loss, model.val_acc_update_op, model.val_acc],
                            feed_dict=val_feed_dict)
                        val_cost += loss
                        val_samples += batch_size
                        val_batch = val_data.next_batch(batch_size)
                    loss = val_cost / val_samples
                    custom_sm = tf.Summary(value=[
                        tf.Summary.Value(tag="accuracy", simple_value=acc)
                    ])
                    writer.add_summary(custom_sm, step)
                    print("#validation: accuracy=%.6f,\t average_batch_loss:%.4f" % (acc, loss))
                    cost_between_val = samples_between_val = 0
        self.save(step)

    def infer(self, infer_data=None, batch_size=None, ckpt_dir=None, dump=False):
        model = Model(args['input_width'], args['input_height'], args['num_class'], 'infer')
        model.build()
        self.restore(ckpt_dir=ckpt_dir)
        print("start inferring")
        batch_size = batch_size or args['batch_size']
        infer_data = infer_data or Data(args['input_height'], args['input_width'], args['num_class']) \
            .load_char_map(args['charmap_path']) \
            .read(args['dir_infer']) \
            .init_indices()

        infer_batch = infer_data.next_batch(batch_size)
        self.sess.run(tf.local_variables_initializer())

        buff = []
        while infer_batch is not None:
            infer_images, infer_labels = infer_batch
            infer_feed_dict = model.feed(infer_images, infer_labels)
            classes = self.sess.run(model.classes,
                                    feed_dict=infer_feed_dict)
            buff += infer_data.unmap(classes.tolist())
            infer_batch = infer_data.next_batch(batch_size)

        if not dump:    return buff

        with open(args['infer_output_path'], 'w', encoding='utf-8') as f:
            for infer, label in zip(buff, infer_data.labels):
                f.write("%s - %s\n" % (infer, infer_data.unmap(label)))
        print("infer result dumped to %s" % args['infer_output_path'])
        return buff

    def restore(self, ckpt_dir=None):
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        ckpt_dir = ckpt_dir or args['ckpt']
        ckpt = tf.train.latest_checkpoint(ckpt_dir)
        if ckpt:
            self.saver.restore(self.sess, ckpt)
            print('successfully restored from %s' % ckpt_dir)
        else:
            print('cannot restore from %s' % ckpt_dir)

    def save(self, step):
        if self.saver is None:
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
        self.saver.save(self.sess, os.path.join(args['ckpt'], '%s_model' % str(args['name'])), global_step=step)
        print('ckpt saved')

    def variable_summaries(self, var):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)

            # 计算参数的标准差
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))

            tf.summary.scalar('histogram', var)


def main(_):
    print('using tensorflow', tf.__version__)
    m = Main()
    if args['gpu'] == -1:
        dev = '/cpu:0'
    else:
        dev = '/gpu:%d' % args['gpu']

    with tf.device(dev):
        m.run(args['mode'])


if __name__ == '__main__':
    tf.logging.set_verbosity('INFO')
    # cmd_args.mode = 'infer'
    tf.app.run()
