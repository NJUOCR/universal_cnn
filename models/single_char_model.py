import tensorflow as tf


class Model:

    def __init__(self, input_width, input_height, num_class, mode):
        self.input_width = input_width
        self.input_height = input_height
        self.num_class = num_class
        self.training = mode.lower() in ('train',)

        self.images = tf.placeholder(tf.float32, [None, input_height, input_width, 1], name='input_img_batch')
        self.labels = tf.placeholder(tf.int32, [None], name='input_lbl_batch')

        # define op
        self.step = None
        self.loss = None
        self.classes = None
        self.train_op = None
        self.val_acc = self.val_acc_update_op = None

    def feed(self, images, labels):
        return {
            self.images: images,
            self.labels: labels
        }

# Todo
    def build(self):
        images = self.images
        labels = self.labels

        input_layer = tf.reshape(images, [-1, self.input_height, self.input_width, 1])

        # cnn block 1
        x = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=3,
            padding='same',
            kernel_initializer=tf.glorot_uniform_initializer()
        )

        x = tf.layers.batch_normalization(
            inputs=x,
            training=self.training
        )

        x = tf.nn.leaky_relu(x, alpha=0.01)

        x = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=2)

        # cnn block 2
        x = tf.layers.conv2d(
            inputs=x,
            filters=64,
            kernel_size=3,
            padding='same',
            kernel_initializer=tf.glorot_uniform_initializer()
        )

        x = tf.layers.batch_normalization(
            inputs=x,
            training=self.training
        )

        x = tf.nn.leaky_relu(x, alpha=0.01)

        x = tf.layers.max_pooling2d(
            inputs=x,
            pool_size=[2, 2],
            strides=2
        )

        # cnn block 3
        x = tf.layers.conv2d(
            inputs=x,
            filters=128,
            kernel_size=3,
            padding='same',
            kernel_initializer=tf.glorot_uniform_initializer()
        )

        x = tf.layers.batch_normalization(
            inputs=x,
            training=self.training
        )

        x = tf.nn.leaky_relu(x, alpha=0.01)

        x = tf.layers.max_pooling2d(
            inputs=x,
            pool_size=[2, 2],
            strides=2
        )

        # cnn block 4
        x = tf.layers.conv2d(
            inputs=x,
            filters=256,
            kernel_size=3,
            padding='same',
            kernel_initializer=tf.glorot_uniform_initializer()
        )

        x = tf.layers.batch_normalization(
            inputs=x,
            training=self.training
        )

        x = tf.nn.leaky_relu(x, alpha=0.01)

        x = tf.layers.max_pooling2d(
            inputs=x,
            pool_size=[2, 2],
            strides=2
        )

        # dense
        x = tf.layers.flatten(
            inputs=x
        )

        x = tf.layers.dense(
            inputs=x,
            units=8192
        )

        logits = tf.layers.dense(
            inputs=x,
            units=self.num_class
        )

        # probabilities = tf.nn.softmax(logits, name='P')
        self.classes = tf.argmax(input=logits, axis=1, name='class')
        self.step = tf.train.get_or_create_global_step()

        self.loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        optimizer = tf.train.AdamOptimizer()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(
                loss=self.loss,
                global_step=self.step
            )

        self.val_acc, self.val_acc_update_op = tf.metrics.accuracy(labels, self.classes)

        return self
