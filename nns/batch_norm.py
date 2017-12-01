import tensorflow as tf


def get_batch_norm_model(x, y_, is_training):
    W = tf.Variable(tf.truncated_normal((784, 10), stddev=0.1),
                    trainable=True)
    b = tf.Variable(tf.constant(0.1, shape=(10,), dtype=tf.float32))

    y = tf.matmul(x, W) + b

    y = tf.contrib.layers.batch_norm(
        y,
        data_format='NHWC',
        center=True,
        scale=False,
        is_training=is_training)

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)) \
                    + 0.05 * tf.nn.l2_loss(W)

    # Training
    train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

    # The fully connected layer
    return tf.nn.softmax(y), train_step
