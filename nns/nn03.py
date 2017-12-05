import tensorflow as tf


def get_linear_nn03_model(x, y_, is_training):    
    with tf.name_scope("linear"):
        W1 = tf.Variable(tf.truncated_normal((784, 1024), stddev=0.1),
                        trainable=True)
        b1 = tf.Variable(tf.constant(0.1, shape=(1024,), dtype=tf.float32))

        y = tf.nn.relu(tf.matmul(x, W1) + b1)

    with tf.name_scope("linear_out"):
        W2 = tf.Variable(tf.truncated_normal((1024, 10), stddev=0.1),
                         trainable=True)
        b2 = tf.Variable(tf.constant(0.1, shape=(10,), dtype=tf.float32))

        y = tf.matmul(y, W2) + b2

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    
    # Training
    train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

    # The fully connected layer
    return tf.nn.softmax(y), train_step
