import tensorflow as tf


def get_conv_linear_model(x, y_, is_training):
    # X has shape (None, 784)
    with tf.name_scope("conv"):
        input_reshaped = tf.reshape(x, (-1, 28, 28, 1))

        # Kernel shape (height, width, input_channels, output_channels)
        K = tf.Variable(tf.truncated_normal((3, 3, 1, 4), stddev=0.1))
        kb = tf.Variable(tf.constant(0.1, shape=[4,]))

        conv_out = tf.nn.conv2d(input_reshaped, K, [1, 1, 1, 1], 'SAME')
        conv_out = tf.nn.bias_add(conv_out, kb)

        # Shape (None, 28, 28, 1)
        conv_out = tf.nn.relu(conv_out)

        # Shape (None, 784)
        conv_out = tf.reshape(conv_out, (-1, 784*4))

    with tf.name_scope("linear_1"):
        W1 = tf.Variable(tf.truncated_normal((784*4, 256), stddev=0.1),
                        trainable=True)
        b1 = tf.Variable(tf.constant(0.1, shape=(256,), dtype=tf.float32))

        linear_out = tf.nn.relu(tf.matmul(conv_out, W1) + b1)

    with tf.name_scope("linear_out"):
        W2 = tf.Variable(tf.truncated_normal((256, 10), stddev=0.1),
                        trainable=True)
        b2 = tf.Variable(tf.constant(0.1, shape=(10,), dtype=tf.float32))

        linear_out = tf.matmul(linear_out, W2) + b2

    y = linear_out

    # The fully connected layer
    # Loss function
    # Note that tf.nn.softmax_cross_entropy_with_logits internally applies the softmax on the
    # model's unnormalized model prediction and sums across all classes, and tf.reduce_mean
    # takes the average over these sums
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)) \
                    + 0.001 * tf.nn.l2_loss(K) \
                    + 0.005 * tf.nn.l2_loss(W1) \
                    + 0.01 * tf.nn.l2_loss(W2)

    # Training
    train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
    # train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

    return tf.nn.softmax(linear_out), train_step
