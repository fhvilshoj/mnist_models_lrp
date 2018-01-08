import tensorflow as tf


def get_max_pool_convolution_model(x, y_, *args):
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

    conv_out = tf.nn.dropout(conv_out, keep_prob=0.8)

    with tf.name_scope("max_pool"):
        max_out = tf.nn.max_pool(conv_out, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

        # Shape (None, 784)
        max_out = tf.reshape(max_out, (-1, 14*14*4))

    with tf.name_scope("linear"):
        W = tf.Variable(tf.truncated_normal((14*14*4, 10), stddev=0.1),
                        trainable=True)
        b = tf.Variable(tf.constant(0.1, shape=(10,), dtype=tf.float32))

        linear_out = tf.matmul(max_out, W) + b

    y = linear_out

    # The fully connected layer
    # Loss function
    # Note that tf.nn.softmax_cross_entropy_with_logits internally applies the softmax on the
    # model's unnormalized model prediction and sums across all classes, and tf.reduce_mean
    # takes the average over these sums
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    # Training
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    # train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

    return tf.nn.softmax(linear_out), train_step
