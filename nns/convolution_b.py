import tensorflow as tf


def get_convolutional_b_model(x, y_, is_training):
    # X has shape (None, 784)
    with tf.name_scope("conv"):
        input_reshaped = tf.reshape(x, (-1, 28, 28, 1))

        # Kernel shape (height, width, input_channels, output_channels)
        K = tf.Variable(tf.truncated_normal((3, 3, 1, 2), stddev=0.1))
        kb = tf.Variable(tf.constant(0.1, shape=[2,]))

        conv_out = tf.nn.conv2d(input_reshaped, K, [1, 1, 1, 1], 'SAME')
        conv_out = tf.nn.bias_add(conv_out, kb)

        conv_out = tf.contrib.layers.batch_norm(
            conv_out,
            center=True,
            scale=False,
            is_training=is_training,
            updates_collections=None,
            decay=0.99
        )

        # Shape (None, 28, 28, 1)
        conv_out = tf.nn.relu(conv_out)

        # Shape (None, 784)
        conv_out = tf.reshape(conv_out, (-1, 784*2))

    with tf.name_scope("linear"):
        W = tf.Variable(tf.truncated_normal((784*2, 10), stddev=0.1),
                        trainable=True)
        b = tf.Variable(tf.constant(0.1, shape=(10,), dtype=tf.float32))

        linear_out = tf.matmul(conv_out, W) + b

    y = linear_out

    # The fully connected layer
    # Loss function
    # Note that tf.nn.softmax_cross_entropy_with_logits internally applies the softmax on the
    # model's unnormalized model prediction and sums across all classes, and tf.reduce_mean
    # takes the average over these sums
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    # Training
    train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
    # train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

    return tf.nn.softmax(linear_out), train_step
