import tensorflow as tf


def get_conv_b_max_lstm_lin_model(x, y_, is_training, *args):
    keep_prob = tf.cond(is_training,
                        true_fn=lambda: 0.5,
                        false_fn=lambda: 1.)

    # X has shape (None, 784)
    with tf.name_scope("conv"):
        output_depth = 4

        input_reshaped = tf.reshape(x, (-1, 28, 28, 1))

        # Kernel shape (height, width, input_channels, output_channels)
        K = tf.Variable(tf.truncated_normal((3, 3, 1, output_depth), stddev=0.1))
        kb = tf.Variable(tf.constant(0.1, shape=[output_depth, ]))

        conv_out = tf.nn.conv2d(input_reshaped, K, [1, 1, 1, 1], 'SAME')
        conv_out = tf.nn.bias_add(conv_out, kb)

        conv_out = tf.contrib.layers.batch_norm(
            conv_out,
            center=True,
            scale=False,
            is_training=is_training,
        )

        # Shape (None, 28, 28, 1)
        conv_out = tf.nn.relu(conv_out)

        conv_out = tf.nn.dropout(conv_out, keep_prob)

    with tf.name_scope("max_pool"):
        max_out = tf.nn.max_pool(conv_out, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

        # Shape (None, 784)
        max_out = tf.reshape(max_out, (-1, 14, 14 * output_depth))

    with tf.name_scope("lstm"):
        lstm_units = 10

        # Create lstm layer
        lstm = tf.contrib.rnn.LSTMCell(lstm_units, forget_bias=0.)
        # Put it into Multi RNN Cell
        lstm = tf.contrib.rnn.MultiRNNCell([lstm])
        # Let dynamic rnn setup the control flow (making while loops and stuff)
        # lstm_output shape: (None, 28, 5)
        lstm_output, _ = tf.nn.dynamic_rnn(lstm, max_out, dtype=tf.float32)

        lstm_output_reshaped = tf.reshape(lstm_output, (-1, 14 * lstm_units))

    with tf.name_scope("linear_1"):
        W1 = tf.Variable(tf.truncated_normal((14 * lstm_units, 256), stddev=0.1),
                         trainable=True)
        b1 = tf.Variable(tf.constant(0.1, shape=(256,), dtype=tf.float32))

        lin_activation = tf.matmul(lstm_output_reshaped, W1) + b1

        linear_out = tf.nn.relu(lin_activation)

        linear_out = tf.nn.dropout(linear_out, keep_prob)

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
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    # Training
    train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
    # train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

    return tf.nn.softmax(linear_out), train_step
