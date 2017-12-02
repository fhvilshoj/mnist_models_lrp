import tensorflow as tf


def get_one_block(*args):
    return get_model_with_bolcks(1, *args)


def get_five_blocks(*args):
    return get_model_with_bolcks(5, *args)


def get_ten_blocks(*args):
    return get_model_with_bolcks(10, *args)


def get_model_with_bolcks(blocks, x, y_, is_training, *args):
    keep_prob = tf.cond(is_training,
                        true_fn=lambda: 0.7,
                        false_fn=lambda: 1.)

    output = x

    for i in range(blocks):
        with tf.name_scope("block_{}".format(i)):
            output = tf.reshape(output, (-1, 28, 28, 1))

            # X has shape (None, 784)
            with tf.name_scope("conv_{}".format(i)):
                # Kernel shape (height, width, input_channels, output_channels)
                K = tf.get_variable('Kernel_{}'.format(i),
                                    shape=(3, 3, 1, 2),
                                    initializer=tf.contrib.layers.variance_scaling_initializer())

                # K = tf.Variable(tf.truncated_normal((3, 3, 1, 2), stddev=0.1))
                kb = tf.Variable(tf.constant(1., shape=[2, ]))

                conv_out = tf.nn.conv2d(output, K, [1, 1, 1, 1], 'SAME')
                conv_out = tf.nn.bias_add(conv_out, kb)

                conv_out = tf.contrib.layers.batch_norm(
                    conv_out,
                    center=True,
                    scale=False,
                    is_training=is_training,
                )

            # Shape (None, 28, 28, 2)
            conv_out = tf.nn.relu(conv_out)

            with tf.name_scope("max_pool_{}".format(i)):
                max_out = tf.nn.max_pool(conv_out, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

            # Shape (None, 784)
            max_out = tf.reshape(max_out, (-1, 14, 14 * 2))

            with tf.name_scope("lstm_{}".format(i)):
                lstm_units = 64
            with tf.variable_scope("lstm_{}".format(i)):
                # Create lstm layer
                lstm = tf.contrib.rnn.LSTMCell(lstm_units, forget_bias=0.)
            # Put it into Multi RNN Cell
            lstm = tf.contrib.rnn.MultiRNNCell([lstm])
            # Let dynamic rnn setup the control flow (making while loops and stuff)
            # lstm_output shape: (None, 28, 5)
            lstm_output, _ = tf.nn.dynamic_rnn(lstm, max_out, dtype=tf.float32)

            lstm_output_reshaped = tf.reshape(lstm_output, (-1, 14 * lstm_units))

            with tf.name_scope("linear_{}".format(i)):
                W1 = tf.get_variable('linear_weight_{}'.format(i),
                                    shape=(14 * lstm_units, 784),
                                    initializer=tf.contrib.layers.variance_scaling_initializer())
                b1 = tf.Variable(tf.constant(0.1, shape=(784,), dtype=tf.float32))

                lin_activation = tf.matmul(lstm_output_reshaped, W1) + b1

            output = tf.nn.relu(lin_activation)

            # linear_out = tf.nn.dropout(linear_out, keep_prob)

            output = tf.nn.dropout(output, keep_prob)

            with tf.name_scope("linear_out"):
                W2 = tf.Variable(tf.truncated_normal((784, 10), stddev=0.1),
                                 trainable=True)
        b2 = tf.Variable(tf.constant(0.1, shape=(10,), dtype=tf.float32))

        linear_out = tf.matmul(output, W2) + b2

        y = linear_out

        # The fully connected layer
        # Loss function
        # Note that tf.nn.softmax_cross_entropy_with_logits internally applies the softmax on the
        # model's unnormalized model prediction and sums across all classes, and tf.reduce_mean
        # takes the average over these sums
        cross_entropy = tf.reduce_mean(

            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))


# Training
train_step = tf.train.AdamOptimizer(8e-5).minimize(cross_entropy)
# train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

return tf.nn.softmax(linear_out), train_step
