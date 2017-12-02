import tensorflow as tf
import math

def get_conv5_max_pool5_lstm_5_lin_5_model(x, y_, is_training, *args):
    keep_prob = tf.cond(is_training,
                        true_fn=lambda: 0.5,
                        false_fn=lambda: 1.)


    # X has shape (None, 784)
    output_depth = 2
    input_depth = 1
    o_size = 28
    output = tf.reshape(x, (-1, o_size, o_size, input_depth))

    for i in range(5):
        with tf.name_scope("conv_{}".format(i)):
            output_depth *= 2

            # Kernel shape (height, width, input_channels, output_channels)
            K = tf.Variable(tf.truncated_normal((3, 3, input_depth, output_depth), stddev=0.1))
            kb = tf.Variable(tf.constant(0.1, shape=[output_depth, ]))

            input_depth = output_depth

            conv_out = tf.nn.conv2d(output, K, [1, 1, 1, 1], 'SAME')
            conv_out = tf.nn.bias_add(conv_out, kb)

            # conv_out = tf.contrib.layers.batch_norm(
            #     conv_out,
            #     center=True,
            #     scale=False,
            #     is_training=is_training,
            # )

            # Shape (None, 28, 28, 1)
            conv_out = tf.nn.relu(conv_out)

            # conv_out = tf.nn.dropout(conv_out, keep_prob)

            output = tf.nn.max_pool(conv_out, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

            o_size = math.ceil(o_size / 2)

    output = tf.reshape(output, (-1, 64))

    for i in range(5):
        with tf.name_scope("lstm_linear_{}".format(i)):
            output = tf.reshape(output, (-1, 8, 8))

            lstm_units = 10

            with tf.variable_scope("lstm_{}".format(i)):
                # Create lstm layer
                lstm = tf.contrib.rnn.LSTMCell(lstm_units, forget_bias=0.)
                # Put it into Multi RNN Cell
                lstm = tf.contrib.rnn.MultiRNNCell([lstm])
                # Let dynamic rnn setup the control flow (making while loops and stuff)
                # lstm_output shape: (None, 28, 5)
                lstm_output, _ = tf.nn.dynamic_rnn(lstm, output, dtype=tf.float32)

            lstm_output_reshaped = tf.reshape(lstm_output, (-1, 8 * lstm_units))

            W = tf.Variable(tf.truncated_normal((8 * lstm_units, 64), stddev=0.1),
                             trainable=True)
            b = tf.Variable(tf.constant(0.1, shape=(64,), dtype=tf.float32))

            lin_activation = tf.matmul(lstm_output_reshaped, W) + b

            output = tf.nn.relu(lin_activation)

            # output = tf.nn.dropout(linear_out, keep_prob)

    output = tf.reshape(output, (-1, 64))
    output = tf.nn.dropout(output, keep_prob)

    with tf.name_scope("linear_out"):
        W2 = tf.Variable(tf.truncated_normal((64, 10), stddev=0.1),
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
    train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
    # train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

    return tf.nn.softmax(linear_out), train_step
