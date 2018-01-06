import tensorflow as tf

KERNEL_DEPTH = 32
LSTM_UNITS = 64
LINEAR_UNITS = 64

def get_one_block(*args):
    return get_model_with_bolcks(1, 14, 0.5, *args)


def get_two_blocks(*args):
    return get_model_with_bolcks(2, 14, 0.5, *args)


def get_four_blocks(*args):
    return get_model_with_bolcks(4, 7, 0.8, *args)


def get_eight_blocks(*args):
    return get_model_with_bolcks(8, 2, 0.9, *args)


def _get_conv_kernel(i, i_depth=KERNEL_DEPTH):
    K = tf.get_variable('Kernel_{}'.format(i),
                        shape=(3, 3, i_depth, KERNEL_DEPTH),
                        initializer=tf.contrib.layers.variance_scaling_initializer())

    # K = tf.Variable(tf.truncated_normal((3, 3, 1, 2), stddev=0.1))
    kb = tf.get_variable('bias_{}'.format(i),
                         shape=(KERNEL_DEPTH,),
                         initializer=tf.constant_initializer(value=1.))
    return K, kb


def get_model_with_bolcks(blocks, sequence_length, keep_prob, x, y_, is_training, *args):
    i_depth = 1
    output = x
    output = tf.reshape(output, (-1, 28, 28, i_depth))

    for i in range(blocks):
        with tf.name_scope("conv_block_{}".format(i)):

            # X has shape (None, 784)
            with tf.name_scope("conv_{}".format(i)):

                K, kb = _get_conv_kernel(i, i_depth)
                i_depth = KERNEL_DEPTH

                conv_out = tf.nn.conv2d(output, K, [1, 1, 1, 1], 'SAME')
                conv_out = tf.nn.bias_add(conv_out, kb)

                output = tf.contrib.layers.batch_norm(
                    conv_out,
                    center=True,
                    scale=False,
                    is_training=is_training,
                    trainable=True,
                    variables_collections=tf.GraphKeys.TRAINABLE_VARIABLES
                )

                # noinspection PyUnboundLocalVariable
                output = tf.nn.relu(output)

        # Add max pool after every second conv and after first conv for nn07 having only one block
        if i % 2 == 1 or blocks == 1:
            with tf.name_scope("max_pool_{}".format(i)):
                output = tf.nn.max_pool(output, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

    lstm_output = tf.reshape(output, (-1, sequence_length, sequence_length * KERNEL_DEPTH))

    lstm_units = LSTM_UNITS

    for i in range(blocks):
        with tf.name_scope("lstm_{}".format(i)):
            with tf.variable_scope("lstm_{}".format(i)):
                # Create lstm layer
                lstm = tf.contrib.rnn.LSTMCell(lstm_units, forget_bias=0.)
                # Put it into Multi RNN Cell
                lstm = tf.contrib.rnn.MultiRNNCell([lstm])
                # Let dynamic rnn setup the control flow (making while loops and stuff)
                # lstm_output shape: (None, 28, 5)
                lstm_output, _ = tf.nn.dynamic_rnn(lstm, lstm_output, dtype=tf.float32)

    linear_input_width = sequence_length * lstm_units
    output = tf.reshape(lstm_output, (-1, linear_input_width))

    for i in range(blocks):
        with tf.name_scope("linear_{}".format(i)):
            W1 = tf.get_variable('linear_weight_{}'.format(i),
                                 shape=(linear_input_width, LINEAR_UNITS),
                                 initializer=tf.contrib.layers.variance_scaling_initializer())
            linear_input_width = LSTM_UNITS

            b1 = tf.Variable(tf.constant(0.1, shape=(LINEAR_UNITS,), dtype=tf.float32))
            lin_activation = tf.matmul(output, W1) + b1

            output = tf.nn.relu(lin_activation)

    # linear_out = tf.nn.dropout(linear_out, keep_prob)

    # output = tf.nn.dropout(output, keep_prob)

    with tf.name_scope("linear_out"):
        W2 = tf.Variable(tf.truncated_normal((LINEAR_UNITS, 10), stddev=0.1),
                         trainable=True)
        b2 = tf.Variable(tf.constant(0.1, shape=(10,), dtype=tf.float32))

        linear_out = tf.matmul(output, W2) + b2

        y = linear_out

    # The fully connected layer
    # Loss function
    # Note that tf.nn.softmax_cross_entropy_with_logits internally applies the softmax on the
    # model's unnormalized model prediction and sums across all classes, and tf.reduce_mean
    # takes the average over these sums
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    # Training
    optimizer = tf.train.AdamOptimizer(1.e-4)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_step = optimizer.minimize(cross_entropy)

    # train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

    return tf.nn.softmax(linear_out), train_step

##### OLD #####
# def get_model_with_bolcks(blocks, x, y_, is_training, *args):
#     output = x
#     for i in range(blocks):
#         with tf.name_scope("block_{}".format(i)):
#             output = tf.reshape(output, (-1, 28, 28, 1))
#
#             # X has shape (None, 784)
#             with tf.name_scope("conv_{}".format(i)):
#                 # Kernel shape (height, width, input_channels, output_channels)
#                 K = tf.get_variable('Kernel_{}'.format(i),
#                                     shape=(3, 3, 1, 2),
#                                     initializer=tf.contrib.layers.variance_scaling_initializer())
#
#                 # K = tf.Variable(tf.truncated_normal((3, 3, 1, 2), stddev=0.1))
#                 kb = tf.Variable(tf.constant(1., shape=[2, ]))
#
#                 conv_out = tf.nn.conv2d(output, K, [1, 1, 1, 1], 'SAME')
#                 conv_out = tf.nn.bias_add(conv_out, kb)
#
#                 conv_out = tf.contrib.layers.batch_norm(
#                     conv_out,
#                     center=True,
#                     scale=False,
#                     is_training=is_training,
#                     trainable=True,
#                     variables_collections=tf.GraphKeys.TRAINABLE_VARIABLES
#                 )
#
#             # Shape (None, 28, 28, 2)
#             conv_out = tf.nn.relu(conv_out)
#
#             with tf.name_scope("max_pool_{}".format(i)):
#                 max_out = tf.nn.max_pool(conv_out, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
#
#             # Shape (None, 784)
#             max_out = tf.reshape(max_out, (-1, 14, 14 * 2))
#
#             with tf.name_scope("lstm_{}".format(i)):
#                 lstm_units = 64
#
#                 with tf.variable_scope("lstm_{}".format(i)):
#                     # Create lstm layer
#                     lstm = tf.contrib.rnn.LSTMCell(lstm_units, forget_bias=0.)
#                     # Put it into Multi RNN Cell
#                     lstm = tf.contrib.rnn.MultiRNNCell([lstm])
#                     # Let dynamic rnn setup the control flow (making while loops and stuff)
#                     # lstm_output shape: (None, 28, 5)
#                     lstm_output, _ = tf.nn.dynamic_rnn(lstm, max_out, dtype=tf.float32)
#
#             lstm_output_reshaped = tf.reshape(lstm_output, (-1, 14 * lstm_units))
#
#             with tf.name_scope("linear_{}".format(i)):
#                 W1 = tf.get_variable('linear_weight_{}'.format(i),
#                                      shape=(14 * lstm_units, 784),
#                                      initializer=tf.contrib.layers.variance_scaling_initializer())
#                 b1 = tf.Variable(tf.constant(0.1, shape=(784,), dtype=tf.float32))
#
#                 lin_activation = tf.matmul(lstm_output_reshaped, W1) + b1
#
#                 output = tf.nn.relu(lin_activation)
#
#                 # linear_out = tf.nn.dropout(linear_out, keep_prob)
#
#     # output = tf.nn.dropout(output, keep_prob)
#
#     with tf.name_scope("linear_out"):
#         W2 = tf.Variable(tf.truncated_normal((784, 10), stddev=0.1),
#                          trainable=True)
#         b2 = tf.Variable(tf.constant(0.1, shape=(10,), dtype=tf.float32))
#
#         linear_out = tf.matmul(output, W2) + b2
#
#         y = linear_out
#
#     # The fully connected layer
#     # Loss function
#     # Note that tf.nn.softmax_cross_entropy_with_logits internally applies the softmax on the
#     # model's unnormalized model prediction and sums across all classes, and tf.reduce_mean
#     # takes the average over these sums
#     cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
#
#     # Training
#     optimizer = tf.train.AdamOptimizer(1.e-4)
#     with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
#         train_step = optimizer.minimize(cross_entropy)
#
#     # train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
#
#     return tf.nn.softmax(linear_out), train_step
