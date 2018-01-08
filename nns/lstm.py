import tensorflow as tf

def get_lstm_model(x, y_, is_training):
    lstm_units = 64

    # x has shape (None, 784)
    x_reshaped = tf.reshape(x, (-1, 28, 28))

    # Create lstm layer
    lstm = tf.contrib.rnn.LSTMCell(lstm_units, forget_bias=0.)
    # Put it into Multi RNN Cell
    lstm = tf.contrib.rnn.MultiRNNCell([lstm])
    # Let dynamic rnn setup the control flow (making while loops and stuff)
    # lstm_output shape: (None, 28, 5)
    lstm_output, _ = tf.nn.dynamic_rnn(lstm, x_reshaped, dtype=tf.float32)

    lstm_output_reshaped = tf.reshape(lstm_output, (-1, 28 * lstm_units))

    # lstm_output_reshaped = tf.nn.dropout(lstm_output_reshaped, keep_prob=0.5)
    
    W = tf.Variable(tf.truncated_normal((28 * lstm_units, 10), stddev=0.1),
                    trainable=True)
    b = tf.Variable(tf.constant(0.1, shape=(10,), dtype=tf.float32))

    y = tf.matmul(lstm_output_reshaped, W) + b

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)) \
                    + 0.03 * tf.nn.l2_loss(W)

    # Training
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

    # The fully connected layer
    return tf.nn.softmax(y), train_step
