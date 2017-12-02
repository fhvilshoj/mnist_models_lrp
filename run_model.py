import argparse
import tensorflow as tf
import numpy as np
import inquirer
import sys

from data_feed import DataFeed

from nns.big_blocks import get_one_block, get_two_blocks, get_four_blocks, get_eight_blocks
from nns.conv_linear import get_conv_linear_model
from nns.conv_max_pool import get_max_pool_convolution_model
from nns.convolution_b import get_convolutional_b_model

from nns.linear import get_linear_model
from nns.convolution import get_convolutional_model
from nns.lstm import get_lstm_model
from nns.tensorflow_guide_99p import get_tensorflow_guide_99p_model

from config_selection import configurations
from config_selection.result_file_writer import ResultWriter
from lrp import lrp

model_dir = "./models"

models = {
    'linear': {
        'nn': get_linear_model
    },
    'tensorflow_guide_99p': {
        'nn': get_tensorflow_guide_99p_model
    },
    'convolution': {
        'nn': get_convolutional_model
    },
    'convolution_b': {
        'nn': get_convolutional_b_model
    },
    'conv_linear': {
        'nn': get_conv_linear_model
    },
    'conv_max_pool': {
        'nn': get_max_pool_convolution_model
    },
    'big_block_01': {
        'nn': get_one_block
    },
    'big_block_02': {
        'nn': get_two_blocks
    },
    'big_block_04': {
        'nn': get_four_blocks
    },
    'big_block_08': {
        'nn': get_eight_blocks
    },
    'lstm': {
        'nn': get_lstm_model
    }
}


def get_initializer(train):
    if train:
        init = tf.global_variables_initializer()  # TF >1.0
    else:
        init = tf.global_variables_initializer()
    return init


def run_model(selected_model_names, train, iterations, batch_size, **kwargs):
    configs = configurations.get_configurations()

    for selected_model_name in selected_model_names:
        print("#"*40)
        print("Evaluating {}".format(selected_model_name.title()))

        selected_model = models[selected_model_name]
        model_file = '%s/%s.ckpt' % (model_dir, selected_model_name)

        if kwargs['lrp']:
            do_lrp_pertubation_tests(configs, selected_model, model_file, **kwargs)
        else:
            test_model(batch_size, iterations, model_file, selected_model, train, **kwargs)


def do_pertubations(config, session, feed, explanation, iterations, batch_size, test_size, writer, x, y):
    for iter in range(iterations):
        x_, y_ = feed.next(batch_size)

        y_hat = session.run(y, feed_dict={x: x_})

        y_hat = np.argmax(y_hat, axis=1)

        explanation = session.run(explanation, feed_dict={x: x_})

        predictions = []
        for i in range(test_size):
            # Print progress
            sys.stdout.write('\r')
            sys.stdout.write("[%-20s] %d%%" % (
                '=' * int((20 * (iter * test_size + i)) / (iterations * test_size)),
                (100 * (iter * test_size + i)) / (iterations * test_size)))
            sys.stdout.flush()

            # Prediction has shape (batch_size, 10)
            prediction = session.run(y, feed_dict={x: x_})
            predictions.append(prediction)

            # Alter input
            expl_argmax = np.argmax(explanation, axis=1)
            x_[np.arange(batch_size), expl_argmax] = 0
            explanation[np.arange(batch_size), expl_argmax] = -100

        # Predictions has shape (test_size, batch_size, 10)
        predictions = np.array(predictions)

        # Reshape predictions to have shape (batch_size, 10, test_size)
        predictions = predictions.transpose([1, 0, 2])

        # Write the results to file
        writer.write_result(config, y_, y_hat, predictions)

    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%%\n" % ('=' * 20, 100))
    sys.stdout.flush()


def do_lrp_pertubation_tests(configs, selected_model, model_file, test_size, batch_size, **kwargs):
    result_writer = ResultWriter('./pertubation/results')
    feed = DataFeed()

    iterations = test_size // batch_size

    for config in configs:
        graph = tf.Graph()
        feed.reset_permutation()

        with graph.as_default():
            x = tf.placeholder(tf.float32, shape=[None, 784])
            y_ = tf.placeholder(tf.float32, shape=[None, 10])

            is_training = tf.constant(False, tf.bool)

            y, _= selected_model['nn'](x, y_, is_training)

            explanation = lrp.lrp(x, y, config)

            init = get_initializer(False)

            important_variables = tf.trainable_variables()
            important_variables.extend([v for v in tf.global_variables() if 'moving_' in v.name])

            saver = tf.train.Saver(important_variables)

            with tf.Session() as s:
                # Initialize stuff and restore model
                s.run(init)
                saver.restore(s, model_file)





def test_model(batch_size, iterations, model_file, selected_model, train, **kwargs):
    graph = tf.Graph()
    with graph.as_default():
        x = tf.placeholder(tf.float32, shape=[None, 784])
        y_ = tf.placeholder(tf.float32, shape=[None, 10])

        is_training = tf.placeholder(tf.bool, ())

        y, train_step = selected_model['nn'](x, y_, is_training)

        # Testing
        # Test trained model
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        feed = DataFeed()

        init = get_initializer(train)

        important_variables = tf.trainable_variables()
        important_variables.extend([v for v in tf.global_variables() if 'moving_' in v.name])

        saver = tf.train.Saver(important_variables)

        with tf.Session() as s:
            s.run(init)

            if train:
                if kwargs['restore']:
                    saver.restore(s, model_file)

                # Train the model
                try:
                    for i in range(iterations):
                        batch = feed.next(batch_size)
                        s.run(train_step, feed_dict={x: batch[0], y_: batch[1], is_training: True})

                        if i % 200 == 0:
                            print(feed.offset)
                            tra_batch = feed.train()
                            val_batch = feed.validation()
                            train_acc = s.run(accuracy,
                                              feed_dict={x: tra_batch[0], y_: tra_batch[1], is_training: False})
                            validation_acc = s.run(accuracy,
                                                   feed_dict={x: val_batch[0], y_: val_batch[1], is_training: False})
                            print()
                            print("{} Accuracy train: {:10f} validation: {:10f}".format(i, train_acc, validation_acc))
                except KeyboardInterrupt:
                    print("Training interrupted. Now saving model")
                finally:
                    saver.save(s, model_file)
            else:
                saver.restore(s, model_file)

            test_batch = feed.test()
            acc = s.run(accuracy, feed_dict={x: test_batch[0],
                                             y_: test_batch[1],
                                             is_training: False})
            print('Final test accuracy: %f%%' % (acc * 100))


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Get model scores')

    parser.add_argument('--train', action='store_true',
                        help='Indicates that we need to train model')
    parser.add_argument('-i', '--iterations', type=int, default=10000,
                        help='Number of iterations for training')
    parser.add_argument('-b', '--batch_size', type=int, default=80,
                        help='Batch size for each iteration')
    parser.add_argument('--restore', action='store_true',
                        help='Restore model to continue training')
    parser.add_argument('--lrp', action='store_true',
                        help='Do lrp pertubation tests on models')
    parser.add_argument('-t', '--test-size', type=int, default=1000,
                        help='Do pertubations on `test-size` samples')

    args = parser.parse_args()

    model_keys = [k for k in models.keys()]
    model_keys.sort()

    questions = [
        inquirer.Checkbox('models',
                          message="Which models to run?",
                          choices=model_keys,
                          ),
    ]
    selected_models = inquirer.prompt(questions)['models']
    print(selected_models)

    # Call config selection with gathered arguments
    run_model(selected_models, **vars(args))
