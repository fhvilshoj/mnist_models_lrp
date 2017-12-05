import argparse
import tensorflow as tf
import numpy as np
import inquirer
import sys
import os

from lrp.configuration import LOG_LEVEL

from data_feed import DataFeed

from nns.big_blocks import get_one_block, get_two_blocks, get_four_blocks, get_eight_blocks
from nns.conv_linear import get_conv_linear_model
from nns.conv_max_pool import get_max_pool_convolution_model
from nns.convolution_b import get_convolutional_b_model
from nns.nn03 import get_linear_nn03_model
from nns.linear import get_linear_model
from nns.convolution import get_convolutional_model
from nns.lstm import get_lstm_model
from nns.tensorflow_guide_99p import get_tensorflow_guide_99p_model

from config_selection import configurations
from config_selection import logger
from config_selection.result_file_writer import ResultWriter
from lrp import lrp

model_dir = "./models"

models = {
    'nn01': {  # NN1
        'nn': get_linear_model,
        #         lin    conv  lstm   maxp   batchnormalization
        'confs': [True, False, False, False, False]
    },
    # 'tensorflow_guide_99p': {
    #     'nn': get_tensorflow_guide_99p_model,
    #     #         lin    conv  lstm   maxp   batchnormalization
    #     'confs': [True, True, False, True, False]
    # },
    'nn02': { # NN2
        'nn': get_convolutional_model,
        #         lin    conv  lstm   maxp   batchnormalization
        'confs': [True, True, False, False, False]
    },
    'nn03': { # NN3
        'nn': get_linear_nn03_model,
        #         lin    conv  lstm  maxp    batchnormalization
        'confs': [True, False, False, False, False]
    },
    'nn05': { # NN5
        'nn': get_convolutional_b_model,
        #         lin    conv  lstm   maxp   batchnormalization
        'confs': [True, True, False, False, True]
    },
    # 'conv_linear': {
    #     'nn': get_conv_linear_model,
    #     #         lin    conv  lstm   maxp   batchnormalization
    #     'confs': [True, True, False, False, False]
    # },
    'nn06': { # NN6
        'nn': get_max_pool_convolution_model,
        #         lin    conv  lstm   maxp   batchnormalization
        'confs': [True, True, False, True, False]
    },
    'nn07': { # NN7
        'nn': get_one_block,
        #         lin    conv  lstm   maxp   batchnormalization
        'confs': [True, True, True, True, True]
    },
    'nn08': { # NN8
        'nn': get_two_blocks,
        #         lin    conv  lstm   maxp   batchnormalization
        'confs': [True, True, True, True, True]
    },
    'nn09': { # NN9
        'nn': get_four_blocks,
        #         lin    conv  lstm   maxp   batchnormalization
        'confs': [True, True, True, True, True]
    },
    'nn10': { # NN10
        'nn': get_eight_blocks,
        #         lin    conv  lstm   maxp   batchnormalization
        'confs': [True, True, True, True, True]
    },
    'nn04': { # NN4
        'nn': get_lstm_model,
        #         lin    conv  lstm   maxp   batchnormalization
        'confs': [True, False, True, False, False]
    }
}


def get_initializer(train):
    if train:
        init = tf.global_variables_initializer()  # TF >1.0
    else:
        init = tf.global_variables_initializer()
    return init


def run_model(selected_model_names, **kwargs):
    destination = "./pertubation/results"

    i = 1
    d = destination
    while os.path.exists(d):
        d = "{}_{:02}".format(destination, i)
        i += 1

    for selected_model_name in selected_model_names:
        print("#"*40)
        print("Evaluating {}".format(selected_model_name.title()))

        selected_model = models[selected_model_name]
        if(kwargs['use_old']):
            configs = configurations.get_configurations()
        else:
            configs = configurations.get_configurations_for_layers(*selected_model['confs'])

        model_file = '%s/%s.ckpt' % (model_dir, selected_model_name)

        if kwargs['lrp']:
            do_lrp_pertubation_tests(configs, selected_model, model_file,
                                     destination="%s/%s" % (d, selected_model_name), **kwargs)
        else:
            test_model(model_file, selected_model, **kwargs)


def do_pertubations(config, session, feed, explanation, iterations, batch_size, pertubations, writer, x_placeholder, y):
    for iteration in range(iterations):
        x_input, y_ = feed.next(batch_size)
        y_ = np.argmax(y_, axis=1)

        l = lambda x: (x, type(x))

        y_hat = session.run(y, feed_dict={x_placeholder: x_input})
        y_hat = np.argmax(y_hat, axis=1)

        expl = session.run(explanation, feed_dict={x_placeholder: x_input})
        batch_range = np.arange(batch_size)

        predictions = []
        for i in range(pertubations):
            # Print progress
            sys.stdout.write('\r')
            sys.stdout.write("[%-20s] %d%%" % (
                '=' * int((20 * (iteration * pertubations + i)) / (iterations * pertubations)),
                (100 * (iteration * pertubations + i)) / (iterations * pertubations)))
            sys.stdout.flush()

            # Prediction has shape (batch_size, 10)
            prediction = session.run(y, feed_dict={x_placeholder: x_input})
            predictions.append(prediction[batch_range, y_hat])

            # Alter input
            expl_argmax = np.argmax(expl, axis=1)

            x_input[batch_range, expl_argmax] = 0
            expl[batch_range, expl_argmax] = -100

        # Predictions has shape (test_size, batch_size, 10)
        predictions = np.array(predictions)
        predictions = predictions.reshape((pertubations, batch_size, 1))

        # Reshape predictions to have shape (batch_size, 10, test_size)
        predictions = predictions.transpose([1, 2, 0])

        # Write the results to file
        writer.write_result(config, y_, y_hat, predictions)

    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%%\n" % ('=' * 20, 100))
    sys.stdout.flush()


def do_lrp_pertubation_tests(configs, selected_model, model_file, destination, **kwargs):
    result_writer = ResultWriter(destination)
    feed = DataFeed()

    iterations = kwargs['test_size'] // kwargs['batch_size']
    
    start = kwargs['start']
    end = kwargs['end']
    if end == -1:
        end = len(configs)
        
    for config_idx, config in enumerate(configs[start:end]):
        graph = tf.Graph()
        feed.reset_permutation()

        with graph.as_default():
            x = tf.placeholder(tf.float32, shape=[None, 784])
            y_ = tf.placeholder(tf.float32, shape=[None, 10])

            is_training = False

            y, _ = selected_model['nn'](x, y_, is_training)

            print("Testing ({}/{}) {}".format(config_idx, end-start, config))
            explanation = lrp.lrp(x, y, config)

            init = get_initializer(False)

            important_variables = tf.trainable_variables()
            important_variables.extend([v for v in tf.global_variables() if 'moving_' in v.name])

            saver = tf.train.Saver(important_variables)

            with tf.Session() as s:
                # Initialize stuff and restore model
                s.run(init)
                saver.restore(s, model_file)

                do_pertubations(config, s, feed, explanation, iterations, kwargs['batch_size'], kwargs['pertubations'], result_writer, x, y)


def test_model(model_file, selected_model, **kwargs):
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

        train = kwargs['train']
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
                    for i in range(kwargs['iterations']):
                        batch = feed.next(kwargs['batch_size'])
                        s.run(train_step, feed_dict={x: batch[0], y_: batch[1], is_training: True})

                        if i % 200 == 0:
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
    parser.add_argument('-p', '--pertubations', type=int, default=100)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)
    parser.add_argument('-t', '--test-size', type=int, default=1000,
                        help='Do pertubations on `test-size` samples')
    parser.add_argument('--use-old', action='store_true')
    
    args = parser.parse_args()

    # Disable console prints
    logger.handlers[0].setLevel(30)

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
