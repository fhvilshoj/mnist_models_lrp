import argparse
import tensorflow as tf
import numpy as np
import inquirer
import sys
import os

from lrp.configuration import LOG_LEVEL

from data_feed import DataFeed
from sensitivity_analysis_relevance import get_sensitivity_relevance
from random_relevance import get_random_relevance

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
    results = []
    for selected_model_name in selected_model_names:
        print("#"*40)
        print("Searching in model: {}".format(selected_model_name.title()))

        selected_model = models[selected_model_name]
        
        configs = configurations.get_configurations_for_layers(*selected_model['confs'])

        model_file = '%s/%s.ckpt' % (model_dir, selected_model_name)

        res = do_nan_searching(configs, selected_model, model_file,
                               model_name=selected_model_name, **kwargs)
        results.append(res)

    logger.info("Summary:")
    for r in results:
        logger.info("\t%s: %3i" % r)
        print("\t%s: %3i" % r)

def do_search(config, session, feed, explanation, iterations, batch_size, x_placeholder):
    found_nan = False
    for iteration in range(iterations):
        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d%%" % (
            '=' * int((20 * iteration) / iterations),
            ((100 * iteration) / (iterations))))
        sys.stdout.flush()

        x_input, _ = feed.next(batch_size)
        expl = session.run(explanation, feed_dict={x_placeholder: x_input})

        if np.isnan(expl).any():
            found_nan = True
            print("")
            print("############################")
            print("")
            print("FOUND A NaN in configuration")
            print(config)
            print("")
            print("############################")
            print("")

    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%%\n" % ('=' * 20, 100))
    sys.stdout.flush()
    return int(found_nan) # Return 1 if found any else 0

def do_nan_searching(configs, selected_model, model_file, model_name, **kwargs):
    iterations = kwargs['test_size'] // kwargs['batch_size']
    feed = DataFeed()
    
    start = kwargs['start']
    end = kwargs['end']
    if end == -1:
        end = len(configs)

    found_nans = 0
    for config_idx, config in enumerate(configs[start:end]):
        graph = tf.Graph()
        feed.reset_permutation()

        with graph.as_default():
            x = tf.placeholder(tf.float32, shape=[None, 784])
            y_ = tf.placeholder(tf.float32, shape=[None, 10])

            is_training = False

            y, _ = selected_model['nn'](x, y_, is_training)

            print("Testing ({}/{}), Nan-cnt {}: {}".format(config_idx, end-start, found_nans, config))
            explanation = lrp.lrp(x, y, config)

            init = get_initializer(False)

            important_variables = tf.trainable_variables()
            important_variables.extend([v for v in tf.global_variables() if 'moving_' in v.name])

            saver = tf.train.Saver(important_variables)

            with tf.Session() as s:
                # Initialize stuff and restore model
                s.run(init)
                saver.restore(s, model_file)

                found_nans += do_search(config, s, feed, explanation, iterations, kwargs['batch_size'], x)
    logger.info("Found nans for {} configurations in total for model {}".format(found_nans, model_name))
    print("Found nans for {} configurations in total for model {}".format(found_nans, model_name))
    return model_name, found_nans


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Get model scores')

    parser.add_argument('-b', '--batch_size', type=int, default=250,
                        help='Batch size for each iteration')

    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)
    parser.add_argument('-t', '--test-size', type=int, default=1000,
                        help='Do pertubations on `test-size` samples')
    
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
