import argparse
import tensorflow as tf

from data_feed import DataFeed
from nns.batch_norm import get_batch_norm_model
from nns.con_b_max_lstm_lin import get_conv_b_max_lstm_lin_model
from nns.conv5_max_pool5_lstm_5_lin_5 import get_conv5_max_pool5_lstm_5_lin_5_model
from nns.conv_linear import get_conv_linear_model
from nns.conv_max_pool import get_max_pool_convolution_model

from nns.linear import get_linear_model
from nns.convolution import get_convolutional_model
from nns.lstm import get_lstm_model

import inquirer

from nns.lstm_conv_max_lin import get_conv_max_lstm_lin_model

model_dir = "./models"

models = {
    'linear': {
        'nn': get_linear_model
    },
    # 'batch_norm': {
    #     'nn': get_batch_norm_model
    # },
    'convolution': {
        'nn': get_convolutional_model
    },
    'conv_linear': {
        'nn': get_conv_linear_model
    },
    'conv_max_pool': {
        'nn': get_max_pool_convolution_model
    },
    'conv_max_lstm_lin': {
        'nn': get_conv_max_lstm_lin_model
    },
    'conv_b_max_lstm_lin': {
        'nn': get_conv_b_max_lstm_lin_model
    },
    'conv5_max_pool5_lstm_5_lin_5': {
        'nn': get_conv5_max_pool5_lstm_5_lin_5_model
    },
    'lstm': {
        'nn': get_lstm_model
    }
}


def get_initializer(train):
    vars_to_train = tf.trainable_variables()  # option-1

    if train:
        init = tf.global_variables_initializer()  # TF >1.0
    else:
        vars_all = tf.global_variables()
        vars_to_init = list(set(vars_all) - set(vars_to_train))
        init = tf.variables_initializer(vars_to_init)  # TF >1.0
    return init


def run_model(selected_model_names, train, iterations, batch_size, **kwargs):

    for selected_model_name in selected_model_names:
        print("#"*40)
        print("Evaluating {}".format(selected_model_name.title()))

        selected_model = models[selected_model_name]
        model_file = '%s/%s.ckpt' % (model_dir, selected_model_name)

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
            saver = tf.train.Saver(tf.trainable_variables())

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
                                train_acc = s.run(accuracy, feed_dict={x: tra_batch[0], y_: tra_batch[1], is_training: False})
                                validation_acc = s.run(accuracy, feed_dict={x: val_batch[0], y_: val_batch[1], is_training: False})
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
