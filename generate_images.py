import numpy as np
import tensorflow as tf
import matplotlib.cm
import skimage.io
import skimage.feature
import skimage.filters
import argparse

from nns.convolution_b import get_convolutional_b_model

import inquirer

from config_selection import score_parser
from lrp import lrp
from lrp.configuration import LRPConfiguration, EpsilonConfiguration, BIAS_STRATEGY, LAYER

from data_feed import DataFeed
from sensitivity_analysis_relevance import get_sensitivity_relevance


def _reshape_to_image(X, shape):
    if len(shape) == 0:
        # Assume that image is square
        shape = list(map(int, [(np.sqrt(X.size))] * 2))

    return X.reshape(shape)


def _enlarge_image(img, scaling=3):
    if len(img.shape) == 2:
        H, W = img.shape

        out = np.zeros((scaling * H, scaling * W))
        for h in range(H):
            fh = scaling * h
            for w in range(W):
                fw = scaling * w
                out[fh:fh + scaling, fw:fw + scaling] = img[h, w]

    elif len(img.shape) == 3:
        H, W, D = img.shape

        out = np.zeros((scaling * H, scaling * W, D))
        for h in range(H):
            fh = scaling * h
            for w in range(W):
                fw = scaling * w
                out[fh:fh + scaling, fw:fw + scaling, :] = img[h, w, :]

    return out


def _digit_to_rgb(X, shape=(), cmap='binary'):
    # create color map object from name string
    cmap = eval('matplotlib.cm.{}'.format(cmap))

    image = _reshape_to_image(X, shape)
    image = _enlarge_image(image, 3)
    image = cmap(image.flatten())[..., 0:3].reshape([image.shape[0], image.shape[1], 3])  # colorize, reshape

    return image


def _hm_to_rgb(R, X=None, shape=(), sigma=2, cmap='seismic', normalize=True): #bwr
    # create color map object from name string
    cmap = eval('matplotlib.cm.{}'.format(cmap))

    R = _reshape_to_image(R, shape)
    R = _enlarge_image(R, 3)

    if normalize:
        R = R / np.max(np.abs(R))  # normalize to [-1,1] wrt to max relevance magnitude
        R = (R + 1.) / 2.  # shift/normalize to    [0,1] for color mapping

    rgb = cmap(R.flatten())[..., 0:3].reshape([R.shape[0], R.shape[1], 3])
    # rgb = repaint_corner_pixels(rgb, scaling) #obsolete due to directly calling the color map with [0,1]-normalized inputs

    if not X is None:  # compute the outline of the input
        X = _reshape_to_image(X, shape)
        X = _enlarge_image(X, 3)
        xdims = X.shape
        Rdims = R.shape

        if not np.all(xdims == Rdims):
            print("R shape {} and X shape {} does not match".format(Rdims, xdims))
        else:
            edges = skimage.feature.canny(X, sigma=sigma)
            edges = np.invert(np.dstack([edges] * 3)) * 1.0
            rgb *= edges  # set outline pixels to black color

    return rgb


def _save_image(rgb_images, path, gap=1):
    sz = []
    image = []
    for i in range(len(rgb_images)):
        if not sz:
            sz = rgb_images[i].shape
            image = rgb_images[i]
            gap = np.zeros((sz[0], gap, sz[2]))
            continue
        if not sz[0] == rgb_images[i].shape[0] and sz[1] == rgb_images[i].shape[2]:
            print('image', i, 'differs in size. unable to perform horizontal alignment')
        else:
            image = np.hstack((image, gap, rgb_images[i]))

    image *= 255
    image = image.astype(np.uint8)

    print('saving image to ', path)
    skimage.io.imsave(path, image)
    return image


def generate_images(images, relevances, titles):
    for img, relevance, title in zip(images, relevances, titles):
        print(img.shape, relevance.shape, title)
        i = _digit_to_rgb(img, shape=(28, 28))
        r = _hm_to_rgb(relevance, img, shape=(28, 28))
        _save_image([i], title + "_img.png")
        _save_image([r], title + "_rel.png")


def get_relevance(input, labels, config):
    graph = tf.Graph()
    feed.reset_permutation()

    with graph.as_default():
        x = tf.placeholder(tf.float32, shape=[None, 784])
        y_ = tf.placeholder(tf.float32, shape=[None, 10])

        model_file = '%s/%s.ckpt' % ("models", "nn05")

        y, _ = get_convolutional_b_model(x, y_, False)

        if isinstance(config, str):
            print("Testing sensitivity analysis")
            explanation = get_sensitivity_relevance(x, y)
        else:
            print(config)
            with tf.name_scope("LRP"):
                explanation = lrp.lrp(x, y, config)

        init = tf.global_variables_initializer()

        important_variables = tf.trainable_variables()
        important_variables.extend([v for v in tf.global_variables() if 'moving_' in v.name])

        saver = tf.train.Saver(important_variables)

        with tf.Session() as s:
            # Initialize stuff and restore model
            s.run(init)
            saver.restore(s, model_file)

            expls = s.run(explanation, feed_dict={
                x: input,
                y_: labels
            })
            return expls


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image printer for configuration")

    parser.add_argument('-d', '--destination', type=str, default='./relevance_images')
    parser.add_argument('-ic', '--image-count', type=int, default=1)
    parser.add_argument('-o', '--offset', type=int, default=0)
    args = parser.parse_args()

    config = LRPConfiguration()
    epsilon = EpsilonConfiguration(1e-12, bias_strategy=BIAS_STRATEGY.NONE)
    config.set(LAYER.LINEAR, epsilon)
    config.set(LAYER.CONVOLUTIONAL, epsilon)
    config.set(LAYER.ELEMENTWISE_LINEAR, epsilon)

    feed = DataFeed(False)
    images_for_lrp, labels = feed.test(False)

    images, _ = feed.test(True)

    images_for_lrp = images_for_lrp[args.offset:args.offset + args.image_count]
    labels = labels[args.offset:args.offset + args.image_count]
    images = images[args.offset:args.offset + args.image_count]

    explanations = get_relevance(images_for_lrp, labels, config)
    generate_images(images, explanations, ["{}/{}".format(args.destination, i) for i in range(args.image_count)])
