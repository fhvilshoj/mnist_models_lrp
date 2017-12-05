import tensorflow as tf
import numpy as np

def get_random_relevance(inp):
    rand_expl = tf.random_uniform(tf.shape(inp), 0., 1., tf.float32)
    return rand_expl
