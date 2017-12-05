import tensorflow as tf
import numpy as np

def get_sensitivity_relevance(inp, out):
    slice_index = tf.argmax(out, axis=-1)
    one_hot_selection = tf.one_hot(slice_index, 10, axis=1)
    out_selected = out * one_hot_selection

    grad = tf.gradients(out_selected, inp)
    sens_expl = tf.squeeze(tf.pow(grad, 2), axis=0)

    return sens_expl

