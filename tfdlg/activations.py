import tensorflow as tf
import math


def gelu(x):
    """
    This implementation is the approximation used in [Hendrycks+, 2016] https://arxiv.org/abs/1606.08415
    """
    x = tf.convert_to_tensor(x)
    pi = tf.cast(math.pi, x.dtype)
    return 0.5 * x * (1.0 + tf.tanh(tf.sqrt(2.0 / pi) * (x + 0.044715 * tf.pow(x, 3))))


def get(identifier):
    """
    Args:
        identifier (str): function name of the activation

    Returns:
        Function corresponding to the argument
    """
    func_map = {
        "gelu": gelu,
        "relu": tf.keras.activations.relu
    }
    return func_map[identifier]
