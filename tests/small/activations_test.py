from tfchat.activations import gelu
from tfchat.activations import get
import tensorflow as tf
import numpy as np


def test_gelu():
    def test_zero():
        got = gelu(0.)
        want = 0.
        np.testing.assert_almost_equal(got, want)

    test_zero()


def test_get():
    assert get("gelu") == gelu
    assert get("relu") == tf.keras.activations.relu