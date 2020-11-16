from tfchat.activations import gelu
import numpy as np


def test_gelu():
    def test_zero():
        got = gelu(0.)
        want = 0.
        np.testing.assert_almost_equal(got, want)

    test_zero()
