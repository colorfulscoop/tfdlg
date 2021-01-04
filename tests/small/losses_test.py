import numpy as np
from tfdlg.losses import PaddingLoss


def test_padding_loss():
    gold = np.array([1, 0, 0], dtype=np.int32)
    pred = np.array([[1, 0], [1, 0], [1, 0]], dtype=np.float32)

    loss_fn = PaddingLoss()
    got = loss_fn(gold, pred)  # == <tf.Tensor: shape=(), dtype=float32, numpy=1.3132616>
    np.testing.assert_almost_equal(got, 1.3132616)
