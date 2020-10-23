import numpy as np
from tfchat.generations import filter_to_topk
from tfchat.generations import filter_to_topp
from tfchat.generations import filter_bad_ids


def test_filter_to_topk():
    res = filter_to_topk(
        top_k=2,
        dist=np.array([[2, 0, 3, 1, -1], [5, 6, 7, 8, 9]], dtype=np.float32)
    )

    inf = float("Inf")
    expected = np.array(
        [[2, -inf, 3, -inf, -inf],
         [-inf, -inf, -inf, 8, 9]],
        dtype=np.float32
    )
    np.testing.assert_almost_equal(res, expected)


def test_filter_to_topk_out_of_index():
    res = filter_to_topk(
        top_k=10,
        dist=np.array([[2, 0, 3, 1, -1], [5, 6, 7, 8, 9]], dtype=np.float32)
    )
    expected = np.array(
        [[2, 0, 3, 1, -1],
         [5, 6, 7, 8, 9]],
        dtype=np.float32
    )
    np.testing.assert_almost_equal(res, expected)


def test_filter_to_topp():
    res = filter_to_topp(
        top_p=0.9,
        dist=np.array([[2, 0, 3, 1, -1], [5, 6, 7, 8, 9]], dtype=np.float32)
    )
    inf = float("Inf")
    expected = np.array(
        [[2, -inf, 3, 1, -inf],
         [-inf, -inf, 7, 8, 9]],
        dtype=np.float32
    )
    np.testing.assert_almost_equal(res, expected)


def test_filter_to_topp_under():
    res = filter_to_topp(
        top_p=0,
        dist=np.array([[2, 0, 3, 1, -1], [5, 6, 7, 8, 9]], dtype=np.float32)
    )
    inf = float("Inf")
    expected = np.array(
        [[-inf, -inf, 3, -inf, -inf],
         [-inf, -inf, -inf, -inf, 9]],
        dtype=np.float32
    )
    np.testing.assert_almost_equal(res, expected)


def test_filter_bad_ids():
    res = filter_bad_ids(
        bad_ids=[1, 2],
        dist=np.array([[2, 0, 3, 1, -1], [5, 6, 7, 8, 9]], dtype=np.float32)
    )
    inf = float("Inf")
    expected = np.array(
        [[2, -inf, -inf, 1, -1],
         [5, -inf, -inf, 8, 9]],
        dtype=np.float32
    )
    np.testing.assert_equal(res, expected)