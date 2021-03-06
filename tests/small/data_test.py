from tfdlg.data import BlockDataset
from tfdlg.data import LineByLineDataset
import numpy as np


def test_BlockDataset_generator():
    def encode(text):
        words = text.split(" ")
        return [int(w) for w in words]

    # Loader test
    texts = ["0 1 2 3 4", "5 6 7 8", "9 10 11 12 13"]
    dataset = BlockDataset.from_generator(lambda: texts, encode_fn=encode, block_size=3, batch_size=2, shuffle=False)

    got_X = np.array([item[0].numpy() for item in dataset])
    want_X = np.array([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]])

    got_y = np.array([item[1].numpy() for item in dataset])
    want_y = np.array([[[1, 2, 3], [4, 5, 6]],
                       [[7, 8, 9], [10, 11, 12]]])

    np.testing.assert_equal(got_X, want_X)
    np.testing.assert_equal(got_y, want_y)


def test_LineByLineDataset_from_generator():
    def encode(text):
        words = text.split(" ")
        return [int(w) for w in words]

    texts = ["0", "0 1", "0 1 2", "0 1 2 3", "0 1 2 3 4"]

    dataset = LineByLineDataset.from_generator(lambda: texts, encode_fn=encode, max_len=5, batch_size=2, shuffle=False)

    got_X = np.array([item[0].numpy() for item in dataset])
    want_X = np.array([[[0, 0, 0, 0], [0, 1, 0, 0]],
                       [[0, 1, 2, 0], [0, 1, 2, 3]]])

    got_y = np.array([item[1].numpy() for item in dataset])
    want_y = np.array([[[0, 0, 0, 0], [1, 0, 0, 0]],
                       [[1, 2, 0, 0], [1, 2, 3, 0]]])

    np.testing.assert_equal(got_X, want_X)
    np.testing.assert_equal(got_y, want_y)


def test_LineByLineDataset_from_generator_large_input():
    def encode(text):
        words = text.split(" ")
        return [int(w) for w in words]

    texts = ["0 1 2 3 4 5 6"]

    dataset = LineByLineDataset.from_generator(lambda: texts, encode_fn=encode, max_len=5, batch_size=1, shuffle=False)

    got_X = np.array([item[0].numpy() for item in dataset])
    want_X = np.array([[[0, 1, 2, 3]]])

    got_y = np.array([item[1].numpy() for item in dataset])
    want_y = np.array([[[1, 2, 3, 4]]])

    np.testing.assert_equal(got_X, want_X)
    np.testing.assert_equal(got_y, want_y)
