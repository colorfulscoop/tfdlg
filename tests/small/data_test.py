from tfchat.data import BlockDataset
from tfchat.data import LineByLineDataset
import numpy as np


def test_BlockDataset_build():
    loader = BlockDataset(block_size=3, batch_size=2)

    # Loader test
    ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    dataset = loader.build(ids, shuffle=False)

    got_X = np.array([item[0].numpy() for item in dataset])
    want_X = np.array([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]])

    got_y = np.array([item[1].numpy() for item in dataset])
    want_y = np.array([[[1, 2, 3], [4, 5, 6]],
                       [[7, 8, 9], [10, 11, 12]]])

    np.testing.assert_equal(got_X, want_X)
    np.testing.assert_equal(got_y, want_y)


def test_BlockDataset_text_generator():
    loader = BlockDataset(block_size=3, batch_size=2)

    def encode(text):
        words = text.split(" ")
        return [int(w) for w in words]

    # Loader test
    texts = ["0 1 2 3 4", "5 6 7 8", "9 10 11 12 13"]
    dataset = loader.from_text_generator(lambda: texts, encode, shuffle=False)

    got_X = np.array([item[0].numpy() for item in dataset])
    want_X = np.array([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]])

    got_y = np.array([item[1].numpy() for item in dataset])
    want_y = np.array([[[1, 2, 3], [4, 5, 6]],
                       [[7, 8, 9], [10, 11, 12]]])

    np.testing.assert_equal(got_X, want_X)
    np.testing.assert_equal(got_y, want_y)


def test_LineByLineDataset():
    loader = LineByLineDataset(max_len=5, batch_size=2)

    ids = np.array([[0], [0, 1], [0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4]])

    dataset = loader.build(ids, shuffle=False)

    got_X = np.array([item[0].numpy() for item in dataset])
    want_X = np.array([[[0, 0, 0, 0], [0, 1, 0, 0]],
                       [[0, 1, 2, 0], [0, 1, 2, 3]]])

    got_y = np.array([item[1].numpy() for item in dataset])
    want_y = np.array([[[0, 0, 0, 0], [1, 0, 0, 0]],
                       [[1, 2, 0, 0], [1, 2, 3, 0]]])

    np.testing.assert_equal(got_X, want_X)
    np.testing.assert_equal(got_y, want_y)
