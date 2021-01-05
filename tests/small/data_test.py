from tfdlg.data import BlockDataset
from tfdlg.data import LineByLineDataset
import numpy as np


def test_BlockDataset_text_generator():
    def encode(text):
        words = text.split(" ")
        return [int(w) for w in words]

    loader = BlockDataset(block_size=3, encode_fn=encode)

    # Loader test
    texts = ["0 1 2 3 4", "5 6 7 8", "9 10 11 12 13"]
    dataset = loader.from_text_generator(lambda: texts, batch_size=2, shuffle=False)

    got_X = np.array([item[0].numpy() for item in dataset])
    want_X = np.array([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]])

    got_y = np.array([item[1].numpy() for item in dataset])
    want_y = np.array([[[1, 2, 3], [4, 5, 6]],
                       [[7, 8, 9], [10, 11, 12]]])

    np.testing.assert_equal(got_X, want_X)
    np.testing.assert_equal(got_y, want_y)


def test_BlockDataset_convert_text_to_ids():
    def encode(text):
        words = text.split(" ")
        return [int(w) for w in words]

    dt = BlockDataset(block_size=3, encode_fn=encode)

    got = dt.convert_text_to_ids(text="0 1 2 3 4 5")
    want = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)

    np.testing.assert_equal(got, want)


def test_LineByLineDataset_from_text_generator():
    def encode(text):
        words = text.split(" ")
        return [int(w) for w in words]

    loader = LineByLineDataset(max_len=5, encode_fn=encode)

    texts = ["0", "0 1", "0 1 2", "0 1 2 3", "0 1 2 3 4"]

    dataset = loader.from_text_generator(lambda: texts, batch_size=2, shuffle=False)

    got_X = np.array([item[0].numpy() for item in dataset])
    want_X = np.array([[[0, 0, 0, 0], [0, 1, 0, 0]],
                       [[0, 1, 2, 0], [0, 1, 2, 3]]])

    got_y = np.array([item[1].numpy() for item in dataset])
    want_y = np.array([[[0, 0, 0, 0], [1, 0, 0, 0]],
                       [[1, 2, 0, 0], [1, 2, 3, 0]]])

    np.testing.assert_equal(got_X, want_X)
    np.testing.assert_equal(got_y, want_y)


def test_lineByLineDataset_convert_text_to_ids():
    def encode(text):
        words = text.split(" ")
        return [int(w) for w in words]

    dt = LineByLineDataset(max_len=5, encode_fn=encode)

    got = dt.convert_text_to_ids(text="0 1 2 3 4 5")
    want = np.array([0, 1, 2, 3, 4], dtype=np.int32)

    np.testing.assert_equal(got, want)


def test_LineByLineDataset_from_text_generator_large_input():
    def encode(text):
        words = text.split(" ")
        return [int(w) for w in words]

    loader = LineByLineDataset(max_len=5, encode_fn=encode)

    texts = ["0 1 2 3 4 5 6"]

    dataset = loader.from_text_generator(lambda: texts, batch_size=1, shuffle=False)

    got_X = np.array([item[0].numpy() for item in dataset])
    want_X = np.array([[[0, 1, 2, 3]]])

    got_y = np.array([item[1].numpy() for item in dataset])
    want_y = np.array([[[1, 2, 3, 4]]])

    np.testing.assert_equal(got_X, want_X)
    np.testing.assert_equal(got_y, want_y)
