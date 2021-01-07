from tfdlg.dialog.data import DialogDataset
import numpy as np
from tfdlg.data import LineByLineDataset
from tfdlg.data import BlockDataset


def test_DialogDataset_test_generator():
    def encode(text):
        words = text.split(" ")
        return [int(w) for w in words]

    loader = DialogDataset(max_len=5, encode_fn=encode, sep_token_id=-1)

    texts = [["1", "1 2"],
             ["1 2", "1 2 3"],
             ["1 2 3 4 5", "1 2 3 4 5 6"],
             ["1 2"],
             ["1"],
             ]

    dataset = loader.from_text_generator(lambda: texts, batch_size=2, shuffle=False)

    got_X = np.array([item[0].numpy() for item in dataset])
    want_X = np.array([[[-1, 1, -1, 1], [-1, 1, 2, -1]],
                       [[-1, 1, 2, 3], [-1, 1, 2, -1]]])

    got_y = np.array([item[1].numpy() for item in dataset])
    want_y = np.array([[[1, -1, 1, 2], [1, 2, -1, 1]],
                       [[1, 2, 3, 4], [1, 2, -1, 0]]])

    np.testing.assert_equal(got_X, want_X)
    np.testing.assert_equal(got_y, want_y)


def test_DialogDataset_convert_text_to_ids():
    def encode(text):
        words = text.split(" ")
        return [int(w) for w in words]

    dt = DialogDataset(max_len=8, encode_fn=encode, sep_token_id=-1)

    # Case: context only
    got = dt.convert_text_to_ids(text="|0 1|3|")
    want = np.array([-1, 0, 1, -1, 3, -1], dtype=np.int32)
    np.testing.assert_equal(got, want)

    # Case: many tokens more than max_len
    got = dt.convert_text_to_ids(text="|0 1 2 3 4|5 6 7|")
    want = np.array([-1, 0, 1, 2, 3, 4, -1, 5], dtype=np.int32)
    np.testing.assert_equal(got, want)

    # Case: context and response
    got = dt.convert_text_to_ids("|0|3|5 6")
    want = np.array([-1, 0, -1, 3, -1, 5, 6], dtype=np.int32)
    np.testing.assert_equal(got, want)

    # Case: resposne only
    got = dt.convert_text_to_ids(text="|5 6")
    want = np.array([-1, 5, 6], dtype=np.int32)
    np.testing.assert_equal(got, want)

    # Case: Nothing given
    got = dt.convert_text_to_ids(text="")
    want = np.array([], dtype=np.int32)
    np.testing.assert_equal(got, want)

    # Case: separator only give
    got = dt.convert_text_to_ids(text="|")
    want = np.array([-1], dtype=np.int32)
    np.testing.assert_equal(got, want)


def test_DialogDataset_convert_context_to_ids():
    def encode(text):
        words = text.split(" ")
        return [int(w) for w in words]

    dt = DialogDataset(max_len=8, encode_fn=encode, sep_token_id=-1)

    # Case: context only
    got = dt.convert_context_to_ids(context=["0 1", "3"])
    want = np.array([-1, 0, 1, -1, 3, -1], dtype=np.int32)
    np.testing.assert_equal(got, want)

    # Case: many tokens more than max_len
    got = dt.convert_context_to_ids(context=["0 1 2 3 4", "5 6 7"])
    want = np.array([-1, 0, 1, 2, 3, 4, -1, 5], dtype=np.int32)
    np.testing.assert_equal(got, want)

    # Case: context and response
    got = dt.convert_context_to_ids(context=["0", "3"], response="5 6")
    want = np.array([-1, 0, -1, 3, -1, 5, 6], dtype=np.int32)
    np.testing.assert_equal(got, want)

    # Case: resposne only
    got = dt.convert_context_to_ids(context=[], response="5 6")
    want = np.array([-1, 5, 6], dtype=np.int32)
    np.testing.assert_equal(got, want)

    # Case: Nothing given
    got = dt.convert_context_to_ids(context=[])
    want = np.array([-1], dtype=np.int32)
    np.testing.assert_equal(got, want)


def test_BlockDataset_convert_context_to_ids():
    def encode(text):
        words = text.split(" ")
        return [int(w) for w in words]

    dt = BlockDataset(block_size=4, encode_fn=encode)

    got = dt.convert_context_to_ids(context=["0 1", "3"], response="4 5")
    want = np.array([0, 1, 3, 4, 5], dtype=np.int32)
    np.testing.assert_equal(got, want)


def test_LineByLineDataset_convert_context_to_ids():
    def encode(text):
        words = text.split(" ")
        return [int(w) for w in words]

    dt = LineByLineDataset(max_len=4, encode_fn=encode)

    got = dt.convert_context_to_ids(context=["0 1", "3"], response="4 5")
    want = np.array([0, 1, 3, 4], dtype=np.int32)
    np.testing.assert_equal(got, want)
