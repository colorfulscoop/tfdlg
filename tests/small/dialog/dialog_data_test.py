from tfchat.dialog.data import ContextDataset
import numpy as np


def test_ContextDataset_test_generator():
    loader = ContextDataset(max_len=5, batch_size=2, sep_token_id=-1)

    def encode(text):
        words = text.split(" ")
        return [int(w) for w in words]

    texts = [["0", "0 1"],
             ["0 1", "0 1 2"],
             ["0 1 2 3 4", "0 1 2 3 4 5"],
             ["1 2 3"],
             ["1"],
             ]

    dataset = loader.from_text_generator(lambda: texts, encode, shuffle=False)

    got_X = np.array([item[0].numpy() for item in dataset])
    want_X = np.array([[[0, -1, 0, 1], [0, 1, -1, 0]],
                       [[0, 1, 2, 3], [1, 2, 3, 0]]])

    got_y = np.array([item[1].numpy() for item in dataset])
    want_y = np.array([[[-1, 0, 1, 0], [1, -1, 0, 1]],
                       [[1, 2, 3, 4], [2, 3, 0, 0]]])

    np.testing.assert_equal(got_X, want_X)
    np.testing.assert_equal(got_y, want_y)

