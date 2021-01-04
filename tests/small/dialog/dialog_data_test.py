from tfdlg.dialog.data import DialogDataset
import numpy as np


def test_DialogDataset_test_generator():
    loader = DialogDataset(max_len=5, batch_size=2, sep_token_id=-1)

    def encode(text):
        words = text.split(" ")
        return [int(w) for w in words]

    texts = [["1", "1 2"],
             ["1 2", "1 2 3"],
             ["1 2 3 4 5", "1 2 3 4 5 6"],
             ["1 2"],
             ["1"],
             ]

    dataset = loader.from_text_generator(lambda: texts, encode, shuffle=False)

    got_X = np.array([item[0].numpy() for item in dataset])
    want_X = np.array([[[-1, 1, -1, 1], [-1, 1, 2, -1]],
                       [[-1, 1, 2, 3], [-1, 1, 2, -1]]])

    got_y = np.array([item[1].numpy() for item in dataset])
    want_y = np.array([[[1, -1, 1, 2], [1, 2, -1, 1]],
                       [[1, 2, 3, 4], [1, 2, -1, 0]]])

    np.testing.assert_equal(got_X, want_X)
    np.testing.assert_equal(got_y, want_y)
