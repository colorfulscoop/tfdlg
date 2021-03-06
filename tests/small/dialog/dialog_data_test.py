from tfdlg.dialog.data import DialogDataset
from tfdlg.dialog.data import DialogClsDataset
from tfdlg.dialog.tokenizers import encode_dialog
import numpy as np


def test_DialogDataset_test_generator():
    def encode(text):
        words = text.split(" ")
        return [int(w) for w in words]

    texts = [["1", "1 2"],
             ["1 2", "1 2 3"],
             ["1 2 3 4 5", "1 2 3 4 5 6"],
             ["1 2"],
             ["1"],
             ]

    dataset = DialogDataset.from_generator(
        lambda: texts,
        context_encode_fn=lambda context: encode_dialog(encode_fn=encode, sep_token_id=-1, context=context, response=None),
        max_len=5,
        batch_size=2,
        shuffle=False
    )

    got_X = np.array([item[0].numpy() for item in dataset])
    want_X = np.array([[[-1, 1, -1, 1], [-1, 1, 2, -1]],
                       [[-1, 1, 2, 3], [-1, 1, 2, -1]]])

    got_y = np.array([item[1].numpy() for item in dataset])
    want_y = np.array([[[1, -1, 1, 2], [1, 2, -1, 1]],
                       [[1, 2, 3, 4], [1, 2, -1, 0]]])

    np.testing.assert_equal(got_X, want_X)
    np.testing.assert_equal(got_y, want_y)


def test_DialogClsDataset_test_generator():
    def encode(text):
        words = text.split(" ")
        return [int(w) for w in words]

    texts = [["1", "1 2"],
             ["1 2", "1 2 3"],
             ["1 2 3 4 5", "1 2 3 4 5 6"],
             ["1 2"],
             ["1"],
             ]

    dataset = DialogClsDataset.from_generator(
        lambda: texts,
        encode_fn=encode,
        max_len=5,
        sep_token_id=-1,
        batch_size=2,
        shuffle=False,
        seed=0,
    )

    got = list(dataset.as_numpy_iterator())
    print(got)

    got_lm_X = np.array([item[0][0] for item in got])
    want_lm_X = np.array([[[-1, 1, -1,  1, 2], [-1, 1, -1,  1, 2]],
                          [[-1, 1,  2, -1, 1], [-1, 1,  2, -1, 1]],
                          [[-1, 1,  2,  3, 4], [-1, 1,  2,  3, 4]],
                          [[-1, 1,  2, -1, 0], [-1, 1, -1,  0, 0]],
                          [[-1, 1, -1,  0, 0], [-1, 1,  2, -1, 0]],
                          ])

    got_lm_y = np.array([item[1][0] for item in got])
    want_lm_y = np.array([[[1, -1,  1, 2, -1], [0, 0, 0, 0, 0]],
                          [[1,  2, -1, 1,  2], [0, 0, 0, 0, 0]],
                          [[1,  2,  3, 4,  5], [0, 0, 0, 0, 0]],
                          [[1,  2, -1, 0,  0], [0, 0, 0, 0, 0]],
                          [[1, -1,  0, 0,  0], [0, 0, 0, 0, 0]],
                          ])
    got_cls_ids = np.array([item[0][1] for item in got])
    want_cls_ids = np.array([[4, 4],
                             [4, 4],
                             [4, 4],
                             [3, 2],
                             [2, 3],
                             ])
    got_label = np.array([item[1][1] for item in got])
    want_label = np.array([[1, 0],
                           [1, 0],
                           [1, 0],
                           [1, 0],
                           [1, 0],
                           ])

    np.testing.assert_equal(got_lm_X, want_lm_X)
    np.testing.assert_equal(got_lm_y, want_lm_y)
    np.testing.assert_equal(got_label, want_label)
    np.testing.assert_equal(got_cls_ids, want_cls_ids)
