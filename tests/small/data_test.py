from tfchat.data import BlockDataLoader
import numpy as np


def test_TestDataLoader():
    ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    loader = BlockDataLoader(block_size=3, batch_size=2, shuffle=False)
    dataset = loader.load(ids)

    got_X = np.array([item[0].numpy() for item in dataset])
    want_X = np.array([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]])

    np.testing.assert_equal(got_X, want_X)
