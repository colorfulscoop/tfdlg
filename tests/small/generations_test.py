import numpy as np
from tfchat.generations import filter_to_topk
from tfchat.generations import filter_to_topp
from tfchat.generations import filter_bad_ids
from tfchat.generations import sample_multinomial
from tfchat.generations import TopKTopPGenerator


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


def test_sample_multinomial():
    dist = np.array([[2, 0, 3, 1, -1], [5, 6, 7, 8, 9]], dtype=np.float32)
    got = sample_multinomial(dist)

    assert got.shape == (dist.shape[0], )


def test_TopKTopPGenerator():
    class ModelMock:
        def __call__(self, inputs):
            vocab_size = 100
            # Add vocab_size axis at the last dimension
            outputs = np.zeros(inputs.shape + (vocab_size, ), dtype=inputs.dtype)  # shape == (batch_size, seq_len, vocab_size)
            return outputs

    max_len = 20

    generator = TopKTopPGenerator(model=ModelMock(),
                                  top_k=10,
                                  top_p=0.5,
                                  bad_ids=[],
                                  max_len=max_len)

    def test_step():
        batch_size = 2
        seq_len = 10
        inputs = np.zeros((batch_size, seq_len))
        outputs = generator.step(inputs)

        assert outputs.shape == (inputs.shape[0], )

    def test_generation():
        batch_size = 2
        seq_len = 10
        inputs = np.zeros((batch_size, seq_len))
        outputs = generator.generate(inputs)

        assert outputs.shape == (inputs.shape[0], inputs.shape[1]+max_len)

    test_step()
    test_generation()
