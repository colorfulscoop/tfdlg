import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tfchat.models import PreLNDecoder
from tfchat.configs import Config
from tfchat.losses import PaddingLoss
from tfchat.utils import set_mixed_precision_policy
import pytest


@pytest.fixture()
def enable_mixed_precision_policy():
    # Preprocess
    set_mixed_precision_policy()

    # Execute test
    yield

    # Postprocess
    set_mixed_precision_policy("float32")


def test_PreLNDecoder(enable_mixed_precision_policy):
    config = Config(
        num_layers=4,
        d_model=128,
        num_heads=8,
        d_ff=256,
        vocab_size=1000,
        context_size=100,
        residual_dropout_rate=0.1,
        attention_dropout_rate=0.1,
        embedding_dropout_rate=0.1,
        activation="relu",
        kernel_initializer="he_normal",
        epsilon=1e-6,
    )

    gpt = PreLNDecoder(config)

    batch_size = 2
    seq_len = 10
    inputs = np.ones((batch_size, seq_len), dtype=np.int32)

    got = gpt(inputs, training=False)

    assert not np.any(tf.math.is_nan(got))
    assert got.shape == (batch_size, seq_len, config.vocab_size)


def test_PreLNDecoder_fit(enable_mixed_precision_policy):
    config = Config(
        num_layers=4,
        d_model=128,
        num_heads=8,
        d_ff=256,
        vocab_size=1000,
        context_size=100,
        residual_dropout_rate=0.1,
        attention_dropout_rate=0.1,
        embedding_dropout_rate=0.1,
        activation="relu",
        kernel_initializer="he_normal",
        epsilon=1e-6,
    )

    model = PreLNDecoder(config)
    model.compile(
        loss=PaddingLoss(),
        optimizer=keras.optimizers.Adam(),
    )

    # Prepare input data
    seq_len = 10
    num_samples = 10

    inputs = np.ones((num_samples, seq_len), dtype=np.int32)
    outputs = inputs

    model.fit(inputs, outputs)
