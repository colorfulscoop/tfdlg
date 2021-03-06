import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
from tfdlg.models import PositionalEncoding
from tfdlg.models import scaled_dot_product_attention
from tfdlg.models import MultiHeadAttention
from tfdlg.models import PointwiseFeedForwardNetwork
from tfdlg.models import PostLN
from tfdlg.models import TransposedEmbedding
from tfdlg.models import Decoder
from tfdlg.models import create_padding_mask
from tfdlg.models import create_look_ahead_mask
from tfdlg.models import create_combined_mask
from tfdlg.models import PostLNDecoder
from tfdlg.models import PreLNDecoder
from tfdlg.losses import PaddingLoss
from tfdlg.configs import Config


def test_position_encoding():
    batch_size = 4
    d_model = 128
    max_steps = 100

    pos_enc_inputs = np.zeros((batch_size, max_steps, d_model))
    pos_enc = PositionalEncoding(embed_size=d_model, max_steps=max_steps)

    got = pos_enc(pos_enc_inputs)

    assert got.shape == (batch_size, max_steps, d_model)


def test_scaled_dot_product_attention():
    seq_len_tgt = 2
    seq_len_src = 3
    d_k = 128

    q = tf.zeros((seq_len_tgt, d_k), dtype=tf.float32)
    k = tf.zeros((seq_len_src, d_k), dtype=tf.float32)
    v = tf.zeros((seq_len_src, d_k), dtype=tf.float32)

    dropout = keras.layers.Dropout(rate=0.1)

    assert scaled_dot_product_attention(q, k, v, mask=None, attention_dropout=dropout).shape == (seq_len_tgt, d_k)


def test_multi_head_attention():
    batch_size = 2
    d_model = 128
    num_heads = 4
    seq_len_src = 10
    seq_len_tgt = 5

    q = np.zeros((batch_size, seq_len_tgt, d_model), dtype=np.float32)
    k = np.zeros((batch_size, seq_len_src, d_model), dtype=np.float32)
    v = np.zeros((batch_size, seq_len_src, d_model), dtype=np.float32)

    mh_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, attention_dropout_rate=0.1)
    got = mh_attn(q, k, v, mask=None).shape

    assert got == (batch_size, seq_len_tgt, d_model)


def test_pointwise_ffn():
    d_model = 128
    d_ff = 256
    fnn = PointwiseFeedForwardNetwork(d_model=d_model, d_ff=d_ff, activation="relu", kernel_initializer="he_normal")

    batch_size = 2
    seq_len = 10
    inputs = np.zeros((batch_size, seq_len, d_model), dtype=np.float32)

    got = fnn(inputs)

    assert got.shape == (batch_size, seq_len, d_model)


def test_post_ln():
    d_model = 128
    num_heads = 8
    d_ff = 256
    layer = PostLN(d_model=d_model, num_heads=num_heads, d_ff=d_ff, residual_dropout_rate=0.1, attention_dropout_rate=0.1,
                   activation="relu", kernel_initializer="he_normal")

    batch_size = 2
    seq_len = 10
    inputs = np.zeros((batch_size, seq_len, d_model), dtype=np.float32)

    got = layer(inputs, training=False, look_ahead_mask=None)

    assert got.shape == (batch_size, seq_len, d_model)


def test_transposed_embedding():
    batch_size = 20
    vocab_size = 100
    embed_dim = 10

    emb = keras.layers.Embedding(vocab_size, embed_dim)
    temb = TransposedEmbedding(embedding_layer=emb)

    emb_inputs = np.zeros((batch_size, vocab_size), dtype=np.float32)

    # Pass inputs to Embedding to generate weight, because
    # Embedding weight is generated after inputting data.
    emb(emb_inputs)

    #temb.build(input_shape=(None, embed_dim))

    inputs = np.zeros((batch_size, embed_dim), dtype=np.float32)
    got = temb(inputs)
    assert got.shape == (batch_size, vocab_size)


def test_decoder():
    transformer_cls = PostLN
    num_layers = 4
    d_model = 128
    num_heads = 8
    d_ff = 256
    vocab_size = 1000
    context_size = 100

    decoder = Decoder(transformer_cls, num_layers, d_model, num_heads,
                      d_ff, vocab_size, context_size,
                      residual_dropout_rate=0.1,
                      attention_dropout_rate=0.1,
                      embedding_dropout_rate=0.1,
                      activation="relu",
                      kernel_initializer="he_normal"
                      )

    batch_size = 2
    seq_len = 10
    inputs = np.zeros((batch_size, seq_len), dtype=np.int32)

    got = decoder(inputs, training=False, look_ahead_mask=None)

    assert got.shape == (batch_size, seq_len, vocab_size)


def test_create_padding_mask():
    seq = np.array([[1, 2, 0], [0, 2, 0]], dtype=np.int32)
    res = create_padding_mask(seq)
    assert np.all(res.numpy() == [[[[0, 0, 1]]], [[[1, 0, 1]]]])


def test_create_look_ahead_mask():
    out = create_look_ahead_mask(3)
    assert np.all(out.numpy() == [[0, 1, 1], [0, 0, 1], [0, 0, 0]])


def test_create_combined_mask():
    seq = np.array([[1, 2, 0], [0, 2, 0]], dtype=np.int32)
    out = create_combined_mask(seq)
    assert np.all(out.numpy() == [[[[0, 1, 1], [0, 0, 1], [0, 0, 1]]],
                                  [[[1, 1, 1], [1, 0, 1], [1, 0, 1]]]])


def test_PostLNDecoder():
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

    gpt = PostLNDecoder(config)

    batch_size = 2
    seq_len = 10
    inputs = np.ones((batch_size, seq_len), dtype=np.int32)

    got = gpt(inputs, training=False)

    assert got.shape == (batch_size, seq_len, config.vocab_size)


def test_PostLNDecoder_fit():
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

    model = PostLNDecoder(config)
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


# Test for the model of classification head

def test_decoder_cls():
    transformer_cls = PostLN
    num_layers = 4
    d_model = 128
    num_heads = 8
    d_ff = 256
    vocab_size = 1000
    context_size = 100

    decoder = Decoder(transformer_cls, num_layers, d_model, num_heads,
                      d_ff, vocab_size, context_size,
                      residual_dropout_rate=0.1,
                      attention_dropout_rate=0.1,
                      embedding_dropout_rate=0.1,
                      activation="relu",
                      kernel_initializer="he_normal"
                      )

    batch_size = 2
    seq_len = 10
    inputs = np.zeros((batch_size, seq_len), dtype=np.int32)
    cls_ids = np.ones((batch_size,), dtype=np.int32)

    got_lm, got_cls = decoder(inputs, cls_ids=cls_ids, training=False, look_ahead_mask=None)

    assert got_lm.shape == (batch_size, seq_len, vocab_size)
    assert got_cls.shape == (batch_size, 1)


def helper_Decoder_fit_cls(cls):
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

    model = cls(config)
    model.compile(
        loss=(PaddingLoss(), tf.keras.losses.BinaryCrossentropy(from_logits=True)),
        optimizer=keras.optimizers.Adam(),
    )

    # Prepare input data
    seq_len = 10
    num_samples = 10

    lm_inputs = np.ones((num_samples, seq_len), dtype=np.int32)
    lm_outputs = lm_inputs

    cls_ids = np.ones((num_samples,), dtype=np.int32)
    cls_outputs = np.ones((num_samples,), dtype=np.int32)

    inputs = (lm_inputs, cls_ids)
    outputs = (lm_outputs, cls_outputs)

    model.fit(inputs, outputs, batch_size=3)


def test_PostLNDecoder_fit_cls():
    helper_Decoder_fit_cls(PostLNDecoder)


def test_PreLNDecoder_fit_cls():
    helper_Decoder_fit_cls(PreLNDecoder)