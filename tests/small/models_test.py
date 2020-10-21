import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tfchat.models import PositionalEncoding
from tfchat.models import scaled_dot_product_attention
from tfchat.models import MultiHeadAttention
from tfchat.models import PointwiseFeedForwardNetwork
from tfchat.models import PostLN
from tfchat.models import TransposedEmbedding


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

    q = np.zeros((seq_len_tgt, d_k), dtype=np.float32)
    k = np.zeros((seq_len_src, d_k), dtype=np.float32)
    v = np.zeros((seq_len_src, d_k), dtype=np.float32)

    assert scaled_dot_product_attention(q, k, v, mask=None).shape == (seq_len_tgt, d_k)


def test_multi_head_attention():
    batch_size = 2
    d_model = 128
    num_heads = 4
    seq_len_src = 10
    seq_len_tgt = 5

    q = np.zeros((batch_size, seq_len_tgt, d_model), dtype=np.float32)
    k = np.zeros((batch_size, seq_len_src, d_model), dtype=np.float32)
    v = np.zeros((batch_size, seq_len_src, d_model), dtype=np.float32)

    mh_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    got = mh_attn(q, k, v, mask=None).shape

    assert got == (batch_size, seq_len_tgt, d_model)


def test_pointwise_ffn():
    d_model = 128
    d_ff = 256
    fnn = PointwiseFeedForwardNetwork(d_model=d_model, d_ff=d_ff)

    batch_size = 2
    seq_len = 10
    inputs = np.zeros((batch_size, seq_len, d_model), dtype=np.float32)

    got = fnn(inputs)

    assert got.shape == (batch_size, seq_len, d_model)


def test_post_ln():
    d_model = 128
    num_heads = 8
    d_ff = 256
    layer = PostLN(d_model=d_model, num_heads=num_heads, d_ff=d_ff)

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