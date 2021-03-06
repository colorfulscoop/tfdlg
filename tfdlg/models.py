import numpy as np
import tensorflow as tf
from tensorflow import keras
from .activations import get as get_activation


class PositionalEncoding(keras.layers.Layer):
    def __init__(self, embed_size, max_steps, dtype=tf.float32):
        """
        Args:
            embed_size (int): Embedding dimension
            max_steps (int): The maximum length of tokens
        """
        super().__init__(dtype=dtype)

        # the embedding dimension must be even
        assert embed_size % 2 == 0

        p, i = np.meshgrid(np.arange(max_steps), np.arange(embed_size // 2))
        pos_emb = np.empty((1, max_steps, embed_size))
        pos_emb[0, :, ::2] = np.sin(p / 10000 ** (2 * i / embed_size)).T
        pos_emb[0, :, 1::2] = np.cos(p / 10000 ** (2 * i / embed_size)).T
        # [TODO] positional encoding is not trained in this implementation
        self._positional_embedding = tf.constant(pos_emb.astype(self.dtype))

    def call(self, inputs):
        shape = tf.shape(inputs)
        return inputs + self._positional_embedding[:, :shape[-2], :shape[-1]]


def scaled_dot_product_attention(q, k, v, mask, attention_dropout):
    """
    where
    - vec_size_q and vec_size k should be equal to calculate scores between query and key
    - seq_len_k and seq_len_v should be equal to calculate the sum of all the vectors in each position

    Args:
        q: shape == (..., seq_len_q, d_k)
        k: shape == (..., seq_len_k, d_k)
        v: shape == (..., seq_len_v, d_k)
        mask: shape == (..., seq_len_q, seq_len_k)

    Output: shape == (..., seq_len_q, d_k)
    """
    # mask value should be smaller when fp16 is enabled https://github.com/NVIDIA/apex/issues/93
    if q.dtype == tf.float32:
        mask_val = -1e9
    elif q.dtype == tf.float16:
        mask_val = -1e4
    else:
        raise Exception(f"Input type {q.dtype} is not float16 or float32")

    # shape: (..., seq_len_q, d_k) x (..., seq_len_k, d_k)t == (..., seq_len_q, seq_len_k)
    # (transpose_b tranpose the last two dimension)
    positionwise_score = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # Scaling by the embed dim
    d_k = tf.cast(tf.shape(k)[-1], k.dtype)  # d_k == vec_size_q == vec_size_k
    scaled_positionwise_score = positionwise_score / tf.math.sqrt(d_k)

    if mask is not None:
        mask = tf.cast(mask, scaled_positionwise_score.dtype)
        scaled_positionwise_score += (mask * mask_val)

    # Calculate weight
    weights = tf.nn.softmax(scaled_positionwise_score, axis=-1)

    # Attention dropout
    weights = attention_dropout(weights)

    output = weights @ v  # dim: (..., seq_len_q, d_k)

    return output


class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, d_model, num_heads, attention_dropout_rate):
        super().__init__()

        assert d_model % num_heads == 0

        # Define query, key, value matrix
        self._wq = tf.keras.layers.Dense(d_model, use_bias=False)  # matrix shape: (d_model, d_model)
        self._wk = tf.keras.layers.Dense(d_model, use_bias=False)  # matrix shape: (d_model, d_model)
        self._wv = tf.keras.layers.Dense(d_model, use_bias=False)  # matrix shape: (d_model, d_model)

        # Output dense layer
        self._dense = tf.keras.layers.Dense(d_model, use_bias=False)  # matrix shape: (d_model, d_model)

        # Attention dropout
        # Do not need to set training argument when using with fit function.
        # fit method automatically set training parameter.
        # -> https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout
        self._attention_dropout = tf.keras.layers.Dropout(rate=attention_dropout_rate)

        # Define other attributes
        self._d_model = d_model
        self._num_heads = num_heads
        self._d_k = d_model // num_heads

    def call(self, q, k, v, mask):
        """
        Args:
            q: shape == (batch_size, seq_len_tgt, d_model)
            k: shape == (batch_size, seq_len_src, d_model)
            v: shape == (batch_size, seq_len_src, d_model)

        Output: shape == (batch_size, seq_len_tgt, d_model)
        """
        batch_size = tf.shape(q)[0]

        # Convert inputs to query, key, value
        q = self._wq(q)  # output shape: (batch_size, seq_len, d_model)
        k = self._wk(k)  # output shape: (batch_size, seq_len, d_model)
        v = self._wv(v)  # output shape: (batch_size, seq_len, d_model)

        # Split head into num_heads
        q = self._split_heads(batch_size, q)  # output shape: (batch_size, num_heads, seq_len, d_k)
        k = self._split_heads(batch_size, k)  # output shape: (batch_size, num_heads, seq_len, d_k)
        v = self._split_heads(batch_size, v)  # output shape: (batch_size, num_heads, seq_len, d_k)

        attn = scaled_dot_product_attention(q, k, v, mask, self._attention_dropout)  # output shape: (batch_size, num_heads, seq_len, d_k)
        attn = tf.transpose(attn, perm=[0, 2, 1, 3])  # output shape: (batch_size, seq_len, num_heads, d_k)

        # Concat attention heads (merge 2nd and 3rd dimensions into one)
        concat_attn = tf.reshape(attn, (batch_size, -1, self._d_model))  # output shape: (batch_size, seq_len, d_model)

        # Apply final dense layer
        output = self._dense(concat_attn)  # output shape: (batch_size, seq_len, d_model)

        return output

    def _split_heads(self, batch_size, tensor):
        """
        Args:
            tensor: shape == (batch_size, seq_len, d_model)

        Returns: shape == (batch_size, num_heads, seq_len, d_k)

        """
        # Reshape the last dimention: (d_model,) -> (num_heads, d_k)
        x = tf.reshape(tensor, (batch_size, -1, self._num_heads, self._d_k))
        # Replace the dimension
        tx = tf.transpose(x, perm=[0, 2, 1, 3])  # (batch_size, num_heads, seq_len, d_k)
        return tx


class PointwiseFeedForwardNetwork(keras.layers.Layer):
    def __init__(self, d_model, d_ff, activation, kernel_initializer):
        super().__init__()
        self._mid = tf.keras.layers.Dense(d_ff,
                                          activation=get_activation(activation),
                                          kernel_initializer=kernel_initializer)
        self._last = tf.keras.layers.Dense(d_model)

    def call(self, inputs):
        mid = self._mid(inputs)
        output = self._last(mid)
        return output


class TransposedEmbedding(keras.layers.Layer):
    """TransposedEmbedding represents the linear layer which shares its weights
    with the Embedding layer specified in the initializer.
    """
    # One solution is suggested in Stack Overflow to tie Embedding and liner layer
    # https://stackoverflow.com/questions/47095673/how-to-tie-word-embedding-and-softmax-weights-in-keras
    def __init__(self, embedding_layer, **kwargs):
        super().__init__(**kwargs)
        self._embedding_layer = embedding_layer

    def call(self, inputs):
        return tf.matmul(inputs, tf.cast(self._embedding_layer.weights[0], inputs.dtype), transpose_b=True)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, transformer_cls, num_layers, d_model, num_heads,
                 d_ff, vocab_size, context_size,
                 residual_dropout_rate, attention_dropout_rate,
                 embedding_dropout_rate,
                 activation, kernel_initializer,
                 epsilon=1e-6):
        super().__init__()

        self._embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self._pos_enc = PositionalEncoding(embed_size=d_model,
                                           max_steps=context_size)

        self._dec_layers = [transformer_cls(d_model=d_model,
                                            num_heads=num_heads,
                                            d_ff=d_ff,
                                            residual_dropout_rate=residual_dropout_rate,
                                            attention_dropout_rate=attention_dropout_rate,
                                            activation=activation,
                                            kernel_initializer=kernel_initializer,
                                            epsilon=epsilon)
                            for _ in range(num_layers)]
        self._dropout = tf.keras.layers.Dropout(rate=embedding_dropout_rate)
        self._output_layer = TransposedEmbedding(embedding_layer=self._embedding)
        #self._output_layer = keras.layers.Dense(vocab_size, use_bias=False)

        self._to_fp32 = tf.keras.layers.Activation('linear', dtype='float32')

        self._num_layers = num_layers
        self._d_model = d_model

        # parameters for cls
        self._cls_output_layer = tf.keras.layers.Dense(1,
                                                       activation=get_activation(activation),
                                                       kernel_initializer=kernel_initializer)

    def call(self, inputs, training, look_ahead_mask, cls_ids=None):
        """
        Args:
            inputs: shape == (batch_size, seq_len)

        Returns: shape == (batch_size, seq_len, d_model)
        """
        # seq_len = tf.shape(inputs)[1]

        # Embedding returns float32 even if you use fp16 mode
        # because the input to the Embedding layer is not float32 but int32
        # You can find the concrete explanation here https://github.com/tensorflow/tensorflow/issues/41614
        x = self._embedding(inputs)  # shape: (batch_size, seq_len, d_model)

        x *= tf.math.sqrt(tf.cast(self._d_model, x.dtype))
        x = self._pos_enc(x)  # dtype == float32

        # Residual dropout: Dropout(Embed(x) + Pos(i))
        x = self._dropout(x, training=training)  # dtype == float32 of float16

        for i in range(self._num_layers):
            # output shape of x: (batch_size, seq_len, d_model)
            x = self._dec_layers[i](inputs=x, training=training,
                                    look_ahead_mask=look_ahead_mask)

        x = self._output_layer(x)
        cls_input = x

        # the final output before Softmax should be float32 when the policy is float16.
        # You can find more detail here https://www.tensorflow.org/guide/mixed_precision
        x = self._to_fp32(x)

        if cls_ids is not None:
            cls_output = tf.gather(cls_input, cls_ids, axis=1, batch_dims=1)  # Shape: (batch_size, d_model)
            cls_output = self._cls_output_layer(cls_output)
            cls_output = self._to_fp32(cls_output)
            return x, cls_output
        else:
            return x


def create_padding_mask(seq):
    """
    Args:
        seq: shape == (batch_size, seq_len)

    Returns: shape == (batch_size, 1, 1, seq_len)
    """
    mask = tf.cast(tf.math.equal(seq, 0), tf.float32)
    batch_size = tf.shape(seq)[0]
    seq_len = tf.shape(seq)[-1]
    out = tf.reshape(mask, (batch_size, 1, 1, seq_len))

    return out


def create_look_ahead_mask(seq_len):
    mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)  # shape == (seq_len, seq_len)
    return mask


def create_combined_mask(seq):
    seq_len = tf.shape(seq)[-1]
    return tf.maximum(create_padding_mask(seq), create_look_ahead_mask(seq_len))


# ====== Define PostLNDecoder model ======


class PostLN(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, residual_dropout_rate,
                 attention_dropout_rate, activation, kernel_initializer,
                 epsilon=1e-6,
                 ):
        """
        [TODO] Need to check the default value of parameters - rate and epsilon
        """
        super().__init__()

        self._mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads,    
                                       attention_dropout_rate=attention_dropout_rate)
        self._fst_layernorm = tf.keras.layers.LayerNormalization(epsilon=epsilon)
        self._fst_dropout = tf.keras.layers.Dropout(rate=residual_dropout_rate)

        self._ffn = PointwiseFeedForwardNetwork(d_model=d_model, d_ff=d_ff, activation=activation, kernel_initializer=kernel_initializer)
        self._snd_layernorm = tf.keras.layers.LayerNormalization(epsilon=epsilon)
        self._snd_dropout = tf.keras.layers.Dropout(rate=residual_dropout_rate)

    def call(self, inputs, training, look_ahead_mask):
        attn = self._mha(inputs, inputs, inputs, mask=look_ahead_mask)
        # Residual dropout: LayerNorm(x + Dropout(Sublayer(x)))
        attn = self._fst_dropout(attn, training=training)
        fst_out = self._fst_layernorm(inputs + attn)

        ffn = self._ffn(fst_out)
        # Residual dropout: LayerNorm(x + Dropout(Sublayer(x)))
        ffn = self._snd_dropout(ffn, training=training)
        snd_out = self._snd_layernorm(fst_out + ffn)

        return snd_out


def parse_input(inputs):
    """Parse input to the top of the mode classes (PostLNDecoder, PreLNDecoder)."""
    if isinstance(inputs, tuple) or isinstance(inputs, list):
        assert len(inputs) <= 2
        lm_inputs = inputs[0]
        if len(inputs) == 2:
            cls_ids = inputs[1]
        else:
            cls_ids = None
    else:
        lm_inputs = inputs
        cls_ids = None
    return lm_inputs, cls_ids


class PostLNDecoder(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self._decoder = Decoder(transformer_cls=PostLN, **config.dict())

    def call(self, inputs, training=None):
        lm_inputs, cls_ids = parse_input(inputs)
        look_ahead_mask = create_combined_mask(lm_inputs)
        return self._decoder(inputs=lm_inputs, training=training, look_ahead_mask=look_ahead_mask, cls_ids=cls_ids)


# ====== Define PreLNDecoder model ======


class PreLN(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, residual_dropout_rate,
                 attention_dropout_rate, activation, kernel_initializer,
                 epsilon=1e-6,
                 ):
        """
        [TODO] Need to check the default value of parameters - rate and epsilon
        """
        super().__init__()

        self._mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads,    
                                       attention_dropout_rate=attention_dropout_rate)
        self._fst_layernorm = tf.keras.layers.LayerNormalization(epsilon=epsilon)
        self._fst_dropout = tf.keras.layers.Dropout(rate=residual_dropout_rate)

        self._ffn = PointwiseFeedForwardNetwork(d_model=d_model, d_ff=d_ff, activation=activation, kernel_initializer=kernel_initializer)
        self._snd_layernorm = tf.keras.layers.LayerNormalization(epsilon=epsilon)
        self._snd_dropout = tf.keras.layers.Dropout(rate=residual_dropout_rate)

    def call(self, inputs, training, look_ahead_mask):
        x = self._fst_layernorm(inputs)
        attn = self._mha(x, x, x, mask=look_ahead_mask)
        # Residual dropout: LayerNorm(x + Dropout(Sublayer(x)))
        attn = self._fst_dropout(attn, training=training)
        fst_out = inputs + attn

        y = self._snd_layernorm(fst_out)
        ffn = self._ffn(y)
        # Residual dropout: LayerNorm(x + Dropout(Sublayer(x)))
        ffn = self._snd_dropout(ffn, training=training)
        snd_out = fst_out + ffn

        return snd_out


class PreLNDecoder(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self._decoder = Decoder(transformer_cls=PreLN, **config.dict())

    def call(self, inputs, training=None):
        lm_inputs, cls_ids = parse_input(inputs)
        look_ahead_mask = create_combined_mask(lm_inputs)
        return self._decoder(inputs=lm_inputs, training=training, look_ahead_mask=look_ahead_mask, cls_ids=cls_ids)
