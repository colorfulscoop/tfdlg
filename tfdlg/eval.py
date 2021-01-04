from .losses import PaddingLoss
import tensorflow as tf
import numpy as np


def perplexity(model, dataset, verbose=True):
    loss_fn = PaddingLoss()

    num_batches = 0
    shapes = []
    num_tokens = []
    losses = []

    for item in dataset:
        X, y_true = item
        shapes.append(y_true.shape)
        mask = tf.math.logical_not(tf.math.equal(y_true, 0))
        y_pred = model(X)

        num_batches += 1

        # - (1/N) sum{ log P(x_t|x<t) }
        loss = loss_fn(y_true, y_pred)

        # Mask calculation to consider padding
        mask = tf.cast(mask, dtype=loss.dtype)
        num_token = tf.reduce_sum(mask)

        # - sum{ log P(x_t|x<t) }
        part_loss = loss * num_token

        num_tokens.append(num_token)
        losses.append(part_loss)

    num_tokens = tf.reduce_sum(num_tokens)
    final_loss = tf.reduce_sum(losses) / num_tokens

    stats = {
        "loss": final_loss.numpy(),
        "perplexity": tf.math.exp(final_loss).numpy(),
        "num_batches": num_batches,
        "num_tokens": num_tokens.numpy().astype(np.int32),
    }
    if verbose:
        print(stats)

    return stats["perplexity"]

