from tfchat.losses import PaddingLoss
import tensorflow as tf


def perplexity(model, dataset):
    loss_fn = PaddingLoss()

    num_tokens = []
    ppls = []

    for item in dataset.take(3):
        X, y_true = item
        mask = tf.math.logical_not(tf.math.equal(y_true, 0))
        y_pred = model(X)

        # - (1/N) sum{ log P(x_t|x<t) }
        loss = loss_fn(y_true, y_pred)

        # Mask calculation to consider padding
        mask = tf.cast(mask, dtype=loss.dtype)
        num_token = tf.reduce_sum(mask)

        # - sum{ log P(x_t|x<t) }
        part_ppl = loss * num_token

        num_tokens.append(num_token)
        ppls.append(part_ppl)

    log_val = tf.reduce_sum(ppls) / tf.reduce_sum(num_tokens)
    return tf.math.exp(log_val)
