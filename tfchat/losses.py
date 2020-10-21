import tensorflow as tf
from tensorflow import keras


class PaddingLoss(keras.losses.Loss):
    def __init__(self, from_logits=True, **kwargs):
        super().__init__(**kwargs)
        self._loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=from_logits,
            reduction="none"
        )
        self._from_logits = from_logits

    def __call__(self, y_true, y_pred, sample_weight=None):
        mask = tf.math.logical_not(tf.math.equal(y_true, 0))
        loss = self._loss_fn(y_true, y_pred)
        # convert boolian mask to number mask
        mask = tf.cast(mask, dtype=loss.dtype)
        # Apply mask
        loss = loss * mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "from_logits": self._from_logits}
