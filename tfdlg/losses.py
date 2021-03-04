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

    # __call__ needs to be wrapped by tf.function because it uses
    # `if` statement. If not wrippared, tf.cond, (or tf.function for more convenience)
    # should be used.
    # Find more details here https://github.com/tensorflow/tensorflow/issues/40916#issuecomment-660305407
    @tf.function
    def __call__(self, y_true, y_pred, sample_weight=None):
        mask = tf.math.logical_not(tf.math.equal(y_true, 0))
        loss = self._loss_fn(y_true, y_pred)
        # convert boolian mask to number mask
        mask = tf.cast(mask, dtype=loss.dtype)
        # Apply mask
        loss = loss * mask

        # If all the target value is masked, return 0.
        mask_sum = tf.reduce_sum(mask)
        if mask_sum == 0:
            return tf.reduce_sum(loss)  # should be equal to 0
        else:
            return tf.reduce_sum(loss) / mask_sum

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "from_logits": self._from_logits}
