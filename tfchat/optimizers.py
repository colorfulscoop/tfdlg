import tensorflow as tf


class TransformerScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    """TransformerScheduler implements the scheduler used in "Attention is All You Need"""
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()

        self._d_model = d_model
        self._warmup_steps = warmup_steps

    def __call__(self, step):
        fst = tf.math.rsqrt(step)
        snd = step * (self._warmup_steps ** -1.5)

        out = tf.math.rsqrt(tf.cast(self._d_model, tf.float32)) * tf.math.minimum(fst, snd)

        return out

    def get_config(self):
        """get_config is for supporing save/load hyper parameters."""

        # Do not call `super().get_config()` to get parent parameters because
        # it will raise NotImplementedError
        # https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/optimizer_v2/learning_rate_schedule.py#L47
        return {
            "d_model": self._d_model,
            "warmup_steps": self._warmup_steps
        }
