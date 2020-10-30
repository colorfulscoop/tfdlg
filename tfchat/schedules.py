import tensorflow as tf


class TransformerSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """TransformerScheduler implements the scheduler used in [Vaswani+, 2017]"""
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


class LinearSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, max_learning_rate, warmup_steps, training_steps):
        self._max_learning_rate = max_learning_rate
        self._warmup_steps = warmup_steps
        self._training_steps = training_steps

    def __call__(self, step):
        fst = step / tf.math.maximum(1.0, self._warmup_steps)
        snd = (self._training_steps - step) / tf.math.maximum(1.0, self._training_steps - self._warmup_steps)
        return self._max_learning_rate * tf.math.maximum(tf.math.minimum(fst, snd), 0)

    def get_config(self):
        # Do not call `super().get_config()` to get parent parameters because
        # it will raise NotImplementedError
        # https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/optimizer_v2/learning_rate_schedule.py#L47
        return {
            "max_learning_rate": self._max_learning_rate,
            "warmup_steps": self._warmup_steps,
            "training_steps": self._training_steps,
        }
