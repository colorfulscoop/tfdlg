import tensorflow as tf
from tfchat.schedules import TransformerSchedule
from tfchat.schedules import WarmupLinearDecay
import numpy as np


def test_scheduler():
    scheduler = TransformerSchedule(d_model=1024)
    out = scheduler(tf.range(50000, dtype=tf.float32))

    np.testing.assert_almost_equal(np.max(out), 0.00049410586)
    np.testing.assert_almost_equal(np.min(out), 0)


def test_warmup_linear_decay():
    schedule = WarmupLinearDecay(max_learning_rate=1e-3, warmup_steps=1000, training_steps=10000)
    out = schedule(tf.range(20000, dtype=tf.float32))

    np.testing.assert_almost_equal(np.max(out), 1e-3)
    np.testing.assert_almost_equal(np.min(out), 0)
