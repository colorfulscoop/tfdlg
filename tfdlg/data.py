import tensorflow as tf
import numpy as np
from typing import List


class BlockDataset:
    @classmethod
    def from_generator(cls, generator, encode_fn, block_size, batch_size=1, buffer_size=10000, shuffle=False):
        """
        Args:
            generator: callable object which returns a generator function.
                The generator functions returns str for each time.
            encode_fn: function which takes str as an input and returns list of integers (i.e. List[int])
        """
        def gen():
            for text in generator():
                for id_ in np.array(encode_fn(text), dtype=np.int32):
                    yield id_

        window_length = block_size + 1
        dataset = tf.data.Dataset.from_generator(gen,
                                                 output_types=tf.int32,
                                                 output_shapes=tf.TensorShape([]))
        dataset = dataset.window(window_length,
                                 shift=block_size,
                                 drop_remainder=True)
        dataset = dataset.flat_map(lambda wd: wd.batch(window_length))
        if shuffle:
            dataset = dataset.shuffle(buffer_size)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.map(lambda x: (x[:, :-1], x[:, 1:]))
        dataset = dataset.prefetch(1)

        return dataset


class LineByLineDataset:
    @classmethod
    def from_generator(cls, generator, encode_fn, max_len, batch_size=1, buffer_size=10000, shuffle=False):
        def gen():
            for text in generator():
                yield np.array(encode_fn(text)[:max_len], dtype=np.int32)

        dataset = tf.data.Dataset.from_generator(gen,
                                                 output_types=tf.int32,
                                                 output_shapes=tf.TensorShape([None]))
        if shuffle:
            dataset = dataset.shuffle(buffer_size)
        # padded_shapes should be greater than the actual length
        dataset = dataset.padded_batch(batch_size=batch_size,
                                       padding_values=0,
                                       padded_shapes=max_len,
                                       drop_remainder=True,
                                       )
        dataset = dataset.map(lambda x: (x[:, :-1], x[:, 1:]))
        dataset = dataset.prefetch(1)

        return dataset
