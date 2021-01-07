import tensorflow as tf
import numpy as np
from typing import List


class BlockDataset:
    def __init__(self, block_size, encode_fn):
        self._block_size = block_size
        self._encode_fn = encode_fn

    def from_text_generator(self, generator, batch_size=1, buffer_size=10000, shuffle=False):
        """
        Args:
            generator: callable object which returns a generator function.
                The generator functions returns str for each time.
            encode_fn: function which takes str as an input and returns list of integers (i.e. List[int])
        """
        window_length = self._block_size + 1

        dataset = tf.data.Dataset.from_generator(self._gen_iter_ids(generator),
                                                 output_types=tf.int32,
                                                 output_shapes=tf.TensorShape([]))
        dataset = dataset.window(window_length,
                                 shift=self._block_size,
                                 drop_remainder=True)
        dataset = dataset.flat_map(lambda wd: wd.batch(window_length))
        if shuffle:
            dataset = dataset.shuffle(buffer_size)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.map(lambda x: (x[:, :-1], x[:, 1:]))
        dataset = dataset.prefetch(1)

        return dataset

    def convert_text_to_ids(self, text: str):
        return np.array(self._encode_fn(text), dtype=np.int32)

    def _gen_iter_ids(self, text_generator):
        def gen():
            for text in text_generator():
                for id_ in self.convert_text_to_ids(text):
                    yield id_
        return gen

    def convert_context_to_ids(self, context: List[str], response: str = ""):
        ids = []
        for text in context:
            ids.extend(self.convert_text_to_ids(text))
        if response:
            ids.extend(self.convert_text_to_ids(response))
        return ids

    def _build(self, ids, shuffle=False):
        """This method will be removed in the next release.

        Args:
            ids (Iterator[np.int32]): Iterator which generates
                a np.int32 value at each time
        """
        window_length = self._block_size+1

        # Example of dataset transformation
        # In the example, {} means dataset, [] means batch
        #
        # Notes:
        # - window makes new dataset consisting of several sub-Dataset
        #
        # Example:
        #   init slice -> {1, 2, 3, 4, 5, 6, 7}
        #   window     -> {{1, 2, 3}, {4, 5, 6}}
        #   map batch  -> {{[1, 2, 3]}, {[4, 5, 6]}}
        #   flat       -> {[1, 2, 3], [4, 5, 6]}
        #   batch      -> {[[1, 2, 3], [4, 5, 6]]}
        dataset = tf.data.Dataset.from_tensor_slices(ids)
        dataset = dataset.window(window_length,
                                 shift=self._block_size,
                                 drop_remainder=True)
        dataset = dataset.flat_map(lambda wd: wd.batch(window_length))
        if shuffle:
            dataset = dataset.shuffle(buffer_size)
        dataset = dataset.batch(self._batch_size, drop_remainder=True)
        dataset = dataset.map(lambda x: (x[:, :-1], x[:, 1:]))
        dataset = dataset.prefetch(1)

        return dataset


class LineByLineDataset:
    def __init__(self, max_len, encode_fn, buffer_size=10000):
        self._max_len = max_len
        self._encode_fn = encode_fn

    def from_text_generator(self, generator, batch_size=1, buffer_size=10000, shuffle=False):
        dataset = tf.data.Dataset.from_generator(self._gen_iter_ids(generator),
                                                 output_types=tf.int32,
                                                 output_shapes=tf.TensorShape([None]))
        if shuffle:
            dataset = dataset.shuffle(buffer_size)
        # padded_shapes should be greater than the actual length
        dataset = dataset.padded_batch(batch_size=batch_size,
                                       padding_values=0,
                                       padded_shapes=self._max_len,
                                       drop_remainder=True,
                                       )
        dataset = dataset.map(lambda x: (x[:, :-1], x[:, 1:]))
        dataset = dataset.prefetch(1)

        return dataset

    def convert_text_to_ids(self, text: str):
        return np.array(self._encode_fn(text)[:self._max_len], dtype=np.int32)

    def _gen_iter_ids(self, text_generator):
        def gen():
            for text in text_generator():
                yield self.convert_text_to_ids(text)
        return gen

    def convert_context_to_ids(self, context: List[str], response: str = ""):
        ids = []
        for text in context:
            ids.extend(self._encode_fn(text))
        if response:
            ids.extend(self._encode_fn(response))
        ids = np.array(ids[:self._max_len], dtype=np.int32)
        return ids

    def _build(self, ids, shuffle=False):
        """This method will be removed in the next release.

        Args:
            ids (Iterator[List[np.int32]]): Iterator which generates
                a List[lnp.int32] value at each time
        """
        dataset = tf.data.Dataset.from_generator(lambda: ids, tf.int32)
        if shuffle:
            dataset = dataset.shuffle(buffer_size)
        dataset = dataset.padded_batch(batch_size=self._batch_size,
                                       padding_values=0,
                                       padded_shapes=self._max_len,
                                       drop_remainder=True,
                                       )
        dataset = dataset.map(lambda x: (x[:, :-1], x[:, 1:]))
        dataset = dataset.prefetch(1)
