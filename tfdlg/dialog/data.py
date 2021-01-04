import tensorflow as tf
import numpy as np


class ContextDataset:
    def __init__(self, max_len, batch_size, sep_token_id, buffer_size=10000):
        self._max_len = max_len
        self._batch_size = batch_size
        self._sep_token_id = sep_token_id
        self._buffer_size = buffer_size

    def _gen_iter_ids(self, text_generator, encode_fn):
        def gen():
            for texts in text_generator():
                ids = []
                for i, txt in enumerate(texts):
                    ids += [self._sep_token_id]
                    ids += encode_fn(txt)
                ids += [self._sep_token_id]
                yield np.array(ids[:self._max_len], dtype=np.int32)
        return gen

    def from_text_generator(self, generator, encode_fn, shuffle=False):
        dataset = tf.data.Dataset.from_generator(self._gen_iter_ids(generator, encode_fn),
                                                 output_types=tf.int32,
                                                 output_shapes=tf.TensorShape([None]))
        if shuffle:
            dataset = dataset.shuffle(self._buffer_size)
        # padded_shapes should be greater than the actual length
        dataset = dataset.padded_batch(batch_size=self._batch_size,
                                       padding_values=0,
                                       padded_shapes=self._max_len,
                                       drop_remainder=True,
                                       )
        dataset = dataset.map(lambda x: (x[:, :-1], x[:, 1:]))
        dataset = dataset.prefetch(1)

        return dataset
