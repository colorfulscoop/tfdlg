import tensorflow as tf
import numpy as np
from typing import List


class DialogDataset:
    def __init__(self, max_len, encode_fn, sep_token_id):
        self._max_len = max_len
        self._encode_fn = encode_fn
        self._sep_token_id = sep_token_id

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

    def convert_text_to_ids(self, text: str, sep_token="|"):
        ids = []
        text_segments = text.split(sep_token)
        for i, txt in enumerate(text_segments):
            if len(txt) > 0:
                ids.extend(self._encode_fn(txt))
            if i < len(text_segments) - 1:
                ids.append(self._sep_token_id)
        return np.array(ids[:self._max_len], dtype=np.int32)

    def convert_context_to_ids(self, context: List[str], response: str = ""):
        ids = []
        for i, txt in enumerate(context):
            ids += [self._sep_token_id]
            ids += self._encode_fn(txt)
        ids += [self._sep_token_id]
        if response:
            ids += self._encode_fn(response)
        return np.array(ids[:self._max_len], dtype=np.int32)

    def _gen_iter_ids(self, text_generator):
        def gen():
            for texts in text_generator():
                yield self.convert_context_to_ids(context=texts)
        return gen
