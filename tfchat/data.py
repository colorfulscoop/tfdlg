import tensorflow as tf


class BlockDataset:
    def __init__(self, block_size, batch_size, buffer_size=10000):
        self._block_size = block_size
        self._batch_size = batch_size
        self._buffer_size = buffer_size

    def build(self, ids, shuffle=False):
        """
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
        dataset = tf.data.Dataset.from_generator(lambda: ids, tf.int32)
        dataset = dataset.window(window_length,
                                 shift=self._block_size,
                                 drop_remainder=True)
        dataset = dataset.flat_map(lambda wd: wd.batch(window_length))
        if shuffle:
            dataset = dataset.shuffle(self._buffer_size)
        dataset = dataset.batch(self._batch_size, drop_remainder=True)
        dataset = dataset.map(lambda x: (x[:, :-1], x[:, 1:]))
        dataset = dataset.prefetch(1)

        return dataset


class LineByLineDataset:
    def __init__(self, max_len, batch_size, buffer_size=10000):
        self._max_len = max_len
        self._batch_size = batch_size
        self._buffer_size = buffer_size

    def build(self, ids, shuffle=False):
        """
        Args:
            ids (Iterator[List[np.int32]]): Iterator which generates
                a List[lnp.int32] value at each time
        """
        dataset = tf.data.Dataset.from_generator(lambda: ids, tf.int32)
        if shuffle:
            dataset = dataset.shuffle(self._buffer_size)
        dataset = dataset.padded_batch(batch_size=self._batch_size,
                                       padding_values=0,
                                       padded_shapes=self._max_len,
                                       drop_remainder=True,
                                       )
        dataset = dataset.map(lambda x: (x[:, :-1], x[:, 1:]))
        dataset = dataset.prefetch(1)

        return dataset
