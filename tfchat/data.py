import tensorflow as tf


class BlockDataLoader:
    def __init__(self, block_size, batch_size, shuffle, buffer_size=10000):
        self._block_size = block_size
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._buffer_size = buffer_size

    def load(self, ids):
        """
        Args:
            text (List[int]):
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
        if self._shuffle:
            dataset = dataset.shuffle(self._buffer_size)
        dataset = dataset.batch(self._batch_size, drop_remainder=True)
        dataset = dataset.map(lambda x: (x[:, :-1], x[:, 1:]))
        dataset = dataset.prefetch(1)

        return dataset
