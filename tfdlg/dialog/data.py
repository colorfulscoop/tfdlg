import tensorflow as tf


class DialogDataset:
    @classmethod
    def from_generator(cls, generator, context_encode_fn, max_len, batch_size=1, buffer_size=10000, shuffle=False):
        def gen():
            for texts in generator():
                yield context_encode_fn(texts)[:max_len]

        dataset = tf.data.Dataset.from_generator(gen,
                                                 output_types=tf.int32,
                                                 output_shapes=tf.TensorShape([None]))

        if shuffle:
            dataset = dataset.shuffle(buffer_size)

        encoded_ds = dataset
        # padded_shapes should be greater than the actual length
        dataset = encoded_ds.padded_batch(batch_size=batch_size,
                                          padding_values=0,
                                          padded_shapes=max_len,
                                          drop_remainder=True,
                                          )
        dataset = dataset.map(lambda x: (x[:, :-1], x[:, 1:]))
        dataset = dataset.prefetch(1)

        return dataset


class DialogClsDataset:
    """
    max_len より長いsequenceは、たとえ終端記号でなくても最後のトークンの位置を cls_id の位置とする。
    """
    @classmethod
    def from_generator(cls, generator, encode_fn, max_len, sep_token_id,
                       batch_size=1, buffer_size=10000, shuffle=False, seed=None):
        context_gen, response_gen = _build_generator(generator, encode_fn, sep_token_id=sep_token_id)

        context_ds = tf.data.Dataset.from_generator(
            context_gen,
            output_types=tf.int32
        )
        response_ds = tf.data.Dataset.from_generator(
            response_gen,
            output_types=tf.int32
        )
        response_shuf_ds = response_ds.shuffle(buffer_size=buffer_size, seed=seed)

        true_ds = tf.data.Dataset.zip((context_ds, response_ds))
        true_ds = true_ds.map(_compose_label(label=1, max_len=max_len+1, pad_value=0))

        false_ds = tf.data.Dataset.zip((context_ds, response_shuf_ds))
        false_ds = false_ds.map(_compose_label(label=0, max_len=max_len+1, pad_value=0))

        ds = tf.data.Dataset.from_tensor_slices([true_ds, false_ds]).interleave(lambda x: x)

        if shuffle:
            ds = ds.shuffle(buffer_size=buffer_size, seed=seed)

        ds = ds.batch(batch_size, drop_remainder=True)  # returns batches of (ids, label)

        # If not set the shape, fit raises the following error
        #
        #   ValueError: Cannot take the length of shape with unknown rank.
        #
        # because LM input tensor shape is set to unknown
        # (<tf.Tensor 'IteratorGetNext:0' shape=<unknown> dtype=int32>, <tf.Tensor 'ExpandDims:0' shape=(4, 1) dtype=int32>)
        #
        # Check details in https://github.com/tensorflow/tensorflow/issues/37193
        def set_shape(ids, target_ids, cls_id, label):
            ids.set_shape([batch_size, max_len+1])
            target_ids.set_shape([batch_size, max_len+1])
            return (ids, target_ids, cls_id, label)

        ds = ds.map(set_shape)

        # returns batches of ((ids, cls_ids), (ids, label))
        ds = ds.map(lambda ids, target_ids, cls_id, label: ((ids[:, :-1], cls_id), (target_ids[:, 1:], label)))
        ds = ds.prefetch(1)

        return ds


def _build_generator(generator, encode_fn, sep_token_id):
    """Helper function for DialogClsDataset"""
    def get_context(gen):
        for item in gen():
            context = [sep_token_id]
            for l in item[:-1]:
                context += encode_fn(l)
                context += [sep_token_id]
            yield context

    def get_response(gen):
        for item in gen():
            yield encode_fn(item[-1]) + [sep_token_id]

    return lambda: get_context(generator), lambda: get_response(generator)


def _compose_label(label, max_len, pad_value):
    """Helper function for DialogClsDataset"""
    def tmp(context, response):
        ids = tf.concat([context, response], axis=0)
        ids = ids[:max_len]
        # Finally we use [:-1] for input. Therefore the max_len is set to +1 than usual. maximum position is set to max_len-2
        cls_id = tf.math.minimum(len(ids) - 1, max_len-2)
        if len(ids) < max_len:
            ids = tf.concat([ids, tf.zeros([max_len - len(ids)], dtype=tf.int32)], axis=0)

        # when label == 0, target should be ignored
        if label == 1:
            target_ids = ids
        elif label == 0:
            target_ids = tf.zeros([len(ids)], dtype=tf.int32)
        else:
            raise Exception(f"{label} should be 1 or 0")
        return (ids, target_ids, cls_id, label)
    return tmp
