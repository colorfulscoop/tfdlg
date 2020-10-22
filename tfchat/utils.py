import tensorflow as tf


def set_memory_growth():
    """Enable to allocase necessary GPU memory
    Document is https://www.tensorflow.org/guide/gpu
    """

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # All the GPUs need to set memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Set memory growth to {gpu}")
        except RuntimeError as e:
            # When memory growth is set after initialization,  RuntimeError should be raised
            print(e)
