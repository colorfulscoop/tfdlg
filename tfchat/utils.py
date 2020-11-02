import tensorflow as tf
from pathlib import Path
import json
import inspect


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


def import_class(name):
    """Import class from string.
    Args:
        name (str): class path
    Example:
        >>> model_cls = import_class("tfchat.models.PreLNDecoder")
    """
    components = name.split(".")
    mod = __import__(".".join(components[:-1]), fromlist=[components[-1]])
    return getattr(mod, components[-1])


def save_model(model_dir, model, config):
    model_dir_path = Path(model_dir)

    if not model_dir_path.exists():
        model_dir_path.mkdir()

    # Save config
    config_path = model_dir_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config.dict(), f)

    # Save model
    model_path = model_dir_path / "model.h5"
    model.save_weights(model_path)

    # Save class name
    class_path = model_dir_path / "class.json"
    with open(class_path, "w") as f:
        class_dict = {
            "config": ".".join([inspect.getmodule(config).__name__, config.__class__.__name__]),
            "model": ".".join([inspect.getmodule(model).__name__, model.__class__.__name__]),
        }
        json.dump(class_dict, f)


def load_model(model_dir):
    model_dir_path = Path(model_dir)

    # Load class
    class_path = model_dir_path / "class.json"
    with open(class_path) as f:
        class_dict = json.load(f)
        config_cls = import_class(class_dict["config"])
        model_cls = import_class(class_dict["model"])

    # Load config
    config_path = model_dir_path / "config.json"
    with open(config_path) as f:
        config_dict = json.load(f)
        config = config_cls(**config_dict)

    # Save model
    model = model_cls(config)
    # Call build to avoid the next error
    #       ValueError: Unable to load weights saved in HDF5 format into a subclassed Model
    #       which has not created its variables yet. Call the Model first, then load the weights.
    model.build(input_shape=(None, config.context_size))

    # Load weights
    model_path = model_dir_path / "model.h5"
    model.load_weights(model_path)

    return model, config
