from tfchat.configs import Config
from tfchat.data import BlockDataset
from tfchat.metrics import perplexity
from tfchat.losses import PaddingLoss
from tfchat.schedules import WarmupLinearDecay
from tfchat.generations import TopKTopPGenerator
from tfchat.models import PreLNDecoder

import tensorflow.keras as keras
import numpy as np

from pathlib import Path
import json


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


def load_model(model_dir, model_cls, config_cls):
    model_dir_path = Path(model_dir)

    # Save config
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


def main(save_model_dir=None, load_model_dir=None, do_eval=True):
    # Prepare save and load
    if load_model_dir:
        model, config = load_model(load_model_dir, PreLNDecoder, Config)
    else:
        # Define model config
        config = Config(num_layers=6, d_model=64, num_heads=1, d_ff=256,
                        vocab_size=100, context_size=64,
                        attention_dropout_rate=0.1, residual_dropout_rate=0.1,
                        embedding_dropout_rate=0.1, epsilon=1e-06)
        model = PreLNDecoder(config)
        model.build(input_shape=(None, config.context_size))
        model.summary()

    # Define training parameters
    batch_size = 2
    epochs = 1

    # Prepare dataset
    train_ids = np.tile(np.arange(10, dtype=np.int32), 1000)  # Prepare token ids for training data
    valid_ids = np.tile(np.arange(10, dtype=np.int32), 100)   # Prepare token ids for validation data

    dataset = BlockDataset(block_size=config.context_size, batch_size=batch_size)
    train_dataset = dataset.build(train_ids, shuffle=True)
    valid_dataset = dataset.build(valid_ids, shuffle=False)

    # Prepare model
    num_steps = len([_ for _ in train_dataset])
    schedule = WarmupLinearDecay(max_learning_rate=1e-4, warmup_steps=0, training_steps=num_steps*epochs)
    optimizer = keras.optimizers.Adam(schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-8, clipnorm=1.0)
    model.compile(loss=PaddingLoss(), optimizer=optimizer)

    # Train
    history = model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=epochs,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=1, restore_best_weights=True),
            # If you want to save chekcpoints, remove the next comment out
            #keras.callbacks.ModelCheckpoint("keras_model/", save_best_only=True)
        ]
    )
    if save_model_dir:
        save_model(save_model_dir, model, config)

    # Evaluate
    if do_eval:
        ppl = perplexity(model, valid_dataset)
        print("Validation PPL:", ppl)

    # Generate
    gen = TopKTopPGenerator(model=model, max_len=3)
    inputs = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)
    print(gen.generate(inputs))


if __name__ == "__main__":
    import fire

    fire.Fire(main)
