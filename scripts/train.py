from tfchat.data import BlockDataset
from tfchat.eval import perplexity
from tfchat.losses import PaddingLoss
from tfchat.schedules import WarmupLinearDecay
from tfchat.generations import TopKTopPGenerator
from tfchat.utils import import_class
from tfchat.utils import save_model
from tfchat.utils import load_model

import tensorflow.keras as keras
import numpy as np


def main(save_model_dir=None, load_model_dir=None, do_eval=True,
         batch_size=2, epochs=1,
         model_cls="tfchat.models.PreLNDecoder",
         config_cls="tfchat.configs.Config"):
    # Prepare save and load
    if load_model_dir:
        model, config = load_model(load_model_dir)
    else:
        config_cls = import_class(config_cls)
        model_cls = import_class(model_cls)
        # Define model config
        config = config_cls(num_layers=6, d_model=64, num_heads=1, d_ff=256,
                            vocab_size=100, context_size=64,
                            attention_dropout_rate=0.1, residual_dropout_rate=0.1,
                            embedding_dropout_rate=0.1, epsilon=1e-06)
        model = model_cls(config)
        model.build(input_shape=(None, config.context_size))
        model.summary()

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
