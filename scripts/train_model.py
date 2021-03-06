from tfdlg.eval import perplexity
from tfdlg.schedules import WarmupLinearDecay
from tfdlg.generations import TopKTopPGenerator
from tfdlg.utils import import_class
from tfdlg.utils import save_model
from tfdlg.utils import load_model
from tfdlg.utils import set_mixed_precision_policy
from tfdlg.utils import set_memory_growth

import tensorflow.keras as keras
import numpy as np


def main(tokenizer_model_dir, task_cls="task.LMTask",
         load_model_dir=None,
         do_train=True, do_eval=True, do_generate=False,
         # model parameters. The default value is the same as GPT2SmallConfig
         config_cls="tfdlg.configs.Config",
         num_layers=12, d_model=768, num_heads=12, d_ff=3072, vocab_size=50257,
         context_size=1024, attention_dropout_rate=0.1,
         residual_dropout_rate=0.1,
         embedding_dropout_rate=0.1, activation="gelu",
         kernel_initializer="he_normal", epsilon=1e-6,
         # Parameters for training
         train_file=None, valid_file=None, save_model_dir=None,
         batch_size=2, epochs=1,
         warmup_steps=0, max_learning_rate=1e-4, patience=1, clipnorm=1.0,
         # Flag to use mixed precision or not
         fp16=False,
         # Set memory growth no to allocate all the memory
         memory_growth=False,
         # Parameters for do_generate
         max_len=20,
         sep_token="|",
         # Tensorboard setting
         tensorboard_dir=None,
         tensorboard_update_freq=100,
         ):
    # memory_growth should be set before any GPU operations
    # (e.g. set_mixed_precision policy)
    if memory_growth:
        print("Set memory growth")
        set_memory_growth()

    if fp16:
        print("Set mixed precision policy")
        set_mixed_precision_policy()

    # Prepare config, model and task
    if load_model_dir:
        model, config = load_model(load_model_dir)
        task = import_class(task_cls)(config=config)
        tokenizer = task.prepare_tokenizer(model_dir=tokenizer_model_dir)
    else:
        # Define model config
        config = import_class(config_cls)(
            num_layers=num_layers, d_model=d_model, num_heads=num_heads,
            d_ff=d_ff, vocab_size=vocab_size, context_size=context_size,
            attention_dropout_rate=attention_dropout_rate,
            residual_dropout_rate=residual_dropout_rate,
            embedding_dropout_rate=embedding_dropout_rate,
            activation=activation, kernel_initializer=kernel_initializer,
            epsilon=epsilon,
        )
        # Define task
        task = import_class(task_cls)(config=config)
        tokenizer = task.prepare_tokenizer(model_dir=tokenizer_model_dir)

        # Override the vocab_size with the number of tokens in tokenizer
        config.vocab_size = len(tokenizer)

        # Prepare model
        model = task.model_cls(config)
        model.build(input_shape=(None, config.context_size))

    print("Model config:", config)
    model.summary()

    if do_train or do_eval:
        valid_dataset = task.prepare_dataset(
            filename=valid_file,
            encode_fn=tokenizer.encode,
            batch_size=batch_size,
            shuffle=False,
            buffer_size=10000,
        )

        # Train
        if do_train:
            train_dataset = task.prepare_dataset(
                filename=train_file,
                encode_fn=tokenizer.encode,
                batch_size=batch_size,
                shuffle=True,
                buffer_size=10000,
            )

            # Prepare model
            print("Calculating num_steps")
            num_steps = sum(1 for _ in train_dataset)
            print("Num steps per epoch:", num_steps)

            schedule = WarmupLinearDecay(
                max_learning_rate=max_learning_rate,
                warmup_steps=warmup_steps,
                training_steps=num_steps*epochs
            )
            optimizer = keras.optimizers.Adam(
                schedule,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-8,
                clipnorm=clipnorm
            )
            model.compile(loss=task.loss_fn, optimizer=optimizer)

            # Define callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(
                    patience=patience,
                    restore_best_weights=True
                ),
                # If you want to save chekcpoints, remove the next comment out
                #keras.callbacks.ModelCheckpoint("keras_model/", save_best_only=True)
            ]
            if tensorboard_dir:
                callbacks.append(keras.callbacks.TensorBoard(
                    log_dir=tensorboard_dir,
                    update_freq=tensorboard_update_freq,
                    profile_batch=0)
                )

            history = model.fit(
                train_dataset,
                validation_data=valid_dataset,
                epochs=epochs,
                callbacks=callbacks,
            )
            if save_model_dir:
                save_model(save_model_dir, model, config)

        # Evaluate
        if do_eval:
            ppl = perplexity(model, valid_dataset)
            print("Validation PPL:", ppl)

    if do_generate:
        # Generate
        generator = TopKTopPGenerator(model=model, max_len=max_len)
        while True:
            try:
                text = input(">>> ")
            except (KeyboardInterrupt, EOFError):
                print("")
                print("Bye")
                break
            # Tokenize input text.
            # The input text is splitted at the `sep_token`.
            # Then splitted texts are tokenized and convertd to ids per wise.
            # Finally the ids are concatenated with `sep_token_id`
            ids = dataset.convert_text_to_ids(text=text)
            output_ids = generator.generate(np.array([ids], dtype=np.int32))
            output_ids = output_ids[0][len(ids):]
            tkns = tokenizer.decode(output_ids.tolist())
            print("Input: ", text)
            print("Encode:", ids)
            print("Gen:   ", output_ids)
            print("Decode:", tkns)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
