from tfchat.data import BlockDataset
from tfchat.data import LineByLineDataset
from tfchat.dialog.data import ContextDataset
from tfchat.eval import perplexity
from tfchat.losses import PaddingLoss
from tfchat.schedules import WarmupLinearDecay
from tfchat.generations import TopKTopPGenerator
from tfchat.utils import import_class
from tfchat.utils import save_model
from tfchat.utils import load_model
from tfchat.utils import set_mixed_precision_policy
from tfchat.utils import set_memory_growth
from tfchat.tokenizers import SentencePieceTokenizer

import tensorflow.keras as keras
import numpy as np


def main(tokenizer_model_dir, load_model_dir=None,
         do_train=True, do_eval=True, do_generate=False,
         # model parameters. The default value is the same as GPT2SmallConfig
         num_layers=12, d_model=768, num_heads=12, d_ff=3072, vocab_size=50257,
         context_size=1024, attention_dropout_rate=0.1, residual_dropout_rate=0.1,
         embedding_dropout_rate=0.1, activation="gelu", kernel_initializer="he_normal",
         epsilon=1e-6,
         # Parameters for do_generate
         max_len=20,
         # Parameters for training
         train_file=None, valid_file=None, save_model_dir=None, batch_size=2, epochs=1,
         model_cls="tfchat.models.PreLNDecoder", config_cls="tfchat.configs.Config",
         dataset_cls="tfchat.data.BlockDataset",
         warmup_steps=0, max_learning_rate=1e-4, patience=1, clipnorm=1.0,
         # Flag to use mixed precision or not
         fp16=False,
         # Set memory growth no to allocate all the memory
         memory_growth=False,
         ):
    # memory_growth should be set before any GPU operations
    # (e.g. set_mixed_precision policy)
    if memory_growth:
        print("Set memory growth")
        set_memory_growth()

    if fp16:
        print("Set mixed precision policy")
        set_mixed_precision_policy()

    # Load tokenizer
    tokenizer = SentencePieceTokenizer.load(model_dir=tokenizer_model_dir)

    # Prepare model
    if load_model_dir:
        model, config = load_model(load_model_dir)
    else:
        config_cls = import_class(config_cls)
        model_cls = import_class(model_cls)
        # Define model config
        config = config_cls(
            num_layers=num_layers, d_model=d_model, num_heads=num_heads,
            d_ff=d_ff, vocab_size=vocab_size, context_size=context_size,
            attention_dropout_rate=attention_dropout_rate,
            residual_dropout_rate=residual_dropout_rate,
            embedding_dropout_rate=embedding_dropout_rate,
            activation=activation, kernel_initializer=kernel_initializer,
            epsilon=epsilon,
        )

        # Override the vocab_size with the number of tokens in tokenizer
        config.vocab_size = len(tokenizer)

        model = model_cls(config)
        model.build(input_shape=(None, config.context_size))

    print("Model config:", config)
    model.summary()

    # Prepare dataset
    if do_train or do_eval:
        assert train_file and valid_file
        dataset_cls = import_class(dataset_cls)

        # Prepare dataset object
        if dataset_cls == BlockDataset:
            dataset = dataset_cls(block_size=config.context_size, batch_size=batch_size)
            def gen(fl):
                return (t.strip("\n") for t in open(fl))
        elif dataset_cls == LineByLineDataset:
            dataset = dataset_cls(max_len=config.context_size, batch_size=batch_size)
            def gen(fl):
                return (t.strip("\n") for t in open(fl))
        elif dataset_cls == ContextDataset:
            dataset = dataset_cls(max_len=config.context_size, batch_size=batch_size, sep_token_id=tokenizer.sep_token_id)
            def gen(fl):
                return (t.strip("\n").split("\t") for t in open(fl))
        else:
            raise Exception(f"{dataset} is not one of BlockDataset, LineByLineDataset")
        print("Dataset class:", dataset_cls)

        train_dataset = dataset.from_text_generator(lambda: gen(train_file), encode_fn=tokenizer.encode, shuffle=True)
        valid_dataset = dataset.from_text_generator(lambda: gen(valid_file), encode_fn=tokenizer.encode, shuffle=False)

        # Train
        if do_train:
            # Prepare model
            print("Calculating num_steps")
            num_steps = sum(1 for _ in train_dataset)
            print("Num steps per epoch:", num_steps)

            schedule = WarmupLinearDecay(max_learning_rate=max_learning_rate, warmup_steps=warmup_steps, training_steps=num_steps*epochs)
            optimizer = keras.optimizers.Adam(schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-8, clipnorm=clipnorm)
            model.compile(loss=PaddingLoss(), optimizer=optimizer)
            history = model.fit(
                train_dataset,
                validation_data=valid_dataset,
                epochs=epochs,
                callbacks=[
                    keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True),
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

    if do_generate:
        # Generate
        gen = TopKTopPGenerator(model=model, max_len=max_len)
        while True:
            try:
                text = input(">>> ")
            except (KeyboardInterrupt, EOFError):
                print("")
                print("Bye")
                break
            ids = tokenizer.encode(text)
            output_ids = gen.generate(np.array([ids], dtype=np.int32))
            tkns = tokenizer.decode(output_ids.tolist())
            print("Input: ", text)
            print("Encode:", ids)
            print("Gen:   ", output_ids)
            print("Decode:", tkns)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
