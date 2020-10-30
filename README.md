# TfChat

**TfChat** is a Python library for transformer-based language model with TensorFlow.

TfChat adopts simple and transparent model implementation to enable users to customize models for their research and interests.
You can find the model implementation in [tfchat/models.py](tfchat/models.py).

## Installation

Prepare your environment with Python >= 3.6 first. Then run `pip` to install this package from GitHub.

```sh
$ pip install git+https://github.com/noriyukipy/tfchat
```

You can run tests with [pytest](https://docs.pytest.org/en/stable/) to make sure your installtion succeeds.

```sh
$ pip install pytest==6.1.1
$ pytest tests/
```

## Usage

Please take a look at [Jupyter Notebook - docs/usage.ipynb](docs/usage.ipynb).

## Example

```py
from tfchat.data import BlockDataset
from tfchat.metrics import perplexity
from tfchat.losses import PaddingLoss
from tfchat.optimizers import TransformerScheduler

# Prepare config

# Prepare dataset
train_ids = [...]  # Prepare token ids for training data
valid_ids = [...]  # Prepare token ids for validation data
dataset = BlockDataset(block_size=config.context_size, batch_size=2)

train_dataset = dataset.build(train_ids, shuffle=True)
valid_dataset = dataset.build(valid_ids, shuffle=False)

# Prepare model
scheduler = TransformerScheduler(d_model=config.d_model, warmup_steps=1000)
optimizer = keras.optimizers.Adam(scheduler,
                                  beta_1=0.9,
                                  beta_2=0.999,
                                  epsilon=1e-8,
                                  clipnorm=1.0)
model.compile(loss=PaddingLoss(), optimizer=optimizer)
model.build(input_shape=(None, config.context_size))
model.summary()

# Train
history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=10,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=1, restore_best_weights=True),
        # If you want to save chekcpoints, remove the next comment out
        #keras.callbacks.ModelCheckpoint("keras_model/", save_best_only=True)
    ]
)

# Evaluate
perplexity(model, valid_dataset)
```

## Models

### tfchat.models.PostLNDecoder

It is the decoder side implementation of [Vaswani+, 2017] .

Difference from [VasWani+, 2017] is

- PostLNDecoder does not share the embedding parameter with the last layer before Softmax.
- Weight is initialized by Grolo's uniform distribution except for layers which uses ReLU. For those which uses the ReLU activation function, He's initialization is used. (The weight initialization method is not mentioned in the paper.)

### tfchat.models.PreLNDecoder

PreLNDecoder replaces Post Layer Normalization architecture of PostLNDecoder with Pre Layer Normalization architecture explained in [Xiong+, 2020].

This architecture is related to GPT-2 introduced in [Radford+, 2019].
Difference against GPT2.

- Not using GeLU

## Reference

* [Vaswani+, 2017] *Attention Is All You Need* by Ashish Vaswani et al. (https://arxiv.org/abs/1706.03762v5)
* [Radford+, 2018] *Improving Language Understanding by Generative Pre-Training* by Alec Radford et al. (https://openai.com/blog/language-unsupervised/)
* [Radford+, 2019] *Language Models are Unsupervised Multitask Learners* by Alec Radford et al. (https://openai.com/blog/better-language-models/)
* [Holtzman+, 2019] *The Curious Case of Neural Text Degeneration* by Ari Holtzman et al. (https://arxiv.org/abs/1904.09751)
* [Xiong+, 2020] *On Layer Normalization in the Transformer Architecture* by Ruibin Xiong et al. (https://arxiv.org/abs/2002.04745)