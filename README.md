# TfChat

**TfChat** is a Python library for transformer-based language model with TensorFlow.

Features:

* **Simple model:** TfChat adopts simple and easy-to-understand model implementation to enable users to customize models for their research and interests. You can find the model implementation in [tfchat/models.py](tfchat/models.py).
* **Useful utilities:** TfChat provides several useful utilities to prepare dataset with `tf.data.Dataset` format, learning rate schedules, loss function with padding consideration and perplexity evaluation metrics. You can find them in [examples/usage.ipynb](examples/usage.ipynb)

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


The next code shows the overview of how to use TfChat. You can find the result of running it in [examples/overview.ipynb](examples/overview.ipynb).

```py
from tfchat.configs import Config
from tfchat.data import BlockDataset
from tfchat.metrics import perplexity
from tfchat.losses import PaddingLoss
from tfchat.schedules import WarmupLinearDecay
from tfchat.generations import TopKTopPGenerator
from tfchat.models import PreLNDecoder

import tensorflow.keras as keras
import numpy as np


# Define model config
config = Config(num_layers=6, d_model=64, num_heads=1, d_ff=256, vocab_size=100,
                context_size=64, attention_dropout_rate=0.1, residual_dropout_rate=0.1,
                embedding_dropout_rate=0.1, epsilon=1e-06)

# You can use predefined config as follows instead of defining config by yourself
#
# from tfchat.configs import GPT2SmallConfig
# config = GPT2SmallConfig()


# Define training parameters
batch_size = 2
epochs = 10

# Prepare dataset
train_ids = np.tile(np.arange(10, dtype=np.int32), 1000)  # Prepare token ids for training data
valid_ids = np.tile(np.arange(10, dtype=np.int32), 100)   # Prepare token ids for validation data

dataset = BlockDataset(block_size=config.context_size, batch_size=batch_size)
train_dataset = dataset.build(train_ids, shuffle=True)
valid_dataset = dataset.build(valid_ids, shuffle=False)

# Prepare model
num_steps = len([_ for _ in train_dataset])
schedule = WarmupLinearDecay(max_learning_rate=1e-3, warmup_steps=0, training_steps=num_steps*epochs)
optimizer = keras.optimizers.Adam(schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-8, clipnorm=1.0)

model = PreLNDecoder(config)
model.compile(loss=PaddingLoss(), optimizer=optimizer)
model.build(input_shape=(None, config.context_size))
model.summary()

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

# Evaluate
ppl = perplexity(model, valid_dataset)
print("Validation PPL:", ppl)

# Generate
gen = TopKTopPGenerator(model=model, max_len=3)
inputs = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)
gen.generate(inputs)
```

Please take a look at [examples/usage.ipynb](examples/usage.ipynb) to get more details of each classes and functions.


## Model Description

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