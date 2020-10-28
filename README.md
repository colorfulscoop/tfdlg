# TfChat

**TfChat** is a Python library for transformer-based language model with TensorFlow.

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

### Models

To use models provided in TfChat, you first need to create a config object which handles hyper-parameters of the model.

```py
>>> from tfchat.configs import GPT2SmallConfig
>>> config = GPT2SmallConfig()
>>> config
GPT2SmallConfig(num_layers=12, d_model=768, num_heads=12, d_ff=3072, vocab_size=50257, context_size=1024, attention_dropout_rate=0.1, residual_dropout_rate=0.1, embedding_dropout_rate=0.1, epsilon=1e-06)
```

Then you can create the model with the config object.
TfChat currently provides two types of models which use decoder-side Transformer called Pre-LN and Post-LN Transformer, which are explained in [Xiong+, 2020].
Here is an example of using Pre-LN Transformer.

```py
>>> from tfchat.models import PreLNDecoder
>>> model = PreLNDecoder(config)
```

The models can be used in the usual manner of TensorFlow Keras API.

```py
>>> import tensorflow.keras as keras
>>> model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=keras.optimizers.Adam())
>>> model.build(input_shape=(None, config.context_size))
>>> model.summary()
Model: "pre_ln_decoder_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
decoder_1 (Decoder)          multiple                  162299473
=================================================================
Total params: 162,299,473
Trainable params: 162,299,473
Non-trainable params: 0
_________________________________________________________________
```

To train the models, you need to provide the data. The data consists of several samples, and each sample is a array of ids with `(config.context_size, )` shape.

For example, assume that context size = 3.
When a sentence "Nice to meet you" can be encoded as `[3, 4, 5, 6]`. Then the sample for input is `[3, 4, 5]` and the target label is `[4, 5, 6]`.

In the following example, `contexts` is a dummy data for four encoded sentences. You can substitute your own data there.

After preparing data, you can use `fit` method to train your model.

```py
>>> import numpy as np
>>> contexts = np.ones((4, config.context_size+1), dtype=np.int32)  # This is dummy data. Replace with your data
>>> train_X = contexts[:,:-1]
>>> train_y = contexts[:,1:]
>>> history = model.fit(train_X, train_y, batch_size=2)
2/2 [==============================] - 7s 4s/step - loss: 6.4654
```

### Utilities

TfChat provides not only models but also utilities to help users train and evaluate models as well as generating sentences.

#### Dataset

Preparing input data usually takes time. For typical input data, TfChat provides useful classes under `tfchat.data`.

One of them is `BlockDataset`.
This class is for creating dataset to train a language model with large documents (e.g. Wikipedia)
After you prepare the ids of encoded documents, `BlockDataset` split the data into `block_size` and batch them.

For example, you have ids as `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`. For applying `BlockDataset.build` method with `block_size=5` and `batch_size=2`,
then the output is [tensorflow.data.Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)
which consists of tuple `([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])`.
The first item is the input, and the second one is the target label.

Let's take a look at this example.

```py
>>> from tfchat.data import BlockDataset
>>> dataset = BlockDataset(block_size=5, batch_size=2)
>>> ids = list(range(11))
>>> ids
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
>>> for item in dataset.build(ids, shuffle=False):
...     print(item)
...
(<tf.Tensor: shape=(2, 5), dtype=int32, numpy=
array([[0, 1, 2, 3, 4],
       [5, 6, 7, 8, 9]], dtype=int32)>, <tf.Tensor: shape=(2, 5), dtype=int32, numpy=
array([[ 1,  2,  3,  4,  5],
       [ 6,  7,  8,  9, 10]], dtype=int32)>)
```

You can set `shuffle` argument for `build` method to decide shuffle the samples or not.
This is useful to build validation data without shuffling as follow.

```py
# Prepare config

# Prepare dataset
train_ids = [...]  # Prepare token ids for training data
valid_ids = [...]  # Prepare token ids for validation data
dataset = BlockDataset(block_size=config.context_size, batch_size=2)

train_dataset = dataset.build(train_ids, shuffle=True)
valid_dataset = dataset.build(valid_ids, shuffle=False)
```

You can use dataset in the `fit` method in the usual manner.

```py
history = model.fit(train_dataset, validation_data=valid_dataset)
```

#### Loss

To ignore padding value `0` in the loss, `PaddingLoss` can be used.

```py
from tfchat.losses import PaddingLoss
model.compile(loss=PaddingLoss(), optimizer=keras.optimizers.Adam())
```

#### Schedule

`TransformerScheduler` is a learning rate scheduler introduced in [Vaswani+, 2017].
The scheduler can be used with optimizer.

```py
from tfchat.optimizers import TransformerScheduler
scheduler = TransformerScheduler(d_model=config.d_model, warmup_steps=1000)
optimizer = keras.optimizers.Adam(scheduler,
                                  beta_1=0.9,
                                  beta_2=0.999,
                                  epsilon=1e-8,
                                  clipnorm=1.0)
```

#### Evaluation

```py
from tfchat.metrics import perplexity
perplexity(model, valid_dataset)
```

#### Generation

`TopKTopPGenerator` generates outputs from inputs with Top-p (Nucleus) Sampling introduced in [Holtzman+, 2019].

```py
>>> from tfchat.generations import TopKTopPGenerator
>>> gen = TopKTopPGenerator(model=model, max_len=3)
>>> inputs = np.array([[1, 3, 2, 4, 1]], dtype=np.int32)
>>> gen.generate(inputs)
array([[1, 3, 2, 4, 1, 1, 1, 1]])
```

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

## Models and Utililies

### tfchat.models

#### PostLNDecoder

It is the decoder side implementation of [Vaswani+, 2017] .

Difference is
- PostLNDecoder does not share the embedding parameter with the last layer before Softmax.
- Weight is initialized by Grolo's uniform distribution except for layers which uses ReLU. For those which uses the ReLU activation function, He's initialization is used. (The weight initialization method is not mentioned in the paper.)

#### PreLNDecoder

PreLNDecoder uses Pre Layer Normalization architecture instad of Ppost Layer Normalization. This architecture is introduced in [Xiong+, 2020] .

Difference against GPT2
- Not using GeLU

### tfchat.losses

#### PaddingLoss


## Reference

* [Vaswani+, 2017] *Attention Is All You Need* by Ashish Vaswani et al. (https://arxiv.org/abs/1706.03762v5)
* [Radford+, 2018] *Improving Language Understanding by Generative Pre-Training* by Alec Radford et al. (https://openai.com/blog/language-unsupervised/)
* [Radford+, 2019] *Language Models are Unsupervised Multitask Learners* by Alec Radford et al. (https://openai.com/blog/better-language-models/)
* [Holtzman+, 2019] *The Curious Case of Neural Text Degeneration* by Ari Holtzman et al. (https://arxiv.org/abs/1904.09751)
* [Xiong+, 2020] *On Layer Normalization in the Transformer Architecture* by Ruibin Xiong et al. (https://arxiv.org/abs/2002.04745)