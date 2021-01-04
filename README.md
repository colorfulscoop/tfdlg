# tfDlg

**tfDlg** is a Python library for transformer-based language models and dialog models with TensorFlow.

Features:

* **Simple model:** tfDlg adopts simple and easy-to-understand model implementation to enable users to customize models for their research and interests. You can find the model implementation in [tfdlg/models.py](tfdlg/models.py). You can utilize these models in the usual manner of tf.keras (e.g. you can call compile and build method for them).
* **Useful utilities:** tfDlg provides several useful utilities. For example,
  * [tfdlg.data](tfdlg/data.py) provides dataset builders to input them to your model. They generate tf.data.Dataset object
  * [tfdlg.schedules](tfdlg/schedules.py) provides learning rate schedules to consider warmup steps as well as linear decay.
  * [tfdlg.losses](tfdlg/losses.py) provides loss function which considers padding.
  * [tfdlg.eval](tfdlg/eval.py) provides function to calculate perplexity.
  * [tfdlg.tokenizers](tfdlg/tokenizers.py) provides SentencePiece tokenizer.
  * [tfdlg.generations](tfdlg/generations.py) provides top-k top-p generator .
* **Utilities for dialog modeling:** Useful utilities for dialog modeling are provided under the `tfdlg.dialog` namespace.
  * [tfdlg.dialog.data](tfdlg/dialog/data.py) provides a dataset builder which considers context of the dialog.

## Installation

Prepare your environment with Python >= 3.8 first. Then run `pip` to install this package from GitHub.

```sh
$ pip install git+https://github.com/noriyukipy/tfdlg
```

You can run tests with [pytest](https://docs.pytest.org/en/stable/) to make sure your installtion succeeds.

```sh
$ pip install pytest==6.1.1
$ pytest tests/
```

## Usage


The next code shows the overview of how to use tfDlg. You can find the result of running it in [examples/overview.ipynb](examples/overview.ipynb).

```py
from tfdlg.configs import Config
from tfdlg.data import BlockDataset
from tfdlg.metrics import perplexity
from tfdlg.losses import PaddingLoss
from tfdlg.schedules import WarmupLinearDecay
from tfdlg.generations import TopKTopPGenerator
from tfdlg.models import PreLNDecoder

import tensorflow.keras as keras
import numpy as np


# Define model config
config = Config(num_layers=6, d_model=64, num_heads=1, d_ff=256, vocab_size=100,
                context_size=64, attention_dropout_rate=0.1, residual_dropout_rate=0.1,
                embedding_dropout_rate=0.1, epsilon=1e-06)

# You can use predefined config as follows instead of defining config by yourself
#
# from tfdlg.configs import GPT2SmallConfig
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

## Scripts

Change directory to `scripts`, and install dependent libraries.

```sh
$ cd scripts
$ pip install -r requirements.txt
```

### Train Model

Train tokenizer first.

```sh
$ python train_tokenizer.py tokenizer_model train.txt --vocab_size=5000
```

Finally, train model.

```sh
$ python train.py --train_file train.txt --valid_file valid.txt --tokenizer_model_dir tokenizer_model --save_model_dir=model --epochs=2 --batch_size=4
```

### Serve Web API

```sh
$ python serve_webapi.py --tokenizer_model_dir=tokenizer_model --load_model_dir=model --host="0.0.0.0" --port="8000"
```

Document is available to access to http://localhost:8000/docs

## Model Description

### tfdlg.models.PostLNDecoder

It is the decoder side implementation of [Vaswani+, 2017] .

Difference from [VasWani+, 2017] is

- Weight is initialized by Grolo's uniform distribution except for layers which uses ReLU. For those which uses the ReLU activation function, He's initialization is used. (The weight initialization method is not mentioned in the paper.)

### tfdlg.models.PreLNDecoder

PreLNDecoder replaces Post Layer Normalization architecture of PostLNDecoder with Pre Layer Normalization architecture explained in [Xiong+, 2020].

This architecture is related to GPT-2 introduced in [Radford+, 2019].
The main differences from GPT2 are;

- activation is not GeLU but ReLU
- weight initialization is not uniform distribution

## Reference

* [Hendrycks+, 2016] *Gaussian Error Linear Units (GELUs)* by Dan Hendrycks and Kevin Gimpel. (https://arxiv.org/abs/1606.08415)
* [Vaswani+, 2017] *Attention Is All You Need* by Ashish Vaswani et al. (https://arxiv.org/abs/1706.03762v5)
* [Radford+, 2018] *Improving Language Understanding by Generative Pre-Training* by Alec Radford et al. (https://openai.com/blog/language-unsupervised/)
* [Radford+, 2019] *Language Models are Unsupervised Multitask Learners* by Alec Radford et al. (https://openai.com/blog/better-language-models/)
* [Holtzman+, 2019] *The Curious Case of Neural Text Degeneration* by Ari Holtzman et al. (https://arxiv.org/abs/1904.09751)
* [Xiong+, 2020] *On Layer Normalization in the Transformer Architecture* by Ruibin Xiong et al. (https://arxiv.org/abs/2002.04745)