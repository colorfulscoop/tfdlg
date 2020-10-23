# TfChat

TfChat provides a **t**rans**f**ormer-based response generation models with **T**ensor**F**low for **chat**bots.

## Installation

Prepare Python >= 3.6 first. Then run `pip` to install this package from GitHub.

```sh
$ pip install git+https://github.com/noriyukipy/tfchat
```

## Test

```sh
$ pip install pytest==6.1.1
$ pytest tests/
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
* [Xiong+, 2020] *On Layer Normalization in the Transformer Architecture* by Ruibin Xiong et al. (https://arxiv.org/abs/2002.04745)