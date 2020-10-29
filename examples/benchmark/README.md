# Benchmark

## Setup

* Tokenizer: [🤗 Transformers](https://github.com/huggingface/transformers)' [GPT2 tokenizer](https://huggingface.co/transformers/model_doc/gpt2.html#gpt2tokenizer)
* Models:
  * PreLNDecoder and PostLNDecoder from TfChat
  * GPT2 from minGPT-TF
  * GPT2 from [🤗 Transformers](https://github.com/huggingface/transformers)
* Model hyper-parameters (e.g. d_model, n_layers, n_heads): The same as GPT2 small model
* Target dataset: [WikiText-103](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/)
* Training parameters:
  * Max epochs: 10
  * Batch size: 2
  * Optimizers: Adam for Tensorflow models and AdamW for [🤗 Transformers](https://github.com/huggingface/transformers)' model
  * Learning rate schedule: linear schedule where lr decreases from `1e-5` to `0`

In summary:

| Name | Model | Optimizer |
| --- | --- | --- |
| TfChat.PreLNDecoder | PreLNDecoder from TfChat | Adam |
| TfChat.PostLNDecoder | PostLNDecoder from TfChat | Adam |
| minGPT-TF.GPT2 | GPT2 from minGPT-TF | Adam |
| Transformers.GPT2 | GPT2 from [🤗 Transformers](https://github.com/huggingface/transformers) | [AdamW](https://huggingface.co/transformers/main_classes/optimizer_schedules.html#adamw-pytorch) |

## Prepare dataset

```sh
python get_wikitext.py 103_raw
```

## Run

### TfChat.PreLNDecoder, TfChat.PostLNDecoder

```sh
$ docker container run --gpus all -v $(pwd):/work -w /work --rm -p 8888:8888 -it tensorflow/tensorflow:2.3.1-gpu
$ pip install jupyter==1.0.0 papermill==2.1.3
$ papermill tfmodel_train_scratch.ipynb output/tfmodel_train_scratch-wikitext_103_raw-pre_ln.ipynb -p train_file wikitext-103-raw/wiki.train.raw -p valid_file wikitext-103-raw/wiki.valid.raw -p epochs 20 -p model_type pre_ln
```

### minGPT-TF.GPT2

```sh
$ docker container run --gpus all -v $(pwd):/work -w /work --rm -p 8888:8888 -it tensorflow/tensorflow:2.3.1-gpu
$ pip install jupyter==1.0.0 papermill==2.1.3
$ git clone https://github.com/kamalkraj/minGPT-TF
$ cp -r minGPT-TF/mingpt .
$ papermill tfmodel_train_scratch.ipynb output/tfmodel_train_scratch-wikitext_103_raw-min_gpt.ipynb -p train_file wikitext-103-raw/wiki.train.raw -p valid_file wikitext-103-raw/wiki.valid.raw -p epochs 20 -p model_type min_gpt
```

### Transformers.GPT2

```sh
$ docker container run --gpus all -v $(pwd):/work -w /work --rm -p8888:8888 -it pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel
$ pip install jupyter==1.0.0 papermill==2.1.3
$ papermill transformers_train_scratch.ipynb output/transformers_train_scratch-wikitext_103_raw.ipynb -p train_file wikitext-103-raw/wiki.train.raw -p valid_file wikitext-103-raw/wiki.valid.raw -p epochs 10 -p warmup_steps 0
```

## Result

| Name | PPL on WikiText-103 | notebook |
| --- | --- | --- |