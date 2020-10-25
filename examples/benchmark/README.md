# Benchmark

Tokenizer: HuggingFace GPT2 tokenizer
Model: TfChat PreLNDecoder and PostLN Decoder, minGPT-TF, HuggingFace GPT2
Model size: same as GPT2 small model
Target dataset: WikiText-2 or WikiText-103

## Prepare dataset

```sh
python get_wikitext.py 2_raw
python get_wikitext.py 103_raw
```

## Run

```sh
```

### TfChat {Pre,Post}LNDecoder

```sh
$ docker container run --gpus all -v $(pwd):/work -w /work --rm -p 8888:8888 -it tensorflow/tensorflow:2.3.1-gpu
$ pip install jupyter==1.0.0 papermill==2.1.3
$ papermill tfmodel_train_scratch.ipynb output/tfmodel_train_scratch-wikitext_103_raw-pre_ln.ipynb -p train_file wikitext-103-raw/wiki.train.raw -p valid_file wikitext-103-raw/wiki.valid.raw -p epochs 20 -p model_type pre_ln
```

### minGPT-TF

```sh
$ docker container run --gpus all -v $(pwd):/work -w /work --rm -p 8888:8888 -it tensorflow/tensorflow:2.3.1-gpu
$ pip install jupyter==1.0.0 papermill==2.1.3
$ git clone https://github.com/kamalkraj/minGPT-TF
$ cp -r minGPT-TF/mingpt .
$ papermill tfmodel_train_scratch.ipynb output/tfmodel_train_scratch-wikitext_103_raw-min_gpt.ipynb -p train_file wikitext-103-raw/wiki.train.raw -p valid_file wikitext-103-raw/wiki.valid.raw -p epochs 20 -p model_type min_gpt
```

### HuggingFace GPT2

```sh
$ docker container run --gpus all -v $(pwd):/work -w /work --rm -p8888:8888 -it pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel
$ pip install jupyter==1.0.0 papermill==2.1.3
$ papermill transformers_train_scratch.ipynb output/transformers_train_scratch-wikitext_103_raw.ipynb -p train_file wikitext-103-raw/wiki.train.raw -p valid_file wikitext-103-raw/wiki.valid.raw -p epochs 20
```
