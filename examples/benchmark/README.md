# Benchmark

## Setup

* Tokenizer: [🤗 Transformers](https://github.com/huggingface/transformers)' [GPT2 tokenizer](https://huggingface.co/transformers/model_doc/gpt2.html#gpt2tokenizer)
* Models:
  * PreLNDecoder from TfChat
  * GPT2 by Tensorflow implementation from [🤗 Transformers](https://github.com/huggingface/transformers)
  * GPT2 from minGPT-TF
* Activations: ReLU or GeLU
* Model hyper-parameters (e.g. d_model, n_layers, n_heads): The same as [GPT2 small model](https://github.com/openai/gpt-2/blob/master/model_card.md)
* Target dataset: [WikiText-103](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/)
* Training parameters:
  * Max epochs: 10
  * Batch size: 2
  * Optimizer: Adam
  * Learning rate schedule: linear schedule where lr decreases from `1e-4` to `0`. `1e-4` was decided by the preliminary experiment.
* Metrics: Perplexity

Model summary:

| Name | Model |
| --- | --- |
| TfChat.PreLNDecoder | PreLNDecoder from TfChat |
| TfChat.PostLNDecoder | PostLNDecoder from TfChat |
| minGPT-TF.GPT2 | GPT2 from minGPT-TF |
| Transformers.GPT2 | GPT2 from [🤗 Transformers](https://github.com/huggingface/transformers) |

Environment:

* OS: Ubuntu 18.04.5 LTS
* GPU: GeForce RTX 2080 Ti

**Notes:**

* According to [Radford+, 2018], they trained their model for 100 epochs on 64 batch sizes. However, accordint to the resource condition, this benchmark could not be conducted for such large number of epochs and batch sizes.
* According to [this Gist](https://gist.github.com/thomwolf/ca135416a30ea387aa20edaa9b21f0ed), the word-level perplexity should be around 29. Because our experiment uses tokenizre-level perplexity, it is not comparative. However, this result can be one of the information about how our ppl should looks like. 

## Prepare dataset

```sh
python get_wikitext.py 103_raw
```

## Run

### TfChat.PreLNDecoder, TfChat.PostLNDecoder, Transformers.GPT2

```sh
$ docker container run --gpus all -v $(pwd):/work -w /work --rm -it tensorflow/tensorflow:2.3.1-gpu
$ pip install jupyter==1.0.0 papermill==2.1.3
$ papermill tfmodel_train_scratch.ipynb output/tfmodel_train_scratch-wikitext_103_raw-pre_ln-lr_e4.ipynb -p save_model_dir tfchat_model-lr_e4
$ papermill tfmodel_train_scratch.ipynb output/tfmodel_train_scratch-wikitext_103_raw-pre_ln-gelu-lr_e4.ipynb -p save_model_dir tfchat_model-gelu-lr_e4
$ papermill tfmodel_train_scratch.ipynb output/tfmodel_train_scratch-wikitext_103_raw-transformers-lr_e4.ipynb -p model_type transformers -p save_model_dir tfchat_transformers-lr_e4
```

### minGPT-TF.GPT2

```sh
$ docker container run --gpus all -v $(pwd):/work -w /work --rm -p 8888:8888 -it tensorflow/tensorflow:2.3.1-gpu
$ pip install jupyter==1.0.0 papermill==2.1.3
$ git clone https://github.com/kamalkraj/minGPT-TF
$ cp -r minGPT-TF/mingpt .
$ papermill tfmodel_train_scratch.ipynb output/tfmodel_train_scratch-wikitext_103_raw-min_gpt.ipynb -p train_file wikitext-103-raw/wiki.train.raw -p valid_file wikitext-103-raw/wiki.valid.raw -p epochs 20 -p model_type min_gpt
```

## Result

| Name | Activation | Share embedding layer with last layer to softmax | WikiText-103 (PPL) | notebook |
| --- | --- | --- | --- | --- |
| TfChat.PreLNDecoder | ReLU | No | 20.76 | [output/tfmodel_train_scratch-wikitext_103_raw-pre_ln-unshare-lr_e4.ipynb](output/tfmodel_train_scratch-wikitext_103_raw-pre_ln-unshare-lr_e4.ipynb) |
| TfChat.PreLNDecoder | ReLU | Yes | 20.47 | [output/tfmodel_train_scratch-wikitext_103_raw-pre_ln-lr_e4.ipynb](output/tfmodel_train_scratch-wikitext_103_raw-pre_ln-lr_e4.ipynb) |
| TfChat.PreLNDecoder | [GELU](https://github.com/noriyukipy/tfchat/blob/change_default_gelu/tfchat/activations.py#L5) | Yes | 20.13 | [output/tfmodel_train_scratch-wikitext_103_raw-pre_ln-gelu-lr_e4.ipynb](output/tfmodel_train_scratch-wikitext_103_raw-pre_ln-gelu-lr_e4.ipynb) |
| TfChat.PostLNDecoder | | | |
| minGPT-TF.GPT2 | | | |
| Transformers.GPT2 | [GELU](https://github.com/huggingface/transformers/blob/v3.4.0/src/transformers/activations_tf.py#L19) | Yes | 19.52 | [output/tfmodel_train_scratch-wikitext_103_raw-transformers-lr_e4.ipynb](output/tfmodel_train_scratch-wikitext_103_raw-transformers-lr_e4.ipynb) |


## Appendix

### Preliminary experiment

Preliminary experment was conducted by the [language modeling script](https://github.com/huggingface/transformers/blob/v3.4.0/examples/language-modeling/run_language_modeling.py) provided by [🤗 Transformers](https://github.com/huggingface/transformers)'

The main differences from the benchmark setting are

* Optimizers: [AdamW](https://huggingface.co/transformers/main_classes/optimizer_schedules.html#adamw-pytorch)  

I tried three types of learning rate;`1e-3`, `1e-4`, `1e-5` in this preliminary experiment first. `1e-3` divsersed loss in the few steps at the begining. `1e-5` did not converged (i.e. still kept improveing at the end of training). Therefore `1e-4` is used for all the experiment.

```sh
$ docker container run --gpus all -v $(pwd):/work -w /work --rm -p8888:8888 -it pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel
$ pip install jupyter==1.0.0 papermill==2.1.3
$ papermill transformers_train_scratch.ipynb output/transformers_train_scratch-wikitext_103_raw-lr_e4.ipynb -p output_dir transformers_output-lr_e4
```

| lr | WikiText-103 (PPL) | notebook |
| --- | --- | --- |
| `e-3` | (Not converged) | [output/transformers_train_scratch-wikitext_103_raw-lr_e3.ipynb](output/transformers_train_scratch-wikitext_103_raw-lr_e3.ipynb) |
| `e-4` | 18.25 | [output/transformers_train_scratch-wikitext_103_raw-lr_e4.ipynb](output/transformers_train_scratch-wikitext_103_raw-lr_e4.ipynb) |
| `e-5` | 18.91 | [output/transformers_train_scratch-wikitext_103_raw-lr_e5.ipynb](output/transformers_train_scratch-wikitext_103_raw-lr_e5.ipynb) |
