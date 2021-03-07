# Benchmark with WikiText

## Benchmark setting

## How to run benchmark

### Prepare environment

```sh
$ docker container run --gpus all -v $(pwd):/work -w /work --rm -it tensorflow/tensorflow:2.4.1-gpu bash
```

Install required packages.


```sh
$ apt update
$ apt install -y git
$ pip install git+https://github.com/colorfulscoop/tfdlg@v0.2.0
$ pip install -r requirements.txt
```

Prepare corpus

```sh
$ python3 get_wikitext.py 103_raw
```

## Result

| Model | Experiment comamnd | Output |
| --- | --- | --- |
| PreLNDecoder | `papermill tfdlg_train.ipynb output/tfmodel_train-pre_ln.ipynb -p model_type pre_ln -p batch_size 4 -p fp16 True` | |

Take a look at Tensorboard to keep track of training process.

```sh
$ docker container run --gpus all -p 6006:6006 -v $(pwd):/work -w /work --rm -it tensorflow/tensorflow:2.4.1-gpu tensorboard --logdir output/tensorboard --host 0.0.0.0
```