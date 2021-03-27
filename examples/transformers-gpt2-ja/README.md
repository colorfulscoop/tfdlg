# Train GPT-2 model for HuggingFace's transformers from scratch

This repository summarizes how the GPT-2 model of HuggingFace's transformers library is trained on Japanese Wikipedia.

## Setting

Check the [model card](model_card.md) in detail.

## Training

### Prepare corpus

```sh
$ docker container run -w /work -v $(pwd):/work --rm -it python:3.8.6-slim-buster bash
(container)$ apt update && apt install -y wget git
```

Check the latest date in the list from https://dumps.wikimedia.org/jawiki/ .
`20210301` is the latest as of March 15th, 2021.

```sh
(container)$ bash get_jawiki.sh 20210301
```

Finally generated data can be found under `data` directory.

```sh
(container)$ ls data/jawiki/20210301/data/
test.txt  train.txt  valid.txt
```

```sh
(container)$ exit
```

### Train tokenizer

Create a directory to save output files.

```sh
$ mkdir output
```

Then train SentencePiece model.

```sh
$ docker container run -w /work -v $(pwd):/work --rm -it python:3.8.6-slim-buster bash
(container)$ pip install -r requirements.txt
(container)$ papermill train_tokenizer.ipynb output/train_tokenizer.ipynb
(container)$ exit
```

### Train model

```sh
$ docker container run --gpus all -v $(pwd):/work -w /work --rm -it tensorflow/tensorflow:2.4.1-gpu bash
(container)$ apt update && apt install -y git
(container)$ pip install -r requirements.txt
(container)$ papermill train_model.ipynb output/train_model.ipynb
(container)$ exit
```

### Convert the TensorFlow model to PyTorch

```sh
$ docker container run -w /work -v $(pwd):/work --rm -it python:3.8.6-slim-buster bash
(container)$ apt update && apt install -y git
(container)$ pip install -r requirements.txt
(container)$ papermill convert_to_pytorch.ipynb output/convert_to_pytorch.ipynb
(container)$ exit
```

## Usage

```sh
$ docker container run -w /work -v $(pwd):/work --rm -it python:3.8.6-slim-buster bash
(container)$ pip install transformers==4.3.3 torch==1.8.0 sentencepiece==0.1.91
(container)$ python
(container)>>> import transformers, torch
(container)>>> tokenizer = transformers.AutoTokenizer.from_pretrained("output/model")
(container)>>> model = transformers.AutoModelForCausalLM.from_pretrained("output/model")
(container)>>> input = tokenizer.encode("近年の機械学習は", return_tensors="pt")
(container)>>> output = model.generate(input, do_sample=True, top_p=0.95, top_k=50, num_return_sequences=3)
(container)>>> tokenizer.batch_decode(output)
['近年の機械学習は、特に、コンピューター学習において重要な概念である。この概念は、教育心理学', '近年の機械学習は時間間隔の短縮、時間間隔の短縮、学習時間の短縮、学習の', '近年の機械 学習は、学生と学生が自分の能力を高め、結果を向上させることを目的としている。それは、']
```


## Upload to model hub

Finally, upload the trained model to HuggingFace's model hub.
Following the [official document](https://huggingface.co/transformers/model_sharing.html), the following process is executed.

First, create a repository named "gpt2-small-ja" from HuggingFace's website.

Then, prepare git lfs. In a MacOS environment, git lfs can be installed as follows.

```sh
$ brew install git-lfs
$ git lfs install
Updated git hooks.
Git LFS initialized.
```

Then, clone the repository.

```sh
$ git clone https://huggingface.co/colorfulscoop/gpt2-small-ja
```

You can notice that the cloned repository already contains `.gitattributes` which identifies which files should be treaded by lfs.

```sh
$ cat gpt2-small-ja/.gitattributes
*.bin.* filter=lfs diff=lfs merge=lfs -text
*.lfs.* filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text
*.tflite filter=lfs diff=lfs merge=lfs -text
*.tar.gz filter=lfs diff=lfs merge=lfs -text
*.ot filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
*.arrow filter=lfs diff=lfs merge=lfs -text
*.ftz filter=lfs diff=lfs merge=lfs -text
*.joblib filter=lfs diff=lfs merge=lfs -text
*.model filter=lfs diff=lfs merge=lfs -text
*.msgpack filter=lfs diff=lfs merge=lfs -text
*.pb filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
```

Therefore, without using `git lfs track` command, all the models with above prefixes will be tracked by `git lfs`.

Then, add models and model cards into the cloned directory.

```sh
$ cp model_card.md gpt2-small-ja/README.md
$ cp output/model/* gpt2-small-ja
```

Then, modify `config.json` to specify default generation values by following diff.

```sh
   "transformers_version": "4.3.3",
   "unk_token_id": 1,
   "use_cache": true,
-  "vocab_size": 32000
+  "vocab_size": 32000,
+  "top_k": 50,
+  "top_p": 0.95,
+  "do_sample": true
 }
 ```


Then, add and commit models and files.

```sh
$ cd gpt2-small-ja
$ git add .
$ git commit -m "Add models and model card"
```


Finally, push the commit to model hub.

```sh
$ git push origin
```