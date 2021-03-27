---
language: ja
datasets: wikipedia
widget:
- text: "近年の機械学習は"
license: cc-by-sa-3.0
---

# GPT-2 small Japanese model

This repository contains a pretrained SentencePiece tokenizer model and GPT-2 small model trained on Japanese Wikipedia dataset.

## Training data

[Japanese Wikipedia](https://ja.wikipedia.org/wiki/Wikipedia:データベースダウンロード) dataset which is released under [Creative Commons Attribution-ShareAlike 3.0](https://creativecommons.org/licenses/by-sa/3.0/) is used for training both the tokenizer and GPT-2 model as of March 1st, 2021.
The dataset is splitted into three subsets - train, valid and test. Both of tokenizer and model are trained with the train split.

## Model description

The model architecture is the same as GPT-2 small model (n_ctx: 1024, n_embd 768, n_head: 12, n_layer: 12) except for a vocabulary size.
The vocabulary size is set to 32,000 instead of an original size of 50,257.
`transformers.GPT2LMHeadModel` is used for training.

## Tokenizer description

[SentencePiece](https://github.com/google/sentencepiece) tokenizer is used as a tokenizer for this model.

In a training, the tokenizer model is trained with 10,000,000 samples which are extracted from the train split of training data.
The vocabulary size is set to 32,000. A `add_dummy_prefix` option is set to `True` because words are not separated by whitespaces in Japanese.

After training, the model is imported to `transformers.BERTGenerationTokenizer` because it supports SentencePiece models and it does not add any special tokens as default, which is useful expecially for a text generation task.

## Training

The model is trained on the train split for 10 epochs with batch size 2 and 1024 tokens for each sample (i.e. 2048 tokens are processed in each batch). Each epoch contains around 250,000 steps.
Adam optimizer is used. The learning rate is linearly decreased from `1e-4` to `0`. A clip norm is also used to set to `1.0`.
After finishing training, the training loss is reached to 3.23, wihle the validation loss is reached to 3.50.

All the code to train tokenizer and GPT-2 models are available in [a repository on GitHub](https://github.com/colorfulscoop/tfdlg/tree/63d9531870af428b747554684b186c6316e34c54/examples/transformers-gpt2-ja)

## Usage

First, install dependecies.

```sh
$ pip install transformers==4.3.3 torch==1.8.0 sentencepiece==0.1.91
```

Then load the pretrained tokenizer and GPT-2 model, and call a `generate` method.

```sh
>>> import transformers
>>> tokenizer = transformers.AutoTokenizer.from_pretrained("colorfulscoop/gpt2-small-ja")
>>> model = transformers.AutoModelForCausalLM.from_pretrained("colorfulscoop/gpt2-small-ja")
>>> input = tokenizer.encode("近年の機械学習は", return_tensors="pt")
>>> output = model.generate(input, do_sample=True, top_p=0.95, top_k=50, num_return_sequences=3)
>>> tokenizer.batch_decode(output)
['近年の機械学習は、特に、コンピューター学習において重要な概念である。この概念は、教育心理学', '近年の機械学習は時間間隔の短縮、時間間隔の短縮、学習時間の短縮、学習の', '近年の機械学習は、学生と学生が自分の能力を高め、結果を向上させることを目的としている。それは、']
```

## License

All the models included in this repository are licensed under [Creative Commons Attribution-ShareAlike 3.0](https://creativecommons.org/licenses/by-sa/3.0/).
