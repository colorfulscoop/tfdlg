# Language Modeling

## Preparation

```sh
pip install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers==4.4.2 sentencepiece==0.1.95 pytorch-lightning==1.2.6 fire==0.4.0
```

## Execution

```sh
python3 train.py --tokenizer_model=colorfulscoop/gpt2-small-ja --save_model_dir=model --train_file=data/train.txt --valid_file=data/valid.txt --gpus=1 --precision=16 --lr=1e-4 --seed=1000 --val_check_interval=100000 --max_steps=1000000
```

Enable scheduler with `--num_training_steps` option

```sh
python3 train.py --tokenizer_model=colorfulscoop/gpt2-small-ja --save_model_dir=model-scheduler --train_file=data/train.txt --valid_file=data/valid.txt --gpus=1 --precision=16 --lr=1e-4 --seed=1000 --val_check_interval=100000 --max_steps=1000000 --num_training_steps=1000000
```