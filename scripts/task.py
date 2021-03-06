from tfdlg.data import BlockDataset
from tfdlg.dialog.data import DialogDataset
from tfdlg.dialog.data import DialogClsDataset
from tfdlg.models import PreLNDecoder
from tfdlg.losses import PaddingLoss
from tfdlg.tokenizers import SentencePieceTokenizer
from tfdlg.dialog.tokenizers import encode_dialog
from typing import List
from tfdlg.generations import TopKTopPGenerator
import tensorflow.keras as keras
import numpy as np


class LMTask:
    def __init__(self, config):
        self._config = config

    @property
    def model_cls(self):
        return PreLNDecoder

    @property
    def loss_fn(self):
        return PaddingLoss()

    def prepare_tokenizer(self, model_dir):
        return SentencePieceTokenizer.load(model_dir=model_dir)

    def prepare_dataset(self, filename, encode_fn, batch_size, shuffle,
                        buffer_size=10000):
        def gen():
            return (t.strip("\n") for t in open(filename))

        ds = BlockDataset.from_generator(
            generator=gen,
            encode_fn=encode_fn,
            block_size=self._config.context_size,
            batch_size=batch_size,
            buffer_size=buffer_size,
            shuffle=shuffle
        )
        return ds


class DialogTask(LMTask):
    def __init__(self, config):
        self._config = config

    @property
    def model_cls(self):
        return PreLNDecoder

    @property
    def loss_fn(self):
        return PaddingLoss()

    def prepare_tokenizer(self, model_dir):
        self._tokenizer = SentencePieceTokenizer.load(model_dir=model_dir)
        return self._tokenizer

    def prepare_dataset(self, filename, encode_fn, batch_size, shuffle,
                        buffer_size=10000):
        def gen():
            return (t.strip("\n").split("\t") for t in open(filename))

        ds = DialogDataset.from_generator(
            generator=gen,
            context_encode_fn=lambda context: encode_dialog(
                encode_fn=encode_fn,
                sep_token_id=self._tokenizer.sep_token_id,
                context=context,
                response=None
            ),
            max_len=self._config.context_size,
            batch_size=batch_size,
            buffer_size=buffer_size,
            shuffle=shuffle
        )
        return ds


class DialogClsTask(LMTask):
    def __init__(self, config):
        self._config = config

    @property
    def model_cls(self):
        return PreLNDecoder

    @property
    def loss_fn(self):
        return (PaddingLoss(), keras.losses.BinaryCrossentropy(from_logits=True))

    def prepare_tokenizer(self, model_dir):
        self._tokenizer = SentencePieceTokenizer.load(model_dir=model_dir)
        return self._tokenizer

    def prepare_dataset(self, filename, encode_fn, batch_size, shuffle,
                        buffer_size=10000):
        def gen():
            return (t.strip("\n").split("\t") for t in open(filename))

        ds = DialogClsDataset.from_generator(
            generator=gen,
            encode_fn=encode_fn,
            max_len=self._config.context_size,
            sep_token_id=self._tokenizer.sep_token_id,
            batch_size=batch_size,
            buffer_size=buffer_size,
            shuffle=shuffle
        )
        return ds
