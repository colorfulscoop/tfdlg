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
        self._tokenizer = SentencePieceTokenizer.load(model_dir=model_dir)
        return self._tokenizer

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

    def handle_request(self, model, context: List[str], response: str):
        # Parameters
        max_len = 20

        # Prepare text to convert to ids
        ids = []
        for item in context:
            ids += self._tokenizer.encode(item)
        if response:
            ids += self._tokenizer.encode(response)

        generator = TopKTopPGenerator(model=model, max_len=max_len)

        if len(ids) > 0:
            output_ids = generator.generate(np.array([ids], dtype=np.int32))
            output_ids = output_ids[0][len(ids):]
            output_text = self._tokenizer.decode(output_ids.tolist())
            print("Input context: ", context)
            print("Input response: ", response)
            print("Encode:", ids)
            print("Gen:   ", output_ids)
            print("Response:", output_text)

            # [TODO] This will ignore the space which is needed after the given response.
            if response:
                output_text = response + output_text
        else:
            print("Respond empty string because of empty ids")
            output_text = ""

        return output_text


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
