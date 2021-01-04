import sentencepiece as spm
from pathlib import Path


class SentencePieceTokenizer:
    """Tokenizer with SentencePiece"""

    sep_token = "<sep>"
    cls_token = "<cls>"
    pad_token = "<pad>"

    def __init__(self, model_dir, vocab_size=32000,
                 pad_id=0, unk_id=1, bos_id=2, eos_id=3,
                 pad_piece="<pad>", unk_piece="<unk>", bos_piece="<s>", eos_piece="</s>",
                 control_symbols=["<cls>", "<sep>"],
                 input_sentence_size=0, shuffle_input_sentence=True,
                 add_dummy_prefix=True,
                 ):
        """SentencePiece tokenizer

        This module utilizes Python module for SentencePiece; https://github.com/google/sentencepiece/blob/master/python/README.md .
        All options can be found here; https://github.com/google/sentencepiece/blob/master/doc/options.md .


        Args:
            add_dummy_prefix (bool): If set to True,
                add the white space before the sentence while normalizing the sentence
        """

        # This attribute is set after loading model
        self._spm = None

        self._model_dir = Path(model_dir)
        self._model_prefix = Path(model_dir) / Path("sp")
        self._model_path = Path(model_dir) / Path("sp.model")

        self._train_args = dict(
            model_prefix=self._model_prefix,
            vocab_size=vocab_size,
            pad_id=pad_id,
            unk_id=unk_id,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_piece=pad_piece,
            unk_piece=unk_piece,
            bos_piece=bos_piece,
            eos_piece=eos_piece,
            control_symbols=control_symbols,
            input_sentence_size=input_sentence_size,
            shuffle_input_sentence=shuffle_input_sentence,
            add_dummy_prefix=add_dummy_prefix,
        )

    def fit_on_file(self, file):
        if not self._model_dir.exists():
            self._model_dir.mkdir()
        spm.SentencePieceTrainer.train(input=file, **self._train_args)
        self.__class__.load(self._model_dir)

    def save(self):
        raise Exception("fit_on_file saves model as well as train a model.")

    @classmethod
    def load(cls, model_dir):
        obj = cls(model_dir=model_dir)
        obj._spm = spm.SentencePieceProcessor()
        obj._spm.load(str(obj._model_path))
        return obj

    def encode(self, *args, **kwargs):
        return self._spm.encode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self._spm.decode(*args, **kwargs)

    @property
    def pad_token_id(self):
        return self._get_id(self.pad_token)

    @property
    def unk_token_id(self):
        return self._spm.unk_id()

    @property
    def cls_token_id(self):
        return self._get_id(self.cls_token)

    @property
    def sep_token_id(self):
        return self._get_id(self.sep_token)

    def _get_id(self, piece):
        id_ = self._spm.piece_to_id(piece)
        if id_ == self._spm.unk_id():
            raise Exception(f"{piece} is not defined")
        return id_

    def __len__(self):
        return self._spm.get_piece_size()
