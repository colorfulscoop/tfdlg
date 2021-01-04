from tfdlg.tokenizers import SentencePieceTokenizer
import tempfile


def test_tokenizer():
    with tempfile.NamedTemporaryFile(mode="w") as fp:
        # Prepare train file
        fp.write("This is a test file for tokenizers. SentencePiece tokeniezr will be tested in this example.")
        fp.seek(0)

        # Prepare directory to save model
        with tempfile.TemporaryDirectory() as model_dir:
            tokenizer = SentencePieceTokenizer(model_dir=model_dir, vocab_size=40)

            # Train test
            tokenizer.fit_on_file(fp.name)

            # Load test
            tokenizer = SentencePieceTokenizer.load(model_dir=model_dir)

            # Encode/decode test
            text = "This is a test"
            encoded = tokenizer.encode(text)
            decoded = tokenizer.decode(encoded)
            assert decoded == text

            # Encode/decode test with list of text
            text_list = ["This is a test"]
            encoded = tokenizer.encode(text_list)
            decoded = tokenizer.decode(encoded)
            assert decoded == text_list

            # special tokens test
            assert tokenizer.pad_token_id == 0
            assert tokenizer.unk_token_id == 1
            assert tokenizer.cls_token_id == 4
            assert tokenizer.sep_token_id == 5

            # Test len method
            assert len(tokenizer) == 40
