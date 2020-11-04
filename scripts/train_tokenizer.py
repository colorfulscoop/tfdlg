from tfchat.tokenizers import SentencePieceTokenizer


def main(model_dir,
         file=None, do_train=True, do_run=False,
         vocab_size=32000, input_sentence_size=0,
         ):
    if do_train:
        assert file
        tokenizer = SentencePieceTokenizer(model_dir=model_dir,
                                           vocab_size=vocab_size,
                                           input_sentence_size=input_sentence_size
                                           )
        tokenizer.fit_on_file(file=file)
    if do_run:
        tokenizer = SentencePieceTokenizer.load(model_dir=model_dir)
        while True:
            try:
                text = input(">>> ")
            except (KeyboardInterrupt, EOFError):
                print("")
                print("Bye")
                break
            ids = tokenizer.encode(text)
            tkns = tokenizer.decode(ids)
            print("Input: ", text)
            print("Encode:", ids)
            print("Decode:", tkns)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
