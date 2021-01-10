from tfdlg.generations import TopKTopPGenerator
from tfdlg.tokenizers import SentencePieceTokenizer
from tfdlg.utils import import_class
from tfdlg.data import BlockDataset
from tfdlg.data import LineByLineDataset
from tfdlg.dialog.data import DialogDataset
from tfdlg.utils import set_mixed_precision_policy
from tfdlg.utils import set_memory_growth
from tfdlg.utils import load_model
from pydantic import BaseModel
from fastapi import FastAPI
from typing import List
import numpy as np
import uvicorn
import enum
import pkg_resources


class Request(BaseModel):
    context: List[str]
    response: str = ""


class Response(BaseModel):
    request: Request
    response: str


class DatasetType(enum.Enum):
    LM = enum.auto()
    DIALOG = enum.auto()


class Handler:
    def __init__(self, dataset, tokenizer, generator):
        self._dataset = dataset
        self._tokenizer = tokenizer
        self._generator = generator

    def generate(self, req: Request):
        tokenizer = self._tokenizer
        generator = self._generator
        context = req.context
        response = req.response

        # Prepare text to convert to ids
        ids = self._dataset.convert_context_to_ids(context=context, response=response)

        if len(ids) > 0:
            output_ids = generator.generate(np.array([ids], dtype=np.int32))
            output_ids = output_ids[0][len(ids):]
            output_text = tokenizer.decode(output_ids.tolist())
            print("Input context: ", context)
            print("Input response: ", response)
            print("Encode:", ids)
            print("Gen:   ", output_ids)
            print("Response:", output_text)

            if response:
                output_text = response + output_text
        else:
            print("Respond empty string because of empty ids")
            output_text = ""
        return Response(request=req, response=output_text)


def get_version():
    pkg_name = "tfdlg"
    try:
        version = pkg_resources.get_distribution(pkg_name).version
    except pkg_resources.DistributionNotFound:
        print(f"Package name not found: {pkg_name}")
        version = "package version info not found"
    return version


def build_api(handler):
    app = FastAPI(
        title="tfDlg",
        description="",
        version=get_version()
    )
    app.add_api_route("/generate", handler.generate, methods=["POST"], response_model=Response)
    return app


def main(
    tokenizer_model_dir, load_model_dir,
    host="0.0.0.0", port="8080",
    # Flag to use mixed precision or not
    fp16=False,
    # Set memory growth no to allocate all the memory
    memory_growth=False,
    # Parameters for do_generate
    max_len=20,
    # dataset_cls defines how to append the context with id 
    dataset_cls="tfdlg.data.BlockDataset"
):
    if memory_growth:
        print("Set memory growth")
        set_memory_growth()

    if fp16:
        print("Set mixed precision policy")
        set_mixed_precision_policy()

    # Load tokenizer
    tokenizer = SentencePieceTokenizer.load(model_dir=tokenizer_model_dir)

    # Load model
    model, config = load_model(load_model_dir)

    # Prepare generator
    generator = TopKTopPGenerator(model=model, max_len=max_len, stop_id=tokenizer.sep_token_id)

    # Prepare dataset
    dataset_cls = import_class(dataset_cls)
    if dataset_cls == BlockDataset:
        dataset = dataset_cls(block_size=config.context_size, encode_fn=tokenizer.encode)
    elif dataset_cls == LineByLineDataset:
        dataset = dataset_cls(max_len=config.context_size, encode_fn=tokenizer.encode)
    elif dataset_cls == DialogDataset:
        dataset = dataset_cls(max_len=config.context_size, encode_fn=tokenizer.encode, sep_token_id=tokenizer.sep_token_id)
    else:
        raise Exception(f"{dataset} is not one of BlockDataset, LineByLineDataset or DialogDataset")
    print("Dataset class:", dataset_cls)

    # Serve API
    handler = Handler(dataset=dataset, tokenizer=tokenizer, generator=generator)
    app = build_api(handler)
    uvicorn.run(app=app, host=host, port=port)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
