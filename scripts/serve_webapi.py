from tfchat.generations import TopKTopPGenerator
from tfchat.tokenizers import SentencePieceTokenizer
from tfchat.utils import set_mixed_precision_policy
from tfchat.utils import set_memory_growth
from tfchat.utils import load_model
from pydantic import BaseModel
from fastapi import FastAPI
from typing import List
import numpy as np
import uvicorn
import enum
import pkg_resources


class Request(BaseModel):
    context: List[str]
    response: str


class Response(BaseModel):
    request: Request
    response: str = ""


class DatasetType(enum.Enum):
    LM = enum.auto()
    DIALOG = enum.auto()


class Handler:
    def __init__(self, dataset_type: DatasetType, tokenizer, generator):
        self._dataset_type = dataset_type
        self._tokenizer = tokenizer
        self._generator = generator

    def generate(self, req: Request):
        dataset_type = self._dataset_type
        tokenizer = self._tokenizer
        generator = self._generator
        context = req.context
        response = req.response

        # Prepare text to convert to ids
        sep_token = "|"
        if dataset_type == DatasetType.LM:
            text = "".join(context) + response
        elif dataset_type == DatasetType.DIALOG:
            if context:
                text = sep_token + sep_token.join(context)
            text = text + sep_token + response
        else:
            raise Exception(f"Invalid dataset_type: {dataset_type}")

        # Convert to id
        ids = []
        text_segments = text.split(sep_token)
        for i, txt in enumerate(text_segments):
            if len(txt) > 0:
                ids.extend(tokenizer.encode(txt))
            if i < len(text_segments) - 1:
                ids.append(tokenizer.sep_token_id)

        output_ids = generator.generate(np.array([ids], dtype=np.int32))
        output_ids = output_ids[0][len(ids):]
        output_text = tokenizer.decode(output_ids.tolist())

        return Response(request=req, response=output_text)


def get_version():
    pkg_name = "tfchat"
    try:
        version = pkg_resources.get_distribution(pkg_name).version
    except pkg_resources.DistInfoDistribution:
        print(f"Package name is not found: {pkg_name}")
        version = pkg_resources
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
    dataset_cls="tfchat.data.BlockDataset"
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
    generator = TopKTopPGenerator(model=model, max_len=max_len)

    # Define dataset_type
    dataset_type_map = {
        "tfchat.data.BlockDataset": DatasetType.LM,
        "tfchat.data.LineByLineDataset": DatasetType.LM,
        "tfchat.dialog.data.ContextDataset": DatasetType.DIALOG,
    }

    # Serve API
    handler = Handler(dataset_type=dataset_type_map[dataset_cls], tokenizer=tokenizer, generator=generator)
    app = build_api(handler)
    uvicorn.run(app=app, host=host, port=port)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
