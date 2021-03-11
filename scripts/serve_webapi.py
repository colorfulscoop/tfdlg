from tfdlg.utils import import_class
from tfdlg.utils import set_mixed_precision_policy
from tfdlg.utils import set_memory_growth
from tfdlg.utils import load_model
from pydantic import BaseModel
from fastapi import FastAPI
from typing import List
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
    def __init__(self, task, model):
        self._task = task
        self._model = model

    def generate(self, req: Request):
        context = req.context
        response = req.response

        res = self._task.handle_request(
            model=self._model,
            context=context,
            response=response
        )

        return Response(request=req, response=res)


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
        description="tfDlg is a Python library for transformer-based language models and dialog models with TensorFlow.",
        version=get_version(),
    )
    app.add_api_route("/generate", handler.generate, methods=["POST"], response_model=Response)
    return app


def main(
    tokenizer_model_dir, model_dir,
    host="0.0.0.0", port="8080",
    # Flag to use mixed precision or not
    fp16=False,
    # Set memory growth no to allocate all the memory
    memory_growth=False,
    # Parameters for do_generate
    max_len=20,
    # Specify task
    task_cls="task.LMTask"
):
    if memory_growth:
        print("Set memory growth")
        set_memory_growth()

    if fp16:
        print("Set mixed precision policy")
        set_mixed_precision_policy()

    # Load task, model, config and tokenizer
    model, config = load_model(model_dir)
    task = import_class(task_cls)(config=config)
    task.prepare_tokenizer(model_dir=tokenizer_model_dir)  # ignore returned tokenizer, which will not be used in this script

    # Serve API
    handler = Handler(task=task, model=model)
    app = build_api(handler)
    uvicorn.run(app=app, host=host, port=port)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
