import sys


import os
from libs.videollava.eval.eval import load_model
from utils.constants import DEFAULT_DEVICE, PATH_TO_MODEL_FOLDER


def teochat_model_loader(
    model_id: str = "jirvin16/TEOChat",
    device: str = DEFAULT_DEVICE,
):
    model_path = os.path.join(PATH_TO_MODEL_FOLDER, model_id)
    model = None
    processor = None
    tokenizer = None

    tokenizer, model, processor = load_model(
        model_path=model_path, model_base=None, device=device
    )
    model_hub = {}
    model_hub[model_id] = {}  # Initialize the dictionary for this model_id
    model_hub[model_id]["model_id"] = model_id
    model_hub[model_id]["model"] = model
    model_hub[model_id]["processor"] = processor
    model_hub[model_id]["tokenizer"] = tokenizer
    return model_hub
