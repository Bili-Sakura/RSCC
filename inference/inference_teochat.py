import sys

sys.path.append("/home/sakura/projects/RSCC/libs")
sys.path.append("/home/sakura/projects/RSCC/utils")
import os
import PIL
from videollava.eval.eval import load_model
from videollava.eval.inference import run_inference_single
from utils.model_teochat import teochat_model_loader

DEFAULT_DEVICE = "cuda:0"
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.0001
PATH_TO_MODEL_FOLDER = "/home/models"


def inference_teochat(
    model_id: str,
    text_prompt: str,
    pre_image: PIL.Image.Image,
    post_image: PIL.Image.Image,
    device=DEFAULT_DEVICE,
    model_hub=None,
):
    if model_hub is not None:
        try:
            model = model_hub[model_id]["model"]
            processor = model_hub[model_id]["processor"]
            tokenizer = model_hub[model_id]["tokenizer"]
        except:
            print(f"Error Loading {model_id}.")
    else:

        model_hub = teochat_model_loader(model_id=model_id, device=device)
        model = model_hub[model_id]["model"]
        processor = model_hub[model_id]["processor"]
        tokenizer = model_hub[model_id]["tokenizer"]

    image_paths = [pre_image, post_image]
    # Note you must include the video tag <video> in the input string otherwise the model will not process the images.

    inp = "<video>" + text_prompt
    response = run_inference_single(
        model,
        processor,
        tokenizer,
        inp,
        image_paths,
        chronological_prefix=True,
        conv_mode="v1",
        temperature=TEMPERATURE,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    change_caption = response
    return change_caption
