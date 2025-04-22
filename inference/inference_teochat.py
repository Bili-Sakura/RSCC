import sys


sys.path.append("./")
import os
import PIL
from libs.videollava.eval.eval import load_model
from libs.videollava.eval.inference import run_inference_single
from utils.model_teochat import teochat_model_loader
from utils.constants import DEFAULT_DEVICE, MAX_NEW_TOKENS


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
            loading_model_id = model_hub[model_id]["model_id"]
            model = model_hub[model_id]["model"]
            processor = model_hub[model_id]["processor"]
            tokenizer = model_hub[model_id]["tokenizer"]
        except:
            print(f"Error Loading {model_id}.")
    else:

        model_hub = teochat_model_loader(model_id=model_id, device=device)
        loading_model_id = model_hub[model_id]["model_id"]
        model = model_hub[model_id]["model"]
        processor = model_hub[model_id]["processor"]
        tokenizer = model_hub[model_id]["tokenizer"]
    if loading_model_id != model_id:
        print(
            f"Inference model_id is {model_id} while loading model checkpoint is {loading_model_id}"
        )
        return ""
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
        # temperature=TEMPERATURE,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    change_caption = response
    return change_caption
