import sys

sys.path.append("/home/sakura/projects/RSCC/libs")
sys.path.append("/home/sakura/projects/RSCC/utils")
import os
import PIL
from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IGNORE_INDEX,
)
from llava.conversation import conv_templates, SeparatorStyle
from utils.model_ccexpert import ccexpert_model_loader
import torch
import copy

DEFAULT_DEVICE = "cuda:0"
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.0001
PATH_TO_MODEL_FOLDER = "/home/models"


def inference_ccexpert(
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
            tokenizer = model_hub[model_id]["tokenizer"]
            image_processor = model_hub[model_id]["image_processor"]
        except:
            print(f"Error Loading {model_id}.")
    else:
        # model_path = os.path.join(PATH_TO_MODEL_FOLDER, model_id)
        model_hub = ccexpert_model_loader(model_id=model_id, device=device)
        model = model_hub[model_id]["model"]
        tokenizer = model_hub[model_id]["tokenizer"]
        image_processor = model_hub[model_id]["image_processor"]

    # Load two images
    images = [pre_image, post_image]
    image_tensors = process_images(images, image_processor, model.config)
    image_tensors = [
        _image.to(dtype=torch.bfloat16, device=device) for _image in image_tensors
    ]

    # Prepare interleaved text-image input
    conv_template = "qwen_1_5"
    question = "<image><image>" + text_prompt

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = (
        tokenizer_image_token(
            prompt_question,
            tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        )
        .unsqueeze(0)
        .to(device)
    )
    image_sizes = [image.size for image in images]

    # tokenizer.batch_decode(torch.clamp(input_ids, min=0))

    # Generate response
    cont = model.generate(
        input_ids,
        images=image_tensors,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=TEMPERATURE,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
    change_caption = text_outputs[0]
    return change_caption
