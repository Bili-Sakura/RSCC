import sys

sys.path.append("/home/sakura/projects/RSCC/libs")
import os
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
import torch
import warnings

warnings.filterwarnings("ignore")

DEFAULT_DEVICE = "cuda:0"
PATH_TO_MODEL_FOLDER = "/home/models"


def ccexpert_model_loader(
    model_id: str = "Meize0729/CCExpert_7b",
    device: str = DEFAULT_DEVICE,
):
    model_path = os.path.join(PATH_TO_MODEL_FOLDER, model_id)
    model = None
    tokenizer = None
    image_processor = None

    llava_model_args = {
        # "ignore_mismatched_sizes":True,
        "multimodal": True,
        "overwrite_config": {
            "vocab_size": 152064 if "7b" in model_path.lower() else 151936
        },
        "device_map": {"": DEFAULT_DEVICE},  # Using the variable here
    }
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        model_path,
        None,
        model_name="llava_qwen_cc",
        attn_implementation=None,
        **llava_model_args
    )

    model.eval()
    model = model.to(torch.bfloat16)

    model_hub = {}
    model_hub[model_id] = {}  # Initialize the dictionary for this model_id
    model_hub[model_id]["model"] = model
    model_hub[model_id]["tokenizer"] = tokenizer
    model_hub[model_id]["image_processor"] = image_processor
    return model_hub
