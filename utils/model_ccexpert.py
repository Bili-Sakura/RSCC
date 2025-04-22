import sys

sys.path.append("./")
import os
from libs.llava.model.builder import load_pretrained_model
from libs.llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from libs.llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IGNORE_INDEX,
)
from libs.llava.conversation import conv_templates, SeparatorStyle
import torch
import warnings
from utils.constants import DEFAULT_DEVICE, PATH_TO_MODEL_FOLDER

warnings.filterwarnings("ignore")


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
            "vocab_size": 152064 if "7b" in model_path.lower() else 151936,
            "mm_vision_tower": os.path.join(
                PATH_TO_MODEL_FOLDER, "google/siglip-so400m-patch14-384"
            ),
        },
        "device_map": {"": device},
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
    model_hub[model_id]["model_id"] = model_id
    model_hub[model_id]["model"] = model
    model_hub[model_id]["tokenizer"] = tokenizer
    model_hub[model_id]["image_processor"] = image_processor
    return model_hub
