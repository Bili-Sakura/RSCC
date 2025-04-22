from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    AutoModelForVision2Seq,
    AutoTokenizer,
    AutoModel,
    AutoImageProcessor,
    StoppingCriteria,
    LlavaOnevisionForConditionalGeneration,
    LlavaForConditionalGeneration,
    AutoConfig,
    AutoModelForCausalLM,
)
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from decord import VideoReader, cpu
from qwen_vl_utils import process_vision_info
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage, TextChunk, ImageChunk
from mistral_common.protocol.instruct.request import ChatCompletionRequest

from typing import Literal, Union, List
import PIL
from PIL import Image
import torch
import os
import numpy as np

import sys

sys.path.append("./")
from utils.constants import (
    DEFAULT_DEVICE,
    PATH_TO_MODEL_FOLDER,
    DEFAULT_DEVICE_MAP,
    MODEL_LIST,
)


def model_loader(
    model_id: str,
    device: str = DEFAULT_DEVICE,
    device_map: dict = DEFAULT_DEVICE_MAP,
):
    if model_id not in MODEL_LIST:
        print(f"Invalid model_id: {model_id}. Supported models are: {MODEL_LIST}")
        return
    model_path = os.path.join(PATH_TO_MODEL_FOLDER, model_id)
    model = None
    processor = None
    tokenizer = None
    image_processor = None

    if model_id == "moonshotai/Kimi-VL-A3B-Instruct":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    elif model_id == "Qwen/Qwen2-VL-72B-Instruct":
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=device_map,
        )
        processor = AutoProcessor.from_pretrained(model_path)
    elif model_id == "Qwen/Qwen2-VL-7B-Instruct":
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(device)
        processor = AutoProcessor.from_pretrained(model_path)
    elif model_id == "Qwen/Qwen2.5-VL-72B-Instruct":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=device_map,
        )
        processor = AutoProcessor.from_pretrained(model_path)
    elif model_id == "Qwen/Qwen2.5-VL-7B-Instruct":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(device)
        processor = AutoProcessor.from_pretrained(model_path)
    elif model_id == "microsoft/Phi-4-multimodal-instruct":

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype="auto",
            _attn_implementation="flash_attention_2",
        ).to(device)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    elif model_id == "Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5":
        config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        config.vision_encoder_config.model_name = os.path.join(
            PATH_TO_MODEL_FOLDER, "google/siglip-so400m-patch14-384"
        )
        model = (
            AutoModelForVision2Seq.from_pretrained(
                model_path,
                config=config,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
            .eval()
            .to(device)
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
            legacy=False,
        )
        image_processor = AutoImageProcessor.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True
        )
        tokenizer = model.update_special_tokens(tokenizer)
        tokenizer.padding_side = "left"
    elif model_id == "OpenGVLab/InternVL3-8B":
        model = (
            AutoModel.from_pretrained(
                model_path,
                local_files_only=True,
                torch_dtype=torch.bfloat16,
                use_flash_attn=True,
                trust_remote_code=True,
            )
            .eval()
            .to(device)
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )
    elif model_id == "OpenGVLab/InternVL3-78B":
        model = AutoModel.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=device_map,
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )
    elif model_id == "llava-hf/llava-interleave-qwen-7b-hf":
        model = (
            LlavaForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
            )
            .eval()
            .to(device)
        )

        processor = AutoProcessor.from_pretrained(model_path)
    elif model_id == "llava-hf/llava-onevision-qwen2-7b-ov-hf":
        model = (
            LlavaOnevisionForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
            )
            .eval()
            .to(device)
        )

        processor = AutoProcessor.from_pretrained(model_path)
    elif model_id == "mistralai/Pixtral-12B-2409":
        tokenizer = MistralTokenizer.from_file(f"{model_path}/tekken.json")
        model = Transformer.from_folder(model_path).eval().to(device)
    else:
        print(f"Model {model_id} in MODEL_LIST has not been implement yet.")
    return model_id, model, processor, tokenizer, image_processor


def model_hub_loader(model_list: List, device_mapping: List):
    model_hub = {}
    for model_id, device in zip(model_list, device_mapping):
        loading_model_id, model, processor, tokenizer, image_processor = model_loader(
            model_id, device
        )
        model_hub[model_id] = {}  # Initialize the dictionary for this model_id
        model_hub[model_id]["model_id"] = loading_model_id
        model_hub[model_id]["model"] = model
        model_hub[model_id]["processor"] = processor
        model_hub[model_id]["tokenizer"] = tokenizer
        model_hub[model_id]["image_processor"] = image_processor
    return model_hub
