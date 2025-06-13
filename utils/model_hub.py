import math
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


def build_transform(input_size: int):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def split_model(model_path):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for _ in range(num_layer):
            device_map[f"language_model.model.layers.{layer_cnt}"] = i
            layer_cnt += 1
    device_map["vision_model"] = 0
    device_map["mlp1"] = 0
    device_map["language_model.model.tok_embeddings"] = 0
    device_map["language_model.model.embed_tokens"] = 0
    device_map["language_model.output"] = 0
    device_map["language_model.model.norm"] = 0
    device_map["language_model.model.rotary_emb"] = 0
    device_map["language_model.lm_head"] = 0
    device_map[f"language_model.model.layers.{num_layers - 1}"] = 0
    return device_map


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
    elif model_id == "Qwen/Qwen2-VL-7B-Instruct":
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(device)
        processor = AutoProcessor.from_pretrained(model_path)
    elif model_id == "Qwen/Qwen2-VL-72B-Instruct":
        print(f"Loading model {model_id} with device_map: {device_map}")
        print(f"Available devices: {torch.cuda.device_count()} GPUs")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=device_map,
        )
        processor = AutoProcessor.from_pretrained(model_path)
    elif model_id == "Qwen/Qwen2.5-VL-3B-Instruct":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(device)
        processor = AutoProcessor.from_pretrained(model_path)
    elif model_id == "Qwen/Qwen2.5-VL-7B-Instruct":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(device)
        processor = AutoProcessor.from_pretrained(model_path)
    elif model_id == "Qwen/Qwen2.5-VL-32B-Instruct":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(device)
        processor = AutoProcessor.from_pretrained(model_path)
    elif model_id == "Qwen/Qwen2.5-VL-72B-Instruct":
        print(f"Loading model {model_id} with device_map: {device_map}")
        print(f"Available devices: {torch.cuda.device_count()} GPUs")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=device_map,
        )
        processor = AutoProcessor.from_pretrained(model_path)
    elif model_id == "BiliSakura/RSCCM":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(device)
        processor = AutoProcessor.from_pretrained(model_path)
    elif model_id == "Skywork/Skywork-R1V-38B":
        device_map = split_model(model_path)
        model = (
            AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                load_in_8bit=False,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                trust_remote_code=True,
                device_map=device_map,
            )
            .eval()
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )
    elif model_id == "akshaydudhane/EarthDial_4B_RGB":
        model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                trust_remote_code=True,
            )
            .eval()
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )
        image_processor = build_transform
    elif model_id == "Efficient-Large-Model/NVILA-8B":
        import llava

        model = llava.load(model_path)
        tokenizer = model.tokenizer
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
    elif (
        model_id == "OpenGVLab/InternVL3-1B"
        or model_id == "OpenGVLab/InternVL3-2B"
        or model_id == "OpenGVLab/InternVL3-8B"
        or model_id == "OpenGVLab/InternVL3-14B"
        or model_id == "OpenGVLab/InternVL3-38B"
    ):
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
