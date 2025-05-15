import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from decord import VideoReader, cpu
from qwen_vl_utils import process_vision_info
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage, TextChunk, ImageChunk
from mistral_common.protocol.instruct.request import ChatCompletionRequest

import transformers
import PIL
from PIL import Image
import torch
import sys
from copy import deepcopy
from typing import List

sys.path.append("./")
from inference_with_cd.correction_decoding_utils.vcd_add_noise import (
    add_diffusion_noise,
)
from inference_with_cd.correction_decoding_utils.vcd_generate import (
    generate as vcd_generate,
)
from inference_with_cd.correction_decoding_utils.deco_generate import (
    generate as deco_generate,
)
from utils.model_hub import model_loader
from utils.constants import (
    DEFAULT_DEVICE,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    DECO_ALPHA,
    DECO_THRESHOLD_TOP_P,
    DECO_THRESHOLD_TOP_K,
    DECO_EARLY_EXIT_LAYERS_DEFAULT,
    DECO_EARLY_EXIT_LAYERS_DICT,
    VCD_ALPHA,
    VCD_BETA,
    DOLA_LAYER,
    VCD_NOISE_STEP,
)

vanilla_generate = transformers.generation.utils.GenerationMixin.generate


def batch_inference_with_cd(
    model_id: str,
    text_prompt: List[str],
    pre_image_list: List[PIL.Image.Image],
    post_image_list: List[PIL.Image.Image],
    device=DEFAULT_DEVICE,
    model_hub=None,
    cd_type: str = "DoLa",  # Literal["DoLa", "VCD", "DeCo"]
):
    if model_hub is not None:
        try:
            loading_model_id = model_hub[model_id]["model_id"]
            model = model_hub[model_id]["model"]
            processor = model_hub[model_id]["processor"]
            tokenizer = model_hub[model_id]["tokenizer"]
            image_processor = model_hub[model_id]["image_processor"]
        except:
            print(f"Error Loading {model_id} from model_hub.")
    else:
        loading_model_id, model, processor, tokenizer, image_processor = model_loader(
            model_id, device
        )
    if loading_model_id != model_id:
        print(
            f"Inference model_id is {model_id} while loading model checkpoint is {loading_model_id}"
        )
        return ""

    elif (
        model_id == "Qwen/Qwen2-VL-7B-Instruct"
        or model_id == "Qwen/Qwen2-VL-72B-Instruct"
        or model_id == "Qwen/Qwen2.5-VL-3B-Instruct"
        or model_id == "Qwen/Qwen2.5-VL-7B-Instruct"
        or model_id == "Qwen/Qwen2.5-VL-32B-Instruct"
        or model_id == "Qwen/Qwen2.5-VL-72B-Instruct"
    ):
        # Create messages for each batch item
        messages = []
        for pre_img, post_img, curr_prompt in zip(
            pre_image_list, post_image_list, text_prompt
        ):
            msg = {
                "role": "user",
                "content": [
                    {"type": "image", "image": pre_img},
                    {"type": "image", "image": post_img},
                    {"type": "text", "text": curr_prompt},
                ],
            }
            messages.append(msg)

        # Process batched messages
        texts = [
            processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            )
            for msg in messages
        ]
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(
            text=texts,
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device, torch.bfloat16)
        # Generate captions for the input image pair

        if cd_type == "DoLa":
            transformers.generation.utils.GenerationMixin.generate = vanilla_generate
            # use default generate
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                dola_layers=DOLA_LAYER,
            )
        elif cd_type == "DeCo":
            lm_head = model.lm_head
            norm = model.model.norm
            if model_id == "Qwen/Qwen2.5-VL-3B-Instruct":
                DECO_EARLY_EXIT_LAYERS = DECO_EARLY_EXIT_LAYERS_DICT["QWEN25VL3B"]
            elif (
                model_id == "Qwen/Qwen2.5-VL-7B-Instruct"
                or model_id == "Qwen/Qwen2-VL-7B-Instruct"
            ):
                DECO_EARLY_EXIT_LAYERS = DECO_EARLY_EXIT_LAYERS_DICT["QWEN25VL7B"]
            elif model_id == "Qwen/Qwen2.5-VL-32B-Instruct":
                DECO_EARLY_EXIT_LAYERS = DECO_EARLY_EXIT_LAYERS_DICT["QWEN25VL32B"]
            elif (
                model_id == "Qwen/Qwen2.5-VL-72B-Instruct"
                or model_id == "Qwen/Qwen2-VL-72B-Instruct"
            ):
                DECO_EARLY_EXIT_LAYERS = DECO_EARLY_EXIT_LAYERS_DICT["QWEN25VL72B"]
            else:
                DECO_EARLY_EXIT_LAYERS = DECO_EARLY_EXIT_LAYERS_DEFAULT
            transformers.generation.utils.GenerationMixin.generate = deco_generate
            generated_ids = model.generate(
                **inputs,
                do_sample=False,
                alpha=DECO_ALPHA,
                threshold_top_p=DECO_THRESHOLD_TOP_P,
                threshold_top_k=DECO_THRESHOLD_TOP_K,
                early_exit_layers=DECO_EARLY_EXIT_LAYERS,
                lm_head=lm_head,
                norm=norm,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            # set back to default if needed
            transformers.generation.utils.GenerationMixin.generate = vanilla_generate

        elif cd_type == "VCD":
            transformers.generation.utils.GenerationMixin.generate = vcd_generate
            raw_img = deepcopy(inputs["pixel_values"])
            noise_img = add_diffusion_noise(raw_img, noise_step=VCD_NOISE_STEP)
            generated_ids = model.generate(
                **inputs,
                noise_img=noise_img,
                do_sample=False,
                cd_alpha=VCD_ALPHA,
                cd_beta=VCD_BETA,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            # set back to default if needed
            transformers.generation.utils.GenerationMixin.generate = vanilla_generate
        else:
            print(
                f"Unknow correction decoding type as {cd_type}. Convert to DoLa as default."
            )
            transformers.generation.utils.GenerationMixin.generate = vanilla_generate
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                dola_layers=DOLA_LAYER,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        captions = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    return captions


def inference_with_cd(
    model_id: str,
    text_prompt: str,
    pre_image: PIL.Image.Image,
    post_image: PIL.Image.Image,
    device=DEFAULT_DEVICE,
    model_hub=None,
    cd_type: str = "DoLa",  # Literal["DoLa", "VCD", "DeCo"]
) -> str:
    if model_hub is not None:
        try:
            loading_model_id = model_hub[model_id]["model_id"]
            model = model_hub[model_id]["model"]
            processor = model_hub[model_id]["processor"]
            tokenizer = model_hub[model_id]["tokenizer"]
            image_processor = model_hub[model_id]["image_processor"]
        except:
            print(f"Error Loading {model_id} from model_hub.")
    else:
        loading_model_id, model, processor, tokenizer, image_processor = model_loader(
            model_id, device
        )
    if loading_model_id != model_id:
        print(
            f"Inference model_id is {model_id} while loading model checkpoint is {loading_model_id}"
        )
        return ""

    if model_id == "moonshotai/Kimi-VL-A3B-Instruct":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": text_prompt,
                    },
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        inputs = processor(
            images=[pre_image, post_image],
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        if cd_type != "DoLa":
            print(
                f"""Unknow correction decoding type as {cd_type} for {model_id}. Convert to DoLa as default.
                  We will implement it later. Try correction decoding with Qwen2.5-VL series first"""
            )
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            dola_layers=DOLA_LAYER,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        change_caption = response[0]

    elif (
        model_id == "Qwen/Qwen2-VL-7B-Instruct"
        or model_id == "Qwen/Qwen2-VL-72B-Instruct"
        or model_id == "Qwen/Qwen2.5-VL-3B-Instruct"
        or model_id == "Qwen/Qwen2.5-VL-7B-Instruct"
        or model_id == "Qwen/Qwen2.5-VL-32B-Instruct"
        or model_id == "Qwen/Qwen2.5-VL-72B-Instruct"
    ):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pre_image},
                    {"type": "image", "image": post_image},
                    {
                        "type": "text",
                        "text": text_prompt,
                    },
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, padding=True, return_tensors="pt"
        ).to(device, torch.bfloat16)
        # Generate captions for the input image pair

        if cd_type == "DoLa":
            transformers.generation.utils.GenerationMixin.generate = vanilla_generate
            # use default generate
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                dola_layers=DOLA_LAYER,
            )
        elif cd_type == "DeCo":
            lm_head = model.lm_head
            norm = model.model.norm
            if model_id == "Qwen/Qwen2.5-VL-3B-Instruct":
                DECO_EARLY_EXIT_LAYERS = DECO_EARLY_EXIT_LAYERS_DICT["QWEN25VL3B"]
            elif (
                model_id == "Qwen/Qwen2.5-VL-7B-Instruct"
                or model_id == "Qwen/Qwen2-VL-7B-Instruct"
            ):
                DECO_EARLY_EXIT_LAYERS = DECO_EARLY_EXIT_LAYERS_DICT["QWEN25VL7B"]
            elif model_id == "Qwen/Qwen2.5-VL-32B-Instruct":
                DECO_EARLY_EXIT_LAYERS = DECO_EARLY_EXIT_LAYERS_DICT["QWEN25VL32B"]
            elif (
                model_id == "Qwen/Qwen2.5-VL-72B-Instruct"
                or model_id == "Qwen/Qwen2-VL-72B-Instruct"
            ):
                DECO_EARLY_EXIT_LAYERS = DECO_EARLY_EXIT_LAYERS_DICT["QWEN25VL72B"]
            else:
                DECO_EARLY_EXIT_LAYERS = DECO_EARLY_EXIT_LAYERS_DEFAULT
            transformers.generation.utils.GenerationMixin.generate = deco_generate
            generated_ids = model.generate(
                **inputs,
                do_sample=False,
                alpha=DECO_ALPHA,
                threshold_top_p=DECO_THRESHOLD_TOP_P,
                threshold_top_k=DECO_THRESHOLD_TOP_K,
                early_exit_layers=DECO_EARLY_EXIT_LAYERS,
                lm_head=lm_head,
                norm=norm,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            # set back to default if needed
            transformers.generation.utils.GenerationMixin.generate = vanilla_generate

        elif cd_type == "VCD":
            transformers.generation.utils.GenerationMixin.generate = vcd_generate
            raw_img = deepcopy(inputs["pixel_values"])
            noise_img = add_diffusion_noise(raw_img, noise_step=VCD_NOISE_STEP)
            generated_ids = model.generate(
                **inputs,
                noise_img=noise_img,
                do_sample=False,
                cd_alpha=VCD_ALPHA,
                cd_beta=VCD_BETA,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            # set back to default if needed
            transformers.generation.utils.GenerationMixin.generate = vanilla_generate
        else:
            print(
                f"Unknow correction decoding type as {cd_type}. Convert to DoLa as default."
            )
            transformers.generation.utils.GenerationMixin.generate = vanilla_generate
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                dola_layers=DOLA_LAYER,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        captions = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        change_caption = captions[0]
    elif model_id == "microsoft/Phi-4-multimodal-instruct":
        messages = [
            {
                "role": "user",
                "content": "<|image_1|>" + "<|image_2|>" + text_prompt,
            },
        ]
        prompt = processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(prompt, [pre_image, post_image], return_tensors="pt").to(
            device
        )
        if cd_type != "DoLa":
            print(
                f"""Unknow correction decoding type as {cd_type} for {model_id}. Convert to DoLa as default.
                    We will implement it later. Try correction decoding with Qwen2.5-VL series first"""
            )
        generated_ids = model.generate(
            **inputs,
            num_logits_to_keep=1,
            max_new_tokens=MAX_NEW_TOKENS,
            dola_layers=DOLA_LAYER,
        )
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
        response = processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        change_caption = response[0]

    elif model_id == "Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5":

        image_list = []
        image_sizes = []
        image_list.append(
            image_processor([pre_image], image_aspect_ratio="anyres")[
                "pixel_values"
            ].to(device, torch.bfloat16)
        )
        image_sizes.append(pre_image.size)
        image_list.append(
            image_processor([post_image], image_aspect_ratio="anyres")[
                "pixel_values"
            ].to(device, torch.bfloat16)
        )
        image_sizes.append(post_image.size)
        inputs = {"pixel_values": [image_list]}
        prompt = f"<|user|>\n{text_prompt}<|end|>\n<|assistant|>\n"
        language_inputs = tokenizer([prompt], return_tensors="pt")
        inputs.update(language_inputs)
        for name, value in inputs.items():
            if isinstance(value, torch.Tensor):
                inputs[name] = value.to(device=device)
                # print(f"Input '{name}' is a tensor of type {value.dtype}")
        if cd_type != "DoLa":
            print(
                f"""Unknow correction decoding type as {cd_type} for {model_id}. Convert to DoLa as default.
                  We will implement it later. Try correction decoding with Qwen2.5-VL series first"""
            )
        generated_text = model.generate(
            **inputs,
            image_size=[image_sizes],
            pad_token_id=tokenizer.pad_token_id,
            # temperature=TEMPERATURE,
            max_new_tokens=MAX_NEW_TOKENS,
            dola_layers=DOLA_LAYER,
            # do_sample=True,
        )
        change_caption = tokenizer.decode(
            generated_text[0], skip_special_tokens=True
        ).split("<|end|>")[0]

    elif (
        model_id == "OpenGVLab/InternVL3-1B"
        or model_id == "OpenGVLab/InternVL3-2B"
        or model_id == "OpenGVLab/InternVL3-8B"
        or model_id == "OpenGVLab/InternVL3-14B"
        or model_id == "OpenGVLab/InternVL3-38B"
        or model_id == "OpenGVLab/InternVL3-78B"
    ):

        def build_transform(input_size):
            IMAGENET_MEAN = (0.485, 0.456, 0.406)
            IMAGENET_STD = (0.229, 0.224, 0.225)
            MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
            transform = T.Compose(
                [
                    T.Lambda(
                        lambda img: img.convert("RGB") if img.mode != "RGB" else img
                    ),
                    T.Resize(
                        (input_size, input_size),
                        interpolation=InterpolationMode.BICUBIC,
                    ),
                    T.ToTensor(),
                    T.Normalize(mean=MEAN, std=STD),
                ]
            )
            return transform

        def find_closest_aspect_ratio(
            aspect_ratio, target_ratios, width, height, image_size
        ):
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

        def dynamic_preprocess(
            image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
        ):
            orig_width, orig_height = image.size
            aspect_ratio = orig_width / orig_height

            # calculate the existing image aspect ratio
            target_ratios = set(
                (i, j)
                for n in range(min_num, max_num + 1)
                for i in range(1, n + 1)
                for j in range(1, n + 1)
                if i * j <= max_num and i * j >= min_num
            )
            target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

            # find the closest aspect ratio to the target
            target_aspect_ratio = find_closest_aspect_ratio(
                aspect_ratio, target_ratios, orig_width, orig_height, image_size
            )

            # calculate the target width and height
            target_width = image_size * target_aspect_ratio[0]
            target_height = image_size * target_aspect_ratio[1]
            blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

            # resize the image
            resized_img = image.resize((target_width, target_height))
            processed_images = []
            for i in range(blocks):
                box = (
                    (i % (target_width // image_size)) * image_size,
                    (i // (target_width // image_size)) * image_size,
                    ((i % (target_width // image_size)) + 1) * image_size,
                    ((i // (target_width // image_size)) + 1) * image_size,
                )
                # split the image
                split_img = resized_img.crop(box)
                processed_images.append(split_img)
            assert len(processed_images) == blocks
            if use_thumbnail and len(processed_images) != 1:
                thumbnail_img = image.resize((image_size, image_size))
                processed_images.append(thumbnail_img)
            return processed_images

        def load_image(image_file, input_size=448, max_num=12):
            if isinstance(image_file, PIL.Image.Image):
                image = image_file.convert("RGB")
            else:
                # Load the image from file
                image = Image.open(image_file).convert("RGB")
            transform = build_transform(input_size=input_size)
            images = dynamic_preprocess(
                image, image_size=input_size, use_thumbnail=True, max_num=max_num
            )
            pixel_values = [transform(image) for image in images]
            pixel_values = torch.stack(pixel_values)
            return pixel_values

        pixel_values1 = load_image(pre_image).to(device, torch.bfloat16)
        pixel_values2 = load_image(post_image).to(device, torch.bfloat16)
        pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0).to(
            device, torch.bfloat16
        )
        num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]
        question = f"<image>\n<image>\n{text_prompt}"

        if cd_type == "DoLa":
            transformers.generation.utils.GenerationMixin.generate = vanilla_generate
            # use default generate
            generation_config = dict(
                max_new_tokens=MAX_NEW_TOKENS,
                dola_layers=DOLA_LAYER,
            )
            change_caption = model.chat(
                tokenizer,
                pixel_values,
                question,
                generation_config,
                num_patches_list=num_patches_list,
            )
        elif cd_type == "DeCo":
            lm_head = model.language_model.lm_head
            norm = model.language_model.model.norm
            if model_id == "OpenGVLab/InternVL3-1B":
                DECO_EARLY_EXIT_LAYERS = DECO_EARLY_EXIT_LAYERS_DICT["INTERNVL3_1B"]
            elif model_id == "OpenGVLab/InternVL3-2B":
                DECO_EARLY_EXIT_LAYERS = DECO_EARLY_EXIT_LAYERS_DICT["INTERNVL3_2B"]
            elif model_id == "OpenGVLab/InternVL3-8B":
                DECO_EARLY_EXIT_LAYERS = DECO_EARLY_EXIT_LAYERS_DICT["INTERNVL3_8B"]
            elif model_id == "OpenGVLab/InternVL3-14B":
                DECO_EARLY_EXIT_LAYERS = DECO_EARLY_EXIT_LAYERS_DICT["INTERNVL3_14B"]
            elif model_id == "OpenGVLab/InternVL3-38B":
                DECO_EARLY_EXIT_LAYERS = DECO_EARLY_EXIT_LAYERS_DICT["INTERNVL3_38B"]
            elif model_id == "OpenGVLab/InternVL3-78B":
                DECO_EARLY_EXIT_LAYERS = DECO_EARLY_EXIT_LAYERS_DICT["INTERNVL3_78B"]
            else:
                DECO_EARLY_EXIT_LAYERS = DECO_EARLY_EXIT_LAYERS_DEFAULT
            transformers.generation.utils.GenerationMixin.generate = deco_generate
            generation_config = dict(
                do_sample=False,
                alpha=DECO_ALPHA,
                threshold_top_p=DECO_THRESHOLD_TOP_P,
                threshold_top_k=DECO_THRESHOLD_TOP_K,
                early_exit_layers=DECO_EARLY_EXIT_LAYERS,
                lm_head=lm_head,
                norm=norm,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            change_caption = model.chat(
                tokenizer,
                pixel_values,
                question,
                generation_config,
                num_patches_list=num_patches_list,
            )
            # set back to default if needed
            transformers.generation.utils.GenerationMixin.generate = vanilla_generate

        elif cd_type == "VCD":
            transformers.generation.utils.GenerationMixin.generate = vcd_generate
            raw_img = deepcopy(pixel_values)
            noise_img = add_diffusion_noise(raw_img, noise_step=VCD_NOISE_STEP)
            generation_config = dict(
                noise_img=noise_img,
                do_sample=False,
                cd_alpha=VCD_ALPHA,
                cd_beta=VCD_BETA,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            change_caption = model.chat(
                tokenizer,
                pixel_values,
                question,
                generation_config,
                num_patches_list=num_patches_list,
            )
            # set back to default if needed
            transformers.generation.utils.GenerationMixin.generate = vanilla_generate
        else:
            print(
                f"Unknow correction decoding type as {cd_type}. Convert to DoLa as default."
            )
            transformers.generation.utils.GenerationMixin.generate = vanilla_generate
            generation_config = dict(
                max_new_tokens=MAX_NEW_TOKENS,
            )
            change_caption = model.chat(
                tokenizer,
                pixel_values,
                question,
                generation_config,
                num_patches_list=num_patches_list,
            )

    elif model_id == "llava-hf/llava-interleave-qwen-7b-hf":

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": text_prompt,
                    },
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = processor(
            images=[pre_image, post_image], text=prompt, return_tensors="pt"
        ).to(device, torch.bfloat16)

        if cd_type != "DoLa":
            print(
                f"""Unknow correction decoding type as {cd_type} for {model_id}. Convert to DoLa as default.
                  We will implement it later. Try correction decoding with Qwen2.5-VL series first"""
            )
        output = model.generate(
            **inputs,
            pad_token_id=processor.tokenizer.eos_token_id,
            max_new_tokens=MAX_NEW_TOKENS,
            dola_layers=DOLA_LAYER,
            # do_sample=True,
            # temperature=TEMPERATURE,
        )

        change_caption = processor.decode(output[0], skip_special_tokens=True).split(
            "assistant\n"
        )[1]
    elif model_id == "llava-hf/llava-onevision-qwen2-7b-ov-hf":
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": text_prompt,
                    },
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = processor(
            images=[pre_image, post_image], text=prompt, return_tensors="pt"
        ).to(device, torch.float16)
        if cd_type != "DoLa":
            print(
                f"""Unknow correction decoding type as {cd_type} for {model_id}. Convert to DoLa as default.
                  We will implement it later. Try correction decoding with Qwen2.5-VL series first"""
            )
        output = model.generate(
            **inputs,
            pad_token_id=processor.tokenizer.eos_token_id,
            max_new_tokens=MAX_NEW_TOKENS,
            dola_layers=DOLA_LAYER,
            # do_sample=True,
            # temperature=TEMPERATURE,
        )

        change_caption = processor.decode(output[0], skip_special_tokens=True).split(
            "assistant\n"
        )[1]
    elif model_id == "mistralai/Pixtral-12B-2409":
        completion_request = ChatCompletionRequest(
            messages=[
                UserMessage(
                    content=[
                        ImageChunk(image=pre_image),
                        ImageChunk(image=post_image),
                        TextChunk(text=text_prompt),
                    ]
                )
            ]
        )
        encoded = tokenizer.encode_chat_completion(completion_request)
        images = encoded.images
        tokens = encoded.tokens

        if cd_type != "DoLa":
            print(
                f"""Unknow correction decoding type as {cd_type} for {model_id}. Convert to DoLa as default.
                  We will implement it later. Try correction decoding with Qwen2.5-VL series first"""
            )
        out_tokens, _ = generate(
            [tokens],
            model,
            images=[images],
            max_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
            dola_layers=DOLA_LAYER,
        )
        change_caption = tokenizer.decode(out_tokens[0])

    return change_caption
