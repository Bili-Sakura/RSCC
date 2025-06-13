import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from decord import VideoReader, cpu
from qwen_vl_utils import process_vision_info
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage, TextChunk, ImageChunk
from mistral_common.protocol.instruct.request import ChatCompletionRequest

import PIL
from PIL import Image
import torch
import sys

sys.path.append("./")
from utils.model_hub import model_loader
from utils.constants import DEFAULT_DEVICE, MAX_NEW_TOKENS, TEMPERATURE


def inference_naive(
    model_id: str,
    text_prompt: str,
    pre_image: PIL.Image.Image,
    post_image: PIL.Image.Image,
    device=DEFAULT_DEVICE,
    model_hub=None,
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
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            # temperature=TEMPERATURE,
            # do_sample=False,
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
        or model_id == "Qwen/Qwen2.5-VL-7B-Instruct"
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
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            # temperature=TEMPERATURE
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
        generate_ids = model.generate(
            **inputs,
            num_logits_to_keep=1,
            max_new_tokens=MAX_NEW_TOKENS,
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
        generated_text = model.generate(
            **inputs,
            image_size=[image_sizes],
            pad_token_id=tokenizer.pad_token_id,
            # temperature=TEMPERATURE,
            max_new_tokens=MAX_NEW_TOKENS,
            # do_sample=True,
        )
        change_caption = tokenizer.decode(
            generated_text[0], skip_special_tokens=True
        ).split("<|end|>")[0]

    elif model_id == "OpenGVLab/InternVL3-8B" or model_id == "OpenGVLab/InternVL3-78B":

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

        generation_config = dict(
            max_new_tokens=MAX_NEW_TOKENS,
            # do_sample=True,
            # temperature=TEMPERATURE
        )
        pixel_values1 = load_image(pre_image).to(device, torch.bfloat16)
        pixel_values2 = load_image(post_image).to(device, torch.bfloat16)
        pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
        num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]
        question = f"<image>\n<image>\n{text_prompt}"
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

        output = model.generate(
            **inputs,
            pad_token_id=processor.tokenizer.eos_token_id,
            max_new_tokens=MAX_NEW_TOKENS,
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

        output = model.generate(
            **inputs,
            pad_token_id=processor.tokenizer.eos_token_id,
            max_new_tokens=MAX_NEW_TOKENS,
            # do_sample=True,
            # temperature=TEMPERATURE,
        )

        change_caption = processor.decode(output[0], skip_special_tokens=True).split(
            "assistant\n"
        )[1]
    elif model_id == "Skywork/Skywork-R1V-38B":
        generation_config = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=0.6, top_p=0.95, repetition_penalty=1.05)
        pixel_values1 = load_image(pre_image).to(device, torch.bfloat16)
        pixel_values2 = load_image(post_image).to(device, torch.bfloat16)
        pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
        num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]
        question = f"<image>\n<image>\n{text_prompt}"
        change_caption = model.chat(
            tokenizer,
            pixel_values,
            question,
            generation_config,
            num_patches_list=num_patches_list,
        )
    elif model_id == "akshaydudhane/EarthDial_4B_RGB":
        transform = build_transform(input_size=448)
        pixel_values = transform(pre_image).unsqueeze(0).to(device, torch.bfloat16)
        generation_config = {
            "num_beams": 5,
            "max_new_tokens": MAX_NEW_TOKENS,
            "min_new_tokens": 1,
            "do_sample": False,
            "temperature": 0.0,
        }
        question = text_prompt
        change_caption = model.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=question,
            generation_config=generation_config,
            verbose=True,
        )
    elif model_id == "Efficient-Large-Model/NVILA-8B":
        from llava import Image as LlavaImage

        prompt = [LlavaImage(pre_image), LlavaImage(post_image), text_prompt]
        change_caption = model.generate_content(prompt)
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
        out_tokens, _ = generate(
            [tokens],
            model,
            images=[images],
            max_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
        )
        change_caption = tokenizer.decode(out_tokens[0])

    return change_caption
