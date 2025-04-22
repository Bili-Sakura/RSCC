import sys

sys.path.append("./libs")
sys.path.append("./")
import os
import json
import re
import glob
from tqdm import tqdm
from shapely import wkt
from PIL import Image, ImageDraw
import torch
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info
from utils.logger import UTC8Formatter
import logging
from utils.constants import (
    DEFAULT_DEVICE,
    PATH_TO_DATASET_FOLDER,
    PART_MAPPING,
    PROMPT_TEMPLATES,
)

IMAGE_DIR_TEMPLATE = os.path.join(
    PATH_TO_DATASET_FOLDER, "xbd_subset/{event_name}/images/"
)


# Configure logging
def setup_logging(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    for handler in logging.getLogger().handlers:
        handler.setFormatter(
            UTC8Formatter(
                fmt="%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    return logging.getLogger(__name__)


def get_image_pairs():
    """Collect all image pairs from the dataset"""
    pairs = []
    for part_num in range(1, 5):
        pattern = os.path.join(
            IMAGE_DIR_TEMPLATE.format(event_name="*"),
            f"*_pre_disaster_part{part_num}.png",
        )
        pairs.extend(glob.glob(pattern))
    return sorted(pairs)


def generate_prompts(disaster_type, damage_counts):
    """Generate all prompt variations"""
    return {
        "zero-shot": PROMPT_TEMPLATES["naive"].strip(),
        "textual_prompt": PROMPT_TEMPLATES["textual"]
        .format(disaster_type=disaster_type, number=damage_counts)
        .strip(),
        "visual_prompt": PROMPT_TEMPLATES["visual"]
        .format(disaster_type=disaster_type, number=damage_counts)
        .strip(),
    }


# Load processed items from output file
def load_processed_items(output_file_path):
    processed_items = set()
    if os.path.exists(output_file_path):
        with open(output_file_path, "r") as infile:
            for line in infile:
                try:
                    entry = json.loads(line)
                    processed_items.add((entry["pre_image"], entry["post_image"]))
                except json.JSONDecodeError:
                    logger.error(f"Error decoding JSON line: {line}")
    return processed_items


# Batch processing utility
def create_batches(items, batch_size):
    return [
        items[i : i + batch_size]
        for i in range(0, len(items), batch_size)
        if len(items[i : i + batch_size]) == batch_size
    ] + ([items[-(len(items) % batch_size) :]] if len(items) % batch_size != 0 else [])


def classify_disaster_type(disaster_name):
    """Map disaster name to type category"""
    disaster_map = {
        "volcano eruption": ["guatemala-volcano", "lower-puna-volcano"],
        "wildfire": [
            "pinery-bushfire",
            "portugal-wildfire",
            "santa-rosa-wildfire",
            "socal-fire",
            "woolsey-fire",
        ],
        "storm": [
            "hurricane-florence",
            "joplin-tornado",
            "moore-tornado",
            "hurricane-matthew",
            "tuscaloosa-tornado",
        ],
        "earthquake": ["mexico-earthquake"],
        "flooding": [
            "hurricane-michael",
            "hurricane-harvey",
            "midwest-flooding",
            "nepal-flooding",
        ],
        "tsunami": ["palu-tsunami", "sunda-tsunami"],
    }
    for category, names in disaster_map.items():
        if disaster_name in names:
            return f"{category}"
    logging.warning(f"Disaster Event - {disaster_name} is set as 'unknown'")
    return "unknown"


def process_label_file(label_path, part_name):
    """Process label JSON file to extract disaster info and damage counts"""
    with open(label_path, "r") as f:
        data = json.load(f)

    disaster_type = classify_disaster_type(data["metadata"]["disaster"])
    damage_counts, polygons = count_damages(
        data["features"]["xy"], PART_MAPPING[part_name]
    )

    return disaster_type, damage_counts, polygons


def process_visual_prompt(image, polygons, part_name):
    """Add damage polygon overlays to post-disaster image"""
    # Create copy to avoid modifying original image
    annotated_img = image.copy()
    draw = ImageDraw.Draw(annotated_img, "RGBA")

    # Get part bounds from mapping
    bounds = PART_MAPPING[part_name]
    x_offset = bounds[0]
    y_offset = bounds[1]

    damage_colors = {
        "no-damage": (0, 255, 0, 100),  # Green
        "minor-damage": (0, 0, 255, 125),  # Blue
        "major-damage": (255, 69, 0, 125),  # Orange
        "destroyed": (255, 0, 0, 125),  # Red
        "un-classified": (255, 255, 255, 125),  # White
    }

    for damage, polygon in polygons:
        try:
            # Convert polygon coordinates to image space
            x, y = polygon.exterior.coords.xy
            x = [coord - x_offset for coord in x]
            y = [coord - y_offset for coord in y]
            coords = list(zip(x, y))
            # Draw boundary instead of fill
            draw.line(coords + [coords[0]], fill=damage_colors[damage], width=2)
        except Exception as e:
            logging.error(f"Error drawing polygon: {str(e)}")

    del draw  # Important for memory management
    return annotated_img


def count_damages(coordinates, img_bounds):
    """Count damage types within specified image bounds"""
    polygons = []
    counts = {
        "no-damage": 0,
        "minor-damage": 0,
        "major-damage": 0,
        "destroyed": 0,
        "un-classified": 0,
        "all": 0,
    }
    bounds_poly = wkt.loads(
        f"POLYGON(({img_bounds[0]} {img_bounds[1]}, {img_bounds[2]} {img_bounds[1]}, "
        f"{img_bounds[2]} {img_bounds[3]}, {img_bounds[0]} {img_bounds[3]}, "
        f"{img_bounds[0]} {img_bounds[1]}))"
    )

    for coord in coordinates:
        damage = coord["properties"].get("subtype")
        poly = wkt.loads(coord["wkt"])
        if poly.intersects(bounds_poly):
            counts[damage] += 1
            polygons.append((damage, poly))  # Store damage type and polygon
    counts["all"] = sum(counts.values())
    return counts, polygons


# Process batch inference
def process_batch_inference(model, processor, messages):
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts, images=image_inputs, padding=True, return_tensors="pt"
    ).to("cuda")
    generated_ids = model.generate(**inputs, max_new_tokens=256, temperature=0.001)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


# Main execution logic
def main():
    # Constants
    log_file = "./logs/xbd_subset_tpvp.log"
    output_file_path = "./output/xbd_subset_qwen25vl72b_tpvp.jsonl"
    # image_dir = "/home/datasets/xbd_subset/{event_name}/images/"
    batch_size = 6
    MODEL_ID = "Qwen/Qwen2.5-VL-72B-Instruct"
    model_name = "/home/models/Qwen/Qwen2.5-VL-72B-Instruct"

    # Setup logging
    global logger
    logger = setup_logging(log_file)

    # Load processed items
    processed_items = load_processed_items(output_file_path)

    # Prepare image pairs
    image_pairs = get_image_pairs()

    # Create batches
    batches = create_batches(image_pairs, batch_size)

    # Load model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)
    processor.tokenizer.padding_side = "left"

    # Iterate through batches
    all_outputs = []
    for batch_idx, batch in enumerate(tqdm(batches, desc="Processing batches")):
        if batch_idx < len(processed_items) // batch_size:
            continue  # Skip already processed batches

        logger.info(f"Processing batch {batch_idx}/{len(batches)}.")
        messages = []

        for pre_event_img_path in batch:
            post_event_img_path = pre_event_img_path.replace(
                "_pre_disaster_part", "_post_disaster_part"
            )
            if (pre_event_img_path, post_event_img_path) in processed_items:
                logger.info(
                    f"Skipping already processed image pair: {pre_event_img_path} and {post_event_img_path}"
                )
                continue

            label_json_path = (
                re.sub(r"_part\d+", "", post_event_img_path)
                .replace("images", "labels")
                .replace("_pre_disaster_part", "_post_disaster")
                .replace(".png", ".json")
            )

            if not all(
                os.path.exists(path) for path in [post_event_img_path, label_json_path]
            ):
                logger.warning(
                    f"Missing files for image pair: {pre_event_img_path}, {post_event_img_path}, or {label_json_path}"
                )
                continue

            part_name = (
                os.path.basename(pre_event_img_path).split("_")[-1].replace(".png", "")
            )

            disaster_type, damage_counts, post_polygons = process_label_file(
                label_json_path, part_name
            )

            pre_image = Image.open(pre_event_img_path)
            post_image = Image.open(post_event_img_path)
            prompts = generate_prompts(disaster_type, damage_counts)
            for prompt_type, prompt_text in prompts.items():
                if prompt_type == "visual_prompt":
                    pre_image_input = pre_image
                    post_image_input = process_visual_prompt(
                        post_image, post_polygons, part_name
                    )
                else:
                    continue
                    pre_image_input = pre_image
                    post_image_input = post_image

            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pre_image_input},
                        {"type": "image", "image": post_image_input},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            messages.append(message)

            if not messages:
                logger.warning("No messages to process in this batch.")
                continue

            # Perform batch inference
            output_texts = process_batch_inference(model, processor, messages)

            # Save outputs
            for idx, output in enumerate(output_texts):
                pre_event_img_path = batch[idx]
                post_event_img_path = pre_event_img_path.replace(
                    "_pre_disaster_part", "_post_disaster_part"
                )
                all_outputs.append(
                    {
                        "model_id": MODEL_ID,
                        "inference_type": "visual_prompt",
                        "pre_image": pre_event_img_path,
                        "post_image": post_event_img_path,
                        "change_caption": output,
                    }
                )

            with open(output_file_path, mode="a") as outfile:
                for entry in all_outputs:
                    json.dump(entry, outfile)
                    outfile.write("\n")
                    outfile.flush()
            all_outputs.clear()


if __name__ == "__main__":
    main()
