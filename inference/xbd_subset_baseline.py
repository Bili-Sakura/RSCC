import sys
import torch
import re
import json
import glob
import logging
from tqdm import tqdm
from shapely import wkt
from PIL import Image, ImageDraw
import os
import argparse

sys.path.append("./libs")
sys.path.append("./")
from utils.logger import UTC8Formatter
from utils.constants import (
    DEFAULT_DEVICE,
    PATH_TO_DATASET_FOLDER,
    PART_MAPPING,
    PROMPT_TEMPLATES,
)


INFERENCE_MODEL_LIST = [
    "moonshotai/Kimi-VL-A3B-Instruct",
    "Qwen/Qwen2-VL-7B-Instruct",
    "Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5",
    # "microsoft/Phi-4-multimodal-instruct",
    "OpenGVLab/InternVL3-8B",
    # "llava-hf/llava-interleave-qwen-7b-hf",
    # "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    # "mistralai/Pixtral-12B-2409",
    # "Meize0729/CCExpert_7b",
    # "jirvin16/TEOChat",
]

IMAGE_DIR_TEMPLATE = os.path.join(
    PATH_TO_DATASET_FOLDER, "xbd_subset/{event_name}/images/"
)


def setup_logging(log_file):
    """Configure logging with UTC8 formatter"""
    formatter = UTC8Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    handlers = [logging.FileHandler(log_file), logging.StreamHandler()]

    for handler in handlers:
        handler.setFormatter(formatter)

    logging.basicConfig(level=logging.INFO, handlers=handlers)
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


def process_label_file(label_path, part_name):
    """Process label JSON file to extract disaster info and damage counts"""
    with open(label_path, "r") as f:
        data = json.load(f)

    disaster_type = classify_disaster_type(data["metadata"]["disaster"])
    damage_counts, polygons = count_damages(
        data["features"]["xy"], PART_MAPPING[part_name]
    )

    return disaster_type, damage_counts, polygons


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


def load_model(model_id, device):
    """Load appropriate model and inference function"""
    if model_id == "Meize0729/CCExpert_7b":
        from inference.inference_ccexpert import inference_ccexpert
        from utils.model_ccexpert import ccexpert_model_loader

        return ccexpert_model_loader(model_id, device), inference_ccexpert
    elif model_id == "jirvin16/TEOChat":
        from inference.inference_teochat import inference_teochat
        from utils.model_teochat import teochat_model_loader

        return teochat_model_loader(model_id, device), inference_teochat
    else:
        from inference.inference_baseline import inference_naive
        from utils.model_hub import model_hub_loader

        return {
            model_id: model_hub_loader([model_id], [device])[model_id]
        }, inference_naive


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


def process_image_pair(pre_img_path, model_id, model, infer_fn, processed, device):
    """Process a single image pair with a model"""
    post_img_path = pre_img_path.replace("_pre_disaster_part", "_post_disaster_part")
    if (pre_img_path, post_img_path, model_id) in processed:
        return None

    part_name = os.path.basename(pre_img_path).split("_")[-1].replace(".png", "")
    pre_label_path = (
        re.sub(r"_part\d+", "", pre_img_path)
        .replace("images", "labels")
        .replace("_pre_disaster_part", "_pre_disaster")
        .replace(".png", ".json")
    )
    post_label_path = (
        re.sub(r"_part\d+", "", post_img_path)
        .replace("images", "labels")
        .replace("_post_disaster_part", "_post_disaster")
        .replace(".png", ".json")
    )
    # Check if corresponding post image and json files exist using relative paths
    if not os.path.exists(pre_img_path):
        logging.warning(f"Missing pre-disaster image file: {pre_img_path}")
        return
    if not os.path.exists(post_img_path):
        logging.warning(f"Missing post-disaster image file: {post_img_path}")
        return
    if not os.path.exists(pre_label_path):
        logging.warning(f"Missing pre-event label JSON file: {pre_label_path}")
        return
    if not os.path.exists(post_label_path):
        logging.warning(f"Missing post-event label JSON file: {post_label_path}")
        return

    # disaster_type, damage_counts, pre_polygons = process_label_file(
    #     pre_label_path, part_name
    # )
    disaster_type, damage_counts, post_polygons = process_label_file(
        post_label_path, part_name
    )
    pre_image = Image.open(pre_img_path)
    post_image = Image.open(post_img_path)
    prompts = generate_prompts(disaster_type, damage_counts)

    results = []
    for prompt_type, prompt_text in prompts.items():
        if prompt_type == "visual_prompt":
            # pre_image_input = process_visual_prompt(pre_image, pre_polygons, part_name)
            post_image_input = process_visual_prompt(
                post_image, post_polygons, part_name
            )

            # temp_dir = os.path.join("/home", "temp")
            # os.makedirs(temp_dir, exist_ok=True)
            # pre_image_input.save(
            #     os.path.join(temp_dir, f"{os.path.basename(pre_img_path)}")
            # )
            # post_image_input.save(
            #     os.path.join(temp_dir, f"{os.path.basename(post_img_path)}")
            # )

        else:
            pre_image_input = pre_image
            post_image_input = post_image
        result = infer_fn(
            model_id=model_id,
            text_prompt=prompt_text,
            pre_image=pre_image_input,
            post_image=post_image_input,
            model_hub=model,
            device=device,
        )
        results.append(
            {
                "model_id": model_id,
                "inference_type": prompt_type,
                "pre_image": pre_img_path,
                "post_image": post_img_path,
                "change_caption": result,
            }
        )
    return results


def load_processed_items(output_file):
    """Load already processed items from output file"""
    processed = set()
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    key = (entry["pre_image"], entry["post_image"], entry["model_id"])
                    processed.add(key)
                except json.JSONDecodeError:
                    logging.error(f"Error decoding line: {line}")
    return processed


def main():
    """Main processing pipeline"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_file",
        type=str,
        default="./output/xbd_subset_baseline.jsonl",
        help="Path to output JSONL file",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="./logs/xbd_subset_baseline.log",
        help="Path to log file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help="device",
    )
    args = parser.parse_args()
    output_file = args.output_file
    log_file = args.log_file
    device = args.device

    # Ensure directories exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = setup_logging(log_file)
    image_pairs = get_image_pairs()
    processed = load_processed_items(output_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    logger.info(f"Saving results to {output_file}.")

    for model_id in INFERENCE_MODEL_LIST:
        logger.info(f"Processing model: {model_id}")
        model, infer_fn = load_model(model_id, device)

        for pre_img_path in tqdm(image_pairs, desc=f"Processing {model_id}"):
            results = process_image_pair(
                pre_img_path, model_id, model, infer_fn, processed, device
            )
            if not results:
                continue

            with open(output_file, "a") as f:
                for result in results:
                    json.dump(result, f)
                    f.write("\n")
                    f.flush()
                    # processed.add((result["pre_image"], result["post_image"], model_id))

        logger.info(f"Completed processing for {model_id}")
        del model  # Explicit cleanup
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
