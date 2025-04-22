# sakura/projects/RSCC/inference/xbd_subset_qvq.py
import argparse
import json
import os
import logging
import glob
import tqdm
import re
import io
from dotenv import load_dotenv
from PIL import Image, ImageDraw
import base64
from openai import OpenAI
from shapely import wkt
import sys
import concurrent.futures
import threading

# Global variables
query_num = 0
query_num_lock = threading.Lock()  # Add this lock
sys.path.append("../")
sys.path.append("./")
from utils.constants import PROMPT_TEMPLATES, PART_MAPPING, PATH_TO_DATASET_FOLDER
from utils.logger import setup_logging
from utils.token_tracker import TokenTracker  # New import


load_dotenv(override=True)
IMAGE_DIR_TEMPLATE = os.path.join(
    PATH_TO_DATASET_FOLDER, "xbd_subset/{event_name}/images/"
)
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL")
QVQ_MODEL_NAME = os.getenv("QVQ_MODEL_NAME")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "10"))
TOKEN_THRESHOLD = int(os.getenv("TOKEN_THRESHOLD", 100000))


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def process_visual_prompt(post_image_path, polygons, part_name):
    img = Image.open(post_image_path)
    draw = ImageDraw.Draw(img, "RGBA")
    bounds = PART_MAPPING[part_name]
    colors = {
        "no-damage": (0, 255, 0, 100),
        "minor-damage": (0, 0, 255, 125),
        "major-damage": (255, 69, 0, 125),
        "destroyed": (255, 0, 0, 125),
    }
    for damage, poly in polygons:
        x, y = poly.exterior.coords.xy
        coords = list(zip([c - bounds[0] for c in x], [c - bounds[1] for c in y]))
        draw.line(coords + [coords[0]], fill=colors[damage], width=2)
    return img


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


def generate_prompts(disaster_type, damage_counts):
    return {
        # "zero-shot": PROMPT_TEMPLATES["naive"].strip(),
        "textual_prompt": PROMPT_TEMPLATES["textual"]
        .format(disaster_type=disaster_type, number=damage_counts)
        .strip(),
        # "visual_prompt": PROMPT_TEMPLATES["visual"]
        # .format(disaster_type=disaster_type, number=damage_counts)
        # .strip(),
    }


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


def process_label_file(label_path, part_name):
    """Process label JSON file to extract disaster info and damage counts"""
    with open(label_path, "r") as f:
        data = json.load(f)

    disaster_type = classify_disaster_type(data["metadata"]["disaster"])
    damage_counts, polygons = count_damages(
        data["features"]["xy"], PART_MAPPING[part_name]
    )

    return disaster_type, damage_counts, polygons


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


def process_image_pair(
    pre_path,
    client,
    logger,
    tracker,
    output_file,
    file_lock,
    token_lock,
    processed,  # Pass the processed set
):
    global query_num, query_num_lock  # Declare both variables as global
    post_path = pre_path.replace("_pre_", "_post_")
    part_name = os.path.basename(pre_path).split("_")[-1].replace(".png", "")

    # Check if this image pair has already been processed
    key = (pre_path, post_path)
    if key in processed:
        logger.info(f"Skipping already processed image pair: {key}")
        return

    # Load post disaster labels
    post_label_path = (
        re.sub(r"_part\d+", "", post_path)
        .replace("images", "labels")
        .replace("_post_disaster_part", "_post_disaster")
        .replace(".png", ".json")
    )
    with open(post_label_path) as f:
        post_data = json.load(f)

    # Process labels
    disaster_type, damage_counts, post_polygons = process_label_file(
        post_label_path, part_name
    )
    # pre_image = Image.open(pre_path)
    # post_image = Image.open(post_path)
    prompts = generate_prompts(disaster_type, damage_counts)

    for prompt_type, prompt in prompts.items():

        # Prepare images
        if prompt_type == "visual_prompt":
            modified_post = process_visual_prompt(post_path, post_polygons, part_name)
            buffer = io.BytesIO()
            modified_post.save(buffer, format="PNG")
            buffer.seek(0)
            post_b64 = base64.b64encode(buffer.read()).decode()
        else:
            post_b64 = encode_image(post_path)

        pre_b64 = encode_image(pre_path)

        # Create API request
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{pre_b64}"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{post_b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        try:
            stream = client.chat.completions.create(
                model=QVQ_MODEL_NAME,
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},
                temperature=0.01,
            )
            full_response = ""
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
            tracker.update_usage(
                api_key=DASHSCOPE_API_KEY,
                model_name=QVQ_MODEL_NAME,
                tokens=chunk.usage.total_tokens,
            )

            with token_lock:
                tracker.update_usage(
                    DASHSCOPE_API_KEY, QVQ_MODEL_NAME, chunk.usage.total_tokens
                )
            remaining = tracker.get_usage(DASHSCOPE_API_KEY, QVQ_MODEL_NAME)
            logger.info(f"'Tokens Left':{remaining}.")

            # Protect file writes with file_lock
            with file_lock:
                with open(output_file, "a") as f:
                    json.dump(
                        {
                            "model_id": QVQ_MODEL_NAME,
                            "inference_type": prompt_type,
                            "pre_image": pre_path,
                            "post_image": post_path,
                            "change_caption": full_response,
                        },
                        f,
                    )
                    f.write("\n")

        except Exception as e:
            logger.error(f"API Error for {pre_path}: {str(e)}", exc_info=True)


def load_processed_items(output_file):
    """Load already processed items from output file"""
    processed = set()
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    key = (entry["pre_image"], entry["post_image"])
                    processed.add(key)
                except json.JSONDecodeError:
                    logging.error(f"Error decoding line: {line}")
    return processed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--log_file", required=True)

    args = parser.parse_args()
    # Ensure directories exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    logger = setup_logging(args.log_file)

    client = OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url=API_BASE_URL,
    )
    tracker = TokenTracker()

    file_lock = threading.Lock()  # Create a lock for file writes
    token_lock = threading.Lock()  # Add this lock
    image_pairs = get_image_pairs()
    processed = load_processed_items(args.output_file)
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for pre_path in image_pairs:
            # Check token availability before submitting
            with token_lock:
                remaining = tracker.get_usage(DASHSCOPE_API_KEY, QVQ_MODEL_NAME)
                if remaining < TOKEN_THRESHOLD:
                    logger.warning("Token limit reached. Stopping submission.")
                    break
            future = executor.submit(  # Pass both locks
                process_image_pair,
                pre_path,
                client,
                logger,
                tracker,
                args.output_file,
                file_lock,  # Add this
                token_lock,  # Add this
                processed,  # Pass the processed set
            )
            futures.append(future)

        # Wait for all tasks to complete
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Task failed: {str(e)}")


if __name__ == "__main__":
    main()
