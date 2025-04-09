import sys

sys.path.append("../")
sys.path.append("./")
import torch
from inference.inference_baseline import inference_naive
from utils.model_hub import model_hub_loader
from utils.logger import UTC8Formatter
from PIL import Image
import os
import json
from shapely import wkt
import glob
import re
import logging
from tqdm import tqdm

# nohup python inference/xbd_baseline.py > ./logs/baseline.log &

# Set up logging
log_file = "./logs/baseline.log"
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
# Apply the custom formatter to the handlers
for handler in logging.getLogger().handlers:
    handler.setFormatter(
        UTC8Formatter(
            fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
    )

logger = logging.getLogger(__name__)

# Generalize File Paths
image_dir = "./data/xbd/images-w512-h512/{event_name}/images/"
# Check if the output file already exists and read processed items
output_file_path = "./output/xbd_baseline.jsonl"

# Prompt Template
prompt_template = """Give change description between two satellite images.\nFocusing on the following objects if present: roads, vehicles, river, buildings, trees, grass and cropland.\nOutput answer in a news style with a few sentences using precise phrases separated by commas"""

MODEL_LIST = [
    "Qwen/Qwen2-VL-7B-Instruct",
    "Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5",
    "OpenGVLab/InternVL2_5-8B",
    "llava-hf/llava-interleave-qwen-7b-hf",
    "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    "mistralai/Pixtral-12B-2409",
]

# Iterate Through All Image Pairs
image_pairs = []
for part_num in range(1, 5):
    pattern = os.path.join(
        image_dir.format(event_name="*"), f"*_pre_disaster_part{part_num}.png"
    )
    image_pairs.extend(glob.glob(pattern))

image_pairs = [img for img in image_pairs]

all_outputs = []

# Sort image pairs by filename
image_pairs.sort()


# Load Model

processed_items = set()
if os.path.exists(output_file_path):
    with open(output_file_path, "r") as infile:
        for line in infile:
            try:
                entry = json.loads(line)
                processed_items.add((entry["pre_image"], entry["post_image"]))
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON line: {line}")

# Calculate the starting batch index
start_idx = len(processed_items)

device_mapping = ["cuda:0", "cuda:0", "cuda:0", "cuda:1", "cuda:1", "cuda:0"]
model_hub = model_hub_loader(MODEL_LIST, device_mapping)

# Iterate
for idx, image_pair in enumerate(tqdm(image_pairs, desc="Processing")):
    if idx < start_idx:
        continue  # Skip batches that have already been processed
    logging.info(f"Processing {idx}/{len(image_pairs)}.")
    messages = []

    pre_event_img_path = image_pair

    # Construct post_event_img_path
    post_event_img_path = pre_event_img_path.replace(
        "_pre_disaster_part", "_post_disaster_part"
    )
    # logging.info(f"{pre_event_img_path}")
    # logging.info(f"{post_event_img_path}")

    # Check if corresponding post image and json files exist using relative paths
    if not os.path.exists(os.path.join(os.getcwd(), post_event_img_path)):
        logger.warning(
            f"Missing files for post image : {os.path.join(os.getcwd(), post_event_img_path)} "
        )
        break

    pre_image = Image.open(pre_event_img_path)
    post_image = Image.open(post_event_img_path)

    items = []
    for model_id, device in zip(MODEL_LIST, device_mapping):
        result = inference_naive(
            model_id=model_id,
            text_prompt=prompt_template,
            pre_image=pre_image,
            post_image=post_image,
            model_hub=model_hub,
            device=device,
        )
        item = {
            "model_id": model_id,
            "pre_image": pre_event_img_path,
            "post_image": post_event_img_path,
            "change_caption": result,
        }
        items.append(item)

    all_outputs.extend(items)

    # Save all outputs to a jsonl file
    with open(output_file_path, mode="a") as outfile:
        for entry in all_outputs:
            json.dump(entry, outfile)
            outfile.write("\n")
            outfile.flush()  # Flush the data to the file
    all_outputs = []
