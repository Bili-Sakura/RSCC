import sys

sys.path.append("../")
sys.path.append("./")
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from utils.logger import UTC8Formatter
import os
import json
from shapely import wkt
import glob
import re
import logging
from tqdm import tqdm

# nohup python inference/xbd_ground_truth_qwen2vl72b.py > ./logs/gt.log &

# Set up logging
log_file = "./logs/gt.log"
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

# Define batch size
batch_size = 6
# Generalize File Paths
image_dir = "./data/xbd/images-w512-h512/{event_name}/images/"
# Check if the output file already exists and read processed items
output_file_path = "./output/xbd.jsonl"

# Prompt Template
prompt_template = """
These two satellite images show a {disaster_type} natural disaster before and after respectively.  

Here is the disaster level descriptions: 
- Disaster Level 0 (No Damage): Undisturbed. No sign of water, structural or shingle damage, or burn marks.
- Disaster Level 1 (Minor Damage): Building partially burnt, water surrounding structure, volcanic flow nearby, roof elements missing, or visible cracks.
- Disaster Level 2 (Major Damage): Partial wall or roof collapse, encroaching volcanic flow, or surrounded by water/mud.
- Disaster Level 3 (Destroyed): Scorched, completely collapsed, partially/completely covered with water/mud, or otherwise no longer present.

We already know that there are {number[all]} buildings. {number[no-damage]} buildings are no damaged. {number[minor-damage]} buildings are minor damaged, {number[major-damage]} building are major damaged, {number[destroyed]} buildings are destroyed. {number[unclassified]} buildings damage are unknown.
Now, give change description between the pre-event and post-event images in a news style with a few sentences using precise phrases separated by commas.
"""

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

batches = [
    image_pairs[i : i + batch_size]
    for i in range(0, len(image_pairs), batch_size)
    if len(image_pairs[i : i + batch_size]) == batch_size
]

# Add any remaining images that do not fill a complete batch
if len(image_pairs) % batch_size != 0:
    batches.append(image_pairs[-(len(image_pairs) % batch_size) :])


# Load Model
model_name = "../models/Qwen/Qwen2-VL-72B-Instruct"
# model_name = "../models/Qwen/Qwen2-VL-7B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_name)


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
start_batch_idx = len(processed_items) // batch_size

# Iterate Through Batches
for batch_idx, batch in enumerate(tqdm(batches, desc="Processing batches")):
    if batch_idx < start_batch_idx:
        continue  # Skip batches that have already been processed
    logging.info(f"Processing {batch_idx}/{len(batches)}.")
    messages = []
    for pre_event_img_path in batch:
        # Construct post_event_img_path
        post_event_img_path = pre_event_img_path.replace(
            "_pre_disaster_part", "_post_disaster_part"
        )

        # Check if the item has already been processed
        # if (pre_event_img_path, post_event_img_path) in processed_items:
        #     logger.info(f"Skipping already processed image pair: {pre_event_img_path} and {post_event_img_path}")
        #     continue

        # Construct label_json_path
        label_json_path = (
            re.sub(r"_part\d+", "", pre_event_img_path)
            .replace("images", "labels")
            .replace("_pre_disaster_part", "_post_disaster")
            .replace(".png", ".json")
            .replace("labels-w512-h512", "images-w512-h512")
        )
        # logger.info(f"Processing image pair: {pre_event_img_path} and {post_event_img_path}")
        # logger.info(f"Label JSON path: {label_json_path}")

        # Check if corresponding post image and json files exist using relative paths
        if not os.path.exists(os.path.join(os.getcwd(), post_event_img_path)):
            logger.warning(
                f"Missing files for post image : {os.path.join(os.getcwd(), post_event_img_path)} "
            )
            break

        if not os.path.exists(os.path.join(os.getcwd(), label_json_path)):
            logger.warning(
                f"Missing files for json : {os.path.join(os.getcwd(), label_json_path)}"
            )
            break

        # Read JSON File and Extract Data
        with open(label_json_path, "r") as file:
            data = json.load(file)
            disaster_name = data["metadata"].get("disaster", "unknown")
            if disaster_name in ["guatemala-volcano", "lower-puna-volcano"]:
                disaster_type = "volcano eruption"
            elif disaster_name in [
                "pinery-bushfire",
                "portugal-wildfire",
                "santa-rosa-wildfire",
                "socal-fire",
                "woolsey-fire",
            ]:
                disaster_type = "wildfire"
            elif disaster_name in [
                "joplin-tornado",
                "moore-tornado",
                "hurricane-florence",
                "hurricane-matthew",
                "tuscaloosa-tornado",
            ]:
                disaster_type = "storm"
            elif disaster_name in ["mexico-earthquake"]:
                disaster_type = "earthquake"
            elif disaster_name in [
                "midwest-flooding",
                "hurricane-harvey",
                "nepal-flooding",
            ]:
                disaster_type = "flooding"
            elif disaster_name in ["hurricane-michael"]:
                disaster_type = "flooding and storm"
            elif disaster_name in ["palu-tsunami", "sunda-tsunami"]:
                disaster_type = "tsunami"

            coords = data["features"]["xy"]
            wkt_polygons = []
            damage_counts = {
                "no-damage": 0,
                "minor-damage": 0,
                "major-damage": 0,
                "destroyed": 0,
                "unclassified": 0,
            }

            # Mapping parts to specific bounds in the 1024x1024 image
            part_mapping = {
                "part1": (0, 0, 512, 512),
                "part2": (0, 512, 512, 1024),
                "part3": (512, 0, 1024, 512),
                "part4": (512, 512, 1024, 1024),
            }

            # Extract part information from filename
            part_name = pre_event_img_path.split("_")[-1].replace(".png", "")
            img_bounds = part_mapping.get(part_name)

            for coord in coords:
                damage = coord["properties"].get("subtype", "no-damage")
                polygon = wkt.loads(coord["wkt"])
                if polygon.intersects(
                    wkt.loads(
                        f"POLYGON(({img_bounds[0]} {img_bounds[1]}, {img_bounds[2]} {img_bounds[1]}, {img_bounds[2]} {img_bounds[3]}, {img_bounds[0]} {img_bounds[3]}, {img_bounds[0]} {img_bounds[1]}))"
                    )
                ):
                    wkt_polygons.append((damage, coord["wkt"]))
                    damage_counts[damage] += 1

            damage_counts["all"] = sum(damage_counts.values())

        # Format the prompt template with the variables
        formatted_prompt = prompt_template.format(
            disaster_type=disaster_type, number=damage_counts
        )

        # Prepare Messages for Batch Processing
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pre_event_img_path},
                    {"type": "image", "image": post_event_img_path},
                    {"type": "text", "text": formatted_prompt},
                ],
            }
        ]
        messages.append(message)

    # if not messages:
    #     logger.warning("No messages to process in this batch.")
    #     continue
    # else :
    #     logger.info(messages)

    # Preparation for Batch Inference
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=texts,
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")  # Ensure inputs are on the same device as the model

    # Batch Inference
    generated_ids = model.generate(**inputs, max_new_tokens=256, temperature=0.001)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    # Display the Output and Parse JSON
    for idx, output in enumerate(output_texts):
        try:

            # Get the corresponding image paths for the current output
            pre_event_img_path = batch[idx]
            post_event_img_path = pre_event_img_path.replace(
                "_pre_disaster_part", "_post_disaster_part"
            )

            all_outputs.append(
                {
                    "pre_image": pre_event_img_path,
                    "post_image": post_event_img_path,
                    "change_caption": output,
                }
            )

        except AttributeError:
            logger.error(f"Error parsing JSON for Sample {idx + 1}: {output}")

    # Save all outputs to a jsonl file
    with open(output_file_path, mode="a") as outfile:
        for entry in all_outputs:
            json.dump(entry, outfile)
            outfile.write("\n")
            outfile.flush()  # Flush the data to the file
    all_outputs = []
