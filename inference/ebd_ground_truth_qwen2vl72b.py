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
image_dir = "./data/xbd//{event_name}/images/"
# Check if the output file already exists and read processed items
output_file_path = "./output/.jsonl"

# Prompt Template
prompt_template = """
These two satellite images show a {disaster_type} natural disaster before and after respectively.  
Now, give change description between the pre-event and post-event images in a news style with a few sentences using precise phrases separated by commas.
"""

# Iterate Through All Image Pairs
image_pairs = []
for part_num in range(1, 5):
    pattern = os.path.join(
        image_dir.format(event_name="*"), f"*_pre_disaster.png"
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

        disaster_types = {
        "EARTHQUAKE-TURKEY":"earthquake",
        "HURRICANE-DELTA":"",
        "HURRICANE-DORIAN":"",
        "HURRICANE-IAN":"",
        "HURRICANE-IRMA":"",
        "HURRICANE-LAURA":"",
        "MOUNT-SEMERU-ERUPTION": "volcano eruption",
        "PAKISTAN-FLOODING": "",
        "STVINCENT-VOLCANO":"volcano eruption",
        "TEXAS-TORNADOES":"",
        "HURRICANE-IDA": ""
        }
        disaster_type = disaster_types.get(subfolder, "disaster")
        
        if 'pre' in input_image_filename:
            return f"a satellite image after {disaster_type}, {edit_prompt}"
        elif 'post' in input_image_filename:
            return f"a satellite image before {disaster_type}, {edit_prompt}"
        else:
            return f"a satellite image during {disaster_type}, {edit_prompt}"


       

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
