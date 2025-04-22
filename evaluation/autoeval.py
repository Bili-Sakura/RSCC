import argparse
from openai import OpenAI
import os
import base64
import json
import random
from dotenv import load_dotenv
import sys
from tqdm import tqdm
import concurrent.futures
import threading

# Global variables
query_num = 0
query_num_lock = threading.Lock()  # Add this lock
sys.path.append("../")
sys.path.append("./")
from utils.constants import EVALUATION_PROMPT_TEMPLATE
from utils.token_tracker import TokenTracker

# nohup python ./evaluation/autoeval.py \
#   --baseline_path ./output/xbd_subset_baseline.jsonl \
#   --gt_path ./output/xbd_subset_qvq.jsonl \
#   --output_dir ./output/evaluation_results_qvq \
#   > ./logs/autoeval_test.log 2>&1 &

# Load environment variables
load_dotenv(
    override=True
)  # Reload environment variables each time to get latest values


def load_jsonl_data(file_path):
    """Load and parse JSONL file containing evaluation data"""
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]


def encode_image(image_path):
    """Base64 encode image file for API consumption"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_unique_image_pairs(data):
    """Extract unique pre/post disaster image pairs from dataset"""
    return sorted(
        {
            (os.path.basename(e["pre_image"]), os.path.basename(e["post_image"]))
            for e in data
        }
    )


def create_evaluation_prompt(image_pair, baseline_data, gt_data):
    """Create evaluation prompt with ground truth as Model_0"""
    # Find ground truth entry
    gt_entry = next(
        e
        for e in gt_data
        if (
            os.path.basename(e["pre_image"]) == image_pair[0]
            and os.path.basename(e["post_image"]) == image_pair[1]
        )
    )

    # Find all baseline entries for this image pair
    baseline_entries = [
        e
        for e in baseline_data
        if (
            os.path.basename(e["pre_image"]) == image_pair[0]
            and os.path.basename(e["post_image"]) == image_pair[1]
            and e["inference_type"] == "visual_prompt"
        )
    ]

    # Select just one baseline model (first one in the list)
    if not baseline_entries:
        print(
            f"No valid prompt for image pairs {gt_entry['pre_image']} and {gt_entry['post_image']}"
        )
        return None, None, None, None

    selected_baseline = random.choice(baseline_entries)

    # Combine entries with ground truth first
    model_entries = [gt_entry, selected_baseline]

    # Encode images using ground truth paths
    pre_img = encode_image(gt_entry["pre_image"])
    post_img = encode_image(gt_entry["post_image"])

    captions = [
        f"Model_{idx}: {entry['change_caption']}"
        for idx, entry in enumerate(model_entries)
    ]
    model_mapping = {
        f"Model_{idx}": entry["model_id"] for idx, entry in enumerate(model_entries)
    }

    prompt = EVALUATION_PROMPT_TEMPLATE.format(captions="\n".join(captions))
    return prompt, model_mapping, pre_img, post_img


def initialize_openai_client():
    """Validate model name before initializing client"""
    model_name = os.getenv("QVQ_MODEL_NAME")
    supported_models = ["qvq-max", "qvq-max-latest", "qvq-max-2025-03-25"]
    if model_name not in supported_models:
        raise ValueError(
            f"Invalid model name: {model_name}. Supported models: {supported_models}"
        )
    return OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("API_BASE_URL"),
    )


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Evaluate change captions using QvQ-Max API"
    )
    parser.add_argument(
        "--baseline_path",
        type=str,
        required=True,
        help="Path to JSONL file containing baseline model outputs",
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        required=True,
        help="Path to JSONL file containing ground truth outputs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save evaluation results",
    )
    return parser.parse_args()


def calculate_win_rates(output_path):
    """Calculate win rates from existing results file"""
    model_stats = {}

    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                try:
                    result = json.loads(line)
                    # Get both models in the comparison
                    models = list(result["all_models"].values())
                    if len(models) != 2:
                        continue  # Skip invalid entries

                    model_a, model_b = models
                    best_model = result["best_model"]

                    # Update totals for both models
                    for model in [model_a, model_b]:
                        model_stats.setdefault(model, {"wins": 0, "total": 0})
                        model_stats[model]["total"] += 1

                    # Update wins for the best model
                    model_stats[best_model]["wins"] += 1
                except json.JSONDecodeError:
                    continue

    # Calculate final win rates
    win_rates = {}
    for model, stats in model_stats.items():
        win_rates[model] = {
            "wins": stats["wins"],
            "total": stats["total"],
            "win_rate": stats["wins"] / stats["total"] if stats["total"] > 0 else 0,
        }

    return win_rates


def process_evaluation(client, baseline_data, gt_data, output_path):
    """Process all samples and save evaluation results"""
    model_stats = {}
    processed_pairs = load_existing_results(output_path)
    file_lock = threading.Lock()  # Create a lock for file writes
    token_lock = threading.Lock()  # Add this lock
    max_workers = int(os.getenv("MAX_WORKERS", "10"))  # Read from .env
    unprocessed_pairs = [
        pair
        for pair in get_unique_image_pairs(baseline_data)
        if (pair[0], pair[1]) not in processed_pairs
    ]

    tracker = TokenTracker(initial_tokens=1000000)
    current_key = os.getenv("DASHSCOPE_API_KEY")
    model_name = os.getenv("QVQ_MODEL_NAME")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with tqdm(
        total=len(unprocessed_pairs), desc="Evaluating pairs", dynamic_ncols=True
    ) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for pair in unprocessed_pairs:
                prompt, model_mapping, pre_img, post_img = create_evaluation_prompt(
                    pair, baseline_data, gt_data
                )
                if prompt is None:
                    pbar.update(1)
                    continue

                # Use lock for thread-safe token check
                with token_lock:
                    remaining = tracker.get_usage(current_key, model_name)
                    if remaining <= 0:
                        pbar.write("Token limit reached - stopping submission")
                        break  # Exit loop early

                future = executor.submit(
                    evaluate_pair,
                    client,
                    prompt,
                    model_mapping,
                    pre_img,
                    post_img,
                    pbar,
                    tracker,
                    current_key,
                    model_name,
                    pair,
                    output_path,
                    model_stats,
                    file_lock,
                    token_lock,
                )
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    pbar.write(f"Error processing: {str(e)}")

    return model_stats


def evaluate_pair(
    client,
    prompt,
    model_mapping,
    pre_img,
    post_img,
    pbar,
    tracker,
    current_key,
    model_name,
    pair,
    output_path,
    model_stats,
    file_lock,  # Add this parameter to accept a lock
    token_lock,
):
    global query_num, query_num_lock  # Declare both variables as global

    # Thread-safe increment
    with query_num_lock:
        query_num += 1

    try:
        stream = client.chat.completions.create(
            model=os.getenv("QVQ_MODEL_NAME"),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{pre_img}"},
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{post_img}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            stream=True,
            stream_options={"include_usage": True},
            temperature=0.01,
        )

        full_response = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
        tracker.update_usage(
            api_key=current_key,
            model_name=model_name,
            tokens=chunk.usage.total_tokens,
        )
        with token_lock:
            tracker.update_usage(
                current_key, model_name, tokens=chunk.usage.total_tokens
            )
        remaining = tracker.get_usage(current_key, model_name)

        pbar.set_postfix({"Tokens Left": remaining, "Total Processed": pbar.n + 1})

        json_start = full_response.find("{")
        json_end = full_response.rfind("}") + 1
        result = json.loads(full_response[json_start:json_end])

        result_entry = {
            "pre_image": pair[0],
            "post_image": pair[1],
            "best_model": model_mapping[result["best_model_id"]],
            "ground_truth": model_mapping["Model_0"],
            "reason": result["reason"],
            "all_models": model_mapping,
        }
        # Add this section to write the result to file safely
        with file_lock:  # Ensure thread-safe file writes
            with open(output_path, "a") as f:
                f.write(json.dumps(result_entry) + "\n")
        pbar.update(1)

    except Exception as e:
        with token_lock:
            # Handle partial token usage tracking for failed requests
            # This is a best-effort approach since we don't know actual usage
            # You might need to track estimated tokens per request
            pbar.write(f"Error processing {pair[0]}: {str(e)}")
            # You could subtract an estimated minimum token cost here
            # tracker.update_usage(...)  # With estimated tokens

        pbar.update(1)
        raise  # Propagate exception after logging


def load_existing_results(output_path):
    """Load already processed image pairs from existing results file"""
    processed_pairs = set()
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                try:
                    result = json.loads(line)
                    processed_pairs.add((result["pre_image"], result["post_image"]))
                except json.JSONDecodeError:
                    continue
    return processed_pairs


# Main execution flow
if __name__ == "__main__":
    args = parse_arguments()
    baseline_data = load_jsonl_data(args.baseline_path)
    gt_data = load_jsonl_data(args.gt_path)
    client = initialize_openai_client()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Process evaluations and save results
    output_path = os.path.join(args.output_dir, "best_model_results_qvq.jsonl")
    process_evaluation(client, baseline_data, gt_data, output_path)

    # Calculate win rates after all evaluations are done
    win_rates = calculate_win_rates(output_path)

    # Save win rates
    with open(os.path.join(args.output_dir, "win_rates.json"), "w") as f:
        json.dump(win_rates, f, indent=2)

    print(
        f"Evaluation complete. Results saved to folder {args.output_dir}. It contains both best model results and win rate results."
    )
