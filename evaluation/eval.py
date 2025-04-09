import json
import logging
import os
from collections import defaultdict
from evaluate import load
import sys
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

sys.path.append("../")
sys.path.append("./")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = SentenceTransformer(
    "../models/sentence-transformers/sentence-t5-xxl", device="cuda"
)


def extract_filename(path):
    """Extract filename from path"""
    return os.path.basename(path)


def load_jsonl(file_path):
    """Helper function to load jsonl files"""
    with open(file_path) as f:
        return [json.loads(line) for line in f]


def calculate_metrics(source_texts, ground_truths):
    # Initialize metrics
    bleu_metric = load("bleu")
    rouge_metric = load("rouge")
    meteor_metric = load("meteor")

    # Calculate metrics
    bleu_results = bleu_metric.compute(
        predictions=source_texts, references=ground_truths
    )
    rouge_results = rouge_metric.compute(
        predictions=source_texts, references=ground_truths
    )
    meteor_results = meteor_metric.compute(
        predictions=source_texts, references=ground_truths
    )

    # Calculate cosine similarity
    source_embeddings = model.encode(source_texts, normalize_embeddings=True)
    gt_embeddings = model.encode(ground_truths, normalize_embeddings=True)
    similarities = []
    for src_emb, gt_emb in zip(source_embeddings, gt_embeddings):
        src_emb = np.expand_dims(src_emb, axis=0)
        gt_emb = np.expand_dims(gt_emb, axis=0)
        similarity = abs(cosine_similarity(src_emb, gt_emb)[0][0] ** 3)
        similarities.append(similarity)
    cosine_sim = np.mean(similarities)

    # Calculate average caption length
    total_length = sum(len(caption) for caption in source_texts)
    avg_length = total_length / len(source_texts)

    # Return all metrics
    return {
        "bleu": bleu_results["bleu"],
        "bleu1": bleu_results["precisions"][0],
        "bleu2": bleu_results["precisions"][1],
        "bleu3": bleu_results["precisions"][2],
        "bleu4": bleu_results["precisions"][3],
        "rouge": rouge_results["rougeL"],
        "meteor": meteor_results["meteor"],
        "cosine_similarity": cosine_sim,
        "avg_caption_length": avg_length,
    }


def evaluate_models(gt_path, pred_path):
    # Load data
    logger.info(f"Loading ground truth from {gt_path}")
    gt_data = load_jsonl(gt_path)
    logger.info(f"Loaded {len(gt_data)} ground truth entries")

    logger.info(f"Loading predictions from {pred_path}")
    pred_data = load_jsonl(pred_path)
    logger.info(f"Loaded {len(pred_data)} prediction entries")

    # Create mapping from filenames to ground truth
    gt_mapping = {
        (
            extract_filename(item["pre_image"]),
            extract_filename(item["post_image"]),
        ): item["change_caption"]
        for item in gt_data
    }
    logger.info(f"Created ground truth mapping with {len(gt_mapping)} entries")

    # Group predictions by model_id
    model_predictions = defaultdict(lambda: {"source_texts": [], "ground_truths": []})
    matched_count = 0

    for pred in pred_data:
        key = (
            extract_filename(pred["pre_image"]),
            extract_filename(pred["post_image"]),
        )
        if key in gt_mapping:
            matched_count += 1
            model_predictions[pred["model_id"]]["source_texts"].append(
                pred["change_caption"]
            )
            model_predictions[pred["model_id"]]["ground_truths"].append(gt_mapping[key])
        else:
            logger.debug(f"No match found for: {key}")

    logger.info(f"Matched {matched_count} predictions with ground truth")
    logger.info(f"Found {len(model_predictions)} models in predictions")

    # Calculate metrics for each model
    results = {}
    for model_id, data in model_predictions.items():
        logger.info(
            f"Calculating metrics for model {model_id} with {len(data['source_texts'])} samples"
        )
        results[model_id] = calculate_metrics(
            data["source_texts"], data["ground_truths"]
        )

    return results


# Example usage:
results = evaluate_models("../data/xbd_gt.jsonl", "../data/xbd_subset_baseline.jsonl")
print(results)
