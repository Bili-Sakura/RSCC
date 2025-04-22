import json
import logging
import os

from collections import defaultdict
from evaluate import load
from sentence_transformers import SentenceTransformer
from bert_score import score as bert_score
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import Dict, List, Tuple
import sys
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

sys.path.append("./")
from utils.constants import (
    T5_MODEL_REPO,
    PATH_TO_MODEL_FOLDER,
    DEFAULT_DEVICE,
    BERT_MODEL_REPO,
)

# nohup python ./evaluation/metrics.py --ground_truth_file ./output/xbd_subset_qvq.jsonl --predictions_file ./output/xbd_subset_baseline.jsonl --output_file ./output/eval_results_qvq_textualprompt.json > ./logs/eval.log
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_T5 = SentenceTransformer(
    os.path.join(PATH_TO_MODEL_FOLDER, T5_MODEL_REPO), device=DEFAULT_DEVICE
)
BERT_MODEL_PATH = os.path.join(PATH_TO_MODEL_FOLDER, BERT_MODEL_REPO)

TOKENIZER_BERT = AutoTokenizer.from_pretrained(BERT_MODEL_PATH, local_files_only=True)
MODEL_BERT = AutoModel.from_pretrained(BERT_MODEL_PATH, local_files_only=True).to(
    DEFAULT_DEVICE
)


def extract_filename(path):
    """Extract filename from path"""
    return os.path.basename(path)


def load_jsonl(file_path):
    """Helper function to load jsonl files"""
    with open(file_path) as f:
        return [json.loads(line) for line in f]


def calculate_bertscore(candidates: List[str], references: List[str]) -> float:
    """Manual BERTScore implementation using Hugging Face transformers"""
    with torch.no_grad():
        # Tokenize
        cand_inputs = TOKENIZER_BERT(
            candidates, return_tensors="pt", padding=True, truncation=True
        ).to(DEFAULT_DEVICE)
        ref_inputs = TOKENIZER_BERT(
            references, return_tensors="pt", padding=True, truncation=True
        ).to(DEFAULT_DEVICE)

        # Get embeddings
        cand_embeddings = MODEL_BERT(**cand_inputs).last_hidden_state.mean(
            dim=1
        )  # Mean pooling
        ref_embeddings = MODEL_BERT(**ref_inputs).last_hidden_state.mean(dim=1)

        # Cosine similarity
        similarities = (
            F.cosine_similarity(cand_embeddings, ref_embeddings).cpu().numpy()
        )

    return similarities.mean()


def calculate_metrics(source_texts: List[str], ground_truths: List[str]) -> Dict:
    """Calculate evaluation metrics for caption generation.

    Args:
        source_texts: List of generated captions
        ground_truths: List of reference captions

    Returns:
        Dictionary containing computed metrics
    """
    # Initialize metrics
    # bleu = load("/home/sakura/projects/RSCC/libs/metrics/bleu")
    # rouge = load("/home/sakura/projects/RSCC/libs/metrics/rouge")
    # meteor = load("/home/sakura/projects/RSCC/libs/metrics/meteor")

    # # Text-based metrics
    # bleu_results = bleu.compute(predictions=source_texts, references=ground_truths)
    # rouge_results = rouge.compute(predictions=source_texts, references=ground_truths)
    # meteor_results = meteor.compute(predictions=source_texts, references=ground_truths)

    # T5 Semantic similarity
    source_embeds = MODEL_T5.encode(source_texts, normalize_embeddings=True)
    gt_embeds = MODEL_T5.encode(ground_truths, normalize_embeddings=True)
    similarities = cosine_similarity(source_embeds, gt_embeds).diagonal()
    cubed_similarities = np.abs(similarities**3)

    # Caption length analysis
    caption_lengths = [
        len(caption.split()) for caption in source_texts
    ]  # Changed from character count to word count

    # Compute BERTScore manually
    bertscore_value = calculate_bertscore(source_texts, ground_truths)

    return {
        # "bleu": bleu_results["bleu"],
        # "bleu1": bleu_results["precisions"][0],
        # "bleu2": bleu_results["precisions"][1],
        # "bleu3": bleu_results["precisions"][2],
        # "bleu4": bleu_results["precisions"][3],
        # "rouge": rouge_results["rougeL"],
        # "meteor": meteor_results["meteor"],
        "cosine_similarity": float(cubed_similarities.mean()),
        "avg_caption_length": float(np.mean(caption_lengths)),
        "bertscore": float(bertscore_value),
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

    # Group predictions by model_id AND inference_type
    model_results = defaultdict(lambda: {"source_texts": [], "ground_truths": []})
    matched_pairs = 0

    for prediction in pred_data:
        key = (
            extract_filename(prediction["pre_image"]),
            extract_filename(prediction["post_image"]),
        )

        if key in gt_mapping:
            matched_pairs += 1
            # Use both model_id and inference_type as unique key
            model_key = (prediction["model_id"], prediction["inference_type"])
            model_results[model_key]["source_texts"].append(
                prediction["change_caption"]
            )
            model_results[model_key]["ground_truths"].append(gt_mapping[key])

    logger.info(f"Matched {matched_pairs} predictions with ground truth")

    # Calculate metrics for each model/inference_type combination
    results = defaultdict(dict)
    for (model_id, inf_type), data in model_results.items():
        results[model_id][inf_type] = calculate_metrics(
            data["source_texts"], data["ground_truths"]
        )

    return dict(results)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model outputs against ground truth"
    )
    parser.add_argument(
        "--ground_truth_file", type=str, help="Path to ground truth JSONL file"
    )
    parser.add_argument(
        "--predictions_file", type=str, help="Path to model predictions JSONL file"
    )
    parser.add_argument("--output_file", type=str, help="Path to result json file")
    args = parser.parse_args()

    results = evaluate_models(args.ground_truth_file, args.predictions_file)
    logger.info("Evaluation results: %s", results)
    # Add JSON output
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
