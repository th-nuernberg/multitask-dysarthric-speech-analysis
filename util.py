# Copyright 2025 Technische Hochschule Nürnberg
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# Author: Dominik Wagner, Technische Hochschule Nürnberg
import csv
import logging
import os

import torch
import re
from collections import Counter, defaultdict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from constants import ETIOLOGY_MAP, CATEGORY_MAP

logger = logging.getLogger(__name__)


def print_max_gpu_memory():
    max_memory = torch.cuda.max_memory_allocated()
    if max_memory < 1024:
        formatted_memory = f"{max_memory}B"
    elif max_memory < 1024**2:
        formatted_memory = f"{max_memory / 1024:.2f}KB"
    elif max_memory < 1024**3:
        formatted_memory = f"{max_memory / 1024**2:.2f}MB"
    else:
        formatted_memory = f"{max_memory / 1024**3:.2f}GB"

    logger.info("*" * 50)
    logger.info(f"Maximum GPU memory used: {formatted_memory}")
    logger.info("*" * 50)


def print_parameter_count(model):
    def format_number(num):
        if num >= 1e9:
            return f"{num / 1e9:.1f}B"
        elif num >= 1e6:
            return f"{num / 1e6:.1f}M"
        return str(num)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("*" * 50)
    logger.info(f"Total Parameters: {format_number(total_params)}")
    logger.info(f"Trainable Parameters: {format_number(trainable_params)}")
    logger.info("*" * 50)


def compute_wer_metric(wer_metric, normalizer, pred_str, label_str) -> dict:
    wer_ortho = wer_metric.compute(predictions=pred_str, references=label_str)
    # normalize everything and re-compute WER
    norm_pred_str = [normalizer(pred) for pred in pred_str]
    norm_label_str = [normalizer(label) for label in label_str]
    # for logging, we need the pred/labels to match the norm_pred/norm_labels,
    # so we discard any filtered samples here
    pred_str = [
        pred_str[i] for i in range(len(norm_pred_str)) if len(norm_label_str[i]) > 0
    ]
    label_str = [
        label_str[i] for i in range(len(norm_label_str)) if len(norm_label_str[i]) > 0
    ]
    # filtering step to only evaluate the samples that correspond to non-zero normalized references:
    norm_pred_str = [
        norm_pred_str[i]
        for i in range(len(norm_pred_str))
        if len(norm_label_str[i]) > 0
    ]
    norm_label_str = [
        norm_label_str[i]
        for i in range(len(norm_label_str))
        if len(norm_label_str[i]) > 0
    ]

    wer = wer_metric.compute(predictions=norm_pred_str, references=norm_label_str)
    return {
        "wer": wer,
        "wer_ortho": wer_ortho,
        "pred_str": pred_str,
        "label_str": label_str,
        "norm_pred_str": norm_pred_str,
        "norm_label_str": norm_label_str,
    }


def compute_wer_metric_for_asr_tasks(
    wer_metric, normalizer, pred_str, label_str, tasks
) -> dict:
    filtered_preds = []
    filtered_labels = []
    for p, l, t in zip(pred_str, label_str, tasks):
        if "asr" in t:
            filtered_preds.append(p)
            filtered_labels.append(l)
    if len(filtered_labels) > 0:
        return compute_wer_metric(
            wer_metric, normalizer, filtered_preds, filtered_labels
        )
    else:
        logger.warning(
            "No labels/predictions remain for WER computation after filtering!"
        )
    return {
        "wer": -1,
        "wer_ortho": -1,
        "pred_str": [],
        "label_str": [],
        "norm_pred_str": [],
        "norm_label_str": [],
    }


def compute_clf_metrics(labels, predictions, name_prefix="") -> dict:
    """
    Micro F1: Best when class imbalance is high, and we want overall model performance.
    Macro F1: Best when each class is equally important, especially for fairness across categories.
    Weighted F1: This alters 'macro' to account for label imbalance; it can result in an F-score that is not between precision and recall
    :param name_prefix: Prefix to distinguish results e.g. 'test_'
    :param labels: Labels
    :param predictions: Predictions
    :return: Metrics dict
    """
    filtered_labels = []
    filtered_predictions = []
    for label, pred in zip(labels, predictions):
        if label != -1:
            filtered_labels.append(label)
            filtered_predictions.append(pred)
    logger.info(
        f"Computing classification metrics for {len(filtered_labels)} filtered instances (originally: {len(labels)})"
    )
    if not filtered_labels:
        return {"accuracy": -1, "f1": -1, "precision": -1, "recall": -1}
    if any(x == -1 for x in filtered_predictions):
        logger.warning("Predictions contain unassigned labels (-1)!")
        logger.warning(filtered_predictions)

    metrics = {
        f"{name_prefix}accuracy": accuracy_score(filtered_labels, filtered_predictions),
        f"{name_prefix}macro_f1": f1_score(
            filtered_labels, filtered_predictions, average="macro"
        ),
        f"{name_prefix}macro_precision": precision_score(
            filtered_labels, filtered_predictions, average="macro", zero_division=0
        ),
        f"{name_prefix}macro_recall": recall_score(
            filtered_labels, filtered_predictions, average="macro", zero_division=0
        ),
        f"{name_prefix}micro_f1": f1_score(
            filtered_labels, filtered_predictions, average="micro"
        ),
        f"{name_prefix}micro_precision": precision_score(
            filtered_labels, filtered_predictions, average="micro", zero_division=0
        ),
        f"{name_prefix}micro_recall": recall_score(
            filtered_labels, filtered_predictions, average="micro", zero_division=0
        ),
        f"{name_prefix}weighted_f1": f1_score(
            filtered_labels, filtered_predictions, average="weighted"
        ),
        f"{name_prefix}weighted_precision": precision_score(
            filtered_labels, filtered_predictions, average="weighted", zero_division=0
        ),
        f"{name_prefix}weighted_recall": recall_score(
            filtered_labels, filtered_predictions, average="weighted", zero_division=0
        ),
        f"{name_prefix}support": len(filtered_labels),
        f"{name_prefix}num_classes": len(set(filtered_labels)),
    }
    return metrics


def compute_clf_metrics_by_task(labels, predictions, tasks, name_prefix="") -> dict:
    task2preds = defaultdict(list)
    task2labels = defaultdict(list)
    results = {}
    for l, p, t in zip(labels, predictions, tasks):
        task2preds[t].append(p)
        task2labels[t].append(l)
    logger.info(f"Found the following tasks: {task2labels.keys()}")
    for task in set(tasks):
        logger.info(f"Computing metrics for task {task}")
        metrics = compute_clf_metrics(task2labels[task], task2preds[task])
        results[f"{name_prefix}{task}"] = metrics
    return results


def clean_repetitions(transcript, ngram_range=(1, 3), repetition_threshold=5):
    """
    Removes excessive repetitions of words and n-grams from a transcript.

    Args:
        transcript (str): The input ASR transcript.
        ngram_range (tuple): The range of n-grams to consider (min_n, max_n).
        repetition_threshold (int): The maximum allowed repetitions before collapsing.

    Returns:
        str: The cleaned transcript.
    """

    def generate_ngrams(text, n):
        """Generate n-grams from the text."""
        tokens = text.split()
        return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    def collapse_repeated_ngrams(text, n):
        """Collapse repeated n-grams exceeding the threshold."""
        ngrams = generate_ngrams(text, n)
        counts = Counter(ngrams)
        for ngram, count in counts.items():
            if count >= repetition_threshold:
                # Replace excessive repetitions of the n-gram
                pattern = (
                    r"(\b"
                    + re.escape(ngram)
                    + r"\b(?:\s+|$)){"
                    + str(repetition_threshold)
                    + r",}"
                )
                text = re.sub(pattern, ngram + " ", text)
        return text

    # Normalize whitespace
    transcript = re.sub(r"\s+", " ", transcript.strip())

    for n in range(ngram_range[0], ngram_range[1] + 1):
        transcript = collapse_repeated_ngrams(transcript, n)

    return transcript.strip()


def extract_text_and_rating(s):
    parts = re.split(r"<\|clf\|>+", s, maxsplit=1)
    text_before = parts[0].strip() if parts else ""
    last_part = parts[1].strip() if len(parts) > 1 else ""
    match = re.search(r"\d+", last_part)
    return text_before, int(match.group()) if match else -1


def extract_text_and_condition(s):
    parts = re.split(r"<\|clf\|>+", s, maxsplit=1)
    text_before = parts[0].strip() if parts else ""
    last_part = parts[1].strip() if len(parts) > 1 else ""
    for condition, index in ETIOLOGY_MAP.items():
        if re.search(rf"\b{re.escape(condition)}\b", last_part, re.IGNORECASE):
            return text_before, index
    return text_before, -1


def extract_text_and_category(s):
    parts = re.split(r"<\|clf\|>+", s, maxsplit=1)
    text_before = parts[0].strip() if parts else ""
    last_part = parts[1].strip() if len(parts) > 1 else ""
    for category, index in CATEGORY_MAP.items():
        if re.search(rf"\b{re.escape(category)}\b", last_part, re.IGNORECASE):
            return text_before, index
    return text_before, -1


def extract_zero_shot_result(s):
    matches = re.findall(r"\b\d+\b", s)
    last_int = int(matches[-1]) if matches else -1
    return "placeholder", last_int


def check_for_clf_labels(generated_texts, tasks, is_zero_shot=False):
    filtered_texts = []
    clf_results = []
    for text, task in zip(generated_texts, tasks):
        checker_fn = extract_text_and_rating
        if "etiology" in task:
            checker_fn = extract_text_and_condition
        elif "category" in task:
            checker_fn = extract_text_and_category
        if is_zero_shot:
            checker_fn = extract_zero_shot_result
        words, clf_result = checker_fn(text)
        filtered_texts.append(words)
        clf_results.append(clf_result)
    return filtered_texts, clf_results


def print_weight_stats(model):
    logger.info("=" * 100)
    for name, param in model.named_parameters():
        if "classification_heads" in name:
            min_val = param.min().item()
            max_val = param.max().item()
            mean_val = param.mean().item()
            logger.info(
                f"Stats for '{name}':\t\t\tmin={min_val:.4f}\tmax={max_val:.4f}\tmean={mean_val:.4f} "
            )
    logger.info("=" * 100)


def write_preds_to_csv(data_tuple, csv_filename):
    """
    Writes a tuple of lists ([utt_id], [pred], [ref]) to a CSV file.

    Args:
        csv_filename (str): The path to the output CSV file.
        data_tuple (tuple): A tuple of lists (utt_id, pred, ref).
    """
    test_subset = "default"
    utt, pred, ref = data_tuple
    if not (len(pred) == len(ref)):
        logger.error("All lists in the tuple must have the same length.")
        return
    header = ["utt_id", "pred", "ref", "test_subset"]
    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, mode="a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(header)
        for i in range(len(pred)):
            writer.writerow([utt[i], pred[i], ref[i], test_subset])
