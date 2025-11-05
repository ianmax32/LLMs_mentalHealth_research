"""
Evaluation metrics for multi-label classification
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    hamming_loss,
    jaccard_score,
    classification_report,
    multilabel_confusion_matrix
)
from typing import Dict


def compute_metrics(eval_pred) -> Dict[str, float]:
    """
    Compute metrics for multi-label classification

    Args:
        eval_pred: Tuple of (predictions, labels)

    Returns:
        Dictionary of metrics
    """
    predictions, labels = eval_pred

    # Convert logits to binary predictions
    # For multi-label, apply sigmoid and threshold at 0.5
    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        predictions = sigmoid(predictions)
        predictions = (predictions > 0.5).astype(int)
    else:
        predictions = (predictions > 0).astype(int)

    # Ensure labels are integers
    labels = labels.astype(int)

    # Compute metrics
    metrics = {}

    # Exact match (all labels must match)
    exact_match = accuracy_score(labels, predictions)
    metrics['exact_match'] = exact_match

    # Hamming loss (fraction of incorrectly predicted labels)
    metrics['hamming_loss'] = hamming_loss(labels, predictions)

    # Jaccard score (intersection over union)
    metrics['jaccard'] = jaccard_score(labels, predictions, average='samples', zero_division=0)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )

    # Macro-averaged metrics (equal weight to each class)
    metrics['precision_macro'] = precision.mean()
    metrics['recall_macro'] = recall.mean()
    metrics['f1_macro'] = f1.mean()

    # Micro-averaged metrics (total true positives, false positives, false negatives)
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        labels, predictions, average='micro', zero_division=0
    )
    metrics['precision_micro'] = precision_micro
    metrics['recall_micro'] = recall_micro
    metrics['f1_micro'] = f1_micro

    # Weighted-averaged metrics (weighted by support)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    metrics['precision_weighted'] = precision_weighted
    metrics['recall_weighted'] = recall_weighted
    metrics['f1_weighted'] = f1_weighted

    return metrics


def compute_detailed_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    categories: list
) -> Dict:
    """
    Compute detailed metrics including per-class performance

    Args:
        predictions: Binary predictions array
        labels: True labels array
        categories: List of category names

    Returns:
        Dictionary with detailed metrics
    """
    # Overall metrics
    metrics = {
        'exact_match': accuracy_score(labels, predictions),
        'hamming_loss': hamming_loss(labels, predictions),
        'jaccard': jaccard_score(labels, predictions, average='samples', zero_division=0)
    }

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )

    per_class_metrics = {}
    for idx, category in enumerate(categories):
        per_class_metrics[category] = {
            'precision': float(precision[idx]),
            'recall': float(recall[idx]),
            'f1': float(f1[idx]),
            'support': int(support[idx])
        }

    metrics['per_class'] = per_class_metrics

    # Confusion matrices per class
    cm = multilabel_confusion_matrix(labels, predictions)
    confusion_matrices = {}
    for idx, category in enumerate(categories):
        tn, fp, fn, tp = cm[idx].ravel()
        confusion_matrices[category] = {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        }

    metrics['confusion_matrices'] = confusion_matrices

    # Generate classification report
    report = classification_report(
        labels, predictions,
        target_names=categories,
        zero_division=0,
        output_dict=True
    )
    metrics['classification_report'] = report

    return metrics


def print_metrics_summary(metrics: Dict, categories: list):
    """
    Print a formatted summary of metrics

    Args:
        metrics: Dictionary of metrics
        categories: List of category names
    """
    print("=" * 80)
    print("EVALUATION METRICS SUMMARY")
    print("=" * 80)

    # Overall metrics
    print("\n📊 Overall Metrics:")
    print(f"  Exact Match Accuracy: {metrics['exact_match']:.4f}")
    print(f"  Hamming Loss:         {metrics['hamming_loss']:.4f}")
    print(f"  Jaccard Score:        {metrics['jaccard']:.4f}")

    # Averaged metrics
    print("\n📈 Averaged Metrics:")
    if 'per_class' in metrics:
        precisions = [metrics['per_class'][cat]['precision'] for cat in categories]
        recalls = [metrics['per_class'][cat]['recall'] for cat in categories]
        f1s = [metrics['per_class'][cat]['f1'] for cat in categories]

        print(f"  Macro Precision:  {np.mean(precisions):.4f}")
        print(f"  Macro Recall:     {np.mean(recalls):.4f}")
        print(f"  Macro F1:         {np.mean(f1s):.4f}")

    # Per-class metrics
    if 'per_class' in metrics:
        print("\n🏷️  Per-Class Metrics:")
        print(f"  {'Category':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
        print("  " + "-" * 65)

        for category in categories:
            cat_metrics = metrics['per_class'][category]
            print(f"  {category:<15} "
                  f"{cat_metrics['precision']:<12.4f} "
                  f"{cat_metrics['recall']:<12.4f} "
                  f"{cat_metrics['f1']:<12.4f} "
                  f"{cat_metrics['support']:<10}")

    # Confusion matrices
    if 'confusion_matrices' in metrics:
        print("\n🔢 Confusion Matrices:")
        for category in categories:
            cm = metrics['confusion_matrices'][category]
            print(f"\n  {category}:")
            print(f"    TP: {cm['true_positives']:<6} FP: {cm['false_positives']:<6}")
            print(f"    FN: {cm['false_negatives']:<6} TN: {cm['true_negatives']:<6}")

    print("\n" + "=" * 80)
