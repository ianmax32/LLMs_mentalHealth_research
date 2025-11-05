"""
Data loader and preprocessor for mental health classification
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


@dataclass
class DataExample:
    """Single training example"""
    text: str
    labels: List[int]  # Binary labels for each category


class MentalHealthDataLoader:
    """Load and preprocess mental health classification data"""

    def __init__(self, categories: List[str]):
        """
        Initialize data loader

        Args:
            categories: List of mental health categories
        """
        self.categories = categories
        self.num_labels = len(categories)
        self.label2id = {label: idx for idx, label in enumerate(categories)}

    def load_from_json(self, json_path: Path) -> List[DataExample]:
        """
        Load data from JSON file

        Expected format:
        {
          "Psychosis": ["sentence 1", "sentence 2", ...],
          "Anxiety": ["sentence 1", "sentence 2", ...],
          ...
        }

        Args:
            json_path: Path to JSON file

        Returns:
            List of DataExample objects
        """
        logger.info(f"Loading data from {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Handle data with metadata
        if "data" in data:
            data = data["data"]

        examples = []

        # Convert to multi-label format
        for category, sentences in data.items():
            if category not in self.categories:
                logger.warning(f"Category '{category}' not in predefined categories, skipping")
                continue

            for sentence in sentences:
                # Create binary labels (1 for this category, 0 for others)
                labels = [0] * self.num_labels
                labels[self.label2id[category]] = 1

                examples.append(DataExample(
                    text=sentence.strip(),
                    labels=labels
                ))

        logger.info(f"Loaded {len(examples)} examples from {len(data)} categories")
        return examples

    def load_multi_label_json(self, json_path: Path) -> List[DataExample]:
        """
        Load data from multi-label JSON format

        Expected format:
        [
          {
            "text": "I feel anxious and depressed",
            "labels": ["Anxiety", "Depression"]
          },
          ...
        ]

        Args:
            json_path: Path to JSON file

        Returns:
            List of DataExample objects
        """
        logger.info(f"Loading multi-label data from {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        examples = []

        for item in data:
            text = item.get("text", "").strip()
            label_names = item.get("labels", [])

            if not text:
                continue

            # Convert label names to binary vector
            labels = [0] * self.num_labels
            for label_name in label_names:
                if label_name in self.label2id:
                    labels[self.label2id[label_name]] = 1
                else:
                    logger.warning(f"Unknown label '{label_name}', skipping")

            examples.append(DataExample(
                text=text,
                labels=labels
            ))

        logger.info(f"Loaded {len(examples)} multi-label examples")
        return examples

    def split_data(
        self,
        examples: List[DataExample],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42
    ) -> Tuple[List[DataExample], List[DataExample], List[DataExample]]:
        """
        Split data into train/val/test sets

        Args:
            examples: List of examples
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            random_state: Random seed

        Returns:
            Tuple of (train_examples, val_examples, test_examples)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"

        # First split: train vs (val + test)
        train_examples, temp_examples = train_test_split(
            examples,
            test_size=(val_ratio + test_ratio),
            random_state=random_state,
            shuffle=True
        )

        # Second split: val vs test
        val_size = val_ratio / (val_ratio + test_ratio)
        val_examples, test_examples = train_test_split(
            temp_examples,
            test_size=(1 - val_size),
            random_state=random_state,
            shuffle=True
        )

        logger.info(f"Data split: train={len(train_examples)}, "
                   f"val={len(val_examples)}, test={len(test_examples)}")

        return train_examples, val_examples, test_examples

    def save_split_data(
        self,
        train_examples: List[DataExample],
        val_examples: List[DataExample],
        test_examples: List[DataExample],
        output_dir: Path
    ):
        """
        Save split data to JSON files

        Args:
            train_examples: Training examples
            val_examples: Validation examples
            test_examples: Test examples
            output_dir: Output directory
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        def save_examples(examples: List[DataExample], filename: str):
            data = [
                {
                    "text": ex.text,
                    "labels": ex.labels
                }
                for ex in examples
            ]

            filepath = output_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved {len(examples)} examples to {filepath}")

        save_examples(train_examples, "train.json")
        save_examples(val_examples, "val.json")
        save_examples(test_examples, "test.json")

    def compute_class_weights(
        self,
        examples: List[DataExample]
    ) -> np.ndarray:
        """
        Compute class weights for imbalanced data

        Args:
            examples: List of examples

        Returns:
            Array of class weights
        """
        # Count positive samples per class
        label_counts = np.zeros(self.num_labels)

        for example in examples:
            label_counts += np.array(example.labels)

        # Compute weights (inverse frequency)
        total_samples = len(examples)
        weights = total_samples / (self.num_labels * label_counts + 1e-6)

        logger.info(f"Class weights: {dict(zip(self.categories, weights))}")
        return weights

    @staticmethod
    def get_label_statistics(examples: List[DataExample], categories: List[str]) -> Dict:
        """
        Get statistics about labels in the dataset

        Args:
            examples: List of examples
            categories: List of category names

        Returns:
            Dictionary with statistics
        """
        total = len(examples)
        label_counts = np.zeros(len(categories))

        multi_label_count = 0

        for example in examples:
            labels_array = np.array(example.labels)
            label_counts += labels_array

            if labels_array.sum() > 1:
                multi_label_count += 1

        stats = {
            "total_examples": total,
            "multi_label_examples": multi_label_count,
            "single_label_examples": total - multi_label_count,
            "label_distribution": {
                cat: int(count) for cat, count in zip(categories, label_counts)
            },
            "label_percentages": {
                cat: f"{100 * count / total:.2f}%"
                for cat, count in zip(categories, label_counts)
            }
        }

        return stats
