"""
Prepare training data for fine-tuning
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from finetuning.data_loader import MentalHealthDataLoader
from finetuning.config import CATEGORIES


def setup_logging():
    """Setup logging"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Prepare and split training data"
    )

    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Input JSON file with mental health data"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="finetuning/data",
        help="Output directory for split data"
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training set ratio"
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation set ratio"
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test set ratio"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("PREPARING TRAINING DATA")
    logger.info("=" * 60)

    # Load data
    data_loader = MentalHealthDataLoader(categories=CATEGORIES)
    examples = data_loader.load_from_json(Path(args.input_file))

    logger.info(f"\nLoaded {len(examples)} examples")

    # Print statistics
    stats = data_loader.get_label_statistics(examples, CATEGORIES)
    logger.info(f"\nDataset statistics:")
    logger.info(f"  Total examples: {stats['total_examples']}")
    logger.info(f"  Single-label: {stats['single_label_examples']}")
    logger.info(f"  Multi-label: {stats['multi_label_examples']}")
    logger.info(f"\nLabel distribution:")
    for category, count in stats['label_distribution'].items():
        percentage = stats['label_percentages'][category]
        logger.info(f"  {category:<15} {count:>5} ({percentage})")

    # Split data
    logger.info(f"\nSplitting data:")
    logger.info(f"  Train: {args.train_ratio:.0%}")
    logger.info(f"  Val:   {args.val_ratio:.0%}")
    logger.info(f"  Test:  {args.test_ratio:.0%}")

    train_examples, val_examples, test_examples = data_loader.split_data(
        examples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.seed
    )

    # Save split data
    output_dir = Path(args.output_dir)
    data_loader.save_split_data(
        train_examples,
        val_examples,
        test_examples,
        output_dir
    )

    logger.info(f"\n✓ Data prepared and saved to {output_dir}")
    logger.info(f"\nOutput files:")
    logger.info(f"  - {output_dir / 'train.json'}")
    logger.info(f"  - {output_dir / 'val.json'}")
    logger.info(f"  - {output_dir / 'test.json'}")

    # Compute class weights
    class_weights = data_loader.compute_class_weights(train_examples)
    logger.info(f"\nRecommended class weights:")
    for category, weight in zip(CATEGORIES, class_weights):
        logger.info(f"  {category:<15} {weight:.4f}")

    logger.info("\n" + "=" * 60)
    logger.info("Ready to train! Use:")
    logger.info(f"  python train_classifier.py --train-file {output_dir / 'train.json'} --val-file {output_dir / 'val.json'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
