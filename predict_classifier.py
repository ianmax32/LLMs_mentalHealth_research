"""
Prediction script for mental health classifier
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from finetuning.predictor import MentalHealthPredictor, format_prediction_output


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Predict mental health categories for text"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to fine-tuned model"
    )

    parser.add_argument(
        "--text",
        type=str,
        help="Single text to classify"
    )

    parser.add_argument(
        "--input-file",
        type=str,
        help="JSON file with texts to classify"
    )

    parser.add_argument(
        "--output-file",
        type=str,
        help="Output file for predictions"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for positive prediction"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing"
    )

    parser.add_argument(
        "--explain",
        action="store_true",
        help="Show detailed explanation of predictions"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Validate arguments
    if not args.text and not args.input_file:
        logger.error("Please provide either --text or --input-file")
        return 1

    # Load model
    logger.info(f"Loading model from {args.model_path}")
    predictor = MentalHealthPredictor(Path(args.model_path))

    # Single text prediction
    if args.text:
        if args.explain:
            explanation = predictor.explain_prediction(args.text, args.threshold)

            print("\n" + "=" * 60)
            print("DETAILED PREDICTION EXPLANATION")
            print("=" * 60)
            print(f"\nText: {explanation['text'][:100]}...")
            print(f"\n{explanation['summary']}")
            print("\n" + "-" * 60)
            print("Category Analysis (sorted by probability):")
            print("-" * 60)

            for cat_info in explanation['categories']:
                status = "✓" if cat_info['predicted'] else "✗"
                confidence = cat_info['confidence'].upper()
                prob = cat_info['probability']
                print(f"  {status} {cat_info['category']:<15} "
                      f"{prob:.4f}  [{confidence}]")

            print("=" * 60)
        else:
            result = predictor.predict(args.text, args.threshold, return_probabilities=True)
            print(format_prediction_output(result))

        return 0

    # Batch prediction from file
    if args.input_file:
        logger.info(f"Loading texts from {args.input_file}")

        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Handle different formats
        if isinstance(data, list):
            if isinstance(data[0], str):
                texts = data
            else:
                texts = [item.get("text", "") for item in data]
        elif isinstance(data, dict):
            # Assume it's generated data format
            if "data" in data:
                data = data["data"]
            # Flatten all texts
            texts = []
            for sentences in data.values():
                texts.extend(sentences)
        else:
            logger.error("Unsupported input file format")
            return 1

        logger.info(f"Found {len(texts)} texts to classify")

        # Predict
        results = predictor.predict_batch(
            texts,
            batch_size=args.batch_size,
            threshold=args.threshold
        )

        # Save or print results
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved predictions to {args.output_file}")
        else:
            # Print summary
            print("\n" + "=" * 60)
            print("BATCH PREDICTION SUMMARY")
            print("=" * 60)
            print(f"Total texts: {len(results)}")

            # Count predictions per category
            category_counts = {}
            for result in results:
                for category in result['predicted_categories']:
                    category_counts[category] = category_counts.get(category, 0) + 1

            print("\nPredictions per category:")
            for category, count in sorted(category_counts.items()):
                percentage = 100 * count / len(results)
                print(f"  {category:<15} {count:>5} ({percentage:.1f}%)")

            print("\n" + "=" * 60)

        return 0


if __name__ == "__main__":
    sys.exit(main())
