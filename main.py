"""
Main entry point for the Mental Health Data Generator
"""

import argparse
import logging
import sys
from pathlib import Path

from data_generator import config
from data_generator.data_generator import MentalHealthDataGenerator


def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=config.LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("data_generator.log")
        ]
    )


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Generate mental health symptom sentences using Ollama deepseek-r1"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate for all categories"
    )

    parser.add_argument(
        "--category",
        type=str,
        help="Generate for a specific category (Psychosis, Anxiety, Depression, Mania)"
    )

    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        help="Generate for multiple specific categories"
    )

    parser.add_argument(
        "--count",
        type=int,
        default=config.DEFAULT_SENTENCES_PER_CATEGORY,
        help=f"Number of sentences to generate per category (default: {config.DEFAULT_SENTENCES_PER_CATEGORY})"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (default: data/output/generated_sentences.json)"
    )

    parser.add_argument(
        "--input",
        type=str,
        help="Input file path with examples (default: data/input/mental_health_sentences.json)"
    )

    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing output file instead of overwriting"
    )

    parser.add_argument(
        "--ollama-host",
        type=str,
        default=config.OLLAMA_HOST,
        help=f"Ollama server host (default: {config.OLLAMA_HOST})"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=config.OLLAMA_MODEL,
        help=f"Ollama model to use (default: {config.OLLAMA_MODEL})"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("Mental Health Data Generator")
    logger.info("=" * 80)

    # Validate arguments
    if not args.all and not args.category and not args.categories:
        logger.error("Please specify --all, --category, or --categories")
        parser.print_help()
        return 1

    # Prepare paths
    input_file = Path(args.input) if args.input else config.INPUT_JSON
    output_file = Path(args.output) if args.output else config.OUTPUT_JSON

    # Determine categories to generate
    categories = None
    if args.category:
        categories = [args.category]
    elif args.categories:
        categories = args.categories
    # If --all, categories stays None (generates for all)

    # Initialize generator
    logger.info(f"Initializing generator with model: {args.model}")
    generator = MentalHealthDataGenerator(
        ollama_host=args.ollama_host,
        model=args.model,
        input_file=input_file,
        output_file=output_file
    )

    # Generate and save
    logger.info(f"Generating {args.count} sentences per category")
    if categories:
        logger.info(f"Target categories: {categories}")
    else:
        logger.info("Target categories: ALL")

    success = generator.generate_and_save(
        num_sentences=args.count,
        categories=categories,
        output_file=output_file,
        append=args.append
    )

    if success:
        logger.info("=" * 80)
        logger.info("Generation completed successfully!")
        logger.info(f"Output saved to: {output_file}")
        logger.info("=" * 80)
        return 0
    else:
        logger.error("=" * 80)
        logger.error("Generation failed!")
        logger.error("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
