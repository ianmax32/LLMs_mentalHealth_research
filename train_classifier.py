"""
Main training script for mental health classifier
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from finetuning.config import (
    ModelConfig,
    TrainingConfig,
    DataConfig,
    CHECKPOINTS_DIR,
    CATEGORIES
)
from finetuning.trainer import MentalHealthTrainer


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("training.log")
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train mental health classification model"
    )

    # Data arguments
    parser.add_argument(
        "--train-file",
        type=str,
        required=True,
        help="Path to training data JSON file"
    )

    parser.add_argument(
        "--val-file",
        type=str,
        help="Path to validation data JSON file (optional)"
    )

    parser.add_argument(
        "--test-file",
        type=str,
        help="Path to test data JSON file (optional)"
    )

    # Model arguments
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Model name or path from HuggingFace"
    )

    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="Use 4-bit quantization (for large models on limited GPU)"
    )

    parser.add_argument(
        "--use-8bit",
        action="store_true",
        help="Use 8-bit quantization"
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size per device"
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(CHECKPOINTS_DIR / "run_1"),
        help="Output directory for checkpoints"
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 mixed precision training"
    )

    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        help="Resume training from checkpoint"
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

    logger.info("=" * 80)
    logger.info("MENTAL HEALTH CLASSIFIER - TRAINING")
    logger.info("=" * 80)

    # Create configurations
    model_config = ModelConfig(
        model_name=args.model_name,
        use_4bit=args.use_4bit,
        use_8bit=args.use_8bit,
        max_length=args.max_length
    )

    training_config = TrainingConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        fp16=args.fp16
    )

    data_config = DataConfig(
        max_length=args.max_length
    )

    # Print configuration
    logger.info(f"\n📋 Configuration:")
    logger.info(f"  Model: {model_config.model_name}")
    logger.info(f"  Categories: {', '.join(CATEGORIES)}")
    logger.info(f"  Epochs: {training_config.num_epochs}")
    logger.info(f"  Batch size: {training_config.batch_size}")
    logger.info(f"  Learning rate: {training_config.learning_rate}")
    logger.info(f"  Max length: {model_config.max_length}")
    logger.info(f"  Output dir: {args.output_dir}")

    # Create trainer
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = MentalHealthTrainer(
        model_config=model_config,
        training_config=training_config,
        data_config=data_config,
        output_dir=output_dir
    )

    # Train
    try:
        train_result, val_metrics = trainer.train(
            train_file=Path(args.train_file),
            val_file=Path(args.val_file) if args.val_file else None,
            test_file=Path(args.test_file) if args.test_file else None,
            resume_from_checkpoint=args.resume_from_checkpoint
        )

        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"\nFinal model saved to: {output_dir / 'final_model'}")
        logger.info(f"\nValidation metrics:")
        for metric, value in val_metrics.items():
            if isinstance(value, float):
                logger.info(f"  {metric}: {value:.4f}")

        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
