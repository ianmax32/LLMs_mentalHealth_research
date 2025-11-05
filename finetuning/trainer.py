"""
Trainer for mental health classification model
"""

import logging
import torch
from pathlib import Path
from typing import Optional, List
from dataclasses import asdict

from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from datasets import Dataset

from .config import ModelConfig, TrainingConfig, DataConfig, CATEGORIES
from .model_loader import MentalHealthModelLoader
from .data_loader import MentalHealthDataLoader, DataExample
from .metrics import compute_metrics

logger = logging.getLogger(__name__)


class MentalHealthTrainer:
    """Trainer for mental health classification"""

    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        data_config: DataConfig,
        output_dir: Path
    ):
        """
        Initialize trainer

        Args:
            model_config: Model configuration
            training_config: Training configuration
            data_config: Data configuration
            output_dir: Output directory for checkpoints
        """
        self.model_config = model_config
        self.training_config = training_config
        self.data_config = data_config
        self.output_dir = output_dir

        self.model_loader = None
        self.data_loader = None
        self.trainer = None

    def prepare_data(
        self,
        train_file: Path,
        val_file: Optional[Path] = None,
        test_file: Optional[Path] = None
    ):
        """
        Prepare datasets

        Args:
            train_file: Training data file
            val_file: Validation data file (optional)
            test_file: Test data file (optional)
        """
        logger.info("Preparing datasets...")

        self.data_loader = MentalHealthDataLoader(categories=CATEGORIES)

        # Load training data
        train_examples = self.data_loader.load_from_json(train_file)

        # Load or split validation data
        if val_file:
            val_examples = self.data_loader.load_from_json(val_file)
            test_examples = None
            if test_file:
                test_examples = self.data_loader.load_from_json(test_file)
        else:
            # Split data
            train_examples, val_examples, test_examples = self.data_loader.split_data(
                train_examples,
                train_ratio=self.data_config.train_ratio,
                val_ratio=self.data_config.val_ratio,
                test_ratio=self.data_config.test_ratio,
                random_state=self.training_config.seed
            )

        # Print statistics
        train_stats = self.data_loader.get_label_statistics(train_examples, CATEGORIES)
        logger.info(f"Training set statistics: {train_stats}")

        # Compute class weights if needed
        if self.data_config.class_weights is None:
            class_weights = self.data_loader.compute_class_weights(train_examples)
            self.data_config.class_weights = class_weights.tolist()

        return train_examples, val_examples, test_examples

    def prepare_model(self):
        """Load model and tokenizer"""
        logger.info("Loading model and tokenizer...")

        self.model_loader = MentalHealthModelLoader(self.model_config)
        model, tokenizer = self.model_loader.load_model_and_tokenizer()

        return model, tokenizer

    def create_datasets(
        self,
        train_examples: List[DataExample],
        val_examples: List[DataExample],
        tokenizer
    ):
        """
        Create HuggingFace Dataset objects

        Args:
            train_examples: Training examples
            val_examples: Validation examples
            tokenizer: Tokenizer

        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        def examples_to_dict(examples: List[DataExample]):
            return {
                "text": [ex.text for ex in examples],
                "labels": [ex.labels for ex in examples]
            }

        def tokenize_function(examples):
            # Tokenize texts
            tokenized = tokenizer(
                examples["text"],
                padding=self.data_config.padding,
                truncation=self.data_config.truncation,
                max_length=self.data_config.max_length,
                return_tensors=None
            )

            # Convert labels to float for BCEWithLogitsLoss
            tokenized["labels"] = [
                [float(label) for label in labels]
                for labels in examples["labels"]
            ]

            return tokenized

        # Create datasets
        train_dict = examples_to_dict(train_examples)
        val_dict = examples_to_dict(val_examples)

        train_dataset = Dataset.from_dict(train_dict)
        val_dataset = Dataset.from_dict(val_dict)

        # Apply tokenization
        train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )

        val_dataset = val_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )

        logger.info(f"Created datasets: train={len(train_dataset)}, val={len(val_dataset)}")

        return train_dataset, val_dataset

    def train(
        self,
        train_file: Path,
        val_file: Optional[Path] = None,
        test_file: Optional[Path] = None,
        resume_from_checkpoint: Optional[str] = None
    ):
        """
        Run training

        Args:
            train_file: Training data file
            val_file: Validation data file
            test_file: Test data file
            resume_from_checkpoint: Path to checkpoint to resume from

        Returns:
            Training results
        """
        # Prepare data
        train_examples, val_examples, test_examples = self.prepare_data(
            train_file, val_file, test_file
        )

        # Prepare model
        model, tokenizer = self.prepare_model()

        # Create datasets
        train_dataset, val_dataset = self.create_datasets(
            train_examples, val_examples, tokenizer
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.training_config.num_epochs,
            per_device_train_batch_size=self.training_config.batch_size,
            per_device_eval_batch_size=self.training_config.batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            learning_rate=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            warmup_ratio=self.training_config.warmup_ratio,
            logging_steps=self.training_config.logging_steps,
            eval_strategy=self.training_config.eval_strategy,
            eval_steps=self.training_config.eval_steps if self.training_config.eval_strategy == "steps" else None,
            save_strategy=self.training_config.save_strategy,
            save_total_limit=self.training_config.save_total_limit,
            load_best_model_at_end=self.training_config.load_best_model_at_end,
            metric_for_best_model=self.training_config.metric_for_best_model,
            greater_is_better=self.training_config.greater_is_better,
            fp16=self.training_config.fp16,
            bf16=self.training_config.bf16,
            optim=self.training_config.optimizer,
            lr_scheduler_type=self.training_config.lr_scheduler_type,
            max_grad_norm=self.training_config.max_grad_norm,
            seed=self.training_config.seed,
            report_to=self.training_config.report_to,
            remove_unused_columns=False
        )

        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # Early stopping
        callbacks = []
        if self.training_config.load_best_model_at_end:
            callbacks.append(
                EarlyStoppingCallback(early_stopping_patience=3)
            )

        # Create trainer
        self.trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks
        )

        # Train
        logger.info("Starting training...")
        train_result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # Save final model
        logger.info("Saving final model...")
        self.trainer.save_model(str(self.output_dir / "final_model"))

        # Evaluate on validation set
        logger.info("Evaluating on validation set...")
        val_metrics = self.trainer.evaluate()

        # Evaluate on test set if available
        if test_examples:
            logger.info("Evaluating on test set...")
            test_dataset, _ = self.create_datasets(test_examples, [], tokenizer)
            test_metrics = self.trainer.evaluate(test_dataset)
            logger.info(f"Test metrics: {test_metrics}")

        logger.info("Training completed!")

        return train_result, val_metrics

    def save_model(self, output_dir: Optional[Path] = None):
        """Save trained model"""
        if self.model_loader is None:
            raise ValueError("Model must be loaded first")

        save_dir = output_dir or (self.output_dir / "final_model")
        self.model_loader.save_model(save_dir)
