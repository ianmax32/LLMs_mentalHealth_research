"""
Model loader for mental health classification
"""

import logging
import torch
from pathlib import Path
from typing import Optional, Dict

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig
)

from .config import ModelConfig, LABEL2ID, ID2LABEL

logger = logging.getLogger(__name__)


class MentalHealthModelLoader:
    """Load and configure models for mental health classification"""

    def __init__(self, config: ModelConfig):
        """
        Initialize model loader

        Args:
            config: Model configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None

    def load_model_and_tokenizer(
        self,
        model_name: Optional[str] = None,
        cache_dir: Optional[Path] = None
    ):
        """
        Load model and tokenizer

        Args:
            model_name: Model name or path (overrides config)
            cache_dir: Cache directory for downloading models

        Returns:
            Tuple of (model, tokenizer)
        """
        model_name = model_name or self.config.model_name

        logger.info(f"Loading model: {model_name}")

        # Configure quantization if needed
        quantization_config = None
        if self.config.use_4bit:
            logger.info("Using 4-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
        elif self.config.use_8bit:
            logger.info("Using 8-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )

        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )

        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Set pad_token to eos_token")

        # Load model with classification head
        logger.info("Loading model with classification head...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=self.config.num_labels,
            problem_type="multi_label_classification",
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            quantization_config=quantization_config,
            cache_dir=cache_dir,
            trust_remote_code=True
        )

        # Configure model
        if hasattr(self.model.config, "pad_token_id"):
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        logger.info(f"Model loaded successfully")
        logger.info(f"Model parameters: {self.count_parameters(self.model):,}")

        return self.model, self.tokenizer

    @staticmethod
    def count_parameters(model) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def save_model(self, output_dir: Path):
        """
        Save model and tokenizer

        Args:
            output_dir: Output directory
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first")

        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving model to {output_dir}")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        logger.info("Model saved successfully")

    @classmethod
    def load_finetuned_model(
        cls,
        model_path: Path,
        config: Optional[ModelConfig] = None
    ):
        """
        Load a fine-tuned model

        Args:
            model_path: Path to saved model
            config: Model configuration (optional)

        Returns:
            Instance with loaded model and tokenizer
        """
        if config is None:
            from .config import DEFAULT_MODEL_CONFIG
            config = DEFAULT_MODEL_CONFIG

        loader = cls(config)

        logger.info(f"Loading fine-tuned model from {model_path}")

        # Load tokenizer
        loader.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        # Load model
        loader.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        logger.info("Fine-tuned model loaded successfully")

        return loader
