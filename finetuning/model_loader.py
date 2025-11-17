"""
Model loader for mental health classification
"""

import logging
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig
)

from .config import ModelConfig, LABEL2ID, ID2LABEL

logger = logging.getLogger(__name__)


class MultiLayerClassifier(nn.Module):
    """
    Multi-layer classifier head for mental health classification

    This creates a deep classifier with multiple hidden layers instead of
    just a single linear layer, allowing for better feature learning.
    """

    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        hidden_dims: list = [256, 128],
        dropout: float = 0.3,
        activation: str = "relu"
    ):
        """
        Initialize multi-layer classifier

        Args:
            input_dim: Input dimension (hidden size from base model)
            num_labels: Number of output labels
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability between layers
            activation: Activation function (relu, gelu, tanh)
        """
        super().__init__()

        # Select activation function
        if activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "gelu":
            self.activation = nn.GELU()
        elif activation.lower() == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation,
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Final classification layer
        layers.append(nn.Linear(prev_dim, num_labels))

        self.classifier = nn.Sequential(*layers)

        logger.info(f"Created multi-layer classifier: {input_dim} -> {hidden_dims} -> {num_labels}")

    def forward(self, features):
        """Forward pass through classifier"""
        return self.classifier(features)


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
        Load model and tokenizer with optional LoRA and custom classifier head

        This method implements true fine-tuning by:
        1. Loading a pre-trained LLaMA model (NOT training from scratch)
        2. Optionally freezing base weights and adding LoRA adapters (parameter-efficient)
        3. Adding classification layers on top (single or multi-layer)

        Args:
            model_name: Model name or path (overrides config)
            cache_dir: Cache directory for downloading models

        Returns:
            Tuple of (model, tokenizer)
        """
        model_name = model_name or self.config.model_name

        logger.info("=" * 60)
        logger.info("LOADING PRE-TRAINED MODEL FOR FINE-TUNING")
        logger.info("=" * 60)
        logger.info(f"Base model: {model_name}")
        logger.info(f"Use LoRA: {self.config.use_lora}")
        logger.info(f"Use custom classifier: {self.config.use_custom_head}")

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

        # Load model - different approach for custom head vs standard head
        if self.config.use_custom_head:
            logger.info("Loading base model for custom classifier head...")
            # Load base model without classification head
            base_model = AutoModel.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                cache_dir=cache_dir,
                trust_remote_code=True
            )

            # Create custom multi-layer classifier
            hidden_size = base_model.config.hidden_size
            classifier = MultiLayerClassifier(
                input_dim=hidden_size,
                num_labels=self.config.num_labels,
                hidden_dims=self.config.classifier_hidden_dims,
                dropout=self.config.classifier_dropout,
                activation=self.config.classifier_activation
            )

            # Combine base model and classifier
            self.model = self._create_model_with_custom_head(base_model, classifier)
        else:
            logger.info("Loading model with standard classification head...")
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

        # Apply LoRA if enabled
        if self.config.use_lora:
            logger.info("Applying LoRA for parameter-efficient fine-tuning...")
            self.model = self._apply_lora(self.model)

        logger.info(f"✓ Model loaded successfully")
        logger.info(f"Total parameters: {self.count_parameters(self.model, trainable=False):,}")
        logger.info(f"Trainable parameters: {self.count_parameters(self.model, trainable=True):,}")

        trainable_percentage = (
            self.count_parameters(self.model, trainable=True) /
            self.count_parameters(self.model, trainable=False) * 100
        )
        logger.info(f"Trainable: {trainable_percentage:.2f}%")
        logger.info("=" * 60)

        return self.model, self.tokenizer

    def _create_model_with_custom_head(self, base_model, classifier):
        """Create a model wrapper that combines base model with custom classifier"""

        class CustomHeadModel(nn.Module):
            def __init__(self, base_model, classifier):
                super().__init__()
                self.base_model = base_model
                self.classifier = classifier
                self.config = base_model.config
                self.config.num_labels = classifier.classifier[-1].out_features
                self.config.problem_type = "multi_label_classification"

            def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
                # Get base model outputs
                outputs = self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **kwargs
                )

                # Get pooled output (use last hidden state's first token [CLS])
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    pooled_output = outputs.pooler_output
                else:
                    pooled_output = outputs.last_hidden_state[:, 0, :]

                # Pass through classifier
                logits = self.classifier(pooled_output)

                # Calculate loss if labels provided
                loss = None
                if labels is not None:
                    loss_fct = nn.BCEWithLogitsLoss()
                    loss = loss_fct(logits, labels.float())

                # Return in HuggingFace format
                from transformers.modeling_outputs import SequenceClassifierOutput
                return SequenceClassifierOutput(
                    loss=loss,
                    logits=logits,
                    hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
                    attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
                )

            def save_pretrained(self, output_dir):
                """Save model"""
                self.base_model.save_pretrained(output_dir)
                # Save classifier separately
                torch.save(self.classifier.state_dict(), f"{output_dir}/classifier.pt")

        return CustomHeadModel(base_model, classifier)

    def _apply_lora(self, model):
        """
        Apply LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning

        LoRA freezes the pre-trained model weights and injects trainable
        rank decomposition matrices, significantly reducing trainable parameters
        while maintaining performance.
        """
        try:
            from peft import LoraConfig, get_peft_model, TaskType

            logger.info("Configuring LoRA...")
            logger.info(f"  LoRA rank (r): {self.config.lora_r}")
            logger.info(f"  LoRA alpha: {self.config.lora_alpha}")
            logger.info(f"  LoRA dropout: {self.config.lora_dropout}")

            # Auto-detect target modules if not specified
            target_modules = self.config.lora_target_modules
            if target_modules is None:
                # Common LLaMA attention modules
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
                logger.info(f"  Auto-detected target modules: {target_modules}")
            else:
                logger.info(f"  Target modules: {target_modules}")

            # Configure LoRA
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=target_modules,
                bias="none",
                inference_mode=False
            )

            # Apply LoRA to model
            model = get_peft_model(model, lora_config)
            logger.info("✓ LoRA applied successfully")

            # Print trainable parameters
            model.print_trainable_parameters()

            return model

        except ImportError:
            logger.error("PEFT library not installed. Install with: pip install peft")
            logger.error("Continuing without LoRA...")
            return model

    @staticmethod
    def count_parameters(model, trainable: bool = True) -> int:
        """
        Count model parameters

        Args:
            model: PyTorch model
            trainable: If True, count only trainable params. If False, count all params.

        Returns:
            Number of parameters
        """
        if trainable:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in model.parameters())

    def save_model(self, output_dir: Path):
        """
        Save model and tokenizer

        Handles both standard models and LoRA models appropriately.

        Args:
            output_dir: Output directory
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first")

        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving model to {output_dir}")

        # Save model (handles LoRA models automatically)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Save config info
        config_info = {
            "use_lora": self.config.use_lora,
            "use_custom_head": self.config.use_custom_head,
            "model_name": self.config.model_name
        }
        import json
        with open(output_dir / "training_config.json", "w") as f:
            json.dump(config_info, f, indent=2)

        logger.info("✓ Model saved successfully")

    @classmethod
    def load_finetuned_model(
        cls,
        model_path: Path,
        config: Optional[ModelConfig] = None
    ):
        """
        Load a fine-tuned model (supports both standard and LoRA models)

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

        # Check if this is a LoRA model
        import json
        config_path = Path(model_path) / "training_config.json"
        is_lora = False
        if config_path.exists():
            with open(config_path, "r") as f:
                training_config = json.load(f)
                is_lora = training_config.get("use_lora", False)

        # Load tokenizer
        loader.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        # Load model
        if is_lora:
            try:
                from peft import PeftModel

                logger.info("Loading LoRA model...")
                # For LoRA, we need to load base model first, then LoRA adapters
                base_model = AutoModelForSequenceClassification.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )
                loader.model = base_model
                logger.info("✓ LoRA model loaded successfully")
            except ImportError:
                logger.warning("PEFT not installed, loading as standard model")
                loader.model = AutoModelForSequenceClassification.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )
        else:
            loader.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                trust_remote_code=True
            )

        logger.info("✓ Fine-tuned model loaded successfully")

        return loader
