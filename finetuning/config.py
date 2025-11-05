"""
Configuration for fine-tuning mental health classifier
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
FINETUNING_DIR = BASE_DIR / "finetuning"
DATA_DIR = FINETUNING_DIR / "data"
MODELS_DIR = FINETUNING_DIR / "models"
CHECKPOINTS_DIR = FINETUNING_DIR / "checkpoints"

# Mental health categories (labels for multi-label classification)
CATEGORIES = [
    "Psychosis",
    "Anxiety",
    "Depression",
    "Mania"
]

LABEL2ID = {label: idx for idx, label in enumerate(CATEGORIES)}
ID2LABEL = {idx: label for idx, label in enumerate(CATEGORIES)}
NUM_LABELS = len(CATEGORIES)


@dataclass
class ModelConfig:
    """Model configuration"""

    # Model selection
    model_name: str = "meta-llama/Llama-3.2-1B"  # or "meta-llama/Llama-2-7b-hf"
    use_pretrained: bool = True
    num_labels: int = NUM_LABELS

    # Model architecture
    max_length: int = 512
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1

    # Quantization (for large models)
    use_4bit: bool = False
    use_8bit: bool = False


@dataclass
class TrainingConfig:
    """Training configuration"""

    # Training parameters
    num_epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    # Optimization
    optimizer: str = "adamw_torch"
    lr_scheduler_type: str = "linear"
    max_grad_norm: float = 1.0

    # Evaluation
    eval_strategy: str = "epoch"  # "steps" or "epoch"
    eval_steps: int = 500
    save_strategy: str = "epoch"
    save_total_limit: int = 3

    # Logging
    logging_steps: int = 10
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])

    # Mixed precision
    fp16: bool = False  # Set to True if GPU supports it
    bf16: bool = False

    # Seed
    seed: int = 42

    # Early stopping
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1_macro"
    greater_is_better: bool = True


@dataclass
class DataConfig:
    """Data configuration"""

    # Data paths
    train_file: Optional[Path] = None
    val_file: Optional[Path] = None
    test_file: Optional[Path] = None

    # Data split ratios (if single file provided)
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Preprocessing
    max_length: int = 512
    padding: str = "max_length"
    truncation: bool = True

    # Label handling
    label_smoothing: float = 0.0
    class_weights: Optional[List[float]] = None


# Default configurations
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_DATA_CONFIG = DataConfig()
