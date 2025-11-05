# Mental Health Classifier - Fine-tuning Guide

Train a LLaMA-based multi-label classifier to categorize mental health symptoms.

## 🎯 Overview

This package fine-tunes LLaMA models for **multi-label binary classification** of mental health categories:
- **Psychosis**
- **Anxiety**
- **Depression**
- **Mania**

Each category has an independent binary classifier (presence/absence).

## 📋 Prerequisites

### 1. Install Dependencies

```bash
pip install -r requirements_finetuning.txt
```

### 2. Get a LLaMA Model

**Option A: Download from HuggingFace**
```bash
# Login to HuggingFace (required for LLaMA)
huggingface-cli login

# The script will automatically download:
# - meta-llama/Llama-3.2-1B (recommended for starting)
# - meta-llama/Llama-2-7b-hf (larger, better performance)
```

**Option B: Use Ollama model** (advanced)
- Ollama models need to be converted to HuggingFace format
- It's easier to use HuggingFace directly

### 3. Prepare Training Data

Your data should be in JSON format:

**Format 1: Category-based (from data generator)**
```json
{
  "Psychosis": ["sentence 1", "sentence 2", ...],
  "Anxiety": ["sentence 1", "sentence 2", ...],
  "Depression": ["sentence 1", "sentence 2", ...],
  "Mania": ["sentence 1", "sentence 2", ...]
}
```

**Format 2: Multi-label (for overlapping categories)**
```json
[
  {
    "text": "I feel anxious and can't sleep",
    "labels": ["Anxiety", "Depression"]
  },
  {
    "text": "I hear voices telling me things",
    "labels": ["Psychosis"]
  }
]
```

## 🚀 Quick Start

### Step 1: Prepare Data

Use your generated data or create training data:

```bash
# Use data generated from the data_generator package
# The output files are already in the correct format!
```

### Step 2: Train the Model

**Basic training (CPU/small GPU):**
```bash
python train_classifier.py \
    --train-file data/output/generated_sentences_v1_all.json \
    --model-name meta-llama/Llama-3.2-1B \
    --epochs 3 \
    --batch-size 8 \
    --output-dir finetuning/checkpoints/run_1
```

**With GPU and FP16:**
```bash
python train_classifier.py \
    --train-file data/output/generated_sentences_v1_all.json \
    --model-name meta-llama/Llama-3.2-1B \
    --epochs 5 \
    --batch-size 16 \
    --fp16 \
    --output-dir finetuning/checkpoints/run_1
```

**Large model with quantization:**
```bash
python train_classifier.py \
    --train-file data/output/generated_sentences_v1_all.json \
    --model-name meta-llama/Llama-2-7b-hf \
    --use-4bit \
    --epochs 3 \
    --batch-size 4 \
    --output-dir finetuning/checkpoints/llama2_7b
```

### Step 3: Make Predictions

**Single text:**
```bash
python predict_classifier.py \
    --model-path finetuning/checkpoints/run_1/final_model \
    --text "I keep hearing voices and feeling paranoid"
```

**With detailed explanation:**
```bash
python predict_classifier.py \
    --model-path finetuning/checkpoints/run_1/final_model \
    --text "I can't sleep and feel hopeless about everything" \
    --explain
```

**Batch prediction:**
```bash
python predict_classifier.py \
    --model-path finetuning/checkpoints/run_1/final_model \
    --input-file data/test_sentences.json \
    --output-file predictions.json \
    --batch-size 32
```

## 📊 Training Options

### Model Selection

| Model | Size | RAM Required | GPU Required | Training Time |
|-------|------|--------------|--------------|---------------|
| Llama-3.2-1B | 1B | 8GB | Optional | ~1 hour |
| Llama-2-7b-hf | 7B | 16GB+ | Recommended | ~4 hours |
| Llama-2-7b-hf (4-bit) | 7B | 8GB | Required | ~6 hours |

### Training Arguments

```bash
python train_classifier.py \
    --train-file <path>          # Training data JSON
    --val-file <path>            # Optional validation data
    --test-file <path>           # Optional test data
    --model-name <name>          # HuggingFace model name
    --epochs <int>               # Number of epochs (default: 3)
    --batch-size <int>           # Batch size (default: 8)
    --learning-rate <float>      # Learning rate (default: 2e-5)
    --max-length <int>           # Max sequence length (default: 512)
    --output-dir <path>          # Output directory
    --fp16                       # Use FP16 training
    --use-4bit                   # Use 4-bit quantization
    --use-8bit                   # Use 8-bit quantization
    --resume-from-checkpoint <path>  # Resume training
```

## 📈 Monitoring Training

### TensorBoard

```bash
tensorboard --logdir finetuning/checkpoints/run_1
```

### Logs

- Console output: Real-time progress
- `training.log`: Detailed training log
- Checkpoints: Saved in `output_dir`

## 🔍 Prediction Options

```bash
python predict_classifier.py \
    --model-path <path>          # Path to trained model
    --text "<text>"              # Single text to classify
    --input-file <path>          # Batch prediction from file
    --output-file <path>         # Save predictions to file
    --threshold <float>          # Probability threshold (default: 0.5)
    --explain                    # Show detailed explanation
    --batch-size <int>           # Batch size (default: 32)
```

## 💡 Examples

### Example 1: Train on Generated Data

```bash
# Generate data first
python main.py --all --count 50

# Train classifier
python train_classifier.py \
    --train-file data/output/generated_sentences_v1_all.json \
    --model-name meta-llama/Llama-3.2-1B \
    --epochs 5 \
    --batch-size 8 \
    --output-dir finetuning/checkpoints/my_classifier
```

### Example 2: Test the Model

```bash
# Test single sentence
python predict_classifier.py \
    --model-path finetuning/checkpoints/my_classifier/final_model \
    --text "I feel so anxious I can barely leave the house" \
    --explain
```

### Example 3: Production Deployment

```python
from finetuning.predictor import MentalHealthPredictor

# Load model
predictor = MentalHealthPredictor("finetuning/checkpoints/my_classifier/final_model")

# Predict
result = predictor.predict("I hear voices talking about me", threshold=0.5)

print(result['predicted_categories'])
# Output: ['Psychosis']

# Get probabilities
for category, info in result['predictions'].items():
    print(f"{category}: {info['probability']:.3f}")
```

## 📁 Project Structure

```
finetuning/
├── __init__.py
├── config.py              # Configuration classes
├── data_loader.py         # Data loading and preprocessing
├── model_loader.py        # Model loading with classification head
├── trainer.py             # Training pipeline
├── predictor.py           # Inference/prediction
├── metrics.py             # Evaluation metrics
├── data/                  # Training data
├── models/                # Downloaded models
└── checkpoints/           # Training checkpoints
    └── run_1/
        ├── checkpoint-100/
        ├── checkpoint-200/
        └── final_model/
```

## 🎛️ Advanced Usage

### Custom Training Loop

```python
from finetuning.trainer import MentalHealthTrainer
from finetuning.config import ModelConfig, TrainingConfig, DataConfig
from pathlib import Path

# Configure
model_config = ModelConfig(model_name="meta-llama/Llama-3.2-1B")
training_config = TrainingConfig(num_epochs=5, batch_size=16)
data_config = DataConfig()

# Train
trainer = MentalHealthTrainer(
    model_config=model_config,
    training_config=training_config,
    data_config=data_config,
    output_dir=Path("finetuning/checkpoints/custom")
)

train_result, val_metrics = trainer.train(
    train_file=Path("data/train.json"),
    val_file=Path("data/val.json")
)
```

### Batch Inference

```python
from finetuning.predictor import MentalHealthPredictor

predictor = MentalHealthPredictor("finetuning/checkpoints/run_1/final_model")

texts = [
    "I can't stop worrying about everything",
    "I see things that others don't see",
    "I have so much energy I don't need sleep",
    "I feel hopeless and empty inside"
]

results = predictor.predict_batch(texts, batch_size=4)

for text, result in zip(texts, results):
    print(f"\nText: {text}")
    print(f"Categories: {result['predicted_categories']}")
```

## 📊 Evaluation Metrics

The model is evaluated using:

- **Exact Match**: All labels must match perfectly
- **Hamming Loss**: Fraction of incorrect labels
- **Jaccard Score**: Intersection over union
- **F1 Score**: Per-class and averaged
- **Precision/Recall**: Per-class metrics

## 🐛 Troubleshooting

### Out of Memory

```bash
# Reduce batch size
--batch-size 4

# Use gradient accumulation
# (Edit finetuning/config.py: gradient_accumulation_steps=8)

# Use quantization
--use-4bit
```

### Slow Training

```bash
# Use FP16
--fp16

# Use smaller model
--model-name meta-llama/Llama-3.2-1B

# Use GPU
# Ensure CUDA is available: torch.cuda.is_available()
```

### Model Access Denied

```bash
# Login to HuggingFace
huggingface-cli login

# Request access to LLaMA models at:
# https://huggingface.co/meta-llama
```

## 🎓 Tips for Best Results

1. **Data Quality**: More diverse, high-quality examples = better performance
2. **Balance Classes**: Ensure each category has similar number of examples
3. **Validation Set**: Always use a separate validation set
4. **Hyperparameters**: Start with defaults, then tune learning rate
5. **Epochs**: 3-5 epochs usually sufficient, more may overfit
6. **Threshold**: Adjust threshold (0.3-0.7) based on precision/recall needs

## 📚 Next Steps

1. Generate more training data using the data generator
2. Combine multiple generated datasets
3. Fine-tune the model
4. Evaluate on held-out test set
5. Deploy for inference

## 🤝 Integration with Data Generator

```bash
# 1. Generate training data
python main.py --all --count 100

# 2. Train classifier
python train_classifier.py \
    --train-file data/output/generated_sentences_v1_all.json \
    --model-name meta-llama/Llama-3.2-1B \
    --output-dir finetuning/checkpoints/v1

# 3. Test classifier
python predict_classifier.py \
    --model-path finetuning/checkpoints/v1/final_model \
    --text "I feel anxious and hear strange voices"
```

Perfect workflow! 🚀
