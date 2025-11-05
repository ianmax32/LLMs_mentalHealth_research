# Complete Workflow: Data Generation → Fine-tuning → Prediction

End-to-end guide for creating a mental health classification system.

## 📋 Overview

This project consists of two integrated packages:

1. **Data Generator**: Generate synthetic mental health symptom data using Ollama deepseek-r1
2. **Classifier Fine-tuning**: Train a LLaMA-based multi-label classifier

## 🎯 Complete Workflow

### Phase 1: Generate Training Data

#### Step 1: Setup Data Generator

```bash
# Install dependencies
pip install -r requirements.txt

# Install Ollama and deepseek-r1
ollama pull deepseek-r1

# Test setup
python test_setup.py
```

#### Step 2: Generate Synthetic Data

```bash
# Quick test (3 sentences per category)
python main.py --all --count 3

# Generate substantial dataset (50 per category = 200 total)
generate_all.bat 50

# Or generate specific categories
python main.py --category "Anxiety" --count 100
python main.py --category "Depression" --count 100
```

**Output**: Versioned JSON files in `data/output/`
- `generated_sentences_v1_Anxiety.json`
- `generated_sentences_v1_Depression.json`
- etc.

### Phase 2: Prepare Training Data

#### Step 1: Combine Generated Data (if needed)

```bash
# Prepare and split data for training
python prepare_training_data.py \
    --input-file data/output/generated_sentences_v1_all.json \
    --output-dir finetuning/data \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15
```

**Output**: Train/val/test splits in `finetuning/data/`

### Phase 3: Fine-tune Classifier

#### Step 1: Install Fine-tuning Dependencies

```bash
pip install -r requirements_finetuning.txt

# Login to HuggingFace (for LLaMA access)
huggingface-cli login
```

#### Step 2: Train the Model

**Option A: Quick training (CPU/small GPU)**
```bash
python train_classifier.py \
    --train-file finetuning/data/train.json \
    --val-file finetuning/data/val.json \
    --model-name meta-llama/Llama-3.2-1B \
    --epochs 3 \
    --batch-size 8 \
    --output-dir finetuning/checkpoints/run_1
```

**Option B: GPU with FP16**
```bash
python train_classifier.py \
    --train-file finetuning/data/train.json \
    --val-file finetuning/data/val.json \
    --model-name meta-llama/Llama-3.2-1B \
    --epochs 5 \
    --batch-size 16 \
    --fp16 \
    --output-dir finetuning/checkpoints/run_1
```

**Output**: Trained model in `finetuning/checkpoints/run_1/final_model/`

### Phase 4: Evaluate and Use the Classifier

#### Step 1: Test Single Prediction

```bash
python predict_classifier.py \
    --model-path finetuning/checkpoints/run_1/final_model \
    --text "I hear voices and feel like people are watching me" \
    --explain
```

#### Step 2: Batch Prediction

```bash
python predict_classifier.py \
    --model-path finetuning/checkpoints/run_1/final_model \
    --input-file finetuning/data/test.json \
    --output-file predictions.json \
    --batch-size 32
```

## 📊 Example End-to-End Run

```bash
# 1. Generate 100 sentences per category (400 total)
generate_all.bat 100

# 2. Prepare training data
python prepare_training_data.py \
    --input-file data/output/generated_sentences_v1_Anxiety.json \
    --output-dir finetuning/data

# 3. Train classifier
python train_classifier.py \
    --train-file finetuning/data/train.json \
    --val-file finetuning/data/val.json \
    --model-name meta-llama/Llama-3.2-1B \
    --epochs 5 \
    --batch-size 8 \
    --output-dir finetuning/checkpoints/model_v1

# 4. Test the model
python predict_classifier.py \
    --model-path finetuning/checkpoints/model_v1/final_model \
    --text "I can't sleep and everything feels hopeless" \
    --explain
```

## 🎨 Advanced Workflows

### Workflow 1: Iterative Improvement

```bash
# Round 1: Initial data
python main.py --all --count 50
python train_classifier.py --train-file data/output/generated_sentences_v1_all.json ...

# Round 2: More data for weak categories
python main.py --category "Psychosis" --count 100
python prepare_training_data.py --input-file data/output/generated_sentences_v2_Psychosis.json ...
python train_classifier.py --resume-from-checkpoint checkpoints/run_1/checkpoint-XXX ...
```

### Workflow 2: Multi-Model Ensemble

```bash
# Train multiple models
python train_classifier.py --model-name meta-llama/Llama-3.2-1B --output-dir checkpoints/llama3_1b
python train_classifier.py --model-name meta-llama/Llama-2-7b-hf --use-4bit --output-dir checkpoints/llama2_7b

# Compare predictions
python predict_classifier.py --model-path checkpoints/llama3_1b/final_model --text "..."
python predict_classifier.py --model-path checkpoints/llama2_7b/final_model --text "..."
```

### Workflow 3: Production Deployment

```python
# Load once, predict many times
from finetuning.predictor import MentalHealthPredictor

predictor = MentalHealthPredictor("finetuning/checkpoints/run_1/final_model")

# Batch inference
texts = [...]
results = predictor.predict_batch(texts, batch_size=32)

# API integration
def classify_text(text: str):
    result = predictor.predict(text, threshold=0.5)
    return {
        "categories": result['predicted_categories'],
        "confidence": result['predictions']
    }
```

## 📈 Performance Optimization

### Data Generation

- **Speed**: Generate categories in parallel
- **Quality**: Use higher counts for better diversity
- **Balance**: Ensure similar counts per category

### Training

- **GPU**: Enable FP16 for 2x speedup
- **Batch Size**: Increase for faster training (if memory allows)
- **Quantization**: Use 4-bit for large models on small GPUs

### Inference

- **Batch Processing**: Use `predict_batch()` for multiple texts
- **Caching**: Keep model loaded for repeated predictions
- **Threshold Tuning**: Adjust based on precision/recall needs

## 🔧 Troubleshooting

### Data Generation Issues

| Issue | Solution |
|-------|----------|
| "No response" | Generate one category at a time |
| Timeout | Reduce --count value |
| Model not found | Run `ollama pull deepseek-r1` |

### Training Issues

| Issue | Solution |
|-------|----------|
| Out of memory | Reduce batch size, use quantization |
| Slow training | Use FP16, smaller model, GPU |
| Model access denied | Login: `huggingface-cli login` |
| Poor performance | More/better training data, longer training |

## 📁 Final Project Structure

```
LLMs_mentalHealth_research/
├── data_generator/              # Data generation package
│   ├── config.py
│   ├── ollama_client.py
│   ├── prompt_builder.py
│   ├── data_generator.py
│   └── output_handler.py
│
├── finetuning/                  # Fine-tuning package
│   ├── config.py
│   ├── data_loader.py
│   ├── model_loader.py
│   ├── trainer.py
│   ├── predictor.py
│   ├── metrics.py
│   ├── data/                    # Training data
│   ├── models/                  # Downloaded models
│   └── checkpoints/             # Training checkpoints
│
├── data/
│   ├── input/                   # Example data
│   └── output/                  # Generated data (versioned)
│
├── main.py                      # Data generation CLI
├── train_classifier.py          # Training CLI
├── predict_classifier.py        # Prediction CLI
├── prepare_training_data.py     # Data preparation
│
├── requirements.txt             # Data gen dependencies
├── requirements_finetuning.txt  # Training dependencies
│
├── README.md                    # Data generator docs
├── FINETUNING_README.md         # Fine-tuning docs
├── VERSIONING.md                # Versioning system
└── COMPLETE_WORKFLOW.md         # This file
```

## 🎓 Best Practices

1. **Data Quality > Quantity**: 100 high-quality examples > 1000 low-quality
2. **Balanced Datasets**: Similar counts per category
3. **Validation**: Always use separate val/test sets
4. **Versioning**: Let the system auto-version your files
5. **Monitoring**: Watch training metrics, check for overfitting
6. **Testing**: Test on real examples before deployment
7. **Iteration**: Generate data → Train → Evaluate → Repeat

## 🚀 Quick Commands Reference

```bash
# Data Generation
python main.py --all --count 50
generate_all.bat 100

# Data Preparation
python prepare_training_data.py --input-file data/output/generated_sentences_v1_all.json

# Training
python train_classifier.py --train-file finetuning/data/train.json --model-name meta-llama/Llama-3.2-1B

# Prediction
python predict_classifier.py --model-path finetuning/checkpoints/run_1/final_model --text "..."

# Monitoring
tensorboard --logdir finetuning/checkpoints/run_1
```

## 📚 Documentation

- **README.md**: Data generator guide
- **FINETUNING_README.md**: Fine-tuning guide
- **VERSIONING.md**: File versioning system
- **QUICKSTART.md**: Quick setup guide

---

**You now have a complete pipeline for mental health text classification!** 🎉

Generate data → Prepare → Train → Predict → Deploy
