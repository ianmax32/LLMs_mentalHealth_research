# Fine-Tuning Methods Guide

Comprehensive guide to fine-tuning LLaMA models for mental health classification.

## 🎯 What is Fine-Tuning?

**Fine-tuning** is the process of adapting a pre-trained Large Language Model (like LLaMA) to a specific task or dataset **without retraining the entire model from scratch**.

### Key Concept

- ✅ **Fine-tuning**: Start with pre-trained LLaMA → Add classification layers → Train on your data
- ❌ **Training from scratch**: Start with random weights → Train everything from zero

**Benefits of Fine-tuning:**
- Much faster (hours vs weeks)
- Requires less data (hundreds vs millions of examples)
- Leverages pre-trained language understanding
- Better performance with limited resources

---

## 🔧 Fine-Tuning Methods Available

This package supports **three fine-tuning approaches**, each with different trade-offs:

### Method 1: Standard Fine-Tuning (Default)

**What it does:**
- Loads pre-trained LLaMA model
- Adds a single linear classification layer
- Trains all model parameters

**When to use:**
- You have a powerful GPU (16GB+ VRAM)
- You want maximum performance
- Training time is not a concern

**Command:**
```bash
python train_classifier.py \
    --train-file data/train.json \
    --model-name meta-llama/Llama-3.2-1B \
    --epochs 3
```

**Trainable parameters:** ~1.2 billion (100%)

---

### Method 2: LoRA Fine-Tuning (Recommended ⭐)

**What it does:**
- Loads pre-trained LLaMA model
- **Freezes** all base model weights (no changes to pre-trained model)
- Adds small trainable "adapter" layers (LoRA)
- Only trains the adapters (~0.1-1% of parameters)

**How LoRA works:**
Instead of modifying the entire model, LoRA injects small rank-decomposition matrices into the attention layers. This achieves similar performance to full fine-tuning while training 100x fewer parameters.

**When to use:**
- Limited GPU memory (4-8GB VRAM)
- Want faster training
- Want to fine-tune large models (7B, 13B)
- **Recommended for most users!**

**Command:**
```bash
python train_classifier.py \
    --train-file data/train.json \
    --model-name meta-llama/Llama-3.2-1B \
    --use-lora \
    --lora-r 8 \
    --lora-alpha 16 \
    --epochs 3
```

**Trainable parameters:** ~2-20 million (0.1-1%)

**LoRA Configuration:**
- `--lora-r`: Rank of LoRA matrices (higher = more parameters, better performance)
  - `r=4`: Ultra-lightweight (~1M params)
  - `r=8`: Recommended balance (~2M params)
  - `r=16`: Higher capacity (~8M params)
- `--lora-alpha`: Scaling factor (typically 2× the rank)

---

### Method 3: Multi-Layer Classifier Head

**What it does:**
- Uses a deep classifier instead of single linear layer
- Architecture: Base Model → Linear → ReLU → Dropout → Linear → Output
- Adds intermediate layers for better feature transformation

**When to use:**
- Complex classification tasks
- Want more expressive classifier
- Combine with LoRA for best results

**Command:**
```bash
python train_classifier.py \
    --train-file data/train.json \
    --model-name meta-llama/Llama-3.2-1B \
    --use-custom-head \
    --epochs 3
```

**Architecture:**
```
LLaMA Base (hidden_size=2048)
    ↓
Linear(2048 → 256)
    ↓
ReLU + Dropout(0.3)
    ↓
Linear(256 → 128)
    ↓
ReLU + Dropout(0.3)
    ↓
Linear(128 → 4)  [4 categories]
```

---

### Method 4: LoRA + Multi-Layer Head (Best of Both Worlds)

**What it does:**
- Combines LoRA for efficient base model adaptation
- Uses multi-layer classifier for expressive classification
- Achieves best performance with minimal trainable parameters

**When to use:**
- Want maximum performance with efficiency
- Limited GPU but complex task
- **Best approach for most production use cases**

**Command:**
```bash
python train_classifier.py \
    --train-file data/train.json \
    --model-name meta-llama/Llama-3.2-1B \
    --use-lora \
    --lora-r 8 \
    --use-custom-head \
    --epochs 3
```

**Trainable parameters:** ~3-25 million (0.2-2%)

---

## 📊 Comparison Table

| Method | Trainable Params | GPU Memory | Training Time | Performance | Use Case |
|--------|-----------------|------------|---------------|-------------|----------|
| Standard | 1.2B (100%) | 16GB+ | Slow | Excellent | Powerful GPU |
| LoRA | 2-20M (0.1-1%) | 4-8GB | Fast | Excellent | **Recommended** |
| Custom Head | 1.2B + head | 16GB+ | Slow | Excellent+ | Complex tasks |
| LoRA + Head | 2-25M (0.2-2%) | 4-8GB | Fast | **Best** | **Production** |

---

## 🚀 Recommended Workflows

### Workflow 1: Quick Testing (Limited GPU)

```bash
# Use smallest LLaMA model + LoRA
python train_classifier.py \
    --train-file data/train.json \
    --model-name meta-llama/Llama-3.2-1B \
    --use-lora \
    --lora-r 4 \
    --epochs 3 \
    --batch-size 4
```

**Requirements:** 4GB GPU VRAM, 30 mins training

---

### Workflow 2: Production Model (Recommended)

```bash
# LLaMA 3.2 1B + LoRA + Custom Head
python train_classifier.py \
    --train-file data/train.json \
    --val-file data/val.json \
    --model-name meta-llama/Llama-3.2-1B \
    --use-lora \
    --lora-r 8 \
    --lora-alpha 16 \
    --use-custom-head \
    --epochs 5 \
    --batch-size 8 \
    --fp16 \
    --output-dir checkpoints/production
```

**Requirements:** 8GB GPU VRAM, 1-2 hours training

---

### Workflow 3: Large Model with Quantization

```bash
# LLaMA 2 7B + LoRA + 4-bit quantization
python train_classifier.py \
    --train-file data/train.json \
    --model-name meta-llama/Llama-2-7b-hf \
    --use-lora \
    --lora-r 16 \
    --use-4bit \
    --epochs 5 \
    --batch-size 4 \
    --output-dir checkpoints/llama2_7b
```

**Requirements:** 8-12GB GPU VRAM, 2-4 hours training

---

## 💡 Understanding the Training Process

### What Happens During Fine-Tuning?

1. **Load Pre-trained LLaMA Model**
   - Model already understands language from pre-training on massive datasets
   - Has 1-7 billion parameters with rich linguistic knowledge

2. **Add Classification Layers**
   - Single layer: `Linear(hidden_size → num_categories)`
   - Multi-layer: `Linear → ReLU → Dropout → Linear → ...`

3. **Configure Parameter Training**
   - **Standard**: All parameters trainable
   - **LoRA**: Freeze base model, only train adapters

4. **Train on Your Data**
   - Model learns to map mental health symptoms to categories
   - Base knowledge is preserved (not trained from scratch)
   - Only task-specific layers are heavily modified

5. **Save Fine-Tuned Model**
   - Can be used for inference on new data
   - Retains both pre-trained knowledge + your task

---

## 📈 Performance Tips

### For Best Results:

1. **Use LoRA** unless you have 16GB+ GPU
   - Set `--use-lora --lora-r 8`
   - Reduces memory by 10x with minimal performance loss

2. **Enable Mixed Precision** if GPU supports it
   - Add `--fp16` flag
   - 2x faster training, 2x less memory

3. **Use Custom Head** for complex tasks
   - Add `--use-custom-head`
   - Especially helpful with multi-label classification

4. **Combine LoRA + Custom Head**
   - `--use-lora --use-custom-head`
   - Best balance of efficiency and performance

5. **Tune LoRA Rank**
   - Start with `r=8`
   - Increase to `r=16` if performance plateaus
   - Decrease to `r=4` if memory is tight

---

## 🔬 Technical Details

### What is LoRA?

**LoRA (Low-Rank Adaptation)** works by decomposing weight updates into low-rank matrices:

```
Original: W_new = W_pretrained + ΔW  (ΔW is full-rank, huge)
LoRA: W_new = W_pretrained + B×A  (B and A are small matrices)

Where:
- W_pretrained: Frozen pre-trained weights
- B: Low-rank matrix (hidden_dim × r)
- A: Low-rank matrix (r × hidden_dim)
- r: LoRA rank (typically 4-16)
```

**Example for LLaMA 1B:**
- Standard fine-tuning: 1.2B trainable params
- LoRA (r=8): ~2M trainable params (600x reduction!)

### Multi-Layer Classifier

Instead of `hidden_state → labels` (single layer), we use:

```python
hidden_state (2048)
    ↓ Linear(2048 → 256)
representation (256)
    ↓ ReLU + Dropout
    ↓ Linear(256 → 128)
representation (128)
    ↓ ReLU + Dropout
    ↓ Linear(128 → 4)
logits (4 categories)
```

This allows the model to learn non-linear transformations of the base model features.

---

## ✅ Verification

After training, you'll see output like:

```
============================================================
LOADING PRE-TRAINED MODEL FOR FINE-TUNING
============================================================
Base model: meta-llama/Llama-3.2-1B
Use LoRA: True
Use custom classifier: True

✓ LoRA applied successfully
trainable params: 2,359,296 || all params: 1,235,896,320 || trainable%: 0.1909

✓ Model loaded successfully
Total parameters: 1,235,896,320
Trainable parameters: 2,359,296
Trainable: 0.19%
============================================================
```

**Key indicators:**
- ✅ "LOADING PRE-TRAINED MODEL" - confirms NOT training from scratch
- ✅ "trainable%: 0.19%" - confirms efficient fine-tuning with LoRA
- ✅ Only 2.4M params trained instead of 1.2B

---

## 🎓 Summary

**Fine-tuning adapts a pre-trained model to your task without retraining from scratch.**

**Recommended approach for this project:**

```bash
python train_classifier.py \
    --train-file finetuning/data/train.json \
    --val-file finetuning/data/val.json \
    --model-name meta-llama/Llama-3.2-1B \
    --use-lora \
    --lora-r 8 \
    --use-custom-head \
    --epochs 5 \
    --batch-size 8 \
    --fp16 \
    --output-dir finetuning/checkpoints/production_v1
```

This command:
- ✅ Uses pre-trained LLaMA 3.2 1B (NOT training from scratch)
- ✅ Applies LoRA for efficient parameter adaptation (0.2% trainable)
- ✅ Uses multi-layer classifier for better performance
- ✅ Trains only mental health classification capability
- ✅ Saves both base knowledge + your task

**The result:** A model that understands language (from pre-training) AND can classify mental health symptoms (from your fine-tuning) with Positive/Negative outputs for each category.

---

For more details, see:
- `FINETUNING_README.md` - General fine-tuning guide
- `COMPLETE_WORKFLOW.md` - End-to-end workflow
- `finetuning/config.py` - Configuration options
