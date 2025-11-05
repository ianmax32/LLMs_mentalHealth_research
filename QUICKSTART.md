# Quick Start Guide

## Step-by-Step Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

If `pip` is not available, use:
```bash
python -m pip install -r requirements.txt
```

### 2. Install and Setup Ollama

**Download Ollama:**
- Visit: https://ollama.ai
- Download and install for your OS

**Pull the deepseek-r1 model:**
```bash
ollama pull deepseek-r1
```

**Verify Ollama is running:**
```bash
ollama list
```

You should see `deepseek-r1` in the list.

### 3. Test Your Setup

Run the test script to verify everything is configured correctly:

```bash
python test_setup.py
```

This will check:
- ✓ Input file exists and is valid
- ✓ Output directory is writable
- ✓ Ollama is running and model is available

### 4. Generate Your First Dataset

**Generate 3 sentences for all categories (quick test):**
```bash
python main.py --all --count 3
```

**Generate 5 sentences for a specific category:**
```bash
python main.py --category "Anxiety" --count 5
```

**Generate 10 sentences for all categories:**
```bash
python main.py --all --count 10
```

### 5. Check the Output

The generated data will be saved to:
```
data/output/generated_sentences.json
```

Open this file to see your generated mental health symptom sentences!

## Example Output

```json
{
  "metadata": {
    "generated_at": "2025-11-03T10:30:00.000000",
    "total_categories": 4,
    "total_sentences": 12
  },
  "data": {
    "Psychosis": [
      "I keep hearing conversations that aren't there...",
      "My thoughts feel like they're being broadcast...",
      "The shadows on the wall seem to form faces..."
    ],
    "Anxiety": [...],
    "Depression": [...],
    "Mania": [...]
  }
}
```

## Troubleshooting

### "Model deepseek-r1 is not available"
```bash
ollama pull deepseek-r1
```

### "Connection refused"
Make sure Ollama is running. On some systems you may need to start it:
```bash
ollama serve
```

### Import errors
Make sure you're running from the project directory:
```bash
cd LLMs_mentalHealth_research
python main.py --all
```

## What's Next?

1. **Customize the prompt** - Edit `data_generator/config.py` to modify the prompt template
2. **Add more examples** - Edit `data/input/mental_health_sentences.json` to add more example sentences
3. **Generate larger datasets** - Use `--count 20` or higher for more data
4. **Combine categories** - Use `--categories "Anxiety" "Depression"` to focus on specific conditions

## Need Help?

Check the full README.md for detailed documentation and all available options.
