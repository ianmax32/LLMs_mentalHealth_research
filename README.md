# Mental Health Data Generator

Generate synthetic mental health symptom sentences using Ollama's deepseek-r1 model.

## Overview

This package uses the deepseek-r1 model running on Ollama to generate realistic mental health symptom descriptions across four categories:
- **Psychosis**: Hallucinations, delusions, disorganized thinking
- **Anxiety**: Worry, panic, physical symptoms
- **Depression**: Low mood, lack of motivation, hopelessness
- **Mania**: Elevated mood, high energy, grandiose thoughts

## Prerequisites

### 1. Install Ollama

Download and install Ollama from [https://ollama.ai](https://ollama.ai)

### 2. Pull the deepseek-r1 model

```bash
ollama pull deepseek-r1
```

### 3. Verify Ollama is running

```bash
ollama list
```

You should see `deepseek-r1` in the list of available models.

## Installation

1. Clone or navigate to the project directory:

```bash
cd LLMs_mentalHealth_research
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```
LLMs_mentalHealth_research/
├── data_generator/
│   ├── __init__.py
│   ├── config.py              # Configuration settings
│   ├── ollama_client.py       # Ollama API client
│   ├── prompt_builder.py      # Prompt templates
│   ├── data_generator.py      # Main generation logic
│   └── output_handler.py      # Output file handling
├── data/
│   ├── input/
│   │   └── mental_health_sentences.json  # Example sentences
│   └── output/
│       └── generated_sentences.json      # Generated output
├── main.py                    # CLI entry point
├── requirements.txt
└── README.md
```

## Usage

### Generate for all categories

Generate 5 sentences for each category (default):

```bash
python main.py --all
```

Generate 10 sentences for each category:

```bash
python main.py --all --count 10
```

### Generate for a specific category

```bash
python main.py --category "Anxiety" --count 5
```

### Generate for multiple categories

```bash
python main.py --categories "Anxiety" "Depression" --count 7
```

### Custom output file

```bash
python main.py --all --output data/output/my_custom_output.json
```

### Append to existing file

```bash
python main.py --category "Mania" --count 3 --append
```

### Use custom Ollama host

```bash
python main.py --all --ollama-host http://192.168.1.100:11434
```

### Enable debug logging

```bash
python main.py --all --log-level DEBUG
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--all` | Generate for all categories | - |
| `--category` | Generate for a specific category | - |
| `--categories` | Generate for multiple categories | - |
| `--count` | Number of sentences per category | 5 |
| `--output` | Output file path | data/output/generated_sentences.json |
| `--input` | Input file with examples | data/input/mental_health_sentences.json |
| `--append` | Append to existing output file | False |
| `--ollama-host` | Ollama server URL | http://localhost:11434 |
| `--model` | Ollama model name | deepseek-r1 |
| `--log-level` | Logging level (DEBUG/INFO/WARNING/ERROR) | INFO |

## Output Format

The generated JSON file includes metadata and the generated sentences:

```json
{
  "metadata": {
    "generated_at": "2025-11-03T10:30:00.000000",
    "total_categories": 4,
    "total_sentences": 20
  },
  "data": {
    "Psychosis": [
      "Generated sentence 1...",
      "Generated sentence 2...",
      ...
    ],
    "Anxiety": [
      "Generated sentence 1...",
      ...
    ],
    ...
  }
}
```

## Examples

### Example 1: Quick generation for all categories

```bash
python main.py --all
```

This will generate 5 sentences for each of the 4 categories (20 total) and save to `data/output/generated_sentences.json`.

### Example 2: Generate more data for anxiety research

```bash
python main.py --category "Anxiety" --count 20 --output data/output/anxiety_dataset.json
```

### Example 3: Incremental data collection

```bash
# First batch
python main.py --all --count 5

# Add more later
python main.py --categories "Depression" "Anxiety" --count 10 --append
```

## Troubleshooting

### "Model deepseek-r1 is not available"

Make sure you've pulled the model:
```bash
ollama pull deepseek-r1
```

### "Connection refused" error

Ensure Ollama is running:
```bash
ollama serve
```

### Generation takes too long

- The deepseek-r1 model can be slow depending on your hardware
- Default timeout is 5 minutes (configurable in `config.py`)
- Consider using a smaller count value for faster results

### Invalid JSON output

- The model sometimes returns malformed JSON
- The generator automatically retries up to 3 times
- Check the logs for details about what went wrong

## Configuration

Edit `data_generator/config.py` to customize:

- Default number of sentences
- Ollama connection settings
- Prompt template
- Logging format

## Logs

Logs are written to:
- Console (stdout)
- `data_generator.log` file