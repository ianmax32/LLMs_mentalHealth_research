"""
Configuration settings for the data generator
"""

import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"

# Input/Output files
INPUT_JSON = INPUT_DIR / "mental_health_sentences.json"
OUTPUT_JSON = OUTPUT_DIR / "generated_sentences.json"

# Ollama settings
OLLAMA_MODEL = "deepseek-r1"
OLLAMA_HOST = "http://localhost:11434"  # Default Ollama host
OLLAMA_TIMEOUT = 300  # 5 minutes timeout for generation

# Generation settings
DEFAULT_SENTENCES_PER_CATEGORY = 5
MIN_SENTENCES = 10
MAX_SENTENCES = 500

# Prompt template
PROMPT_TEMPLATE = """[INST]
Your task is to analyze the provided JSON file, which describes symptoms of various mental health conditions.

1. **Identify the categories** of mental health conditions present in the JSON file.

2. **For each category**, analyze the provided example sentences and **generate {num_sentences} new sentences** that could also be indicative of that particular condition.

**Example JSON File:**

{example_json}

**Output:**

Present your findings in a structured format, where each key represents a category and the corresponding value is a list of {num_sentences} generated sentences for that category.

For example:

```json
{{
  "Mood Disorders": [
    "I feel a deep sense of emptiness and hopelessness.",
    "I have lost all interest in things I used to enjoy.",
    "I experience significant changes in my appetite and sleep patterns.",
    "I have difficulty concentrating and making decisions.",
    "I feel guilty and worthless most of the time."
  ],
  "Anxiety Disorders": [
    "I experience frequent and intense feelings of worry and nervousness.",
    "I have difficulty relaxing and often feel on edge.",
    "I avoid social situations due to fear of judgment or embarrassment.",
    "I experience physical symptoms such as sweating, trembling, and rapid heartbeat.",
    "I have intrusive thoughts that I cannot seem to control."
  ]
}}
```

**IMPORTANT**: Return ONLY valid JSON. Do not include any explanatory text before or after the JSON output.
[/INST]"""

# Mental health categories
CATEGORIES = [
    "Psychosis",
    "Anxiety",
    "Depression",
    "Mania"
]

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
