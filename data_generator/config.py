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
OLLAMA_TIMEOUT = 1200  # 5 minutes timeout for generation

# Generation settings
DEFAULT_SENTENCES_PER_CATEGORY = 5
MIN_SENTENCES = 100
MAX_SENTENCES = 500

# Static prompt template - only category changes
PROMPT_TEMPLATE = """You are an expert in mental health symptoms and clinical psychology. Your task is to generate realistic first-person sentences that describe symptoms of {category}.

Generate exactly {num_sentences} unique paragraphs of sentences with a good description of what is happening, that someone experiencing {category} symptoms might say, think or feel. Also try to simulate different tones. Each sentence should:
- Be written in first person perspective
- Sound natural and conversational
- Describe a specific symptom, feeling, or experience related to {category}
- Vary in tone and severity

Return ONLY a valid JSON object with this exact format:
{{
  "{category}": [
    "sentence 1",
    "sentence 2",
    ...
  ]
}}

Do not include any explanatory text, markdown formatting, or code blocks. Output only the raw JSON."""

# Mental health categories
CATEGORIES = [
    "Psychosis",
    "Anxiety",
    "Depression",
    "Mania",
    "Bipolar Disorder",
    "Post-Traumatic Stress Disorder",
    "Easting Disorder",
    "Disruptive behavior and dissocial disorders",
    "Schizophrenia",
    "Neurodevelopmental disorders"
]

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
