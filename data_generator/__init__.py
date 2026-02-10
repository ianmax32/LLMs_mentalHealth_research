"""
Data Generator Package for Mental Health Sentences
Uses Ollama deepseek-r1 model to generate synthetic mental health symptom data
"""

__version__ = "1.0.0"

from .data_combiner import DataCombiner, CombineResult, combine_json_files

__all__ = [
    "DataCombiner",
    "CombineResult",
    "combine_json_files",
]
