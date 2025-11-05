"""
Prompt builder for generating mental health sentences
"""

import json
import logging
from typing import Dict, List, Optional
from . import config

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Build prompts for the deepseek-r1 model"""

    def __init__(self, template: str = None):
        """
        Initialize prompt builder

        Args:
            template: Custom prompt template (default from config)
        """
        self.template = template or config.PROMPT_TEMPLATE

    def build_prompt(
        self,
        example_data: Dict[str, List[str]],
        num_sentences: int = None,
        categories: List[str] = None
    ) -> str:
        """
        Build a prompt for sentence generation

        Args:
            example_data: Dictionary of category -> example sentences
            num_sentences: Number of sentences to generate per category
            categories: Specific categories to include (None for all)

        Returns:
            Formatted prompt string
        """
        num_sentences = num_sentences or config.DEFAULT_SENTENCES_PER_CATEGORY

        # Validate number of sentences
        if num_sentences < config.MIN_SENTENCES:
            logger.warning(f"num_sentences {num_sentences} < minimum {config.MIN_SENTENCES}, using minimum")
            num_sentences = config.MIN_SENTENCES
        elif num_sentences > config.MAX_SENTENCES:
            logger.warning(f"num_sentences {num_sentences} > maximum {config.MAX_SENTENCES}, using maximum")
            num_sentences = config.MAX_SENTENCES

        # Filter categories if specified
        if categories:
            filtered_data = {k: v for k, v in example_data.items() if k in categories}
            if not filtered_data:
                logger.error(f"None of the specified categories {categories} found in example data")
                return ""
            example_data = filtered_data

        # Format example JSON
        example_json = json.dumps(example_data, indent=2)

        # Build prompt
        prompt = self.template.format(
            num_sentences=num_sentences,
            example_json=example_json
        )

        logger.info(f"Built prompt for {len(example_data)} categories, {num_sentences} sentences each")
        logger.debug(f"Prompt length: {len(prompt)} characters")

        return prompt

    def build_category_specific_prompt(
        self,
        category: str,
        examples: List[str],
        num_sentences: int = None
    ) -> str:
        """
        Build a prompt for a specific category

        Args:
            category: Mental health category name
            examples: Example sentences for this category
            num_sentences: Number of sentences to generate

        Returns:
            Formatted prompt string
        """
        example_data = {category: examples}
        return self.build_prompt(example_data, num_sentences)

    @staticmethod
    def validate_example_data(data: Dict[str, List[str]]) -> bool:
        """
        Validate example data structure

        Args:
            data: Dictionary to validate

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(data, dict):
            logger.error("Example data must be a dictionary")
            return False

        for category, sentences in data.items():
            if not isinstance(category, str):
                logger.error(f"Category key must be string, got {type(category)}")
                return False

            if not isinstance(sentences, list):
                logger.error(f"Category '{category}' must have list of sentences, got {type(sentences)}")
                return False

            if not sentences:
                logger.error(f"Category '{category}' has no example sentences")
                return False

            if not all(isinstance(s, str) for s in sentences):
                logger.error(f"All sentences in '{category}' must be strings")
                return False

        logger.info(f"Validated example data with {len(data)} categories")
        return True
