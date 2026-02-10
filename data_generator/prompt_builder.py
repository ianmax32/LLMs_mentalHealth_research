"""
Prompt builder for generating mental health sentences
"""

import logging
from typing import List
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

    def build_prompt(self, category: str, num_sentences: int = None) -> str:
        """
        Build a static prompt for sentence generation - only category changes

        Args:
            category: Mental health category (Psychosis, Anxiety, Depression, Mania)
            num_sentences: Number of sentences to generate

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

        # Build prompt with only category and num_sentences
        prompt = self.template.format(
            category=category,
            num_sentences=num_sentences
        )

        logger.info(f"Built prompt for category '{category}', {num_sentences} sentences")
        logger.debug(f"Prompt length: {len(prompt)} characters")

        return prompt

    @staticmethod
    def get_available_categories() -> List[str]:
        """
        Get list of available mental health categories

        Returns:
            List of category names
        """
        return config.CATEGORIES.copy()

    @staticmethod
    def validate_category(category: str) -> bool:
        """
        Validate if a category is valid

        Args:
            category: Category name to validate

        Returns:
            True if valid, False otherwise
        """
        if category not in config.CATEGORIES:
            logger.error(f"Invalid category '{category}'. Available: {config.CATEGORIES}")
            return False
        return True
