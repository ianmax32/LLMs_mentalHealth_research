"""
Main data generator for mental health sentences
"""

import logging
from typing import Dict, List, Optional
from pathlib import Path

from .ollama_client import OllamaClient
from .prompt_builder import PromptBuilder
from .output_handler import OutputHandler
from . import config

logger = logging.getLogger(__name__)


class MentalHealthDataGenerator:
    """Generate mental health symptom sentences using Ollama deepseek-r1"""

    def __init__(
        self,
        model: str = None,
        output_file: Path = None
    ):
        """
        Initialize data generator

        Args:
            model: Model name to use
            output_file: Path to save generated output
        """
        self.ollama_client = OllamaClient(model=model)
        self.prompt_builder = PromptBuilder()
        self.output_handler = OutputHandler(output_path=output_file or config.OUTPUT_JSON)

        logger.info("Initialized MentalHealthDataGenerator")

    def check_prerequisites(self) -> bool:
        """
        Check if all prerequisites are met

        Returns:
            True if ready to generate, False otherwise
        """
        logger.info("Checking prerequisites...")

        # Check if model is available
        if not self.ollama_client.check_model_available():
            logger.error(f"Model {self.ollama_client.model} is not available in Ollama")
            logger.info("Please install it using: ollama pull deepseek-r1")
            return False

        logger.info("All prerequisites met")
        return True

    def generate_for_category(
        self,
        category: str,
        num_sentences: int = None,
        max_retries: int = 3
    ) -> Optional[Dict[str, List[str]]]:
        """
        Generate sentences for a specific category

        Args:
            category: Mental health category name
            num_sentences: Number of sentences to generate
            max_retries: Maximum number of retries on failure

        Returns:
            Dictionary with category -> generated sentences, or None if error
        """
        logger.info(f"Generating sentences for category: {category}")

        # Validate category
        if not self.prompt_builder.validate_category(category):
            return None

        # Build static prompt for this category
        prompt = self.prompt_builder.build_prompt(
            category=category,
            num_sentences=num_sentences
        )

        # Generate with retries
        for attempt in range(1, max_retries + 1):
            logger.info(f"Generation attempt {attempt}/{max_retries} for {category}")

            # Call Ollama model via command line
            response = self.ollama_client.generate(prompt)

            if not response:
                logger.error(f"Generation attempt {attempt} failed: No response")
                continue

            # Extract JSON from response
            generated_data = self.ollama_client.extract_json_from_response(response)

            if not generated_data:
                logger.error(f"Generation attempt {attempt} failed: Could not extract JSON")
                logger.debug(f"Response preview: {response[:500]}...")
                continue

            # Validate generated data
            if not self.output_handler.validate_output_data(generated_data):
                logger.error(f"Generation attempt {attempt} failed: Invalid data structure")
                continue

            # Ensure the category key matches what we asked for
            if category not in generated_data:
                # Try to find a similar key and rename it
                if len(generated_data) == 1:
                    old_key = list(generated_data.keys())[0]
                    generated_data[category] = generated_data.pop(old_key)
                    logger.info(f"Renamed key '{old_key}' to '{category}'")
                else:
                    logger.error(f"Expected category '{category}' not found in response")
                    continue

            logger.info(f"Successfully generated {len(generated_data[category])} sentences for {category}")
            return generated_data

        logger.error(f"Failed to generate data for {category} after {max_retries} attempts")
        return None

    def generate(
        self,
        num_sentences: int = None,
        categories: List[str] = None,
        max_retries: int = 3
    ) -> Optional[Dict[str, List[str]]]:
        """
        Generate new mental health sentences for one or more categories

        Args:
            num_sentences: Number of sentences to generate per category
            categories: Specific categories to generate for (None for all)
            max_retries: Maximum number of retries on failure

        Returns:
            Dictionary of category -> generated sentences, or None if error
        """
        logger.info("Starting sentence generation...")

        # Determine which categories to generate
        if categories is None:
            categories = config.CATEGORIES

        all_generated = {}

        for category in categories:
            result = self.generate_for_category(
                category=category,
                num_sentences=num_sentences,
                max_retries=max_retries
            )

            if result:
                all_generated.update(result)
            else:
                logger.warning(f"Skipping category {category} due to generation failure")

        if not all_generated:
            logger.error("Failed to generate any data")
            return None

        return all_generated

    def generate_and_save(
        self,
        num_sentences: int = None,
        categories: List[str] = None,
        output_file: Path = None,
        append: bool = False
    ) -> bool:
        """
        Generate sentences and save to file

        Args:
            num_sentences: Number of sentences to generate per category
            categories: Specific categories to generate for
            output_file: Output file path (overrides default)
            append: Whether to append to existing file

        Returns:
            True if successful, False otherwise
        """
        logger.info("Starting generation and save process...")

        # Check prerequisites
        if not self.check_prerequisites():
            return False

        # Generate data
        generated_data = self.generate(
            num_sentences=num_sentences,
            categories=categories
        )

        if not generated_data:
            logger.error("Generation failed")
            return False

        # Determine output path with versioning
        if output_file:
            # User specified a custom output file - use as is
            output_path = output_file
        else:
            # Generate versioned filename
            base_path = self.output_handler.output_path or config.OUTPUT_JSON
            output_path = self.output_handler.generate_versioned_filename(
                base_path=base_path,
                categories=categories
            )

        if append:
            success = self.output_handler.append_to_existing(
                new_data=generated_data,
                output_path=output_path
            )
        else:
            success = self.output_handler.save_json(
                data=generated_data,
                output_path=output_path
            )

        if success:
            logger.info(f"Successfully saved generated data to {output_path}")
        else:
            logger.error("Failed to save generated data")

        return success
