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
        ollama_host: str = None,
        model: str = None,
        input_file: Path = None,
        output_file: Path = None
    ):
        """
        Initialize data generator

        Args:
            ollama_host: Ollama server host
            model: Model name to use
            input_file: Path to input JSON with examples
            output_file: Path to save generated output
        """
        self.ollama_client = OllamaClient(host=ollama_host, model=model)
        self.prompt_builder = PromptBuilder()
        self.output_handler = OutputHandler(output_path=output_file or config.OUTPUT_JSON)
        self.input_file = input_file or config.INPUT_JSON

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

        # Check if input file exists
        if not self.input_file.exists():
            logger.error(f"Input file not found: {self.input_file}")
            return False

        logger.info("All prerequisites met")
        return True

    def load_example_data(self) -> Optional[Dict[str, List[str]]]:
        """
        Load example data from input file

        Returns:
            Dictionary of category -> example sentences, or None if error
        """
        logger.info(f"Loading example data from {self.input_file}")

        data = self.output_handler.load_json(self.input_file)

        if not data:
            logger.error("Failed to load example data")
            return None

        if not self.prompt_builder.validate_example_data(data):
            logger.error("Example data validation failed")
            return None

        logger.info(f"Loaded {len(data)} categories with example sentences")
        return data

    def generate(
        self,
        num_sentences: int = None,
        categories: List[str] = None,
        max_retries: int = 3
    ) -> Optional[Dict[str, List[str]]]:
        """
        Generate new mental health sentences

        Args:
            num_sentences: Number of sentences to generate per category
            categories: Specific categories to generate for (None for all)
            max_retries: Maximum number of retries on failure

        Returns:
            Dictionary of category -> generated sentences, or None if error
        """
        logger.info("Starting sentence generation...")

        # Load example data
        example_data = self.load_example_data()
        if not example_data:
            return None

        # Build prompt
        prompt = self.prompt_builder.build_prompt(
            example_data=example_data,
            num_sentences=num_sentences,
            categories=categories
        )

        if not prompt:
            logger.error("Failed to build prompt")
            return None

        # Generate with retries
        for attempt in range(1, max_retries + 1):
            logger.info(f"Generation attempt {attempt}/{max_retries}")

            # Call Ollama model
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

            logger.info(f"Successfully generated data on attempt {attempt}")
            return generated_data

        logger.error(f"Failed to generate data after {max_retries} attempts")
        return None

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

    def generate_for_category(
        self,
        category: str,
        num_sentences: int = None
    ) -> Optional[List[str]]:
        """
        Generate sentences for a specific category

        Args:
            category: Mental health category name
            num_sentences: Number of sentences to generate

        Returns:
            List of generated sentences, or None if error
        """
        logger.info(f"Generating sentences for category: {category}")

        # Load example data
        example_data = self.load_example_data()
        if not example_data:
            return None

        if category not in example_data:
            logger.error(f"Category '{category}' not found in example data")
            logger.info(f"Available categories: {list(example_data.keys())}")
            return None

        # Build category-specific prompt
        prompt = self.prompt_builder.build_category_specific_prompt(
            category=category,
            examples=example_data[category],
            num_sentences=num_sentences
        )

        # Generate
        response = self.ollama_client.generate(prompt)
        if not response:
            logger.error("Generation failed")
            return None

        # Extract and return sentences
        generated_data = self.ollama_client.extract_json_from_response(response)
        if not generated_data or category not in generated_data:
            logger.error("Failed to extract category data from response")
            return None

        return generated_data[category]
