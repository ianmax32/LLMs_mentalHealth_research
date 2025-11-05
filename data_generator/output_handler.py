"""
Output handler for saving generated data
"""

import json
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class OutputHandler:
    """Handle saving generated data to files"""

    def __init__(self, output_path: Path = None):
        """
        Initialize output handler

        Args:
            output_path: Path to save output file
        """
        self.output_path = output_path

    @staticmethod
    def generate_versioned_filename(
        base_path: Path,
        categories: Optional[List[str]] = None
    ) -> Path:
        """
        Generate a versioned filename with category information

        Args:
            base_path: Base output path
            categories: List of categories (None means "all")

        Returns:
            Versioned filename like: generated_sentences_v1_Anxiety.json
                                     generated_sentences_v2_all.json
        """
        output_dir = base_path.parent
        base_name = base_path.stem  # e.g., "generated_sentences"
        extension = base_path.suffix  # e.g., ".json"

        # Determine category suffix
        if not categories or len(categories) == 0:
            category_suffix = "all"
        elif len(categories) == 1:
            category_suffix = categories[0]
        else:
            # Multiple specific categories
            category_suffix = "_".join(categories)

        # Find existing versioned files with same category
        pattern = f"{base_name}_v*_{category_suffix}{extension}"
        existing_files = list(output_dir.glob(pattern))

        # Extract version numbers
        version_numbers = []
        for file in existing_files:
            match = re.search(r'_v(\d+)_', file.stem)
            if match:
                version_numbers.append(int(match.group(1)))

        # Determine next version
        next_version = max(version_numbers, default=0) + 1

        # Generate new filename
        new_filename = f"{base_name}_v{next_version}_{category_suffix}{extension}"
        new_path = output_dir / new_filename

        logger.info(f"Generated versioned filename: {new_filename}")
        return new_path

    def save_json(
        self,
        data: Dict[str, List[str]],
        output_path: Path = None,
        include_metadata: bool = True
    ) -> bool:
        """
        Save generated data to JSON file

        Args:
            data: Dictionary of category -> generated sentences
            output_path: Path to save file (overrides instance path)
            include_metadata: Whether to include generation metadata

        Returns:
            True if successful, False otherwise
        """
        try:
            path = output_path or self.output_path
            if not path:
                logger.error("No output path specified")
                return False

            # Ensure output directory exists
            path.parent.mkdir(parents=True, exist_ok=True)

            # Prepare output data
            output_data = {}

            if include_metadata:
                output_data["metadata"] = {
                    "generated_at": datetime.now().isoformat(),
                    "total_categories": len(data),
                    "total_sentences": sum(len(sentences) for sentences in data.values())
                }

            output_data["data"] = data

            # Write to file
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Successfully saved output to {path}")
            logger.info(f"Generated {output_data.get('metadata', {}).get('total_sentences', 0)} sentences across {len(data)} categories")

            return True

        except IOError as e:
            logger.error(f"IO error writing file: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error saving output: {e}")
            return False

    def load_json(self, input_path: Path) -> Dict[str, List[str]]:
        """
        Load data from JSON file

        Args:
            input_path: Path to input JSON file

        Returns:
            Dictionary of category -> sentences
        """
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Handle both direct data and data with metadata
            if "data" in data:
                logger.info(f"Loaded data with metadata from {input_path}")
                return data["data"]
            else:
                logger.info(f"Loaded data from {input_path}")
                return data

        except FileNotFoundError:
            logger.error(f"File not found: {input_path}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error loading file: {e}")
            return {}

    def append_to_existing(
        self,
        new_data: Dict[str, List[str]],
        output_path: Path = None
    ) -> bool:
        """
        Append new data to existing output file

        Args:
            new_data: New generated data to append
            output_path: Path to output file

        Returns:
            True if successful, False otherwise
        """
        try:
            path = output_path or self.output_path
            if not path:
                logger.error("No output path specified")
                return False

            # Load existing data if file exists
            if path.exists():
                existing_data = self.load_json(path)
                logger.info(f"Loaded existing data from {path}")
            else:
                existing_data = {}
                logger.info("No existing file found, creating new one")

            # Merge data
            for category, sentences in new_data.items():
                if category in existing_data:
                    existing_data[category].extend(sentences)
                    logger.info(f"Appended {len(sentences)} sentences to existing category '{category}'")
                else:
                    existing_data[category] = sentences
                    logger.info(f"Added new category '{category}' with {len(sentences)} sentences")

            # Save merged data
            return self.save_json(existing_data, path)

        except Exception as e:
            logger.error(f"Error appending to existing file: {e}")
            return False

    @staticmethod
    def validate_output_data(data: Dict[str, List[str]]) -> bool:
        """
        Validate output data structure

        Args:
            data: Data to validate

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(data, dict):
            logger.error("Output data must be a dictionary")
            return False

        for category, sentences in data.items():
            if not isinstance(category, str):
                logger.error(f"Category must be string, got {type(category)}")
                return False

            if not isinstance(sentences, list):
                logger.error(f"Sentences must be a list, got {type(sentences)}")
                return False

            if not all(isinstance(s, str) for s in sentences):
                logger.error(f"All sentences must be strings in category '{category}'")
                return False

        logger.info(f"Validated output data with {len(data)} categories")
        return True
