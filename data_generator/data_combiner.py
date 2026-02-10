"""
Data Combiner Module for Mental Health Training Data

This module provides functionality to combine multiple JSON files containing
mental health sentences into a single consolidated training data file.

Usage:
    from data_generator.data_combiner import DataCombiner

    combiner = DataCombiner(input_dir="data/output")
    result = combiner.combine(output_file="training_data.json")
    print(result.summary())
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Optional


@dataclass
class CombineResult:
    """Results from a data combination operation."""
    success: bool
    output_file: str
    files_processed: int
    total_sentences_before_dedup: int
    total_sentences_after_dedup: int
    duplicates_removed: int
    sentences_per_category: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)

    def summary(self) -> str:
        """Return a formatted summary string."""
        lines = [
            "=" * 50,
            "Data Combination Summary",
            "=" * 50,
            f"Status: {'Success' if self.success else 'Failed'}",
            f"Output file: {self.output_file}",
            f"Files processed: {self.files_processed}",
            f"Total sentences before dedup: {self.total_sentences_before_dedup}",
            f"Duplicates removed: {self.duplicates_removed}",
            f"Total unique sentences: {self.total_sentences_after_dedup}",
            "",
            "Sentences per category:",
        ]
        for cat, count in self.sentences_per_category.items():
            lines.append(f"  {cat}: {count}")

        if self.errors:
            lines.append("")
            lines.append("Errors:")
            for error in self.errors:
                lines.append(f"  - {error}")

        lines.append("=" * 50)
        return "\n".join(lines)


class DataCombiner:
    """
    Combines multiple JSON data files into a single training data file.

    Attributes:
        input_dir: Directory containing source JSON files
        file_pattern: Glob pattern for matching JSON files (default: "*.json")
        exclude_files: List of filenames to exclude from combination
        categories: List of categories to include (None = all categories)
        deduplicate: Whether to remove duplicate sentences (default: True)
        verbose: Whether to print progress messages (default: False)
    """

    DEFAULT_CATEGORIES = ["Anxiety",
                          "Depression",
                          "Mania",
                          "Psychosis",
                          "Bipolar Disorder",
                          "Bipolar Disorder",
                          "Post-Traumatic Stress Disorder",
                          "Easting Disorder",
                          "Disruptive behavior and dissocial disorders",
                          "Schizophrenia",
                          "Neurodevelopmental disorders"
                          ]

    def __init__(
        self,
        input_dir: str,
        file_pattern: str = "*.json",
        exclude_files: Optional[list] = None,
        categories: Optional[list] = None,
        deduplicate: bool = True,
        verbose: bool = False
    ):
        self.input_dir = Path(input_dir)
        self.file_pattern = file_pattern
        self.exclude_files = exclude_files or []
        self.categories = categories
        self.deduplicate = deduplicate
        self.verbose = verbose

    def _log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)

    def _get_json_files(self) -> list:
        """Get list of JSON files to process."""
        pattern = str(self.input_dir / self.file_pattern)
        files = glob(pattern)

        # Filter out excluded files
        files = [
            f for f in files
            if os.path.basename(f) not in self.exclude_files
        ]

        return files

    def _read_json_file(self, filepath: str) -> tuple:
        """
        Read a JSON file and extract data.

        Returns:
            tuple: (data_dict, error_message or None)
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data, None
        except json.JSONDecodeError as e:
            return None, f"JSON decode error in {filepath}: {e}"
        except Exception as e:
            return None, f"Error reading {filepath}: {e}"

    def _remove_duplicates(self, sentences: list) -> list:
        """Remove duplicate sentences while preserving order."""
        seen = set()
        unique = []
        for sentence in sentences:
            if sentence not in seen:
                seen.add(sentence)
                unique.append(sentence)
        return unique

    def combine(
        self,
        output_file: str = "training_data.json",
        output_dir: Optional[str] = None
    ) -> CombineResult:
        """
        Combine all JSON files into a single training data file.

        Args:
            output_file: Name of the output file
            output_dir: Directory for output file (default: same as input_dir)

        Returns:
            CombineResult: Object containing combination statistics
        """
        # Initialize result tracking
        errors = []
        files_processed = 0
        total_before = 0

        # Initialize combined data structure
        combined_data = {}
        if self.categories:
            for cat in self.categories:
                combined_data[cat] = []

        # Add output file to exclude list
        exclude = self.exclude_files + [output_file]
        self.exclude_files = exclude

        # Get files to process
        json_files = self._get_json_files()
        self._log(f"Found {len(json_files)} JSON files to process...")

        # Process each file
        for json_file in json_files:
            data, error = self._read_json_file(json_file)

            if error:
                errors.append(error)
                self._log(f"  Error: {error}")
                continue

            if 'data' not in data:
                self._log(f"  Skipping {json_file}: No 'data' key found")
                continue

            for category, sentences in data['data'].items():
                # Skip categories not in filter list (if filter is set)
                if self.categories and category not in self.categories:
                    continue

                # Initialize category if not exists
                if category not in combined_data:
                    combined_data[category] = []

                combined_data[category].extend(sentences)
                total_before += len(sentences)

            files_processed += 1
            self._log(f"  Processed: {os.path.basename(json_file)}")

        # Remove duplicates if enabled
        if self.deduplicate:
            self._log("\nRemoving duplicates...")
            for category in combined_data:
                combined_data[category] = self._remove_duplicates(combined_data[category])

        # Calculate statistics
        sentences_per_category = {cat: len(sentences) for cat, sentences in combined_data.items()}
        total_after = sum(sentences_per_category.values())
        duplicates_removed = total_before - total_after

        # Determine output path
        if output_dir:
            output_path = Path(output_dir) / output_file
        else:
            output_path = self.input_dir / output_file

        # Create output structure
        output = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "source_files_processed": files_processed,
                "total_categories": len([cat for cat in combined_data if combined_data[cat]]),
                "total_sentences": total_after,
                "duplicates_removed": duplicates_removed,
                "sentences_per_category": sentences_per_category
            },
            "data": combined_data
        }

        # Write output file
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            self._log(f"\nOutput saved to: {output_path}")
            success = True
        except Exception as e:
            errors.append(f"Failed to write output file: {e}")
            success = False

        return CombineResult(
            success=success,
            output_file=str(output_path),
            files_processed=files_processed,
            total_sentences_before_dedup=total_before,
            total_sentences_after_dedup=total_after,
            duplicates_removed=duplicates_removed,
            sentences_per_category=sentences_per_category,
            errors=errors
        )

    def combine_specific_files(
        self,
        files: list,
        output_file: str = "training_data.json",
        output_dir: Optional[str] = None
    ) -> CombineResult:
        """
        Combine specific JSON files into a single training data file.

        Args:
            files: List of file paths to combine
            output_file: Name of the output file
            output_dir: Directory for output file (default: same as input_dir)

        Returns:
            CombineResult: Object containing combination statistics
        """
        errors = []
        files_processed = 0
        total_before = 0
        combined_data = {}

        self._log(f"Processing {len(files)} specified files...")

        for json_file in files:
            data, error = self._read_json_file(json_file)

            if error:
                errors.append(error)
                continue

            if 'data' not in data:
                continue

            for category, sentences in data['data'].items():
                if self.categories and category not in self.categories:
                    continue

                if category not in combined_data:
                    combined_data[category] = []

                combined_data[category].extend(sentences)
                total_before += len(sentences)

            files_processed += 1
            self._log(f"  Processed: {os.path.basename(json_file)}")

        if self.deduplicate:
            for category in combined_data:
                combined_data[category] = self._remove_duplicates(combined_data[category])

        sentences_per_category = {cat: len(sentences) for cat, sentences in combined_data.items()}
        total_after = sum(sentences_per_category.values())

        if output_dir:
            output_path = Path(output_dir) / output_file
        else:
            output_path = self.input_dir / output_file

        output = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "source_files_processed": files_processed,
                "total_categories": len([cat for cat in combined_data if combined_data[cat]]),
                "total_sentences": total_after,
                "duplicates_removed": total_before - total_after,
                "sentences_per_category": sentences_per_category
            },
            "data": combined_data
        }

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            success = True
        except Exception as e:
            errors.append(f"Failed to write output file: {e}")
            success = False

        return CombineResult(
            success=success,
            output_file=str(output_path),
            files_processed=files_processed,
            total_sentences_before_dedup=total_before,
            total_sentences_after_dedup=total_after,
            duplicates_removed=total_before - total_after,
            sentences_per_category=sentences_per_category,
            errors=errors
        )


def combine_json_files(
    input_dir: str,
    output_file: str = "training_data.json",
    output_dir: Optional[str] = None,
    exclude_files: Optional[list] = None,
    categories: Optional[list] = None,
    deduplicate: bool = True,
    verbose: bool = False
) -> CombineResult:
    """
    Convenience function to combine JSON files in a directory.

    Args:
        input_dir: Directory containing source JSON files
        output_file: Name of the output file
        output_dir: Directory for output file (default: same as input_dir)
        exclude_files: List of filenames to exclude
        categories: List of categories to include (None = all)
        deduplicate: Whether to remove duplicates
        verbose: Whether to print progress

    Returns:
        CombineResult: Object containing combination statistics

    Example:
        from data_generator.data_combiner import combine_json_files

        result = combine_json_files(
            input_dir="data/output",
            output_file="training_data.json",
            verbose=True
        )
        print(result.summary())
    """
    combiner = DataCombiner(
        input_dir=input_dir,
        exclude_files=exclude_files,
        categories=categories,
        deduplicate=deduplicate,
        verbose=verbose
    )
    return combiner.combine(output_file=output_file, output_dir=output_dir)


if __name__ == "__main__":
    # Example usage when run directly
    import argparse

    parser = argparse.ArgumentParser(description="Combine JSON data files into training data")
    parser.add_argument("--input-dir", "-i", default=".", help="Input directory containing JSON files")
    parser.add_argument("--output", "-o", default="training_data.json", help="Output filename")
    parser.add_argument("--output-dir", "-d", help="Output directory (default: same as input)")
    parser.add_argument("--no-dedup", action="store_true", help="Disable deduplication")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--categories", "-c", nargs="+", help="Categories to include")

    args = parser.parse_args()

    result = combine_json_files(
        input_dir=args.input_dir,
        output_file=args.output,
        output_dir=args.output_dir,
        categories=args.categories,
        deduplicate=not args.no_dedup,
        verbose=args.verbose
    )

    print(result.summary())
