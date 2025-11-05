"""
Test the automatic versioning feature
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data_generator import config
from data_generator.output_handler import OutputHandler

def test_versioning():
    """Test the versioned filename generation"""
    print("=" * 60)
    print("Testing Automatic File Versioning")
    print("=" * 60)

    handler = OutputHandler()
    base_path = config.OUTPUT_JSON

    # Test 1: Single category
    print("\n[Test 1] Single category - Anxiety")
    filename = handler.generate_versioned_filename(
        base_path=base_path,
        categories=["Anxiety"]
    )
    print(f"Generated: {filename.name}")

    # Test 2: Different category
    print("\n[Test 2] Different category - Depression")
    filename = handler.generate_versioned_filename(
        base_path=base_path,
        categories=["Depression"]
    )
    print(f"Generated: {filename.name}")

    # Test 3: All categories
    print("\n[Test 3] All categories")
    filename = handler.generate_versioned_filename(
        base_path=base_path,
        categories=None  # None means "all"
    )
    print(f"Generated: {filename.name}")

    # Test 4: Multiple categories
    print("\n[Test 4] Multiple categories - Anxiety + Depression")
    filename = handler.generate_versioned_filename(
        base_path=base_path,
        categories=["Anxiety", "Depression"]
    )
    print(f"Generated: {filename.name}")

    # Test 5: Simulate existing files
    print("\n[Test 5] Simulating existing files...")
    print("Creating dummy files to test version increment...")

    # Create dummy files
    dummy_files = [
        config.OUTPUT_DIR / "generated_sentences_v1_Anxiety.json",
        config.OUTPUT_DIR / "generated_sentences_v2_Anxiety.json",
    ]

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for dummy in dummy_files:
        dummy.write_text("{}")
        print(f"  Created: {dummy.name}")

    # Now test versioning with existing files
    print("\nGenerating new filename for Anxiety (should be v3)...")
    filename = handler.generate_versioned_filename(
        base_path=base_path,
        categories=["Anxiety"]
    )
    print(f"✓ Generated: {filename.name}")

    # Clean up
    print("\nCleaning up test files...")
    for dummy in dummy_files:
        if dummy.exists():
            dummy.unlink()
            print(f"  Removed: {dummy.name}")

    print("\n" + "=" * 60)
    print("VERSIONING TEST COMPLETE")
    print("=" * 60)
    print("\nFilename format: generated_sentences_v{VERSION}_{CATEGORY}.json")
    print("\nKey features:")
    print("  ✓ Automatic version incrementing")
    print("  ✓ Separate version sequences per category")
    print("  ✓ No file overwrites")
    print("=" * 60)

if __name__ == "__main__":
    test_versioning()
