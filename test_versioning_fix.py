"""
Test script to verify automatic versioning works correctly
"""

from pathlib import Path
from data_generator.output_handler import OutputHandler

def test_versioning():
    """Test versioning filename generation"""

    print("=" * 60)
    print("Testing Automatic Versioning System")
    print("=" * 60)

    # Test base path
    base_path = Path("data/output/generated_sentences.json")

    # Test 1: Single category
    print("\n1. Testing single category (Anxiety):")
    for i in range(1, 4):
        versioned = OutputHandler.generate_versioned_filename(
            base_path=base_path,
            categories=["Anxiety"]
        )
        print(f"   Run {i}: {versioned.name}")

    # Test 2: Different category
    print("\n2. Testing different category (Depression):")
    for i in range(1, 4):
        versioned = OutputHandler.generate_versioned_filename(
            base_path=base_path,
            categories=["Depression"]
        )
        print(f"   Run {i}: {versioned.name}")

    # Test 3: All categories
    print("\n3. Testing all categories:")
    for i in range(1, 4):
        versioned = OutputHandler.generate_versioned_filename(
            base_path=base_path,
            categories=None  # None means "all"
        )
        print(f"   Run {i}: {versioned.name}")

    # Test 4: Multiple specific categories
    print("\n4. Testing multiple specific categories:")
    for i in range(1, 4):
        versioned = OutputHandler.generate_versioned_filename(
            base_path=base_path,
            categories=["Anxiety", "Depression"]
        )
        print(f"   Run {i}: {versioned.name}")

    print("\n" + "=" * 60)
    print("Versioning Test Complete!")
    print("=" * 60)
    print("\nExpected behavior:")
    print("- Each category gets its own version sequence")
    print("- Version numbers increment based on existing files")
    print("- Files with different categories don't interfere")
    print("\nNOTE: Actual version numbers depend on existing files in data/output/")
    print("=" * 60)

if __name__ == "__main__":
    test_versioning()
