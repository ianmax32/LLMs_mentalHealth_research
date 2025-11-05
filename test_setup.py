"""
Test script to verify the setup and connectivity
"""

import sys
import json
from pathlib import Path

# Add data_generator to path
sys.path.insert(0, str(Path(__file__).parent))

from data_generator import config
from data_generator.ollama_client import OllamaClient
from data_generator.output_handler import OutputHandler

def test_ollama_connection():
    """Test connection to Ollama"""
    print("=" * 60)
    print("Testing Ollama Connection")
    print("=" * 60)

    client = OllamaClient()
    print(f"Ollama Host: {client.host}")
    print(f"Model: {client.model}")

    print("\nChecking if model is available...")
    if client.check_model_available():
        print("✓ Model is available!")
        return True
    else:
        print("✗ Model not found!")
        print("\nPlease install it using:")
        print("  ollama pull deepseek-r1")
        return False

def test_input_file():
    """Test if input file exists and is valid"""
    print("\n" + "=" * 60)
    print("Testing Input File")
    print("=" * 60)

    if not config.INPUT_JSON.exists():
        print(f"✗ Input file not found: {config.INPUT_JSON}")
        return False

    print(f"✓ Input file exists: {config.INPUT_JSON}")

    # Load and validate
    handler = OutputHandler()
    data = handler.load_json(config.INPUT_JSON)

    if not data:
        print("✗ Failed to load input data")
        return False

    print(f"✓ Loaded {len(data)} categories")
    for category, sentences in data.items():
        print(f"  - {category}: {len(sentences)} example sentences")

    return True

def test_output_directory():
    """Test if output directory is writable"""
    print("\n" + "=" * 60)
    print("Testing Output Directory")
    print("=" * 60)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    test_file = config.OUTPUT_DIR / "test_write.txt"
    try:
        test_file.write_text("test")
        test_file.unlink()
        print(f"✓ Output directory is writable: {config.OUTPUT_DIR}")
        return True
    except Exception as e:
        print(f"✗ Cannot write to output directory: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("MENTAL HEALTH DATA GENERATOR - SETUP TEST")
    print("=" * 60 + "\n")

    results = []

    # Test 1: Input file
    results.append(("Input File", test_input_file()))

    # Test 2: Output directory
    results.append(("Output Directory", test_output_directory()))

    # Test 3: Ollama connection
    results.append(("Ollama Connection", test_ollama_connection()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED! Ready to generate data.")
        print("\nTry running:")
        print("  python main.py --all --count 3")
    else:
        print("SOME TESTS FAILED. Please fix the issues above.")
    print("=" * 60 + "\n")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
