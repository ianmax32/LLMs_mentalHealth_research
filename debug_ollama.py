"""
Debug script to test Ollama connectivity and responses
"""

import requests
import json

OLLAMA_HOST = "http://localhost:11434"

def test_ollama_basic():
    """Test basic Ollama connectivity"""
    print("=" * 60)
    print("Testing Ollama Basic Connectivity")
    print("=" * 60)

    # Test 1: Check if Ollama is running
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
        print(f"✓ Ollama is running at {OLLAMA_HOST}")
        print(f"Status code: {response.status_code}")

        models = response.json().get("models", [])
        print(f"\nAvailable models: {len(models)}")
        for model in models:
            print(f"  - {model.get('name')}")

        return True
    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to Ollama at {OLLAMA_HOST}")
        print("Make sure Ollama is running: ollama serve")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_simple_generation():
    """Test a simple generation request"""
    print("\n" + "=" * 60)
    print("Testing Simple Generation with deepseek-r1")
    print("=" * 60)

    prompt = "Generate exactly 2 sentences about feeling anxious. Return only the sentences, no other text."

    payload = {
        "model": "deepseek-r1",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 200  # Limit output length
        }
    }

    print(f"\nSending request...")
    print(f"Prompt: {prompt[:80]}...")

    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json=payload,
            timeout=120  # 2 minutes for simple test
        )

        print(f"Response status: {response.status_code}")

        if response.status_code != 200:
            print(f"✗ Bad status code: {response.status_code}")
            print(f"Response: {response.text[:500]}")
            return False

        result = response.json()
        print(f"\n✓ Got response!")
        print(f"Response keys: {list(result.keys())}")

        if "response" in result:
            generated_text = result["response"]
            print(f"\nGenerated text length: {len(generated_text)} chars")
            print(f"\nGenerated text preview:")
            print("-" * 60)
            print(generated_text[:500])
            print("-" * 60)

            if not generated_text or generated_text.strip() == "":
                print("\n⚠ WARNING: Response is empty!")
                return False

            return True
        else:
            print(f"✗ No 'response' field in result")
            print(f"Result: {result}")
            return False

    except requests.exceptions.Timeout:
        print(f"✗ Request timed out after 120 seconds")
        return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Request error: {e}")
        return False
    except json.JSONDecodeError as e:
        print(f"✗ JSON decode error: {e}")
        print(f"Response text: {response.text[:500]}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_streaming_generation():
    """Test streaming generation"""
    print("\n" + "=" * 60)
    print("Testing Streaming Generation")
    print("=" * 60)

    prompt = "Say 'Hello world' in one sentence."

    payload = {
        "model": "deepseek-r1",
        "prompt": prompt,
        "stream": True
    }

    print(f"\nSending streaming request...")

    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json=payload,
            timeout=60,
            stream=True
        )

        print(f"Response status: {response.status_code}")

        if response.status_code != 200:
            print(f"✗ Bad status code")
            return False

        print("\n✓ Streaming response:")
        print("-" * 60)

        full_response = ""
        chunk_count = 0

        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                if "response" in chunk:
                    text = chunk["response"]
                    full_response += text
                    print(text, end="", flush=True)
                    chunk_count += 1

                if chunk.get("done", False):
                    break

        print()
        print("-" * 60)
        print(f"\nReceived {chunk_count} chunks")
        print(f"Total response length: {len(full_response)} chars")

        if not full_response:
            print("⚠ WARNING: Empty response!")
            return False

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "=" * 60)
    print("OLLAMA DEBUG TOOL")
    print("=" * 60 + "\n")

    # Test 1: Basic connectivity
    if not test_ollama_basic():
        print("\n⚠ Fix connectivity issues before proceeding")
        return 1

    # Test 2: Simple generation
    if not test_simple_generation():
        print("\n⚠ Generation test failed")
        return 1

    # Test 3: Streaming
    test_streaming_generation()

    print("\n" + "=" * 60)
    print("DEBUG COMPLETE")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
