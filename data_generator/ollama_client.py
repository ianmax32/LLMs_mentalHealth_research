"""
Ollama client for interacting with the deepseek-r1 model via command line
"""

import json
import logging
import subprocess
import re
from typing import Dict, Any, Optional
from . import config

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with Ollama via command line"""

    def __init__(self, model: str = None, timeout: int = None):
        """
        Initialize Ollama client

        Args:
            model: Model name (default from config)
            timeout: Request timeout in seconds (default from config)
        """
        self.model = model or config.OLLAMA_MODEL
        self.timeout = timeout or config.OLLAMA_TIMEOUT

        logger.info(f"Initialized Ollama client with model: {self.model}")

    def check_model_available(self) -> bool:
        """
        Check if the specified model is available

        Returns:
            bool: True if model is available, False otherwise
        """
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                logger.error(f"Failed to list models: {result.stderr}")
                return False

            # Check if model name appears in the output
            is_available = self.model in result.stdout

            if is_available:
                logger.info(f"Model {self.model} is available")
            else:
                logger.warning(f"Model {self.model} not found in available models")
                logger.debug(f"Available models:\n{result.stdout}")

            return is_available

        except subprocess.TimeoutExpired:
            logger.error("Timeout while checking model availability")
            return False
        except FileNotFoundError:
            logger.error("Ollama command not found. Please ensure Ollama is installed and in PATH")
            return False
        except Exception as e:
            logger.error(f"Error checking model availability: {e}")
            return False

    def generate(self, prompt: str) -> Optional[str]:
        """
        Generate text using the Ollama model via command line

        Args:
            prompt: Input prompt for generation

        Returns:
            Generated text or None if error occurs
        """
        try:
            logger.info(f"Running ollama {self.model} command...")
            logger.debug(f"Prompt length: {len(prompt)} characters")

            # Run ollama with the prompt via stdin
            result = subprocess.run(
                ["ollama", "run", self.model],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                encoding='utf-8',
                errors='replace'
            )

            if result.returncode != 0:
                logger.error(f"Ollama command failed: {result.stderr}")
                return None

            response = result.stdout.strip()
            logger.info(f"Received response ({len(response)} characters)")
            logger.debug(f"Response preview: {response[:500]}...")

            return response

        except subprocess.TimeoutExpired:
            logger.error(f"Request timed out after {self.timeout} seconds")
            return None
        except FileNotFoundError:
            logger.error("Ollama command not found. Please ensure Ollama is installed and in PATH")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during generation: {e}")
            return None

    def extract_json_from_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from model response

        Args:
            response: Raw response from model

        Returns:
            Parsed JSON dict or None if parsing fails
        """
        try:
            # Remove thinking tags if present (deepseek-r1 uses <think>...</think>)
            cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
            cleaned = cleaned.strip()

            # Try to find JSON block in response
            if "```json" in cleaned:
                # Extract content between ```json and ```
                start = cleaned.find("```json") + 7
                end = cleaned.find("```", start)
                json_str = cleaned[start:end].strip()
            elif "```" in cleaned:
                # Extract content between ``` and ```
                start = cleaned.find("```") + 3
                end = cleaned.find("```", start)
                json_str = cleaned[start:end].strip()
            else:
                # Try to find JSON object directly
                # Look for { and find the matching }
                brace_start = cleaned.find("{")
                if brace_start != -1:
                    brace_count = 0
                    brace_end = brace_start
                    for i, char in enumerate(cleaned[brace_start:], brace_start):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                brace_end = i + 1
                                break
                    json_str = cleaned[brace_start:brace_end]
                else:
                    json_str = cleaned

            # Parse JSON
            parsed = json.loads(json_str)
            logger.info("Successfully extracted JSON from response")
            return parsed

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from response: {e}")
            logger.debug(f"Response content: {response[:1000]}...")
            return None
        except Exception as e:
            logger.error(f"Unexpected error extracting JSON: {e}")
            return None
