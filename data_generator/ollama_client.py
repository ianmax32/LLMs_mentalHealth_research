"""
Ollama client for interacting with the deepseek-r1 model
"""

import json
import logging
import requests
from typing import Dict, Any, Optional
from . import config

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with Ollama API"""

    def __init__(self, host: str = None, model: str = None, timeout: int = None):
        """
        Initialize Ollama client

        Args:
            host: Ollama host URL (default from config)
            model: Model name (default from config)
            timeout: Request timeout in seconds (default from config)
        """
        self.host = host or config.OLLAMA_HOST
        self.model = model or config.OLLAMA_MODEL
        self.timeout = timeout or config.OLLAMA_TIMEOUT
        self.api_url = f"{self.host}/api/generate"

        logger.info(f"Initialized Ollama client with model: {self.model}")

    def check_model_available(self) -> bool:
        """
        Check if the specified model is available

        Returns:
            bool: True if model is available, False otherwise
        """
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=10)
            response.raise_for_status()
            models = response.json().get("models", [])
            available_models = [m.get("name", "") for m in models]

            is_available = any(self.model in model for model in available_models)

            if is_available:
                logger.info(f"Model {self.model} is available")
            else:
                logger.warning(f"Model {self.model} not found. Available models: {available_models}")

            return is_available
        except Exception as e:
            logger.error(f"Error checking model availability: {e}")
            return False

    def generate(self, prompt: str, stream: bool = False) -> Optional[str]:
        """
        Generate text using the Ollama model

        Args:
            prompt: Input prompt for generation
            stream: Whether to stream the response

        Returns:
            Generated text or None if error occurs
        """
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": stream,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                }
            }

            logger.info(f"Sending generation request to Ollama...")

            response = requests.post(
                self.api_url,
                json=payload,
                timeout=self.timeout,
                stream=stream
            )
            response.raise_for_status()

            if stream:
                # Handle streaming response
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        if "response" in chunk:
                            full_response += chunk["response"]
                        if chunk.get("done", False):
                            break
                return full_response
            else:
                # Handle non-streaming response
                result = response.json()
                return result.get("response", "")

        except requests.exceptions.Timeout:
            logger.error(f"Request timed out after {self.timeout} seconds")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
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
            # Try to find JSON block in response
            if "```json" in response:
                # Extract content between ```json and ```
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
            elif "```" in response:
                # Extract content between ``` and ```
                start = response.find("```") + 3
                end = response.find("```", start)
                json_str = response[start:end].strip()
            else:
                # Assume entire response is JSON
                json_str = response.strip()

            # Parse JSON
            parsed = json.loads(json_str)
            logger.info("Successfully extracted JSON from response")
            return parsed

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from response: {e}")
            logger.debug(f"Response content: {response[:500]}...")
            return None
        except Exception as e:
            logger.error(f"Unexpected error extracting JSON: {e}")
            return None
