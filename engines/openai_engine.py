"""
OpenAI API Engine implementation (v1 compatible)
Supports OpenAI, OpenRouter, VectorEngine and other OpenAI-compatible APIs
"""

import base64
import time
from typing import Dict, Any, Optional, List
from pathlib import Path
from io import BytesIO

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("[OpenAIV1Engine] Warning: requests package not installed")

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("[OpenAIV1Engine] Warning: Pillow package not installed")

from .engines import BaseEngine

# Default configuration
DEFAULT_TIMEOUT = 120  # seconds
DEFAULT_MAX_RETRIES = 2
RETRY_DELAY = 1.0  # seconds
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class OpenAIV1Engine(BaseEngine):
    """
    OpenAI API Engine for text and image generation
    Compatible with OpenAI, OpenRouter, VectorEngine and other OpenAI-compatible APIs
    """

    def __init__(self, api_key: str, base_url: str, **kwargs):
        super().__init__(api_key, base_url, **kwargs)
        self.provider = kwargs.get("provider", "openai")
        self.model_config = kwargs.get("model_config", None)
        self.timeout = kwargs.get("timeout", DEFAULT_TIMEOUT)
        self.max_retries = kwargs.get("max_retries", DEFAULT_MAX_RETRIES)

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with API key"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Add extra headers from model config if available
        if self.model_config and self.model_config.extra_headers:
            headers.update(self.model_config.extra_headers)

        return headers

    def _make_request(
        self, method: str, url: str, **kwargs
    ) -> requests.Response:
        """
        Make HTTP request with retry logic for transient errors.

        Args:
            method: HTTP method (get, post, etc.)
            url: Request URL
            **kwargs: Additional arguments for requests

        Returns:
            Response object

        Raises:
            requests.exceptions.RequestException: If all retries fail
        """
        kwargs.setdefault("timeout", self.timeout)
        kwargs.setdefault("headers", self._get_headers())

        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                response = getattr(requests, method)(url, **kwargs)

                # Check if we should retry on this status code
                if response.status_code in RETRYABLE_STATUS_CODES:
                    if attempt < self.max_retries:
                        wait_time = RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                        print(
                            f"[OpenAIV1Engine] Retrying after {response.status_code} "
                            f"(attempt {attempt + 1}/{self.max_retries + 1}), "
                            f"waiting {wait_time:.1f}s..."
                        )
                        time.sleep(wait_time)
                        continue

                return response

            except requests.exceptions.Timeout as e:
                last_exception = e
                if attempt < self.max_retries:
                    wait_time = RETRY_DELAY * (2 ** attempt)
                    print(
                        f"[OpenAIV1Engine] Timeout, retrying "
                        f"(attempt {attempt + 1}/{self.max_retries + 1})..."
                    )
                    time.sleep(wait_time)
                    continue
                raise

            except requests.exceptions.ConnectionError as e:
                last_exception = e
                if attempt < self.max_retries:
                    wait_time = RETRY_DELAY * (2 ** attempt)
                    print(
                        f"[OpenAIV1Engine] Connection error, retrying "
                        f"(attempt {attempt + 1}/{self.max_retries + 1})..."
                    )
                    time.sleep(wait_time)
                    continue
                raise

        # Should not reach here, but just in case
        if last_exception:
            raise last_exception
        raise requests.exceptions.RequestException("Max retries exceeded")

    def _parse_error_response(self, response: requests.Response) -> str:
        """Parse error details from API response."""
        try:
            error_data = response.json()
            if "error" in error_data:
                error_info = error_data["error"]
                if isinstance(error_info, dict):
                    return error_info.get("message", str(error_info))
                return str(error_info)
            return response.text[:500]  # Truncate long error messages
        except Exception:
            return f"HTTP {response.status_code}: {response.reason}"

    def _encode_image(self, image_data) -> str:
        """Encode image to base64 string"""
        if isinstance(image_data, str):
            # File path
            with open(image_data, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        elif isinstance(image_data, bytes):
            # Raw bytes
            return base64.b64encode(image_data).decode("utf-8")
        elif PIL_AVAILABLE and hasattr(image_data, "mode"):
            # PIL Image
            buffer = BytesIO()
            image_data.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        else:
            raise ValueError(f"Unsupported image type: {type(image_data)}")

    def generate(
        self,
        prompt: str,
        model_id: str,
        output_path: Optional[str] = None,
        aspect_ratio: str = "16:9",
        image_size: str = "2K",
        reference_images: Optional[List] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate image using OpenAI-compatible API

        Returns:
            Dict with:
            - success: bool
            - image_data: bytes (if output_path not provided)
            - image_path: str (if output_path provided)
            - error: str (if failed)
        """
        if not REQUESTS_AVAILABLE:
            return {"success": False, "error": "requests package not installed"}

        try:
            # Map aspect ratio to size
            size_map = {
                "1:1": "1024x1024",
                "16:9": "1792x1024",
                "9:16": "1024x1792",
                "4:3": "1024x768",
                "3:4": "768x1024",
            }
            size = size_map.get(aspect_ratio, "1024x1024")

            # Prepare request
            url = f"{self.base_url.rstrip('/')}/images/generations"
            payload = {
                "model": model_id,
                "prompt": prompt,
                "n": 1,
                "size": size,
                "response_format": "b64_json",
            }

            # Make request with retry
            response = self._make_request("post", url, json=payload)

            # Check for errors
            if not response.ok:
                error_msg = self._parse_error_response(response)
                return {
                    "success": False,
                    "error": f"API error ({response.status_code}): {error_msg}",
                }

            # Parse response
            data = response.json()

            if data.get("data"):
                item = data["data"][0]

                # Get image data
                if "b64_json" in item:
                    image_data = base64.b64decode(item["b64_json"])
                elif "url" in item:
                    img_response = self._make_request("get", item["url"], timeout=60)
                    if not img_response.ok:
                        return {
                            "success": False,
                            "error": f"Failed to download image: HTTP {img_response.status_code}",
                        }
                    image_data = img_response.content
                else:
                    return {
                        "success": False,
                        "error": "API response missing image data (no b64_json or url)",
                    }

                # Return or save
                if output_path:
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    with open(output_path, "wb") as f:
                        f.write(image_data)
                    return {
                        "success": True,
                        "image_path": output_path,
                        "image_data": image_data,
                        "model": model_id,
                    }
                else:
                    return {
                        "success": True,
                        "image_data": image_data,
                        "model": model_id,
                    }

            return {
                "success": False,
                "error": "Unexpected API response: missing 'data' field",
            }

        except requests.exceptions.Timeout:
            return {"success": False, "error": f"Request timeout after {self.timeout}s"}
        except requests.exceptions.ConnectionError as e:
            return {"success": False, "error": f"Connection error: {str(e)}"}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"API request failed: {str(e)}"}
        except Exception as e:
            return {"success": False, "error": f"Unexpected error: {str(e)}"}

    def generate_text(
        self,
        prompt: str,
        model_id: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        images: Optional[List] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate text using OpenAI-compatible API

        Returns:
            Dict with:
            - success: bool
            - text: str
            - error: str (if failed)
        """
        if not REQUESTS_AVAILABLE:
            return {"success": False, "error": "requests package not installed"}

        try:
            # Build messages
            messages = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            # Build user message
            if images:
                # Multimodal input
                content = [{"type": "text", "text": prompt}]

                for img in images:
                    b64_image = self._encode_image(img)
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                        }
                    )

                messages.append({"role": "user", "content": content})
            else:
                # Text only
                messages.append({"role": "user", "content": prompt})

            # Prepare request
            url = f"{self.base_url.rstrip('/')}/chat/completions"
            payload = {
                "model": model_id,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            # Make request with retry
            response = self._make_request("post", url, json=payload)

            # Check for errors
            if not response.ok:
                error_msg = self._parse_error_response(response)
                return {
                    "success": False,
                    "error": f"API error ({response.status_code}): {error_msg}",
                }

            # Parse response
            data = response.json()

            if data.get("choices") and len(data["choices"]) > 0:
                choice = data["choices"][0]

                # Check for content filter
                if choice.get("finish_reason") == "content_filter":
                    return {
                        "success": False,
                        "error": "Content blocked by safety filter",
                    }

                if "message" in choice and "content" in choice["message"]:
                    content = choice["message"]["content"]
                    if content is not None:
                        return {
                            "success": True,
                            "text": content,
                            "model": model_id,
                            "finish_reason": choice.get("finish_reason"),
                        }

            return {
                "success": False,
                "error": "Unexpected API response: missing or empty content",
            }

        except requests.exceptions.Timeout:
            return {"success": False, "error": f"Request timeout after {self.timeout}s"}
        except requests.exceptions.ConnectionError as e:
            return {"success": False, "error": f"Connection error: {str(e)}"}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"API request failed: {str(e)}"}
        except Exception as e:
            return {"success": False, "error": f"Unexpected error: {str(e)}"}
