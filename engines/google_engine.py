"""
Google Gemini Engine implementation
Supports both native Google API and OpenAI-compatible endpoints
"""

from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from io import BytesIO
import base64

# Handle imports gracefully
try:
    from google import genai
    from google.genai import types

    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False
    print("[GoogleEngine] Warning: google-genai package not installed")

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("[GoogleEngine] Warning: Pillow package not installed")

from .engines import BaseEngine


class GoogleEngine(BaseEngine):
    """
    Google Gemini Engine for text and image generation
    Supports both native Google API and OpenAI-compatible endpoints
    """

    def __init__(self, api_key: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(api_key, base_url, **kwargs)
        self.client = None
        self._init_client()

    def _init_client(self):
        """Initialize Google GenAI client"""
        if not GOOGLE_GENAI_AVAILABLE:
            raise ImportError("google-genai package is required for GoogleEngine")

        http_options = {}
        if self.base_url:
            http_options["base_url"] = self.base_url

        try:
            self.client = genai.Client(api_key=self.api_key, http_options=http_options)
        except Exception as e:
            print(f"[GoogleEngine] Error initializing client: {e}")
            raise

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
        Generate image using Google Gemini API

        Returns:
            Dict with:
            - success: bool
            - image_data: bytes (if output_path not provided)
            - image_path: str (if output_path provided)
            - error: str (if failed)
        """
        if not GOOGLE_GENAI_AVAILABLE:
            return {"success": False, "error": "google-genai package not installed"}

        if not PIL_AVAILABLE:
            return {"success": False, "error": "Pillow package not installed"}

        try:
            # Build contents list
            contents = [prompt]

            # Add reference images
            if reference_images:
                for img in reference_images:
                    if isinstance(img, (str, Path)):
                        contents.append(Image.open(img))
                    elif isinstance(img, bytes):
                        contents.append(Image.open(BytesIO(img)))
                    elif isinstance(img, Image.Image):
                        contents.append(img)

            # Make API call
            response = self.client.models.generate_content(
                model=model_id,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(
                        aspect_ratio=aspect_ratio, image_size=image_size
                    ),
                ),
            )

            # Check for valid response
            if not response.candidates or len(response.candidates) == 0:
                return {"success": False, "error": "No candidates in API response"}

            candidate = response.candidates[0]

            # Check finish reason for safety filters
            if hasattr(candidate, "finish_reason") and candidate.finish_reason:
                finish_reason = str(candidate.finish_reason)
                if "SAFETY" in finish_reason or "BLOCKED" in finish_reason:
                    return {
                        "success": False,
                        "error": f"Content blocked by safety filter: {finish_reason}",
                    }

            # Check for valid content
            if not hasattr(candidate, "content") or not candidate.content:
                return {"success": False, "error": "No content in response candidate"}

            if not hasattr(candidate.content, "parts") or not candidate.content.parts:
                return {"success": False, "error": "No parts in response content"}

            # Extract image data
            for part in candidate.content.parts:
                if part.inline_data:
                    image_data = part.inline_data.data

                    # Save to file or return bytes
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

            return {"success": False, "error": "No image data in response parts"}

        except Exception as e:
            return {"success": False, "error": str(e)}

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
        Generate text using Google Gemini API

        Returns:
            Dict with:
            - success: bool
            - text: str
            - error: str (if failed)
        """
        if not GOOGLE_GENAI_AVAILABLE:
            return {"success": False, "error": "google-genai package not installed"}

        if not PIL_AVAILABLE:
            return {"success": False, "error": "Pillow package not installed"}

        try:
            # Build contents
            contents = [prompt]

            # Add images if provided
            if images:
                for img in images:
                    if isinstance(img, (str, Path)):
                        contents.append(Image.open(img))
                    elif isinstance(img, bytes):
                        contents.append(Image.open(BytesIO(img)))
                    elif isinstance(img, Image.Image):
                        contents.append(img)

            # Build config
            config_kwargs = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }

            if system_prompt:
                config_kwargs["system_instruction"] = system_prompt

            # Make API call
            response = self.client.models.generate_content(
                model=model_id,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs),
            )

            # Check for valid response
            if not response.candidates or len(response.candidates) == 0:
                return {"success": False, "error": "No candidates in API response"}

            candidate = response.candidates[0]

            # Check finish reason for safety filters
            if hasattr(candidate, "finish_reason") and candidate.finish_reason:
                finish_reason = str(candidate.finish_reason)
                if "SAFETY" in finish_reason or "BLOCKED" in finish_reason:
                    return {
                        "success": False,
                        "error": f"Content blocked by safety filter: {finish_reason}",
                    }

            # Check for valid content
            if not hasattr(candidate, "content") or not candidate.content:
                return {"success": False, "error": "No content in response candidate"}

            if not hasattr(candidate.content, "parts") or not candidate.content.parts:
                return {"success": False, "error": "No parts in response content"}

            # Extract text
            text = ""
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    text += part.text

            if text:
                return {"success": True, "text": text, "model": model_id}
            else:
                return {"success": False, "error": "No text in response parts"}

        except Exception as e:
            return {"success": False, "error": str(e)}
