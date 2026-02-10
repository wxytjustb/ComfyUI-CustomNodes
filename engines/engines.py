"""
Engine factory and base classes for ComfyUI Gemini Nodes
Integrates with config_manager for configuration management
"""

import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union
from pathlib import Path

from ..utils import config_manager


class BaseEngine(ABC):
    """Base class for all AI engines"""

    def __init__(self, api_key: str, base_url: Optional[str] = None, **kwargs):
        self.api_key = api_key
        self.base_url = base_url
        self.config = kwargs

    @abstractmethod
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
        Generate content using the engine

        Args:
            prompt: Text prompt
            model_id: Model identifier
            output_path: Optional output path for generated image
            aspect_ratio: Aspect ratio for image generation
            image_size: Image size (1K, 2K, 4K)
            reference_images: Optional reference images
            **kwargs: Additional parameters

        Returns:
            Dict with success status and result data
        """
        pass

    @abstractmethod
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
        Generate text content

        Args:
            prompt: Text prompt
            model_id: Model identifier
            system_prompt: Optional system instructions
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            images: Optional input images
            **kwargs: Additional parameters

        Returns:
            Dict with success status and generated text
        """
        pass


class EngineFactory:
    """Factory for creating engine instances based on configuration"""

    @classmethod
    def create_engine(cls, provider_model: str) -> Optional[BaseEngine]:
        """
        Create an engine instance from provider/model string

        Args:
            provider_model: Format "provider/model_key" (e.g., "google/pro")

        Returns:
            Engine instance or None if creation fails
        """
        model_info = config_manager.get_model_info(provider_model)
        if not model_info:
            print(f"[EngineFactory] Error: Unknown provider/model: {provider_model}")
            return None

        provider, model_config = model_info
        model_key = provider_model.split("/", 1)[1] if "/" in provider_model else None

        # Get API key（model_key 为 provider 下 models 的 key，用于取模型级 api_key_env）
        api_key = provider.get_api_key(model_key)
        print(
                api_key
            )
        if not api_key:
            key_name = provider.get_primary_api_key_name(model_key)
            print(
                f"[EngineFactory] Error: API key not configured. Set {key_name} environment variable."
            )
            return None

        # Get base URL
        base_url = provider.base_url

        # Determine engine type
        engine_type = (
            model_config.type
            if model_config and model_config.type
            else provider.type.value
        )

        # Create appropriate engine
        try:
            if engine_type == "google":
                from .google_engine import GoogleEngine

                return GoogleEngine(
                    api_key=api_key,
                    base_url=base_url,
                    provider=provider.name,
                    model_config=model_config,
                )
            elif engine_type in ["openai", "openai_v1"]:
                from .openai_engine import OpenAIV1Engine

                return OpenAIV1Engine(
                    api_key=api_key,
                    base_url=base_url or "https://api.openai.com/v1",
                    provider=provider.name,
                    model_config=model_config,
                )
            else:
                print(f"[EngineFactory] Error: Unsupported engine type: {engine_type}")
                return None

        except ImportError as e:
            print(f"[EngineFactory] Error: Failed to import engine module: {e}")
            return None
        except Exception as e:
            print(f"[EngineFactory] Error: Failed to create engine: {e}")
            return None

    @classmethod
    def get_available_engines(cls) -> List[str]:
        """Get list of all available engine types"""
        return ["google", "openai", "openai_v1"]


# Re-export for backward compatibility
__all__ = ["BaseEngine", "EngineFactory"]
