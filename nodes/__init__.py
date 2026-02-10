# Nodes package initialization
from .gemini_node import (
    GeminiNode,
    GeminiImageNode,
    GeminiImageProNode,
    GeminiInputFilesNode,
)

NODE_CLASS_MAPPINGS = {
    "GeminiNode": GeminiNode,
    "GeminiImageNode": GeminiImageNode,
    "GeminiImageProNode": GeminiImageProNode,
    "GeminiInputFilesNode": GeminiInputFilesNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiNode": "Gemini Text Generation",
    "GeminiImageNode": "Gemini Image Generation",
    "GeminiImageProNode": "Gemini Image Pro",
    "GeminiInputFilesNode": "Gemini Input Files",
}


__all__ = [
    "GeminiNode",
    "GeminiImageNode",
    "GeminiImageProNode",
    "GeminiInputFilesNode",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
