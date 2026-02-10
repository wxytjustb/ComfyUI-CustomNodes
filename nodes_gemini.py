"""
ComfyUI Gemini Nodes - Backward Compatibility Module

This file is kept for backward compatibility.
The main entry point is now __init__.py
"""

# Re-export from main package for backward compatibility
from . import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
