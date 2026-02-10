"""
ComfyUI Gemini Custom Nodes
支持 Gemini、OpenAI、OpenRouter、VectorEngine 等多提供商 API 的 ComfyUI AI 节点系统

This is the main entry point for ComfyUI to load custom nodes.
"""

from .nodes.gemini_node import (
    GeminiNode,
    GeminiImageNode,
    GeminiImageProNode,
    GeminiInputFilesNode,
)

# 在根模块显式构建映射，确保 ComfyUI 只从本入口加载到全部 4 个节点
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

# ComfyUI 需要这两个变量来注册节点
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# 版本信息
__version__ = "1.0.0"
__author__ = "ComfyUI Custom API Nodes"

print("[ComfyUI Custom API Nodes] Registered {} nodes: {}".format(
    len(NODE_CLASS_MAPPINGS),
    list(NODE_CLASS_MAPPINGS.keys()),
))
print("[ComfyUI Custom API Nodes] Edit config/config.yaml to customize providers and models")
