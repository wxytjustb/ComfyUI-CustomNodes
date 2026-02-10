"""
ComfyUI Nodes for Gemini Multimodal LLM
Uses engines module for API interactions
"""

import os
import torch
from io import BytesIO
from typing import List, Optional, Tuple
from pathlib import Path
from .global_config import ROOT_CATEGORY

import folder_paths

# Import from parent package using relative imports
from ..engines import EngineFactory
from ..utils import config_manager


GEMINI_IMAGE_SYS_PROMPT = (
    "you are an expert image-generation engine. You must ALWAYS produce an image.\n"
    "Interpret all user input—regardless of "
    "format, intent, or abstraction—as literal visual directives for image composition.\n"
    "If a prompt is conversational or lacks specific visual details, "
    "you must creatively invent a concrete visual scenario that depicts the concept.\n"
    "Prioritize generating the visual representation above any text, formatting, or conversational requests."
)


def get_available_models() -> List[str]:
    """所有平台下的模型列表（兼容旧用法）"""
    return config_manager.get_all_models()


def get_models_for_node(node_name: str) -> List[str]:
    """根据 node_config 获取该节点可选的 provider/model 列表"""
    return config_manager.get_models_for_node(node_name)


def get_default_model_for_node(node_name: str) -> Optional[str]:
    """该节点在 node_config 中的默认选项"""
    return config_manager.get_default_model_for_node(node_name)


def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    """Convert image tensor to PNG bytes"""
    from PIL import Image
    import numpy as np

    tensor = tensor.cpu()

    if len(tensor.shape) == 4:
        tensor = tensor[0]

    if tensor.shape[-1] == 4:  # RGBA
        arr = tensor.numpy()
        arr = (arr * 255).astype(np.uint8)
        img = Image.fromarray(arr, "RGBA")
        img_rgb = Image.new("RGB", img.size, (255, 255, 255))
        img_rgb.paste(img, mask=img.split()[3])
        img = img_rgb
    elif tensor.shape[-1] == 3:  # RGB
        arr = (tensor.numpy() * 255).astype(np.uint8)
        img = Image.fromarray(arr, "RGB")
    else:
        arr = (tensor.numpy() * 255).astype(np.uint8)
        img = Image.fromarray(arr, "L")

    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def bytes_to_tensor(image_bytes: bytes) -> torch.Tensor:
    """Convert image bytes to tensor"""
    from PIL import Image
    import numpy as np

    img = Image.open(BytesIO(image_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")

    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr)
    tensor = tensor.unsqueeze(0)

    return tensor


def validate_string(text: str, min_length: int = 0) -> bool:
    """Validate string input"""
    if text is None:
        return False
    return len(text.strip()) >= min_length


class GeminiNode:
    """文本生成节点，按 node_config 使用不同平台的模型"""

    @classmethod
    def INPUT_TYPES(cls):
        models = get_models_for_node(cls.__name__)
        if not models:
            models = get_available_models() or ["google_gemini/flash"]
        default = get_default_model_for_node(cls.__name__) or (models[0] if models else "google_gemini/flash")

        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Enter your prompt here...",
                    },
                ),
                "provider_model": (
                    models,
                    {"default": default},
                ),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "images": ("IMAGE",),
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Optional system instructions...",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.1,
                    },
                ),
                "max_tokens": ("INT", {"default": 2048, "min": 1, "max": 8192}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate"
    CATEGORY = ROOT_CATEGORY + "/Gemini"

    def generate(
        self,
        prompt: str,
        provider_model: str,
        seed: int,
        images: Optional[torch.Tensor] = None,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Tuple[str]:
        """Generate text response"""

        if not validate_string(prompt, min_length=1):
            return ("Error: Prompt cannot be empty",)

        # Create engine
        engine = EngineFactory.create_engine(provider_model)
        if not engine:
            return (f"Error: Failed to create engine for {provider_model}",)

        # Convert images to bytes
        image_bytes_list = None
        if images is not None:
            image_bytes_list = []
            if len(images.shape) == 4:
                for i in range(images.shape[0]):
                    img_bytes = tensor_to_bytes(images[i : i + 1])
                    image_bytes_list.append(img_bytes)
            else:
                image_bytes_list = [tensor_to_bytes(images)]

        # 使用配置中的 model key（provider_model 中 / 后的部分）作为 API 的 model_id，model_name 仅作展示
        model_info = config_manager.get_model_info(provider_model)
        if not model_info:
            return (f"Error: Unknown model: {provider_model}",)

        model_id = provider_model.split("/", 1)[1] if "/" in provider_model else "gemini-pro"

        # Generate text
        result = engine.generate_text(
            prompt=prompt,
            model_id=model_id,
            system_prompt=system_prompt if system_prompt else None,
            temperature=temperature,
            max_tokens=max_tokens,
            images=image_bytes_list,
        )

        if result.get("success"):
            return (result.get("text", ""),)
        else:
            return (f"Error: {result.get('error', 'Unknown error')}",)


class GeminiImageNode:
    """图像生成节点，按 node_config 使用不同平台的模型"""

    @classmethod
    def INPUT_TYPES(cls):
        models = get_models_for_node(cls.__name__)
        if not models:
            models = [
                m for m in get_available_models()
                if any(k in m.lower() for k in ["image", "dall", "pro", "flash"])
            ] or get_available_models()[:5]
        default = get_default_model_for_node(cls.__name__) or (models[0] if models else "google_gemini/pro")

        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Describe the image you want to generate...",
                    },
                ),
                "provider_model": (
                    models,
                    {"default": default},
                ),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "reference_images": ("IMAGE",),
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": GEMINI_IMAGE_SYS_PROMPT,
                    },
                ),
                "aspect_ratio": (
                    ["1:1", "4:3", "3:4", "16:9", "9:16"],
                    {"default": "1:1"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "generate"
    CATEGORY = ROOT_CATEGORY + "/Gemini"

    def generate(
        self,
        prompt: str,
        provider_model: str,
        seed: int,
        reference_images: Optional[torch.Tensor] = None,
        system_prompt: str = "",
        aspect_ratio: str = "1:1",
    ) -> Tuple[torch.Tensor, str]:
        """Generate image"""

        if not validate_string(prompt, min_length=1):
            empty_img = torch.zeros((1, 512, 512, 3))
            return (empty_img, "Error: Prompt cannot be empty")

        # Create engine
        engine = EngineFactory.create_engine(provider_model)
        if not engine:
            empty_img = torch.zeros((1, 512, 512, 3))
            return (empty_img, f"Error: Failed to create engine for {provider_model}")

        # Convert reference images to bytes
        image_bytes_list = None
        if reference_images is not None:
            image_bytes_list = []
            if len(reference_images.shape) == 4:
                for i in range(reference_images.shape[0]):
                    img_bytes = tensor_to_bytes(reference_images[i : i + 1])
                    image_bytes_list.append(img_bytes)
            else:
                image_bytes_list = [tensor_to_bytes(reference_images)]

        # 使用配置中的 model key 作为 API 的 model_id
        model_info = config_manager.get_model_info(provider_model)
        if not model_info:
            empty_img = torch.zeros((1, 512, 512, 3))
            return (empty_img, f"Error: Unknown model: {provider_model}")

        model_id = provider_model.split("/", 1)[1] if "/" in provider_model else "gemini-pro"

        # Generate image
        result = engine.generate(
            prompt=prompt,
            model_id=model_id,
            aspect_ratio=aspect_ratio,
            image_size="2K",
            reference_images=image_bytes_list,
        )

        if result.get("success"):
            image_data = result.get("image_data")
            if image_data:
                img_tensor = bytes_to_tensor(image_data)
                info = f"Generated image using {provider_model} (model: {model_id})"
                return (img_tensor, info)
            else:
                empty_img = torch.zeros((1, 512, 512, 3))
                return (empty_img, "Error: No image data in response")
        else:
            empty_img = torch.zeros((1, 512, 512, 3))
            return (empty_img, f"Error: {result.get('error', 'Unknown error')}")


class GeminiImageProNode:
    """专业图像生成节点，按 node_config 使用不同平台的模型"""

    @classmethod
    def INPUT_TYPES(cls):
        models = get_models_for_node(cls.__name__)
        if not models:
            models = [
                m for m in get_available_models()
                if any(k in m.lower() for k in ["image", "pro", "dall"])
            ] or get_available_models()[:5]
        default = get_default_model_for_node(cls.__name__) or (models[0] if models else "vectorengine/gemini-3-pro-image-preview")

        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Detailed description of the image...",
                    },
                ),
                "provider_model": (
                    models,
                    {"default": default},
                ),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "resolution": (
                    ["1K", "2K", "4K"],
                    {"default": "2K"},
                ),
            },
            "optional": {
                "reference_images": ("IMAGE",),
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "What to avoid in the image...",
                    },
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": GEMINI_IMAGE_SYS_PROMPT,
                    },
                ),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "info")
    FUNCTION = "generate"
    CATEGORY = ROOT_CATEGORY + "/Gemini"

    def generate(
        self,
        prompt: str,
        provider_model: str,
        seed: int,
        resolution: str,
        reference_images: Optional[torch.Tensor] = None,
        negative_prompt: str = "",
        system_prompt: str = "",
        num_images: int = 1,
    ) -> Tuple[torch.Tensor, str]:
        """Generate professional images"""

        if not validate_string(prompt, min_length=1):
            empty_img = torch.zeros((1, 512, 512, 3))
            return (empty_img, "Error: Prompt cannot be empty")

        # Create engine
        engine = EngineFactory.create_engine(provider_model)
        if not engine:
            empty_img = torch.zeros((1, 512, 512, 3))
            return (empty_img, f"Error: Failed to create engine for {provider_model}")

        # Convert reference images
        image_bytes_list = None
        if reference_images is not None:
            image_bytes_list = []
            if len(reference_images.shape) == 4:
                for i in range(min(reference_images.shape[0], 4)):
                    img_bytes = tensor_to_bytes(reference_images[i : i + 1])
                    image_bytes_list.append(img_bytes)
            else:
                image_bytes_list = [tensor_to_bytes(reference_images)]

        # Build enhanced prompt
        enhanced_prompt = prompt
        if negative_prompt:
            enhanced_prompt += f"\n\nAvoid: {negative_prompt}"

        # 使用配置中的 model key 作为 API 的 model_id
        model_info = config_manager.get_model_info(provider_model)
        if not model_info:
            empty_img = torch.zeros((1, 512, 512, 3))
            return (empty_img, f"Error: Unknown model: {provider_model}")

        model_id = provider_model.split("/", 1)[1] if "/" in provider_model else "gemini-pro"

        # Generate images
        img_tensors = []
        for i in range(num_images):
            result = engine.generate(
                prompt=enhanced_prompt,
                model_id=model_id,
                image_size=resolution,
                reference_images=image_bytes_list,
            )

            if result.get("success"):
                image_data = result.get("image_data")
                if image_data:
                    img_tensor = bytes_to_tensor(image_data)
                    img_tensors.append(img_tensor)

        if img_tensors:
            combined = torch.cat(img_tensors, dim=0)
            info = f"Generated {len(img_tensors)} image(s) using {provider_model} (model: {model_id}) at {resolution}"
            return (combined, info)
        else:
            empty_img = torch.zeros((1, 512, 512, 3))
            return (empty_img, "Error: Failed to generate images")


class GeminiInputFilesNode:
    """Node for loading and preparing input files"""

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        try:
            files = [f.name for f in os.scandir(input_dir) if f.is_file()]
        except OSError:
            files = []

        return {
            "required": {
                "file": (sorted(files),),
            },
            "optional": {
                "additional_files": ("GEMINI_INPUT_FILES",),
            },
        }

    RETURN_TYPES = ("GEMINI_INPUT_FILES",)
    RETURN_NAMES = ("files",)
    FUNCTION = "load_files"
    CATEGORY = ROOT_CATEGORY + "/Gemini"

    def load_files(self, file: str, additional_files: Optional[List] = None):
        """Load and combine input files"""
        input_dir = folder_paths.get_input_directory()
        file_path = os.path.join(input_dir, file)

        loaded_files = []

        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                content = f.read()
                mime_type = "application/octet-stream"

                ext = Path(file_path).suffix.lower()
                mime_map = {
                    ".txt": "text/plain",
                    ".pdf": "application/pdf",
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".gif": "image/gif",
                    ".mp3": "audio/mp3",
                    ".mp4": "video/mp4",
                }
                mime_type = mime_map.get(ext, mime_type)

                loaded_files.append(
                    {
                        "name": file,
                        "content": content,
                        "mime_type": mime_type,
                    }
                )

        if additional_files:
            loaded_files.extend(additional_files)

        return (loaded_files,)


# Node class mappings
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

# Print loaded configuration on module load
print("=" * 50)
print("ComfyUI Gemini Nodes Loaded")
print("=" * 50)
providers = config_manager.get_all_providers()
print(f"Loaded {len(providers)} providers:")
for name, provider in providers.items():
    print(f"  - {name}: {len(provider.models)} models")
print("=" * 50)
