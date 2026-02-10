"""
Configuration Manager for ComfyUI Custom API Nodes
基于 config.yaml.example 标准格式：env_api_key / providers / node_config
同时兼容旧格式（顶层 provider 名为 key 的字典）
"""

import os
import yaml
from typing import Dict, Any, Optional, List, Tuple, Callable
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum


class ProviderType(str, Enum):
    """Supported provider types"""

    GOOGLE = "google"
    OPENAI = "openai"
    OPENAI_V1 = "openai_v1"


@dataclass
class ApiKeyConfig:
    """
    API Key 配置
    - name: 环境变量名称
    - value: 直接值（可选，支持 ${ENV_VAR}）
    - required: 是否必需
    - description: 描述
    """

    name: str
    value: Optional[str] = None
    required: bool = True
    description: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ApiKeyConfig":
        return cls(
            name=data.get("name", ""),
            value=data.get("value"),
            required=data.get("required", True),
            description=data.get("description"),
        )

    def get_api_key(self) -> Optional[str]:
        if self.value is not None:
            if self.value.startswith("${") and self.value.endswith("}"):
                env_var_name = self.value[2:-1]
                return os.environ.get(env_var_name)
            return self.value
        if self.name:
            return os.environ.get(self.name)
        return None

    def validate(self) -> Tuple[bool, Optional[str]]:
        if self.required and not self.get_api_key():
            return False, f"缺少必需的 API 密钥: {self.name}"
        return True, None


@dataclass
class ModelConfig:
    """单个模型配置"""

    model_name: str
    api_key_env: Optional[str] = None
    env_api_key: List[ApiKeyConfig] = field(default_factory=list)
    extra_headers: Optional[Dict[str, str]] = None
    type: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any], global_key_resolver: Optional[Callable[[str], Optional[str]]] = None) -> "ModelConfig":
        env_api_key_list: List[ApiKeyConfig] = []
        if "env_api_key" in data:
            for item in (
                data["env_api_key"]
                if isinstance(data["env_api_key"], list)
                else [data["env_api_key"]]
            ):
                env_api_key_list.append(ApiKeyConfig.from_dict(item))
        elif data.get("api_key_env"):
            if not global_key_resolver:
                env_api_key_list.append(
                    ApiKeyConfig(name=data["api_key_env"], required=True)
                )
            # 有 global_key_resolver 时在 get_api_key 中按需解析，不写死 value

        return cls(
            model_name=data.get("model_name", ""),
            api_key_env=data.get("api_key_env"),
            env_api_key=env_api_key_list,
            extra_headers=data.get("extra_headers"),
            type=data.get("type"),
        )

    def get_api_key(self, index: int = 0, key_resolver: Optional[Callable[[str], Optional[str]]] = None) -> Optional[str]:
        if key_resolver and self.api_key_env:
            v = key_resolver(self.api_key_env)
            if v is not None:
                return v
        if self.env_api_key and len(self.env_api_key) > index:
            return self.env_api_key[index].get_api_key()
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        return None

    def get_primary_api_key_name(self) -> Optional[str]:
        if self.env_api_key:
            return self.env_api_key[0].name
        return self.api_key_env

    def validate(self) -> Tuple[bool, Optional[str]]:
        for c in self.env_api_key:
            ok, err = c.validate()
            if not ok:
                return ok, err
        return True, None


@dataclass
class ProviderConfig:
    """单个平台(provider)配置"""

    name: str
    type: ProviderType
    api_key_env: Optional[str] = None
    env_api_key: List[ApiKeyConfig] = field(default_factory=list)
    base_url: Optional[str] = None
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    _key_resolver: Optional[Callable[[str], Optional[str]]] = field(default=None, repr=False)

    @classmethod
    def from_dict(
        cls,
        name: str,
        data: Dict[str, Any],
        global_key_resolver: Optional[Callable[[str], Optional[str]]] = None,
    ) -> "ProviderConfig":
        models = {}
        for model_key, model_data in data.get("models", {}).items():
            models[model_key] = ModelConfig.from_dict(
                model_data, global_key_resolver
            )

        env_api_key_list: List[ApiKeyConfig] = []
        if "env_api_key" in data:
            for item in (
                data["env_api_key"]
                if isinstance(data["env_api_key"], list)
                else [data["env_api_key"]]
            ):
                env_api_key_list.append(ApiKeyConfig.from_dict(item))
        elif data.get("api_key_env"):
            if not global_key_resolver:
                env_api_key_list.append(
                    ApiKeyConfig(name=data["api_key_env"], required=True)
                )

        p = cls(
            name=name,
            type=ProviderType(data.get("type", "google")),
            api_key_env=data.get("api_key_env"),
            env_api_key=env_api_key_list,
            base_url=data.get("base_url"),
            models=models,
        )
        p._key_resolver = global_key_resolver
        return p

    def get_api_key(
        self, model_key: Optional[str] = None, index: int = 0
    ) -> Optional[str]:
        if model_key and model_key in self.models:
            key = self.models[model_key].get_api_key(
                index, self._key_resolver
            )
            if key:
                return key
        if self.env_api_key and len(self.env_api_key) > index:
            return self.env_api_key[index].get_api_key()
        if self._key_resolver and self.api_key_env:
            return self._key_resolver(self.api_key_env)
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        return None

    def get_primary_api_key_name(
        self, model_key: Optional[str] = None
    ) -> Optional[str]:
        if model_key and model_key in self.models:
            n = self.models[model_key].get_primary_api_key_name()
            if n:
                return n
        if self.env_api_key:
            return self.env_api_key[0].name
        return self.api_key_env

    def get_model_config(self, model_key: str) -> Optional[ModelConfig]:
        return self.models.get(model_key)

    def get_all_model_names(self) -> List[str]:
        return [f"{self.name}/{k}" for k in self.models.keys()]

    def validate(self, model_key: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        if model_key and model_key in self.models:
            ok, err = self.models[model_key].validate()
            if not ok:
                return ok, err
        for c in self.env_api_key:
            ok, err = c.validate()
            if not ok:
                return ok, err
        return True, None


@dataclass
class NodeModelEntry:
    """node_config 中某一项的 models 列表中的一条"""
    provider: str
    model: str
    is_default: bool = False


class ConfigManager:
    """
    配置管理器。
    支持两种格式：
    - 标准格式：顶层有 env_api_key、providers（列表）、node_config（列表）
    - 旧格式：顶层 key 为 provider 名，value 为 provider 配置
    """

    _instance = None
    _config: Dict[str, ProviderConfig] = {}
    _node_config: Dict[str, List[NodeModelEntry]] = {}  # node_name -> [NodeModelEntry]
    _global_env_api_key: Dict[str, ApiKeyConfig] = {}
    _loaded = False
    _config_path: Optional[Path] = None
    _use_standard_format: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._loaded:
            self.load_config()

    def _get_global_key_value(self, env_name: str) -> Optional[str]:
        """从全局 env_api_key 或环境变量解析密钥"""
        if env_name in self._global_env_api_key:
            return self._global_env_api_key[env_name].get_api_key()
        return os.environ.get(env_name)

    def load_config(self, config_path: Optional[str] = None):
        if config_path is None:
            current_dir = Path(__file__).parent.parent
            for name in ("config.yaml", "config.yaml.example"):
                p = current_dir / "config" / name
                if p.exists():
                    config_path = p
                    break
            else:
                config_path = current_dir / "config" / "config.yaml"

        self._config_path = Path(config_path)

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f) or {}

            self._config = {}
            self._node_config = {}
            self._global_env_api_key = {}

            # 标准格式：env_api_key + providers + node_config
            if "providers" in config_data:
                self._use_standard_format = True
                # 全局 API 密钥
                for item in config_data.get("env_api_key") or []:
                    c = ApiKeyConfig.from_dict(item)
                    self._global_env_api_key[c.name] = c
                key_resolver = self._get_global_key_value
                # 提供商列表 -> 按 name 建索引
                for p in config_data.get("providers") or []:
                    if not isinstance(p, dict):
                        continue
                    name = p.get("name")
                    if not name:
                        continue
                    self._config[name] = ProviderConfig.from_dict(
                        name, p, global_key_resolver=key_resolver
                    )
                # 节点配置
                for nc in config_data.get("node_config") or []:
                    if not isinstance(nc, dict):
                        continue
                    node_name = nc.get("name")
                    if not node_name:
                        continue
                    entries = []
                    for m in nc.get("models") or []:
                        if isinstance(m, dict) and "provider" in m and "model" in m:
                            entries.append(
                                NodeModelEntry(
                                    provider=m["provider"],
                                    model=m["model"],
                                    is_default=m.get("is_default", False),
                                )
                            )
                    self._node_config[node_name] = entries
            else:
                # 旧格式：顶层 key 为 provider 名
                self._use_standard_format = False
                for provider_name, provider_data in config_data.items():
                    if provider_name.startswith("#") or provider_name.startswith("_"):
                        continue
                    if isinstance(provider_data, dict):
                        self._config[provider_name] = ProviderConfig.from_dict(
                            provider_name, provider_data
                        )

            self._loaded = True
            print(f"[Custom API Nodes] 配置加载成功: {config_path}")
            print(f"[Custom API Nodes] 已加载 {len(self._config)} 个提供商")

            self._validate_all()
            if self._config:
                self.print_config_summary()

        except FileNotFoundError:
            print(f"[Custom API Nodes] 警告: 配置文件未找到: {config_path}")
            self._config = {}
        except yaml.YAMLError as e:
            print(f"[Custom API Nodes] 配置文件解析错误: {e}")
            self._config = {}

    def _validate_all(self):
        for provider_name, provider in self._config.items():
            ok, err = provider.validate()
            if not ok:
                print(f"[Custom API Nodes] 警告 [{provider_name}]: {err}")

    def get_provider(self, name: str) -> Optional[ProviderConfig]:
        return self._config.get(name)

    def get_all_providers(self) -> Dict[str, ProviderConfig]:
        return self._config.copy()

    def get_all_models(self) -> List[str]:
        """所有 provider/model 列表"""
        out = []
        for provider in self._config.values():
            out.extend(provider.get_all_model_names())
        return out

    def get_models_for_node(self, node_name: str) -> List[str]:
        """
        根据 node_config 返回该节点可用的 provider/model 列表。
        若该节点未在 node_config 中配置，则返回全部模型（兼容旧行为）。
        """
        entries = self._node_config.get(node_name)
        if not entries:
            return self.get_all_models()
        return [f"{e.provider}/{e.model}" for e in entries]

    def get_default_model_for_node(self, node_name: str) -> Optional[str]:
        """返回该节点在 node_config 中的默认选项（is_default: true 或第一个）。"""
        entries = self._node_config.get(node_name)
        if not entries:
            all_models = self.get_all_models()
            return all_models[0] if all_models else None
        for e in entries:
            if e.is_default:
                return f"{e.provider}/{e.model}"
        return f"{entries[0].provider}/{entries[0].model}"

    def get_model_info(
        self, provider_model: str
    ) -> Optional[Tuple[ProviderConfig, Optional[ModelConfig]]]:
        """从 'provider/model_key' 解析出 (ProviderConfig, ModelConfig)。"""
        if "/" not in provider_model:
            return None
        provider_name, model_key = provider_model.split("/", 1)
        provider = self.get_provider(provider_name)
        if not provider:
            return None
        model_config = provider.get_model_config(model_key)
        if not model_config:
            return None
        return (provider, model_config)

    def get_api_key_status(self) -> Dict[str, Dict[str, Any]]:
        status = {}
        for provider_name, provider in self._config.items():
            provider_status = {"provider_keys": [], "models": {}}
            if provider.env_api_key:
                for c in provider.env_api_key:
                    provider_status["provider_keys"].append(
                        {
                            "name": c.name,
                            "configured": c.get_api_key() is not None,
                            "required": c.required,
                            "description": c.description,
                        }
                    )
            elif provider.api_key_env:
                v = provider.get_api_key()
                provider_status["provider_keys"].append(
                    {
                        "name": provider.api_key_env,
                        "configured": v is not None,
                        "required": True,
                        "description": None,
                    }
                )
            for model_key, model_config in provider.models.items():
                if model_config.env_api_key or model_config.api_key_env:
                    names = [c.name for c in model_config.env_api_key] or [
                        model_config.api_key_env
                    ]
                    provider_status["models"][model_key] = [
                        {"name": n, "configured": provider.get_api_key(model_key) is not None}
                        for n in names
                    ]
            status[provider_name] = provider_status
        return status

    def reload_config(self):
        self._loaded = False
        self.load_config(self._config_path)

    def print_config_summary(self):
        print("=" * 60)
        print("ComfyUI Custom API 节点配置摘要")
        print("=" * 60)
        for provider_name, provider in self._config.items():
            print(f"\n📦 {provider_name} ({provider.type.value})")
            if provider.env_api_key:
                for c in provider.env_api_key:
                    ok = "✅" if c.get_api_key() else "❌"
                    print(f"   API密钥: {c.name} {ok}")
            elif provider.api_key_env:
                ok = "✅" if provider.get_api_key() else "❌"
                print(f"   API密钥: {provider.api_key_env} {ok}")
            if provider.base_url:
                print(f"   基础URL: {provider.base_url}")
            if provider.models:
                print(f"   模型: {list(provider.models.keys())}")
        if self._node_config:
            print("\n节点模型配置 (node_config):")
            for node_name, entries in self._node_config.items():
                defaults = [f"{e.provider}/{e.model}" for e in entries if e.is_default]
                all_ = [f"{e.provider}/{e.model}" for e in entries]
                print(f"   {node_name}: {all_} 默认={defaults or all_[:1]}")
        print("=" * 60)


config_manager = ConfigManager()
