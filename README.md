# ComfyUI Gemini 自定义节点

一个支持 Gemini、OpenAI、OpenRouter 和 VectorEngine 等多提供商 API 的 ComfyUI AI 节点系统。

## 特性

- **多提供商支持**：Google Gemini、OpenAI、OpenRouter、VectorEngine
- **配置驱动**：通过 `config.yaml` 轻松管理提供商和模型
- **文本和图像生成**：适用于不同场景的多个节点
- **环境变量 API 密钥**：安全的 API 密钥管理方式
- **模块化设计**：整洁、可扩展的架构

## 安装

1. 将本仓库克隆到 ComfyUI 的自定义节点文件夹中：
```bash
cd ComfyUI/custom_nodes
git clone <你的仓库地址> ComfyUI-Gemini-Nodes
```

2. 安装依赖：
```bash
cd ComfyUI-Gemini-Nodes
pip install -r requirements.txt
```
（或按需安装：`pip install pyyaml pillow google-genai requests`）

3. 设置 API 密钥的环境变量：
```bash
export GEMINI_API_KEY="你的-gemini-api-key"
export OPENAI_API_KEY="你的-openai-api-key"
export OPENROUTER_API_KEY="你的-openrouter-api-key"
export VECTORENGINE_API_KEY="你的-vectorengine-api-key"
```

## 配置

编辑 `config/config.yaml` 来自定义提供商和模型。

### 新的 env_api_key 配置格式（推荐）- 数组形式

支持多密钥的数组配置：

```yaml
google:
  type: google
  # 新的 env_api_key 格式（推荐）- 数组形式
  env_api_key:
    - name: GEMINI_API_KEY          # 环境变量名称
      # value: "xxx"                # 可选：直接设置值（不推荐）
      required: true                # 是否必需
      description: "Google Gemini API 密钥"
  # 兼容旧版本的 api_key_env
  api_key_env: GEMINI_API_KEY
  base_url: null
  models:
    flash:
      model_name: gemini-2.5-flash-preview-05-20
    pro:
      model_name: gemini-2.5-pro-preview-06-05
    experimental:
      model_name: gemini-exp-1206
      # 模型级别的 env_api_key（覆盖 provider 级别）- 数组形式
      env_api_key:
        - name: GEMINI_TEST_KEY_ENV
          required: false
          description: "实验模型 API 密钥"
      api_key_env: GEMINI_TEST_KEY_ENV
```

### env_api_key 配置项说明

| 配置项 | 类型 | 必需 | 说明 |
|--------|------|------|------|
| `name` | string | 是 | 环境变量名称 |
| `value` | string | 否 | 直接设置值（支持 `${ENV_VAR}` 语法） |
| `required` | boolean | 否 | 是否必需，默认 `true` |
| `description` | string | 否 | 配置说明 |

### 提供商类型

- `google`：原生 Gemini API
- `openai`：OpenAI API
- `openai_v1`：OpenAI 兼容 API（OpenRouter、VectorEngine）

### 模型配置选项

```yaml
models:
  model_key:
    model_name: "actual-model-name"  # 必填
    # 新的 env_api_key 格式 - 数组形式
    env_api_key:
      - name: "MODEL_SPECIFIC_KEY"
        required: true
        description: "模型专用密钥"
    # 兼容旧版本
    api_key_env: "CUSTOM_KEY_ENV"
    type: "google"                     # 可选：覆盖提供商类型
    extra_headers:                     # 可选：额外请求头
      X-Custom-Header: "value"
```

## 可用节点

### 1. Gemini 文本生成

使用多模态输入（文本、图像）生成文本回复。

**输入：**
- `prompt`（必填）：文本提示
- `provider_model`：提供商/模型选择（如 "google/pro"）
- `seed`：随机种子，用于结果复现
- `images`（可选）：参考图像
- `system_prompt`（可选）：系统指令
- `temperature`：采样温度（0.0-2.0）
- `max_tokens`：最大输出 token 数

**输出：**
- `text`：生成的文本回复

### 2. Gemini 图像生成

根据文本描述生成图像。

**输入：**
- `prompt`：图像描述
- `provider_model`：模型选择
- `seed`：随机种子
- `reference_images`（可选）：用于风格/上下文的参考图像
- `system_prompt`：系统指令
- `aspect_ratio`：输出宽高比（1:1、4:3、16:9 等）

**输出：**
- `image`：生成的图像
- `info`：生成信息和 token 使用情况

### 3. Gemini 图像生成专业版

具有更多选项的高级图像生成功能。

**输入：**
- `prompt`：详细描述
- `provider_model`：模型选择
- `seed`：随机种子
- `resolution`：输出分辨率（1024x1024、1792x1024、1024x1792）
- `reference_images`：参考图像
- `negative_prompt`：需要避免的内容
- `system_prompt`：系统指令
- `num_images`：要生成的图像数量（1-4）

**输出：**
- `images`：批量生成的图像
- `info`：生成信息

### 4. Gemini 输入文件

加载并准备输入文件供其他节点使用。

**输入：**
- `file`：输入目录中的文件
- `additional_files`：链式连接多个文件

**输出：**
- `files`：文件数据列表

## 环境变量配置

### 方式一：系统环境变量（推荐）

设置 API 密钥环境变量：

```bash
export GEMINI_API_KEY="你的-gemini-api-key"
export OPENAI_API_KEY="你的-openai-api-key"
export OPENROUTER_API_KEY="你的-openrouter-api-key"
export VECTORENGINE_API_KEY="你的-vectorengine-api-key"
```

### 方式二：使用 ${} 语法引用环境变量

在配置文件中引用环境变量：

```yaml
env_api_key:
  - name: GEMINI_API_KEY
    value: "${GEMINI_API_KEY}"  # 从系统环境变量读取
    required: true
```

### 方式三：为特定模型设置密钥

```yaml
models:
  experimental:
    model_name: gemini-exp-1206
    env_api_key:
      - name: GEMINI_EXPERIMENTAL_KEY
        required: false
        description: "实验模型专用密钥"
    # 兼容旧版本
    api_key_env: GEMINI_EXPERIMENTAL_KEY
```

### 配置优先级

获取 API 密钥时的优先级：
1. **模型级别的 `env_api_key.value`**（如果设置了直接值）
2. **模型级别的 `env_api_key.name`**（环境变量）
3. **Provider 级别的 `env_api_key.value`**
4. **Provider 级别的 `env_api_key.name`**
5. **旧版本的 `api_key_env`**（兼容）

### 环境变量列表

| 变量名 | 说明 |
|----------|-------------|
| `GEMINI_API_KEY` | Google Gemini API 密钥 |
| `OPENAI_API_KEY` | OpenAI API 密钥 |
| `OPENROUTER_API_KEY` | OpenRouter API 密钥 |
| `VECTORENGINE_API_KEY` | VectorEngine API 密钥 |

## 使用示例

### 基础文本生成

1. 添加 "Gemini 文本生成" 节点
2. 将 provider_model 设置为 "google/pro" 或 "google/flash"
3. 输入你的提示词
4. 将输出连接到文本显示节点或保存节点

### 图像生成

1. 添加 "Gemini 图像生成" 节点
2. 选择一个支持图像生成的模型
3. 描述你想要的图像
4. 可选择添加参考图像
5. 设置宽高比

### 使用自定义提供商

1. 在 `config/config.yaml` 中添加你的提供商
2. 设置相应的 `api_key_env`
3. 重启 ComfyUI
4. 你的提供商和模型将出现在下拉菜单中

## 故障排除

### "配置文件未找到"
- 确保 `config/config.yaml` 文件存在
- 检查文件权限

### "API 密钥无效"
- 验证环境变量是否正确设置
- 检查 API 密钥是否有效并具有必要的权限

### "超出速率限制"
- 稍等片刻后再试
- 考虑升级你的 API 套餐
- 查看提供商的速率限制

### 模型未显示
- 检查 config.yaml 语法是否正确
- 验证所有必填字段是否已填写
- 查看 ComfyUI 控制台的错误消息

## 文件结构

```
ComfyUI-CustomNodes/
├── __init__.py              # ComfyUI 主入口
├── config/
│   └── config.yaml          # 提供商和模型配置
├── engines/
│   ├── __init__.py
│   ├── engines.py           # 引擎工厂与基类
│   ├── google_engine.py     # Google Gemini 引擎
│   └── openai_engine.py     # OpenAI 兼容引擎
├── nodes/
│   ├── __init__.py
│   └── gemini_node.py       # 节点实现
├── utils/
│   ├── __init__.py
│   └── config_manager.py    # 配置管理
├── nodes_gemini.py          # 向后兼容入口
├── requirements.txt         # Python 依赖
├── .gitignore               # Git 忽略配置
└── README.md                # 本文件
```

## 扩展

### 添加新提供商

1. 在 `config/config.yaml` 中添加提供商配置：

```yaml
mynewprovider:
  type: openai_v1
  # 使用新的 env_api_key 数组格式
  env_api_key:
    - name: MY_PROVIDER_KEY
      required: true
      description: "我的自定义提供商 API 密钥"
  # 兼容旧版本
  api_key_env: MY_PROVIDER_KEY
  base_url: https://api.myprovider.com/v1
  models:
    model1:
      model_name: my-model-v1
      env_api_key:
        - name: MODEL_SPECIFIC_KEY
          required: false
          description: "模型专用密钥（可选）"
```

2. 设置你的 API 密钥：
```bash
export MY_PROVIDER_KEY="你的-api-key"
export MODEL_SPECIFIC_KEY="模型专用密钥"
```

3. 重启 ComfyUI

### 添加新模型

使用新的 `env_api_key` 数组格式：

```yaml
google:
  models:
    my-custom-model:
      model_name: gemini-custom-v1
      env_api_key:
        - name: CUSTOM_MODEL_KEY
          required: true
          description: "自定义模型 API 密钥"
      # 兼容旧版本
      api_key_env: CUSTOM_MODEL_KEY
```

## 完整配置示例

以下是一个包含所有功能的完整配置示例（方案B - 数组形式）：

```yaml
# 完整配置示例
my_provider:
  type: openai_v1
  
  # 新的 env_api_key 格式（推荐）- 数组形式
  env_api_key:
    - name: MY_API_KEY           # 环境变量名称
      value: "${MY_API_KEY}"     # 使用 ${} 从环境变量读取（可选）
      required: true             # 是否必需
      description: "我的 API 密钥"
    - name: MY_BACKUP_KEY        # 可以配置多个密钥
      required: false
      description: "备用 API 密钥"
  
  # 兼容旧版本（可选，与 env_api_key 等效）
  api_key_env: MY_API_KEY
  
  base_url: https://api.example.com/v1
  
  models:
    # 基础模型配置
    basic_model:
      model_name: gpt-4
    
    # 带独立密钥的模型
    premium_model:
      model_name: gpt-4-turbo
      env_api_key:
        - name: PREMIUM_API_KEY
          required: true
          description: "高级模型专用密钥"
      api_key_env: PREMIUM_API_KEY  # 兼容旧版本
    
    # 带额外请求头的模型
    custom_model:
      model_name: custom-v1
      extra_headers:
        X-Custom-Header: "custom-value"
        X-Another-Header: "another-value"
```

### 配置验证

启动时，节点会自动验证配置并显示状态：

```
============================================================
ComfyUI Gemini 节点配置摘要
============================================================

📦 google (google)
   API密钥1: GEMINI_API_KEY ✅ [必需] - Google Gemini API 密钥
   基础URL: null
   模型 (3个):
     • flash
     • pro
     • experimental [GEMINI_TEST_KEY_ENV ❌]

📦 openai (openai)
   API密钥: OPENAI_API_KEY ✅
   模型 (1个):
     • dall-e-3

============================================================
```

✅ = 已配置  
❌ = 未配置

## 许可证

MIT 许可证

## 贡献

欢迎贡献！请确保：
- 代码遵循现有风格
- 新功能包含文档
- 测试通过（如适用）
