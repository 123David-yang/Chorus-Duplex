# TTS 模型与音色

生产入口默认使用 MOSS ONNX、CPU 和固定的 `Junhao` 助手音色。可以选择后端、设备和中英文：

```bash
python -m scripts.production.scale_moss --help
python -m scripts.production.scale_moss --tts-backend moss_onnx --tts-device cpu --language en --check
python -m scripts.production.scale_moss --tts-backend moss_onnx --tts-device cuda --language en --root /path/to/output --hours 0.05
```

CUDA 需要包含 CUDAExecutionProvider 的 ONNX Runtime 和兼容的 CUDA/cuDNN 环境。MOSS 模型运行时从 `MOSS_DIR` 加载，模型目录可通过 `MOSS_MODEL_DIR` 指定。请求 CUDA 时，运行时无法使用 CUDA 会报错。当前 ONNX 适配器支持 `cpu`、`cuda` 和 `cuda:0`，多卡设备选择需要运行时进一步支持。

`--tts-backend` 支持 `moss_onnx`、`qwen3`、`kokoro` 或实现 `synthesize(text, voice, reference_audio, max_seconds)` 的 `module:factory`。自定义后端使用下方的 `VOICE_REGISTRY` 注册音色。也可用 `TTS_BACKEND` 和 `TTS_DEVICE` 设置默认值，命令行参数优先。旧剧本会保留原来的模型与音色信息，切换模型应使用新目录和新剧本。

`module:factory` 和 `MOSS_DIR` 会从指定位置加载并执行 Python 代码，使用可信的代码来源。

Qwen 的预设音色使用 CustomVoice 模型，参考音色使用 Base 模型。两种模型按需要分别加载，混合使用会占用两份模型内存。默认助手音色为 Qwen 预设 `Ryan`。可通过 `QWEN_PRESET_MODEL` 和 `TTS_MODEL` 指定各自模型路径或模型 ID。Kokoro 不支持参考音色克隆。

Qwen 使用模型的语音编码帧率将单条发言预算转换成 `max_new_tokens`。上游接口不返回结束原因，因此音频长度接近上限时会保守标记为疑似截断；这项检查无法识别提前停止或发音错误，仍需抽听或另行评估。

## 注册音色

设置 `VOICE_REGISTRY` 为 JSON 文件路径，可为每个模型指定固定的助手预设、用户预设和参考音频。预设的 `speaker` 是该模型接受的原始音色 ID，`id` 是项目内使用的名称。参考路径支持相对于清单的位置，`sha256` 可选，提供时会核对音频内容；未提供时会计算并记录。若引用来自公开音色库，填写来源与许可。

```json
{
  "schema_version": 1,
  "backends": {
    "moss_onnx": {
      "assistant": "assistant",
      "voices": [
        {"id": "assistant", "kind": "preset", "speaker": "Junhao", "gender": "男声"},
        {"id": "participant_a", "kind": "preset", "speaker": "Xiaoyu", "gender": "女声"},
        {"id": "participant_b", "kind": "reference", "reference_audio": "voices/b.wav",
         "gender": "女声", "source_url": "https://example.org/source", "license": "CC BY 4.0"}
      ]
    }
  }
}
```

清单必须为当前后端指定助手和至少一个参与者。助手使用预设音色，参与者从其余音色中确定性分配。不同后端可以在同一文件中各有自己的配置。未设置 `VOICE_REGISTRY` 时，MOSS 保留原有预设和 `VOICE_MANIFEST` 参考音色清单的用法。

标注中的参考来源 URL 会去掉用户名、密码、查询参数和片段。应在 `source_url` 中提供可公开访问的来源页面。

新剧本会记录每个注册音色对应的预设或参考音频摘要。续跑时，如果实际映射与剧本记录不一致，会要求使用原配置或重新生成剧本。
