# TTS 模型与音色

## 准备环境

在 Python 3.12 虚拟环境中，CPU 使用 `pip install -r requirements-moss.txt`。
干净 CPU 环境的端到端测试在 Windows x64 完成；当前固定的 ONNX Runtime 版本没有 Intel macOS 的 CPython 3.12 wheel。
CUDA 13 / cuDNN 9 环境使用 `requirements-moss-gpu.txt`，不要在同一环境同时安装 CPU 和 GPU 两种 ONNX Runtime。
已在 AutoDL CUDA 13 Linux 镜像中新建 Python 3.12 venv，按 GPU 依赖清单安装并验证 MOSS 中英文推理；该测试依赖镜像中的 CUDA/cuDNN 系统库，不代表其他 GPU 环境已验证。
CUDA/cuDNN 动态库必须与所选 ONNX Runtime 兼容；必要时参考
[ONNX Runtime 官方安装说明](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)。

将仓库根目录的 `.env.example` 复制成 `.env`，填写剧本服务的 `LLM_API_KEY`、`LLM_BASE_URL` 和 `LLM_MODEL`。不要提交或公开 `.env`。

下载公开模型和测试素材：

```bash
python -m scripts.sound.tools.download_moss
python -m scripts.sound.tools.download_vad
python -m scripts.sound.tools.download_background_sample
```

MOSS 权重固定到上游修订版；Silero VAD 默认优先使用下载的 v5.1.2 并校验 SHA-256，尚未下载时兼容旧版路径。
仓库附带经过测试的轻量 MOSS ONNX 运行时及其 Apache-2.0 许可，无需另行复制本机运行时。
`download_background_sample` 只下载一个约五秒的 VOiCES 片段，适合冒烟测试；
正式生产应从 [VOiCES 原始数据](https://iqtlabs.github.io/voices/downloads/)准备多样化素材，
按 `rm*/babb/*.wav` 组织并通过 `BG_SOURCE_DIR` 指定，保留 CC BY 4.0 署名。
默认资源根可通过 `DUPLEX_RESOURCES_DIR` 修改；在 Windows 上默认为 D 盘。模型下载的 HF 缓存默认也位于资源根内，已设置的 HF 缓存环境变量优先。
模型、VAD 和背景素材可分别通过 `MOSS_MODEL_DIR`、`VAD_MODEL` 和 `BG_SOURCE_DIR` 覆盖。
模型下载器默认使用 `MOSS_MODEL_DIR`（如已设置）；VAD 和背景测试片段的下载器默认写入资源根。如需直接下载到自定义位置，分别传 `download_vad --output PATH`、`download_background_sample --output-root DIR`，或将运行时配置指向默认下载位置。
两个小素材安装器通过同目录临时文件和不覆盖操作完成发布；非 Windows 系统上的目标文件系统需支持硬链接。

## 选择模型和设备

生产入口默认使用 MOSS ONNX、CPU 和固定的 `Junhao` 助手音色。可以选择后端、设备和中英文：

```bash
python -m scripts.production.scale_moss --help
python -m scripts.production.scale_moss --tts-backend moss_onnx --tts-device cpu --language en --check
python -m scripts.production.scale_moss --tts-backend moss_onnx --tts-device cuda --language en --root /path/to/output --hours 0.05
```

CUDA 需要包含 CUDAExecutionProvider 的 ONNX Runtime 和兼容的 CUDA/cuDNN 环境。默认加载仓库内的 MOSS ONNX 运行时；`MOSS_DIR` 和 `MOSS_MODEL_DIR` 可分别覆盖运行时及模型目录。请求 CUDA 时，运行时无法使用 CUDA 会报错。当前 ONNX 适配器支持 `cpu`、`cuda` 和 `cuda:0`，多卡设备选择需要运行时进一步支持。

`--check` 检查必需路径和已知权重文件是否存在，不校验所有权重内容，也不创建 CUDA session；首次使用 GPU 仍需运行小样本确认 CUDA 与 cuDNN 可用。

如通过 `MOSS_DIR` 指向未修改的上游运行时，按该运行时版本另行安装所需的 PyTorch/torchaudio；仓库附带的运行时不需要它们。

`--tts-backend` 支持 `moss_onnx`、`qwen3`、`kokoro` 或实现 `synthesize(text, voice, reference_audio, max_seconds)` 的 `module:factory`。自定义后端使用下方的 `VOICE_REGISTRY` 注册音色。也可用 `TTS_BACKEND` 和 `TTS_DEVICE` 设置默认值，命令行参数优先。旧剧本会保留原来的模型与音色信息，切换模型应使用新目录和新剧本。

`module:factory` 和 `MOSS_DIR` 会从指定位置加载并执行 Python 代码，使用可信的代码来源。

启动时会完整解码当前音色清单中的所有参考音频，格式需受 Python SoundFile 支持；清单中任何损坏条目都会阻止启动，大型清单的首次检查可能较慢。这只验证文件可解码，不代表音频非静音或克隆效果合格。

Qwen 的预设音色使用 CustomVoice 模型，参考音色使用 Base 模型。两种模型按需要分别加载，混合使用会占用两份模型内存。默认助手音色为 Qwen 预设 `Ryan`。可通过 `QWEN_PRESET_MODEL` 和 `TTS_MODEL` 指定各自模型路径或模型 ID。Kokoro 不支持参考音色克隆。

Qwen 适配器已在 AutoDL 的 RTX 4080 SUPER 上用 `qwen-tts==0.1.1`、官方 0.6B CustomVoice 和 Base 本地快照试运行中文与英文各一条预设和参考音色克隆；均完成生成及解码检查，尚未做听感或长时生产评估。请在独立虚拟环境中按 [Qwen 官方安装说明](https://github.com/QwenLM/Qwen3-TTS)配置与设备匹配的 PyTorch、torchaudio 和 `qwen-tts`，不要在同一环境混装本项目的 MOSS CPU/GPU 依赖清单。官方 Qwen 包目前依赖 CPU 版 ONNX Runtime 且要求旧于 1.0 的 Hugging Face Hub，而 MOSS GPU 清单使用 GPU 版 ONNX Runtime 和 Hub 1.7.1；组合前应确认具体版本兼容。实际生产会在调用剧本服务前加载所需的 Qwen 模型，缺少权重时上游可能下载；`--check` 不会加载或下载模型。

适配器在 Ampere 及更新的 CUDA 设备上使用 bfloat16，其他设备使用 float32；所用 AutoDL GPU 上 float16 推理曾出现数值异常，其他 GPU 仍需单独验证。离线运行时应先完整下载模型，可通过 `QWEN_PRESET_MODEL` 和 `TTS_MODEL` 指向对应本地快照目录。设置 `HF_HUB_OFFLINE=1` 后若传入 Hub 模型 ID，适配器先仅从指定缓存解析本地快照；缓存缺失时失败。AutoDL 上已用模型 ID 和既有缓存验证解析、实体化及后续 GPU 推理。

`scale_moss --tts-backend qwen3` 可选择 Qwen 适配器；`produce_and_upload` 目前固定使用 MOSS，尚未支持 Qwen 自动生产并上传。

Qwen 生产入口会在恢复交付、加载权重和生成剧本前，为预设与启用的克隆模型分别核对内容摘要，并以 `model-lock.json` 绑定生产目录。标准 HF 缓存快照会先复制到资源目录的 `qwen_materialized` 下，首次需要额外磁盘和复制时间，后续复用实体快照并重新校验摘要。模型内容变化时须用新生产目录；已有音频、待完成交付或旧 staging 内容但无身份锁的目录不能补绑。已加载的后端实例不能重新绑定。

该锁只验证内容连续性，不认证模型来源，也不保证不同设备或运行版本的听感一致。快照应来自可信、静止的缓存；它不提供对抗性文件系统隔离。AutoDL RTX 4080 SUPER 上已验证双模型首次绑定、全新后端实例续跑比对，以及中英文预设和克隆四条短音频生成及解码；该小样不含完整对话交付、上传或长时生产。模型或缓存准备失败后保留现场，不自动覆盖或删除。

使用内置 Qwen 后端新生成的交付，在可确认设备与精度时于标注的 `profile.tts_runtime` 中记录引擎、实际设备编号和计算精度；该记录不包含模型版本或权重身份，旧交付不会补写。

为避免配置错误时持续消耗剧本服务额度，生产入口在首次生成剧本前加载 Qwen 所需模型，并在连续 12 个候选失败时停止；修正问题后可从同一目录续跑。

Qwen 模型权重缓存遵循显式 `HF_HUB_CACHE` 或 `HF_HOME`；未配置时放入资源目录的 `hf_cache/hub`。Xet 缓存默认使用资源目录（或显式 `HF_HOME`）下的 `hf_cache/xet`（`HF_HOME` 已设置时为其 `xet` 子目录），可在进程启动前通过 `HF_XET_CACHE` 覆盖。若 Hugging Face 库已在同一进程中导入，请在导入前设置这些缓存变量；这不会迁移或更改 SDK 的登录凭据。

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

MOSS 新生产目录可用 `scale_moss --bind-moss-model` 绑定模型内容；中英生产上传协调器启用此检查。绑定在恢复交付、加载模型与启动上传之前完成，加载后再次核对路径和摘要。已有绑定的目录续跑时自动复核，无须再次传选项。模型内容变化会要求新的生产目录。

已有音频但没有绑定记录的目录不能补写模型身份；使用新的输出目录开始绑定生产，旧交付保留。旧版协调器目录因此不能直接续接到绑定生产；可独立验证和上传既有交付，但不能将其当作本次新模型身份的证据。`--check` 仍只做输入预检，不创建绑定或加载模型。

指纹验证内容连续性，不认证模型来源或质量。当前 MOSS ONNX 布局的模型树文件和清单声明资源被覆盖，清单资源不得位于树外；本机制不解析任意 ONNX 图的依赖，也不提供对抗并发篡改的文件系统隔离。请保持模型目录静态。音色参考仍使用独立的参考音频摘要校验。
