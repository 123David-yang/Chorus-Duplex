# ChorusDuplex（和声·双工）· 中文多人全双工语音对话

[![GitHub](https://img.shields.io/badge/GitHub-ChorusDuplex-181717?logo=github&logoColor=white)](https://github.com/123David-yang/ChorusDuplex) [![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Dataset-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/datasets/BiggestWave/ChorusDuplex-SideTalk)

[English](README.md) · **简体中文** · [数据集与音频试听](https://huggingface.co/datasets/BiggestWave/ChorusDuplex-SideTalk)

ChorusDuplex 构建多位说话人与语音助手共同参与的普通话合成对话。它记录谁在说、对谁说、回应哪一句，以及助手何时该开口、何时该听。每条对话配对交付干声和背景人声版本，并附上剧本与根据合成音频得到的时间线。

可以试听[未完成片段](https://huggingface.co/datasets/BiggestWave/ChorusDuplex-SideTalk/resolve/main/demos/incomplete-preview.wav)和[两人同时打断](https://huggingface.co/datasets/BiggestWave/ChorusDuplex-SideTalk/resolve/main/demos/simultaneous-interruption.wav)。未完成样例截取了停顿附近的短片段。

## 参与框架与发言指向

多人会话中，在场的人未必是每句话的对象。助手可以根据发言指向和对话历史判断何时参与。

| 参与情境 | 例子 | 助手的行为 |
| --- | --- | --- |
| 参与且被称呼（Ratified, addressed） | 两位用户请助手协调安排，或一人替另一人提问。 | 回答提问者或被指向的几位用户。 |
| 在场但未被称呼（Unaddressed） | 两位用户先互相核对时间，准备再向助手提问。 | 等他们说完。 |
| 非参与型插曲（Unratified side talk） | 外卖员和一位用户简短交谈。 | 记住原话题，插曲过后接回主线，合适时可简短承接。 |
| 社交过渡（Social bridge） | 用户在共同话题中随口感叹。 | 可以短暂附和，把话轮留给原说话人。 |

这些是项目的设计目标，也包括识别新成员、点名调度和根据对话历史建立称呼。比如新成员先说“叫我小林”，助手随后就可以用“小林”；在此之前可以称“刚才提到预算的那位”。剧本也可以用对话中已经建立的角色或关系帮助指认。这为说话人跟踪、发言指向和对小组成员的回应提供训练信号。

## 多人双工互动

| 类型 | 对话中的表现 |
| --- | --- |
| 普通接话 `speech` | 说话人接过话轮，继续主线。 |
| 未完成 `incomplete` | 同一人话说到一半停 0.5–2 秒再续说，可自然带上“嗯”“那个”等犹豫词。 |
| 附和 `backchannel` | 听众对已经说出的内容短暂表示认同，原说话人继续讲。 |
| 多人同时或连续附和 | A 和 B 同时附和同一观点，或 A 先说“对，我也这么想”，B 随后说“确实”。每人有独立音色和时间。 |
| 打断 `interruption` | 一位说话人在对方话轮结束前抢话，回应已听到的内容。 |
| 多人同时或连续打断 | 两人打断同一位说话人，或 A 打断助手、助手回应 A、B 再打断助手的补充。 |
| 第三方插话 `side_talk` | 第三方短暂参与插曲，随后恢复主线。第三方话轮单独标注。 |

## 和公开双工数据集对比

[Full-Duplex-Bench v1.0 和 v1.5](https://github.com/kevinhu-nv/Full-Duplex-Bench-NV/blob/main/dataset/README.md) 是以一位主要用户和助手为中心的双工评测集，部分重叠评测含其他人声。[MultiTalkFT](https://huggingface.co/datasets/MultiTalk/MultiTalkFT) 是开放的中英多人双工训练语料，提供按说话人分轨的音频与词级时间对齐。[MultiTalk 论文](https://arxiv.org/html/2609.36903v1)

| 特征 | Full-Duplex-Bench | MultiTalkFT | ChorusDuplex |
| --- | :---: | :---: | :---: |
| 多人主对话 | 一对一 | ✓ | ✓ |
| **附和** | ✓ | ✓ | ✓ |
| ↳ 多人附和 | 未单列 | 未单列 | ✓ |
| **打断** | ✓ | ✓ | ✓ |
| ↳ 同时或连续打断 | 未单列 | 未单列 | ✓ |
| **未完成** | ✓ | 有静音轮 | ✓ |
| ↳ 两段分别标注 | 未单列 | 未单列 | ✓ |
| ↳ 带犹豫词续说 | 未单列 | 未单列 | ✓ |
| 第三方交流 | ✓ | 有指向切换 | ✓ |
| 背景人声配对 | ✓ | 未见配对 | ✓ |

✓ 表示公开资料明确支持。未单列表示公开资料未将该项列为独立类型。Full-Duplex-Bench 取 v1.0 和 v1.5，ChorusDuplex 依据生成器和标注格式。

ChorusDuplex 把短附和对齐到主轮中已经说出口的内容。第三方插话有自己的说话人、发言指向和时间，环境人声则作为独立的声学条件加入配对音频。因此可以在同一条对话上比较：有无背景人声时，助手应该回应谁、何时应该保持安静。

## 剧本如何产生多样性

对话索引在 15 个话题领域中轮换。带种子的抽样改变用户人数（1–6 人）、关系、场景、目标轮数和人声音色。剧本模型创作具体情节、人设和表达。助手跨对话固定使用同一个 MOSS 音色，可用参考录音扩展人类说话人的音色。内置人声音色有五种。六位用户和一位第三方需要各有不同音色时，请额外提供至少两个不同的参考人声音色。

未完成、附和、打断、第三方插话独立抽样，计划概率依次为 40%、60%、27%、25%，首条使用预设场景。人数足够时再抽多人互动。提示词只加入选中事件的规则，生成后校验结构和引用。这里的概率用于规划剧本，剧本和音频筛选会改变最终分布。

## 合成链路与交付格式

```text
场景和音色规划 → 剧本生成与校验
→ MOSS-TTS-Nano ONNX 逐轮合成 → VAD 和话轮排布
→ 双声道渲染及 VOiCES 混音 → 音频、标注和时间线
```

`participants` 描述主对话中的人类参与者。助手用 `assistant`，第三方用 `third_party_01`。每轮有 `speaker`、`addressee`、`kind` 和 `responds_to`，其中 `responds_to` 指向此前轮次。附和的 `anchor_text` 取自主轮，用来把附和放到相关话已经说出口的位置。时间线记录实际的 `start_sec`、`end_sec`、重叠及停顿区间等事件细节。

```text
annotations.jsonl
audio/
  <dialogue_id>.flac
  <dialogue_id>_babble.flac
  <dialogue_id>.timeline.json
```

两个版本都是 48 kHz 双声道。左声道是用户和第三方，右声道是助手。背景版本把 VOiCES 人声混进左声道，目标信噪比为 10 dB。剧本中的第三方发言有独立的说话人编号和时间。配对音频只计一次有效时长。本地运行还会写入 `status.json`、`events.jsonl` 和逐条完成记录。

## 用 CLI 运行

使用 Python 3.11 或更新版本。

```bash
git clone https://github.com/123David-yang/ChorusDuplex.git
cd ChorusDuplex
python -m pip install -r requirements.txt
```

在同一个 Python 环境里，按所用 MOSS 运行时版本的说明安装依赖。本仓库的 `requirements.txt` 列的是 ChorusDuplex 链路依赖。如果所用的 MOSS 代码会导入 PyTorch 或 torchaudio，即便用 CPU 推理，也要安装对应的兼容包。[MOSS ONNX 安装说明](https://github.com/OpenMOSS/MOSS-TTS-Nano#onnx-cpu-inference)

下载 [TTS 权重](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX)和[音频分词器权重](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX)。把 `MOSS_MODEL_DIR` 指向它们的共同父目录，里面应有 `MOSS-TTS-Nano-100M-ONNX/` 和 `MOSS-Audio-Tokenizer-Nano-ONNX/`，保留各目录中的相对路径及文件名。

把 [`.env.example`](.env.example) 复制为 `.env`。Windows PowerShell 用 `Copy-Item .env.example .env`，macOS 或 Linux 用 `cp .env.example .env`。然后填写本地配置。

| 配置项 | 填写内容 |
| --- | --- |
| `LLM_BASE_URL`、`LLM_MODEL`、`LLM_API_KEY` | Chat Completions API 基础地址、模型名和密钥。客户端会拼接 `/chat/completions`。 |
| `MOSS_DIR`、`MOSS_MODEL_DIR` | 含 `onnx_tts_runtime.py` 的 [MOSS-TTS-Nano ONNX](https://github.com/OpenMOSS/MOSS-TTS-Nano) 运行时目录，以及本地模型目录。 |
| `VAD_MODEL` | 兼容的 [Silero VAD](https://github.com/snakers4/silero-vad) ONNX 文件。 |
| `BG_SOURCE_DIR` | [VOiCES](https://registry.opendata.aws/lab41-voices/) 素材根目录，素材放在 `rm*/babb/*.wav`。 |
| `VOICE_MANIFEST` | 可选音色参考清单。格式为 `schema_version: 1` 和 `voices` 数组，每项包含 `id`、`gender`、`reference_audio`、`source_url`、`license` 和 `sha256`。相对 WAV 路径以清单所在目录为基准。 |

```bash
python -m scripts.production.scale_moss --help
python -m scripts.production.scale_moss --check
python -m scripts.production.scale_moss --root output/moss --hours 0.05
```

`--check` 仅核对本地输入，无需加载模型或调用 API。最后一条命令的目标是至少三分钟有效音频。续跑使用同一个 `--root`，程序按完成记录跳过已交付条目。

## 目录结构

| 路径 | 内容 |
| --- | --- |
| `scripts/production/scale_moss.py` | CLI 入口 |
| `scripts/production/moss_scale/` | 剧本规划与校验、渲染、调度和交付 |
| `scripts/text/` | 对话结构、剧本生成组件和音色 |
| `scripts/sound/` | 合成后端、VAD 和背景混音 |
| `data/` | 话题、场景和附和配置 |

## 数据限制与素材来源

数据是合成普通话。生成时检查剧本结构、音频时序、TTS 截断和文件格式。当前 CLI 未接入 ASR。已开放的音频、试听和来源详见数据集卡片。[Hugging Face 数据集](https://huggingface.co/datasets/BiggestWave/ChorusDuplex-SideTalk)

背景素材来自 SRI International / Lab41, In-Q-Tel 的 VOiCES，适用 CC BY 4.0。分发混音时请保留署名。使用音色参考录音时，请确保有相应使用权，并在清单中保留来源和许可。[VOiCES 语料](https://registry.opendata.aws/lab41-voices/)

维护者 · David
