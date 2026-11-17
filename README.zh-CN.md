# Chorus-Duplex（和声·双工）· 中英文多人全双工语音对话

[![GitHub](https://img.shields.io/badge/GitHub-Chorus--Duplex-181717?logo=github&logoColor=white)](https://github.com/123David-yang/Chorus-Duplex) [![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Dataset-FFD21E?logo=huggingface&logoColor=FFD21E)](https://huggingface.co/datasets/BiggestWave/Chorus-Duplex)

[English](README.md) · **简体中文** · [数据集与音频试听](https://huggingface.co/datasets/BiggestWave/Chorus-Duplex)

Chorus-Duplex 构建多位说话人与语音助手共同参与的中英文合成对话。它记录谁在说、对谁说、回应哪一句，以及助手何时该开口、何时该听。每条对话配对交付干声和背景人声版本，并附上剧本与根据合成音频得到的时间线。

[完整对话：犹豫接话](https://huggingface.co/datasets/BiggestWave/Chorus-Duplex/blob/main/demos/nov16-hesitation-handoff.mp3) · [连续附和](https://huggingface.co/datasets/BiggestWave/Chorus-Duplex/blob/main/demos/nov16-sequential-backchannels.mp3) · [同时附和](https://huggingface.co/datasets/BiggestWave/Chorus-Duplex/blob/main/demos/nov16-simultaneous-backchannels.mp3) · [同时打断](https://huggingface.co/datasets/BiggestWave/Chorus-Duplex/blob/main/demos/nov16-simultaneous-interruption.mp3) · [附和转打断](https://huggingface.co/datasets/BiggestWave/Chorus-Duplex/blob/main/demos/nov16-backchannel-to-interruption.mp3)

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
| 同时附和 | 不同用户同时开口，可说相同或不同的短反馈。 |
| 连续附和 | 同一用户或不同用户先后开口，下一句等上一句说完。 |
| 打断 `interruption` | 一位说话人在对方话轮结束前抢话，回应已听到的内容。 |
| 多人同时打断 | 两位用户打断同一话轮。 |
| 同时表达不同偏好 | 多人同时表达不同选择；助手听完后请求澄清。后续可达成一致，也可先保留分歧继续比较。 |
| 附和转打断 | 用户先短暂附和，助手继续说。用户转到「不过」「但是」或「等一下」时，助手停止。再次打断助手的新话轮，另记一个事件。 |
| 第三方插话 `side_talk` | 第三方短暂参与插曲，随后恢复主线。第三方话轮单独标注。 |

**犹豫接话** · 用户话说到一半停住，助手等 2.5–3.2 秒，再根据已听到的内容温和地给出建议。用户轮标为 `incomplete`，加上 `resolution: assistant_handoff`。助手作为独立的下一轮引用该用户轮。

## 和公开双工数据集对比

[Full-Duplex-Bench v1.0 和 v1.5](https://github.com/kevinhu-nv/Full-Duplex-Bench-NV/blob/main/dataset/README.md) 是以一位主要用户和助手为中心的双工评测集，部分重叠评测含其他人声。[MultiTalkFT](https://huggingface.co/datasets/MultiTalk/MultiTalkFT) 是开放的中英多人双工训练语料，提供按说话人分轨的音频与词级时间对齐。[MultiTalk 论文](https://arxiv.org/html/2609.36903v1)

| 特征 | Full-Duplex-Bench | MultiTalkFT | Chorus-Duplex |
| --- | :---: | :---: | :---: |
| 多人主对话 | ✗ | ✓ | ✓ |
| 第三方交流 | ✓ | — | ✓ |
| **附和** | ✓ | ✓ | ✓ |
| ↳ 同时附和 | — | — | ✓ |
| ↳ 连续附和 | — | — | ✓ |
| **打断** | ✓ | ✓ | ✓ |
| ↳ 同时打断 | — | — | ✓ |
| ↳ 附和转打断 | — | — | ✓ |
| **未完成** | ✓ | — | ✓ |
| ↳ 犹豫接话 | — | — | ✓ |
| 背景人声配对 | ✓ | — | ✓ |
<div><sub>✓ 支持 · — 公开资料未明确 · ✗ 明确不支持</sub></div>

## 剧本如何产生多样性

对话索引在 15 个中文或 12 个英文话题领域中轮换。带种子的抽样改变用户人数（1–6 人）、关系、场景、目标轮数和人声音色。剧本模型创作具体情节、人设和表达。助手跨对话固定使用同一个 MOSS 音色，可用参考录音扩展人类说话人的音色。内置人声音色有五种。生产规划会按不同的预设或参考录音配置限制人数；不同录音不一定对应听感不同的说话人。如需六位用户加一位第三方，请至少补充两个合适的参考人声音色。

规划器初始抽样未完成、附和、打断、第三方插话的概率依次为 70%、95%、66%、55%，通常至少选择三种事件；首条及可用音色情况有特殊处理。多人互动需要足够的参与者。这些是规划概率，最终数据分布还受生成与校验影响。

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

CPU 依赖组合已在 Python 3.12 下验证。

```bash
git clone https://github.com/123David-yang/Chorus-Duplex.git
cd Chorus-Duplex
python -m pip install -r requirements-moss.txt
```

已验证的 CPU 链路默认使用仓库附带的 MOSS ONNX 运行时。下载固定版本模型、Silero VAD 和一段背景测试素材：

```bash
python -m scripts.sound.tools.download_moss
python -m scripts.sound.tools.download_vad
python -m scripts.sound.tools.download_background_sample
```

这段约五秒的 VOiCES 人声仅用于冒烟测试；正式生产应准备多样的 VOiCES 背景录音。可用 `DUPLEX_RESOURCES_DIR` 指定模型与缓存的位置。

把 [`.env.example`](.env.example) 复制为 `.env`。Windows PowerShell 用 `Copy-Item .env.example .env`，macOS 或 Linux 用 `cp .env.example .env`。然后填写本地配置。

| 配置项 | 填写内容 |
| --- | --- |
| `LLM_BASE_URL`、`LLM_MODEL`、`LLM_API_KEY` | Chat Completions API 基础地址、模型名和密钥。客户端会拼接 `/chat/completions`。 |
| `MOSS_DIR`、`MOSS_MODEL_DIR` | 可选：分别覆盖仓库内运行时与下载的模型目录。 |
| `VAD_MODEL` | 兼容的 [Silero VAD](https://github.com/snakers4/silero-vad) ONNX 文件。 |
| `BG_SOURCE_DIR` | [VOiCES](https://registry.opendata.aws/lab41-sri-voices/) 素材根目录，素材放在 `rm*/babb/*.wav`。 |
| `VOICE_MANIFEST` | 可选音色参考清单。格式为 `schema_version: 1` 和 `voices` 数组，每项包含 `id`、`gender`、`reference_audio`、`source_url`、`license` 和 `sha256`。相对 WAV 路径以清单所在目录为基准。 |

```bash
python -m scripts.production.scale_moss --help
python -m scripts.production.scale_moss --check
python -m scripts.production.scale_moss --root output/moss --hours 0.05
```

GPU 环境使用 `requirements-moss-gpu.txt`，不要与 CPU 依赖文件混装。已在预装 CUDA/cuDNN 的 AutoDL Linux 镜像中新建 Python 3.12 环境，按该文件安装并验证 MOSS CUDA；其他 GPU 环境仍需自行验证。`--check` 不创建 CUDA session，首次使用请先运行一条小样。

生成英文时加 `--language en`；CUDA 环境配置完成后可选择 `--tts-backend moss_onnx --tts-device cuda`。多 TTS 后端、固定助手音色与参考音色见[音色配置说明](scripts/sound/TTS.md)。

`--check` 仅核对本地输入，无需加载模型或调用 API。最后一条命令的目标是至少三分钟有效音频。续跑使用同一个 `--root`，程序按完成记录跳过已交付条目。

## 边生产边上传

先在源码仓库之外准备新的输出目录，并指定有写入权限的数据集。下面是中英文各约三分钟的小规模示例：

```bash
python -m scripts.production.moss_scale.produce_and_upload --root /path/to/output --repo OWNER/DATASET --hours-zh 0.05 --hours-en 0.05 --tts-device cuda --execute
```

先在本地配置剧本服务和 Hugging Face 凭据。生产时已完成的干声/背景声配对会分批上传，输出目录保存唯一批次标识和已验证记录，用于续跑；结束时会检查远端数据集修订，配对音频只计一次时长。不加 `--execute` 时仅显示计划。显存和吞吐允许时可选 `--workers 2` 并行两种语言，需先在自己的设备上测试。在对应语言输出目录创建 `STOP` 文件可停止该生产进程。

`BG_SOURCE_DIR` 应指向 VOiCES 背景录音；新标注会记录带背景版本的来源、署名、许可和混音信息。精确干声音频哈希去重仅覆盖本次运行，不代表对整个已有数据集去重。

可选定向复查默认关闭。设置 `CHORUS_TARGETED_SAFETY_REVIEW=1` 后，仅对本地提示可能存在电器故障回应问题的剧本，每次生成调用最多额外调用一次已配置的语言模型；最多发送四轮助手可见的合成台词。敏感内容过滤并非完整保证，请只用于合成剧本。费用会随命中剧本数增加，剧本保存前崩溃可能导致续跑时再次调用。固定、附和锚点或被打断的台词会跳过，警示仍会保留。这不代表安全性已获保证。远程语言模型地址须使用 HTTPS，本机回环地址可用 HTTP。

协调器已通过本地及故障模拟测试；大规模无人值守 GPU 生产和所有网络故障尚未验证。扩量前请抽查有代表性的剧本与音频。

默认情况下，某个生产进程 30 分钟没有新增交付，或单次运行超过 24 小时，协调器会停止；这两项时限均可配置。协调器只能停止自己的进程，不能代替云平台停止计费；无人值守前还需单独配置云端关机或费用限制。

设置 STOP 时已经发出的剧本预取请求无法撤回。协调器会给生产进程短暂的退出时间，然后结束仍在运行的自有进程；续跑时使用同一输出目录。

## 目录结构

| 路径 | 内容 |
| --- | --- |
| `scripts/production/scale_moss.py` | CLI 入口 |
| `scripts/production/moss_scale/` | 剧本规划与校验、渲染、调度和交付 |
| `scripts/text/` | 对话结构、剧本生成组件和音色 |
| `scripts/sound/` | 合成后端、VAD 和背景混音 |
| `data/` | 话题、场景和附和配置 |

## 快速读取

运行 `pip install huggingface_hub soundfile`。下面读取一条标注、两份配对音频及时间线。音频数组按「采样点 × 声道」排列。

```python
import json

import soundfile as sf
from huggingface_hub import hf_hub_download

REPO = "BiggestWave/Chorus-Duplex"


def fetch(path):
    return hf_hub_download(REPO, path, repo_type="dataset")


with open(fetch("annotations.jsonl"), encoding="utf-8") as f:
    example = json.loads(next(line for line in f if line.strip()))
dialogue_id = example["dialogue_id"]

with open(fetch(f"audio/{dialogue_id}.timeline.json"), encoding="utf-8") as f:
    timeline = json.load(f)

clean, sr = sf.read(fetch(f"audio/{dialogue_id}.flac"), always_2d=True)
babble, _ = sf.read(fetch(f"audio/{dialogue_id}_babble.flac"), always_2d=True)

print(dialogue_id, sr, clean.shape, babble.shape)
print(timeline[0]["speaker"], timeline[0]["start_sec"], timeline[0]["end_sec"])
```

## 数据限制与素材来源

数据是中英文合成对话。生成时检查剧本结构、音频时序、TTS 截断和文件格式。当前 CLI 未接入 ASR。已开放的音频、试听和来源详见数据集卡片。[Hugging Face 数据集](https://huggingface.co/datasets/BiggestWave/Chorus-Duplex)

背景素材来自 SRI International / Lab41, In-Q-Tel 的 VOiCES，适用 CC BY 4.0。分发混音时请保留署名。使用音色参考录音时，请确保有相应使用权，并在清单中保留来源和许可。[VOiCES 语料](https://registry.opendata.aws/lab41-sri-voices/)

维护者 · David

## 数据引用

使用数据时，请引用 Hugging Face 数据卡中的 [BibTeX 条目](https://huggingface.co/datasets/BiggestWave/Chorus-Duplex#citation)。
