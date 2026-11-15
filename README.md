# ChorusDuplex

ChorusDuplex generates synthetic Mandarin multi-party duplex dialogue with MOSS-TTS-Nano. The dataset includes dialogue annotations, clean audio, paired background-speech audio, and realized turn timing.

[Dataset, documentation, and audio previews on Hugging Face](https://huggingface.co/datasets/BiggestWave/ChorusDuplex-SideTalk)

## 多人双工设计

ChorusDuplex 用合成语音构造多人会话，记录谁在说、对谁说、根据哪一轮回应，以及助手什么时候该介入。早期设计把场景分成三类：

| 场景 | 助手的参与方式 |
|---|---|
| 参与型 | 用户向助手提问，或邀请助手把多人意见整理成建议。 |
| 非参与型 | 用户之间或与第三方有自己的交流，指向和旁听信息可以用来区分助手何时保持安静。 |
| 社交过渡 | 助手可用短附和表明在听，完整回答留给明确需要它的时机。 |

`speaker` 记录说话人，`addressee` 记录发言指向，可指向一人或多人。用 `responds_to` 标出依赖的早先轮次。剧本中的回应、他人姓名和安排应根据对话中已经提供的信息。

| 互动 | 示例与标注 |
|---|---|
| 普通接话 | 前一轮结束后接话，`speech`。 |
| 未完成 | 说到一半停 0.5–2 秒，随后同一人续说，`incomplete` 带 `parts` 和 `pause_ms`。 |
| 单人附和 | 听众简短回应，主轮继续，`backchannel`。 |
| 同时或连续附和 | 多人回应同一主轮中的事实，各有自己的音色和时刻。 |
| 打断 | 用户在原轮结束前接过话轮，`interruption`。 |
| 同时或连续打断 | 两人抢同一个话轮，或者 A 打断助手，助手回应，B 再打断补充。 |
| 第三方插话 | 第三方参与短交流后主线恢复，第三方发言标作 `side_talk`。 |

## 合成与交付

```text
场景和音色采样
    → LLM 剧本和因果引用校验
    → MOSS ONNX 逐轮合成、VAD 和响度处理
    → 排布停顿、重叠和双声道
    → VOiCES 人声混音
    → 干声、背景声、时间线、标注及完成记录
```

剧本记录计划中的相对间隔，时间线记录实际合成后的 `start_sec`、`end_sec` 和重叠。主持事实的 `anchor_text` 用于放置附和。同一话轮上的多人事件各自有独立记录，便于分析多人语音活动和助手介入时机。

```text
annotations.jsonl
audio/
  <dialogue_id>.flac
  <dialogue_id>_babble.flac
  <dialogue_id>.timeline.json
```

本地运行生成 `status.json` 和 `events.jsonl`。有效时长按已交付的干声计入，背景声是同一对话的配对版本。

公开数据可用于研究多说话人话轮切换、打断和附和识别，以及助手发言时机。数据集卡片提供样本试听、下载和来源信息。[Hugging Face 数据集与试听](https://huggingface.co/datasets/BiggestWave/ChorusDuplex-SideTalk)

## Run locally

Use Python 3.10 or newer. Install the dependencies and copy `.env.example` to `.env`. Configure these inputs locally.

- A Chat Completions endpoint, model, and API key for script generation.
- The [MOSS-TTS-Nano ONNX runtime and models](https://github.com/OpenMOSS/MOSS-TTS-Nano). Set `MOSS_DIR` to the runtime checkout and `MOSS_MODEL_DIR` to the model directory.
- A compatible [Silero VAD](https://github.com/snakers4/silero-vad) ONNX model at `VAD_MODEL`.
- [VOiCES](https://registry.opendata.aws/lab41-voices/) babble clips. Set `BG_SOURCE_DIR` to a directory containing one or more `rm*/babb/*.wav` folders.
- Optionally, set `VOICE_MANIFEST` for voice cloning. Use the manifest format below and supply each WAV separately.

```bash
python -m pip install -r requirements.txt
python -m scripts.production.scale_moss --root output/moss --hours 0.05
```

The output includes `annotations.jsonl`, per-dialogue timelines and paired 48 kHz stereo FLAC files under `audio/`. The left channel contains users and third parties, and the right channel contains the assistant. For the babble variant, VOiCES speech is selected per dialogue and mixed into the left channel at a target SNR of 10 dB. VOiCES is from SRI International / Lab41, In-Q-Tel, under CC BY 4.0. Preserve attribution when distributing the mixed audio.

The runner checks script structure, turn timing, TTS truncation and file format. It does not run ASR. A single synthesis instance is reused and completed IDs are skipped on restart.

## Code layout

| Path | Purpose |
|---|---|
| `scripts/production/moss_scale/` | Script validation, synthesis timeline, paired delivery and batch scheduling |
| `scripts/text/` | Dialogue schema, generation and voice assignment |
| `scripts/sound/` | TTS adapters, VAD and background mixing |
| `data/` | Topic, scenario and backchannel configuration |

A voice manifest uses `schema_version: 1` and a `voices` array. Each item includes `id`, `gender`, `reference_audio`, `source_url`, `license` and `sha256`. Relative audio paths resolve from the manifest directory. Use reference audio with appropriate permissions, such as selected clips from [AISHELL-3](https://www.openslr.org/93/).

Project contact and maintainer display name: David.
