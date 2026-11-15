# ChorusDuplex — Multi-party full-duplex Mandarin dialogue

**English** · [简体中文](README.zh-CN.md) · [Dataset and audio demos](https://huggingface.co/datasets/BiggestWave/ChorusDuplex-SideTalk)

ChorusDuplex builds synthetic Mandarin conversations between a voice assistant and multiple people. It captures who is talking, whom they are addressing, what prompted each reply, and when an assistant should speak or listen. Each delivery pairs clean audio with a background-speech version and includes a script and the timing measured from the rendered audio.

Listen to [an incomplete-turn excerpt](https://huggingface.co/datasets/BiggestWave/ChorusDuplex-SideTalk/resolve/main/demos/incomplete-preview.wav) or [two users interrupting simultaneously](https://huggingface.co/datasets/BiggestWave/ChorusDuplex-SideTalk/resolve/main/demos/simultaneous-interruption.wav). The first link plays the short excerpt around the pause.

## Participation and addressees

The design follows the distinction between participation and address. Being present in a group conversation still leaves room for two other people to address each other.

| Situation | Example | Assistant behaviour |
| --- | --- | --- |
| Ratified and addressed | Two users ask the assistant to reconcile their plans, or one relays another's question. | Respond to the person or subgroup that asked. |
| Present but unaddressed | Two users compare schedules before coming back to the assistant. | Listen and allow them to finish. |
| Unratified side talk | A participant speaks briefly with a delivery person. | Keep track of the original topic and return to it afterward. A brief acknowledgement can mark the transition. |
| Social bridge | Someone makes a brief aside in a shared discussion. | A short acknowledgement can signal attention while the speaker retains the floor. |

These are design targets. They also cover newcomers, turn coordination and grounding references in dialogue history. For example, after a newcomer says “Call me Xiaolin,” the assistant can say “Xiaolin.” An earlier reference can use “the person who mentioned the budget.” Scripts can also use a role or relationship established by the speakers. This gives a model cues to track speakers and address a subgroup instead of the entire room.

## Full-duplex interactions

| Type | What happens |
| --- | --- |
| Ordinary speech, `speech` | A speaker takes a turn in the main conversation. |
| Incomplete turn, `incomplete` | One speaker pauses for 0.5–2 seconds mid-thought and then continues, sometimes with a natural filler such as 嗯 or 那个. |
| Backchannel, `backchannel` | A listener acknowledges something already said while the main speaker continues. |
| Simultaneous or consecutive backchannels | Two listeners acknowledge the same point together or in quick succession, each with their own voice and timing. |
| Interruption, `interruption` | A speaker takes the floor during an unfinished turn, reacting to something already audible. |
| Simultaneous or successive interruptions | Two users interrupt the same speaker, or user A interrupts the assistant, the assistant responds, and user B interrupts the follow-up. |
| Third-party speech, `side_talk` | A third party speaks briefly in a side exchange before the participants return to the main thread. |

## How the scripts vary

The dialogue index cycles through 15 topic domains. Seeded sampling varies the one to six users, relationships, settings, target turn counts and human voice assignments. The script model writes the particular situation, personas and wording. The assistant has a fixed MOSS voice across dialogues. Optional reference recordings expand the human voice pool. The built-in pool has five human voices. With six users and one third party, supply at least two additional distinct reference voices to avoid reusing a voice.

The scenario plan samples incomplete turns, backchannels, interruptions and side talk independently at 40%, 60%, 27% and 25%, with a curated first case. Group events require enough participants. The prompt includes the rules for the selected events, and validation checks the required structure and references. These are planning probabilities. Script and audio rejection can change the delivered distribution.

## Pipeline and delivery format

```text
Scenario and voice planning → script generation and validation
→ MOSS-TTS-Nano ONNX synthesis → VAD and turn placement
→ stereo rendering and VOiCES mixing → audio, annotations and timeline
```

`participants` describes the humans in the main conversation. The assistant uses `assistant`, and third-party speech uses `third_party_01`. A turn has a `speaker`, `addressee`, `kind` and `responds_to`. The last field refers to earlier turn indices. A backchannel's `anchor_text` comes from the host turn, so the acknowledgement can be placed after the relevant words. The timeline records actual `start_sec` and `end_sec`, overlap, and event-specific details such as pause boundaries.

```text
annotations.jsonl
audio/
  <dialogue_id>.flac
  <dialogue_id>_babble.flac
  <dialogue_id>.timeline.json
```

Both versions are 48 kHz stereo. Users and third parties are on the left, and the assistant is on the right. The babble variant mixes VOiCES background speech into the left channel at a target SNR of 10 dB. An annotated third-party turn has its own timing and speaker ID. Each pair counts once toward the credited duration. Local runs also write `status.json`, `events.jsonl` and completion receipts.

## Run locally

Use Python 3.11 or newer.

```bash
git clone https://github.com/123David-yang/ChorusDuplex.git
cd ChorusDuplex
python -m pip install -r requirements.txt
```

Install the MOSS runtime's dependencies in the same Python environment using the instructions for the checkout you use. This repository's `requirements.txt` covers the ChorusDuplex pipeline. A MOSS checkout that imports PyTorch or torchaudio also needs compatible packages, even when inference runs on CPU. [MOSS ONNX setup](https://github.com/OpenMOSS/MOSS-TTS-Nano#onnx-cpu-inference)

Download the [TTS weights](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX) and [audio tokenizer weights](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX). Set `MOSS_MODEL_DIR` to their shared parent, containing `MOSS-TTS-Nano-100M-ONNX/` and `MOSS-Audio-Tokenizer-Nano-ONNX/`. Keep the relative filenames within each directory.

Copy [`.env.example`](.env.example) to `.env`. In Windows PowerShell use `Copy-Item .env.example .env`; on macOS or Linux use `cp .env.example .env`. Fill in the local settings.

| Setting | Required input |
| --- | --- |
| `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY` | Chat Completions API base URL, model ID and key. The client appends `/chat/completions`. |
| `MOSS_DIR`, `MOSS_MODEL_DIR` | [MOSS-TTS-Nano ONNX](https://github.com/OpenMOSS/MOSS-TTS-Nano) runtime checkout containing `onnx_tts_runtime.py` and the local model directory. |
| `VAD_MODEL` | Path to a compatible [Silero VAD](https://github.com/snakers4/silero-vad) ONNX file. |
| `BG_SOURCE_DIR` | [VOiCES](https://registry.opendata.aws/lab41-voices/) clips under `rm*/babb/*.wav`. |
| `VOICE_MANIFEST` | Optional reference voices. Use `schema_version: 1` and a `voices` array. Each item has `id`, `gender`, `reference_audio`, `source_url`, `license` and `sha256`. Relative WAV paths resolve from the manifest directory. |

```bash
python -m scripts.production.scale_moss --help
python -m scripts.production.scale_moss --check
python -m scripts.production.scale_moss --root output/moss --hours 0.05
```

`--check` validates local inputs without loading models or calling the API. The final command requests at least three minutes of credited audio. Reuse the same `--root` to resume from completion receipts.

## Repository layout

| Path | Purpose |
| --- | --- |
| `scripts/production/scale_moss.py` | CLI entry point |
| `scripts/production/moss_scale/` | Script planning, validation, rendering, scheduling and delivery |
| `scripts/text/` | Dialogue schema, generation components and voices |
| `scripts/sound/` | TTS backends, VAD and background mixing |
| `data/` | Topic, scenario and backchannel configuration |

## Data, limitations and attribution

The data is synthetic Mandarin. Generation checks script structure, timing, truncation and audio format. The CLI currently runs without ASR. The dataset card lists available recordings, audio demos and sources. [Hugging Face dataset](https://huggingface.co/datasets/BiggestWave/ChorusDuplex-SideTalk)

VOiCES is from SRI International / Lab41, In-Q-Tel, under CC BY 4.0. Preserve attribution when redistributing audio mixes. For human voice references, use appropriately licensed clips and keep their source and licence in the manifest. [VOiCES corpus](https://registry.opendata.aws/lab41-voices/)

Maintainer · David
