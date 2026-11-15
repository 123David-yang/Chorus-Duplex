# ChorusDuplex

ChorusDuplex generates synthetic Mandarin multi-party duplex dialogue with MOSS-TTS-Nano. The dataset includes dialogue annotations, clean audio, paired background-speech audio, and realized turn timing.

[Dataset, documentation, and audio previews on Hugging Face](https://huggingface.co/datasets/BiggestWave/ChorusDuplex-SideTalk)

## Dialogue design

The script generator selects interactions to suit each scenario. A listener can acknowledge an ongoing turn without taking the floor, or interrupt and redirect the exchange. Multiple listeners can acknowledge the same turn together or successively. Multiple users can interrupt the same turn, or interrupt different assistant turns in a sequence. Incomplete turns use a 0.5–2 second mid-thought pause before the same speaker continues. A brief third-party exchange can sit within the main discussion.

The assistant uses the built-in Junhao voice. Other speakers can use built-in voices or licensed reference clips. The dataset card describes the publicly available audio and shows selected pilot examples.

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
