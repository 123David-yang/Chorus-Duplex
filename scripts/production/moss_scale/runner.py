"""Run one MOSS instance with script prefetch and resumable delivery."""
import argparse
import importlib.util
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
import re
from pathlib import Path
import time

from scripts.config import MOSS_DIR, load_env, resolve_vad_model
from scripts.production.state.lock import ProductionLock

from .delivery import DeliveryIndex, DuplicateDeliveryError, background_rooms, deliver, recover_deliveries
from .errors import safe_diagnostic
from .render import FastSynth
from .script import dialogue_id, generate


def check_moss_assets(root: Path) -> list[str]:
    """Check referenced model files exist without loading models or claiming integrity."""
    candidates = ("browser_poc_manifest.json", "MOSS-TTS-Nano-100M-ONNX/browser_poc_manifest.json",
                  "MOSS-TTS-Nano-ONNX-CPU/browser_poc_manifest.json")
    manifest = next((root / name for name in candidates if (root / name).is_file()), None)
    if manifest is None:
        return ["MOSS ONNX models are missing; run python -m scripts.sound.tools.download_moss or set MOSS_MODEL_DIR."]
    aliases = {"MOSS-TTS-Nano-ONNX-CPU": "MOSS-TTS-Nano-100M-ONNX",
               "MOSS-Audio-Tokenizer-Nano-ONNX-CPU": "MOSS-Audio-Tokenizer-Nano-ONNX"}

    def resolve(relative: str) -> Path:
        candidate = manifest.parent / relative
        if candidate.exists():
            return candidate
        for old, new in aliases.items():
            if old in relative:
                candidate = manifest.parent / relative.replace(old, new)
                if candidate.exists():
                    return candidate
        return candidate

    try:
        model_files = json.loads(manifest.read_text(encoding="utf-8"))["model_files"]
        tts_meta = resolve(model_files["tts_meta"])
        codec_meta = resolve(model_files["codec_meta"])
        tokenizer = resolve(model_files.get("tokenizer_model", "tokenizer.model"))
        required = [tts_meta, codec_meta, tokenizer]
        for meta, mandatory in ((tts_meta, ("prefill", "decode_step", "local_decoder")),
                                (codec_meta, ("encode", "decode_full", "decode_step"))):
            files = json.loads(meta.read_text(encoding="utf-8"))["files"]
            if not all(isinstance(files.get(key), str) and files[key] for key in mandatory):
                raise ValueError("required ONNX entries are missing")
            required.extend(meta.parent / name for name in files.values() if isinstance(name, str) and name.endswith(".onnx"))
            known_external = {
                "moss_tts_prefill.onnx": "moss_tts_global_shared.data",
                "moss_tts_local_decoder.onnx": "moss_tts_local_shared.data",
                "moss_audio_tokenizer_encode.onnx": "moss_audio_tokenizer_encode.data",
                "moss_audio_tokenizer_decode_full.onnx": "moss_audio_tokenizer_decode_shared.data",
            }
            required.extend(meta.parent / data for graph, data in known_external.items() if graph in files.values())
        missing = [path.name for path in required if not path.is_file() or path.stat().st_size == 0]
        return ["MOSS model files missing or empty: " + ", ".join(sorted(set(missing)))] if missing else []
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return ["MOSS model metadata is missing or invalid; rerun the model downloader."]


def check_inputs():
    errors = []
    backend = os.environ.get("TTS_BACKEND", "moss_onnx")
    device = os.environ.get("TTS_DEVICE", "cpu").lower()
    if backend == "moss_onnx":
        for module in ("onnxruntime", "sentencepiece"):
            if importlib.util.find_spec(module) is None:
                errors.append(f"MOSS requires the {module} Python package.")
        runtime = Path(os.environ.get("MOSS_DIR", MOSS_DIR))
        if not (runtime / "onnx_tts_runtime.py").is_file():
            errors.append("MOSS_DIR must point to a checkout containing onnx_tts_runtime.py.")
        from .config import moss_model_root
        model_root = moss_model_root(runtime)
        errors.extend(check_moss_assets(model_root))
        if device in ("cuda", "cuda:0", "gpu"):
            try:
                import onnxruntime as ort
                if "CUDAExecutionProvider" not in ort.get_available_providers():
                    errors.append("MOSS CUDA requires ONNX Runtime CUDAExecutionProvider.")
            except ImportError:
                errors.append("MOSS CUDA requires onnxruntime-gpu.")
        elif device != "cpu":
            errors.append("MOSS ONNX device must be cpu or cuda:0.")
    elif backend == "qwen3":
        for module in ("torch", "qwen_tts"):
            if importlib.util.find_spec(module) is None:
                errors.append(f"Qwen requires the {module} Python package.")
        if re.fullmatch(r"cuda(?::[0-9]+)?", device):
            try:
                import torch
                if not torch.cuda.is_available():
                    errors.append("Qwen CUDA was requested but CUDA is unavailable.")
                else:
                    index = int(device.split(":", 1)[1]) if ":" in device else torch.cuda.current_device()
                    if index < 0 or index >= torch.cuda.device_count():
                        errors.append("Requested Qwen CUDA device index is unavailable.")
            except ImportError:
                pass
            except RuntimeError:
                errors.append("Qwen CUDA could not be initialized.")
        elif device != "cpu":
            errors.append("Qwen device must be cpu or cuda[:index].")
    elif backend == "kokoro":
        if device != "cpu":
            errors.append("Kokoro uses CPU here; configure OpenVINO device separately.")
        if importlib.util.find_spec("sherpa_onnx") is None:
            errors.append("Kokoro requires the sherpa_onnx Python package.")
        from .config import kokoro_model_root
        folder = kokoro_model_root()
        required = ("voices.bin", "tokens.txt", "lexicon-us-en.txt", "lexicon-zh.txt")
        if not any((folder / model).is_file() for model in ("model.onnx", "model.int8.onnx")) or any(
            not (folder / name).is_file() for name in required
        ):
            errors.append("Kokoro model assets are missing; set KOKORO_MODEL_DIR to a complete model directory.")
    elif ":" not in backend:
        errors.append("TTS_BACKEND must be moss_onnx, qwen3, kokoro, or module:factory.")
    if not resolve_vad_model().is_file():
        errors.append("VAD_MODEL must point to a Silero VAD ONNX file.")
    if not os.environ.get("LLM_API_KEY"):
        errors.append("LLM_API_KEY is missing.")
    try:
        background_rooms()
        from scripts.text.voices import reference_voices, voices_for_dialogue
        reference_voices()
        voices_for_dialogue("preflight", 1)
    except (OSError, ValueError, KeyError) as exc:
        errors.append(str(exc))
    return errors


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, help="output directory")
    parser.add_argument("--hours", type=float, default=100, help="target credited duration in hours")
    parser.add_argument("--start-index", type=int, default=0, help="first candidate index")
    parser.add_argument("--run-tag", help="unique public-safe namespace for a new production root")
    parser.add_argument("--continuous", action="store_true", help="continue until stopped")
    parser.add_argument("--draft-mode", choices=("direct", "interactive"), default="direct")
    parser.add_argument("--language", choices=("zh", "en"), default="zh")
    parser.add_argument("--tts-backend", help="moss_onnx, qwen3, kokoro, or module:factory")
    parser.add_argument("--tts-device", help="cpu or cuda (CUDA support depends on backend)")
    parser.add_argument("--bind-moss-model", action="store_true", help="bind a new MOSS root to model contents; never backfill existing audio")
    parser.add_argument("--check", action="store_true", help="validate local inputs without generation or API calls")
    args=parser.parse_args()
    if not math.isfinite(args.hours) or args.hours < 0:
        parser.error("--hours must be finite and nonnegative")
    if args.start_index < 0:
        parser.error("--start-index must be nonnegative")
    load_env()
    requested_tag = args.run_tag if args.run_tag is not None else os.environ.get("CHORUS_RUN_TAG", "")
    if requested_tag and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}", requested_tag):
        parser.error("--run-tag must be 1–32 letters, digits, underscores or hyphens")
    from .config import default_output_root, resources_root
    default_root = default_output_root()
    root = args.root or (default_root.with_name(default_root.name + "_en")
                        if args.language == "en" else default_root)
    os.environ["TTS_BACKEND"] = args.tts_backend or os.environ.get("TTS_BACKEND", "moss_onnx")
    if args.bind_moss_model and os.environ["TTS_BACKEND"] != "moss_onnx":
        parser.error("--bind-moss-model requires moss_onnx")
    os.environ.update(TTS_STRICT="1", MOSS_TEXT_LANGUAGE=args.language)
    if args.tts_device is not None:
        os.environ["TTS_DEVICE"] = args.tts_device
    legacy_voices = resources_root() / "voices/AISHELL-3/manifest.json"
    if (os.environ["TTS_BACKEND"] == "moss_onnx" and legacy_voices.is_file()
            and not os.environ.get("VOICE_REGISTRY")):
        os.environ.setdefault("VOICE_MANIFEST", str(legacy_voices))
    if args.check:
        errors = check_inputs()
        if errors:
            parser.error("\n".join(errors))
        print("Input check passed. No synthesis performed.")
        return
    for subdir in ("audio", "scripts", "completed", "staging", "duplicates"):
        (root/subdir).mkdir(parents=True, exist_ok=True)
    with ProductionLock(root):
        tag_file = root / "run-tag.json"
        if tag_file.exists():
            saved = json.loads(tag_file.read_text(encoding="utf-8"))
            saved_tag = saved["run_tag"]
            if requested_tag and requested_tag != saved_tag:
                parser.error("This production root belongs to another --run-tag.")
            if saved.get("language") != args.language or saved.get("tts_backend") != os.environ["TTS_BACKEND"]:
                parser.error("Choose a separate --root for each production language and TTS backend.")
            requested_tag = saved_tag
        elif requested_tag:
            if any(any((root / name).iterdir()) for name in ("audio", "scripts", "completed", "staging")):
                parser.error("Use an empty production root when introducing --run-tag.")
            from .storage import atomic_json
            atomic_json(tag_file, {"run_tag": requested_tag, "language": args.language,
                                   "tts_backend": os.environ["TTS_BACKEND"]})
        if requested_tag:
            os.environ["CHORUS_RUN_TAG"] = requested_tag
        else:
            os.environ.pop("CHORUS_RUN_TAG", None)
        dedupe_db = Path(os.environ["CHORUS_DEDUPE_DB"]) if os.environ.get("CHORUS_DEDUPE_DB") else None
        production_backend = None
        moss_binding = None
        if os.environ["TTS_BACKEND"] == "moss_onnx" and (args.bind_moss_model or (root / "model-lock.json").exists() or (root / "model-lock.json").is_symlink()):
            from .moss_binding import prepare_moss_models
            moss_binding = prepare_moss_models(root)
        if os.environ["TTS_BACKEND"] == "qwen3":
            for receipt_path in (root / "completed").glob("*.json"):
                profile = json.loads(receipt_path.read_text(encoding="utf-8"))["annotation"].get("profile", {})
                if profile.get("tts_backend", "moss_onnx") != "qwen3":
                    parser.error("Choose a separate --root for each TTS backend.")
                if profile.get("language", "zh") != args.language:
                    parser.error("Choose a separate --root for each production language.")
            from scripts.sound.synthesis.backends import create_backend
            from scripts.text.voices import reference_voices
            production_backend = create_backend(root / "staging" / "_binding.wav")
            production_backend.prepare_production(root, include_references=bool(reference_voices()))
        recover_deliveries(root, dedupe_db)
        for receipt_path in (root / "completed").glob("*.json"):
            annotation = json.loads(receipt_path.read_text(encoding="utf-8"))["annotation"]
            if annotation.get("profile", {}).get("language", "zh") != args.language:
                parser.error("Choose a separate --root for each production language.")
            existing_backend = annotation.get("profile", {}).get("tts_backend", "moss_onnx")
            if existing_backend != os.environ["TTS_BACKEND"]:
                parser.error("Choose a separate --root for each TTS backend.")
        target = None if args.continuous else args.hours
        index_state = DeliveryIndex(root, target)
        hours = index_state.hours
        if target is not None and hours >= target:
            return
        errors = check_inputs()
        if errors:
            parser.error("\n".join(errors))
        stop_files = (root / "STOP", root / ".chorus-coordinator-stop")
        def stop_requested():
            return any(path.exists() for path in stop_files)
        if stop_requested():
            return
        synth=FastSynth(root/"staging")
        if moss_binding is not None:
            from .moss_binding import verify_loaded_moss
            verify_loaded_moss(synth.backend, *moss_binding)
        if os.environ["TTS_BACKEND"] == "qwen3":
            from scripts.text.voices import reference_voices
            synth.backend = production_backend
            synth.backend.prepare(include_references=bool(reference_voices()))
        failures=0
        def next_index(index):
            while any((root/name/f"{dialogue_id(index, args.language)}.json").exists()
                      for name in ("completed", "duplicates")):
                index+=1
            return index
        log_path = root / "events.jsonl"
        start_index = args.start_index
        if args.continuous and log_path.is_file():
            for line in log_path.read_text(encoding="utf-8").splitlines():
                try:
                    start_index = max(start_index, json.loads(line)["index"] + 1)
                except (ValueError, KeyError, TypeError):
                    continue
        index = next_index(start_index)
        with ThreadPoolExecutor(max_workers=1) as pool, log_path.open("a",encoding="utf-8") as log:
            future = None
            while (target is None or hours < target) and not stop_requested():
                started=time.monotonic()
                following=next_index(index+1)
                if future is None:
                    future=pool.submit(generate,index,root,args.draft_mode,args.language)
                try:
                    try:
                        d = future.result()
                    finally:
                        future = None
                        # Avoid a paid extra draft after STOP or near a finite target.
                        if not stop_requested() and (target is None or target - hours > .1):
                            future = pool.submit(generate, following, root, args.draft_mode, args.language)
                    if stop_requested():
                        break  # Preserve the drafted script for a later resume; do not start synthesis.
                    receipt=deliver(d,synth,root,index_state.clean_hashes,dedupe_db)
                    hours=index_state.add(receipt)
                    failures=0
                    row={"index":index,"state":"delivered","credited_seconds":receipt["credited_seconds"],"hours":hours}
                except DuplicateDeliveryError:
                    failures += 1
                    row={"index":index,"state":"duplicate","error":"candidate retained in staging"}
                except Exception as exc:
                    failures+=1
                    diagnostic = safe_diagnostic(exc)
                    row={"index":index,"state":"failed","error":diagnostic["hint"],
                         "type":diagnostic["type"], "code":diagnostic["code"]}
                row["wall_seconds"]=round(time.monotonic()-started,2)
                log.write(json.dumps(row,ensure_ascii=False)+"\n")
                log.flush()
                print(json.dumps(row,ensure_ascii=False),flush=True)
                if failures >= 12:
                    raise RuntimeError("连续12个候选出错，已停止以避免持续消耗；详见 events.jsonl")
                index=following
            if future is not None:
                future.cancel()
