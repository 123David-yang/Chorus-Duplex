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

from scripts.config import MOSS_DIR, VAD_MODEL, load_env
from scripts.production.state.lock import ProductionLock

from .delivery import DeliveryIndex, background_rooms, deliver, recover_deliveries
from .render import FastSynth
from .script import dialogue_id, generate


def check_inputs():
    errors = []
    backend = os.environ.get("TTS_BACKEND", "moss_onnx")
    device = os.environ.get("TTS_DEVICE", "cpu").lower()
    if backend == "moss_onnx":
        runtime = Path(os.environ.get("MOSS_DIR", MOSS_DIR))
        if not (runtime / "onnx_tts_runtime.py").is_file():
            errors.append("MOSS_DIR must point to a checkout containing onnx_tts_runtime.py.")
        model_dir = os.environ.get("MOSS_MODEL_DIR")
        if model_dir:
            model_root = Path(model_dir).expanduser()
            manifest_paths = (
                "browser_poc_manifest.json",
                "MOSS-TTS-Nano-100M-ONNX/browser_poc_manifest.json",
                "MOSS-TTS-Nano-ONNX-CPU/browser_poc_manifest.json",
            )
            if not any((model_root / name).is_file() for name in manifest_paths):
                errors.append("MOSS_MODEL_DIR must contain the ONNX browser_poc_manifest.json.")
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
            except ImportError:
                pass
        elif device != "cpu":
            errors.append("Qwen device must be cpu or cuda[:index].")
    elif backend == "kokoro":
        if device != "cpu":
            errors.append("Kokoro uses CPU here; configure OpenVINO device separately.")
    elif ":" not in backend:
        errors.append("TTS_BACKEND must be moss_onnx, qwen3, kokoro, or module:factory.")
    if not Path(os.environ.get("VAD_MODEL", VAD_MODEL)).is_file():
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
    parser.add_argument("--continuous", action="store_true", help="continue until stopped")
    parser.add_argument("--draft-mode", choices=("direct", "interactive"), default="direct")
    parser.add_argument("--language", choices=("zh", "en"), default="zh")
    parser.add_argument("--tts-backend", help="moss_onnx, qwen3, kokoro, or module:factory")
    parser.add_argument("--tts-device", help="cpu or cuda (CUDA support depends on backend)")
    parser.add_argument("--check", action="store_true", help="validate local inputs without generation or API calls")
    args=parser.parse_args()
    if not math.isfinite(args.hours) or args.hours < 0:
        parser.error("--hours must be finite and nonnegative")
    if args.start_index < 0:
        parser.error("--start-index must be nonnegative")
    load_env()
    from .config import default_output_root, resources_root
    default_root = default_output_root()
    root = args.root or (default_root.with_name(default_root.name + "_en")
                        if args.language == "en" else default_root)
    os.environ["TTS_BACKEND"] = args.tts_backend or os.environ.get("TTS_BACKEND", "moss_onnx")
    os.environ.update(TTS_STRICT="1", MOSS_TEXT_LANGUAGE=args.language)
    if args.tts_device is not None:
        os.environ["TTS_DEVICE"] = args.tts_device
    legacy_moss = Path("D:/Project_Vibe/moss-tts-official")
    if legacy_moss.is_dir():
        os.environ.setdefault("MOSS_DIR", str(legacy_moss))
    legacy_voices = resources_root() / "voices/AISHELL-3/manifest.json"
    if legacy_voices.is_file() and not os.environ.get("VOICE_REGISTRY"):
        os.environ.setdefault("VOICE_MANIFEST", str(legacy_voices))
    if args.check:
        errors = check_inputs()
        if errors:
            parser.error("\n".join(errors))
        print("Input check passed. No synthesis performed.")
        return
    for subdir in ("audio", "scripts", "completed", "staging"):
        (root/subdir).mkdir(parents=True, exist_ok=True)
    with ProductionLock(root):
        recover_deliveries(root)
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
        stop_file = root / "STOP"
        if args.continuous and stop_file.exists():
            return
        synth=FastSynth(root/"staging")
        failures=0
        def next_index(index):
            while (root/"completed"/f"{dialogue_id(index, args.language)}.json").exists():
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
            future=pool.submit(generate,index,root,args.draft_mode,args.language)
            while (target is None or hours < target) and not (args.continuous and stop_file.exists()):
                started=time.monotonic()
                following=next_index(index+1)
                try:
                    try:
                        d = future.result()
                    finally:
                        future = pool.submit(generate, following, root, args.draft_mode, args.language)
                    receipt=deliver(d,synth,root)
                    hours=index_state.add(receipt)
                    failures=0
                    row={"index":index,"state":"delivered","credited_seconds":receipt["credited_seconds"],"hours":hours}
                except Exception as exc:
                    failures+=1
                    row={"index":index,"state":"failed","error":str(exc)[:400],"type":type(exc).__name__}
                row["wall_seconds"]=round(time.monotonic()-started,2)
                log.write(json.dumps(row,ensure_ascii=False)+"\n")
                log.flush()
                print(json.dumps(row,ensure_ascii=False),flush=True)
                if failures >= 12:
                    if not args.continuous:
                        raise RuntimeError("连续12个候选出错，详见 events.jsonl")
                    time.sleep(60)
                    failures = 0
                index=following
            future.cancel()
