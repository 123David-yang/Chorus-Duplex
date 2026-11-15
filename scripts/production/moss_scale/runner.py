"""Run one MOSS instance with script prefetch and resumable delivery."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import time

from scripts.config import MOSS_DIR, VAD_MODEL, load_env
from scripts.production.state.lock import ProductionLock

from .config import DEFAULT_ROOT, RESOURCES
from .delivery import background_rooms, deliver, refresh
from .render import FastSynth
from .script import generate


def check_inputs():
    errors = []
    runtime = Path(os.environ.get("MOSS_DIR", MOSS_DIR))
    if not (runtime / "onnx_tts_runtime.py").is_file():
        errors.append("MOSS_DIR must point to a checkout containing onnx_tts_runtime.py.")
    model_dir = os.environ.get("MOSS_MODEL_DIR")
    if model_dir and not Path(model_dir).is_dir():
        errors.append("MOSS_MODEL_DIR does not exist.")
    if not Path(os.environ.get("VAD_MODEL", VAD_MODEL)).is_file():
        errors.append("VAD_MODEL must point to a Silero VAD ONNX file.")
    if not os.environ.get("LLM_API_KEY"):
        errors.append("LLM_API_KEY is missing.")
    try:
        background_rooms()
        from scripts.text.voices import reference_voices
        reference_voices()
    except (OSError, ValueError, KeyError) as exc:
        errors.append(str(exc))
    return errors


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="output directory")
    parser.add_argument("--hours", type=float, default=100, help="target credited duration in hours")
    parser.add_argument("--check", action="store_true", help="validate local inputs without generation or API calls")
    args=parser.parse_args()
    if not math.isfinite(args.hours) or args.hours < 0:
        parser.error("--hours must be finite and nonnegative")
    root = args.root
    load_env()
    os.environ.update(TTS_BACKEND="moss_onnx", TTS_STRICT="1")
    default_manifest = RESOURCES / "voices" / "AISHELL-3" / "manifest.json"
    if default_manifest.is_file():
        os.environ.setdefault("VOICE_MANIFEST", str(default_manifest))
    os.environ.pop("MOSS_VOICE_POOL",None)
    if args.check:
        errors = check_inputs()
        if errors:
            parser.error("\n".join(errors))
        print("Input check passed. No synthesis performed.")
        return
    for subdir in ("audio", "scripts", "completed", "staging"):
        (root/subdir).mkdir(parents=True, exist_ok=True)
    with ProductionLock(root):
        hours=refresh(root,args.hours)
        if hours>=args.hours:
            return
        synth=FastSynth(root/"staging")
        failures=0
        def next_index(index):
            while (root/"completed"/f"moss-scale-20261115-{index:06d}.json").exists():
                index+=1
            return index
        index=next_index(0)
        with ThreadPoolExecutor(max_workers=1) as pool, (root/"events.jsonl").open("a",encoding="utf-8") as log:
            future=pool.submit(generate,index,root)
            while hours<args.hours:
                started=time.monotonic()
                following=next_index(index+1)
                try:
                    try:
                        d = future.result()
                    finally:
                        future = pool.submit(generate, following, root)
                    receipt=deliver(d,synth,root)
                    hours=refresh(root,args.hours)
                    failures=0
                    row={"index":index,"state":"delivered","credited_seconds":receipt["credited_seconds"],"hours":hours}
                except Exception as exc:
                    failures+=1
                    row={"index":index,"state":"failed","error":str(exc)[:400],"type":type(exc).__name__}
                row["wall_seconds"]=round(time.monotonic()-started,2)
                log.write(json.dumps(row,ensure_ascii=False)+"\n")
                log.flush()
                print(json.dumps(row,ensure_ascii=False),flush=True)
                if failures>=12:
                    raise RuntimeError("连续12个候选出错，详见 events.jsonl")
                index=following
            future.cancel()
