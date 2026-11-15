"""Run one MOSS instance with script prefetch and resumable delivery."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time

from scripts.config import load_env
from scripts.production.state.lock import ProductionLock

from .config import DEFAULT_ROOT, RESOURCES
from .delivery import deliver, refresh
from .render import FastSynth
from .script import generate


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=DEFAULT_ROOT)
    parser.add_argument("--hours",type=float,default=100)
    args=parser.parse_args()
    root=args.root
    for subdir in ("audio","scripts","completed","staging"):
        (root/subdir).mkdir(parents=True,exist_ok=True)
    load_env()
    os.environ.update(TTS_BACKEND="moss_onnx", TTS_STRICT="1")
    default_manifest = RESOURCES / "voices" / "AISHELL-3" / "manifest.json"
    if default_manifest.is_file():
        os.environ.setdefault("VOICE_MANIFEST", str(default_manifest))
    os.environ.pop("MOSS_VOICE_POOL",None)
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
                    d=future.result()
                    future=pool.submit(generate,following,root)
                    receipt=deliver(d,synth,root)
                    hours=refresh(root,args.hours)
                    failures=0
                    row={"index":index,"state":"delivered","credited_seconds":receipt["credited_seconds"],"hours":hours}
                except Exception as exc:
                    failures+=1
                    row={"index":index,"state":"failed","error":str(exc)[:400],"type":type(exc).__name__}
                    if future.done() and future.exception():
                        future=pool.submit(generate,following,root)
                row["wall_seconds"]=round(time.monotonic()-started,2)
                log.write(json.dumps(row,ensure_ascii=False)+"\n")
                log.flush()
                print(json.dumps(row,ensure_ascii=False),flush=True)
                if failures>=12:
                    raise RuntimeError("连续12个候选出错，详见 events.jsonl")
                index=following
            future.cancel()
