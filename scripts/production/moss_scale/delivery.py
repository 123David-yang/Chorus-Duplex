"""Build the clean and babble pair and update the delivery index."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random

import soundfile as sf

from scripts.sound.background.clips import stitch_clips
from scripts.sound.background.mix import mix
from scripts.sound.synthesis.dialogue_to_wav import SAMPLE_RATE

from .config import BABBLE
from .render import render
from .storage import atomic_json


def deliver(d, synth, root):
    ident = d["dialogue_id"]
    stage = root / "staging" / ident
    stage.mkdir(exist_ok=True)
    clean = stage / f"{ident}.flac"
    noisy = stage / f"{ident}_babble.flac"
    timing = stage / f"{ident}.timeline.json"
    frames, pause_ms = render(d, synth, clean, timing)
    background_root = Path(os.environ.get("BG_SOURCE_DIR", BABBLE))
    rooms = sorted(p for p in background_root.glob("rm*/babb") if any(p.glob("*.wav")))
    if not rooms:
        raise FileNotFoundError("Set BG_SOURCE_DIR to the VOiCES babble directory containing rm*/babb.")
    room = random.Random(ident).choice(rooms)
    bg = stage / "background.wav"
    stitch_clips(room, bg, frames / SAMPLE_RATE + 2, seed_key=ident)
    mix(clean,bg,snr_db=10,out_mixed=noisy,out_bg=False,seed_key=ident)
    for path in (clean,noisy):
        info = sf.info(path)
        if (info.frames,info.samplerate,info.channels)!=(frames,SAMPLE_RATE,2):
            raise ValueError("干声与背景版本长度或格式不符")
    payload = {"annotation":d,"frames":frames,"sample_rate":SAMPLE_RATE,
               "credited_seconds":frames/SAMPLE_RATE-pause_ms/1000-d["turns"][0]["gap_before_ms"]/1000,
               "background":{"corpus":"VOiCES","license":"CC BY 4.0","room":room.parent.name,
                             "mix_channel":"left","target_snr_db":10,"seed":ident},
               "sha256":{p.name:hashlib.file_digest(p.open("rb"),"sha256").hexdigest() for p in (clean,noisy,timing)}}
    for path in (clean,noisy,timing):
        dest=root/"audio"/path.name
        if dest.exists():
            with dest.open("rb") as file:
                if hashlib.file_digest(file,"sha256").hexdigest()!=payload["sha256"][path.name]:
                    raise ValueError(f"交付目录已经存在不同文件 {dest.name}")
        else:
            path.replace(dest)
    bg.unlink()
    atomic_json(root/"completed"/f"{ident}.json",payload)
    return payload


def refresh(root, target):
    rows=[json.loads(p.read_text(encoding="utf-8")) for p in sorted((root/"completed").glob("*.json"))]
    temp=root/"annotations.jsonl.tmp"
    with temp.open("w",encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row["annotation"],ensure_ascii=False)+"\n")
    temp.replace(root/"annotations.jsonl")
    seconds=sum(r["credited_seconds"] for r in rows)
    kinds=Counter(t["kind"] for r in rows for t in r["annotation"]["turns"])
    atomic_json(root/"status.json",{"dialogues":len(rows),"credited_hours":seconds/3600,
                                    "target_hours":target,"turn_kinds":kinds,"asr_enabled":False})
    return seconds/3600
