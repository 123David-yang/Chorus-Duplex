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


def background_rooms():
    root = Path(os.environ.get("BG_SOURCE_DIR", BABBLE))
    rooms = sorted(path for path in root.glob("rm*/babb") if any(path.glob("*.wav")))
    if not rooms:
        raise FileNotFoundError(f"VOiCES WAV clips not found under {root}/rm*/babb. Set BG_SOURCE_DIR.")
    return rooms


def file_sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def finish_delivery(root, pending):
    payload = json.loads(pending.read_text(encoding="utf-8"))
    ident = pending.parent.name
    names = (f"{ident}.flac", f"{ident}_babble.flac", f"{ident}.timeline.json")
    if payload["annotation"]["dialogue_id"] != ident or set(payload["sha256"]) != set(names):
        raise ValueError(f"Invalid pending delivery: {ident}")
    for name in names:
        staged = pending.parent / name
        destination = root / "audio" / name
        existing = destination if destination.exists() else staged
        if file_sha256(existing) != payload["sha256"][name]:
            raise ValueError(f"Delivery file conflicts with pending receipt: {name}")
        if existing == staged:
            staged.replace(destination)
    pending.replace(root / "completed" / f"{ident}.json")
    return payload


def recover_deliveries(root):
    for pending in sorted((root / "staging").glob("*/delivery.pending.json")):
        finish_delivery(root, pending)


def deliver(d, synth, root):
    ident = d["dialogue_id"]
    stage = root / "staging" / ident
    stage.mkdir(exist_ok=True)
    pending = stage / "delivery.pending.json"
    if pending.is_file():
        return finish_delivery(root, pending)
    clean = stage / f"{ident}.flac"
    noisy = stage / f"{ident}_babble.flac"
    timing = stage / f"{ident}.timeline.json"
    rooms = background_rooms()
    frames, pause_ms = render(d, synth, clean, timing)
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
               "sha256":{p.name:file_sha256(p) for p in (clean,noisy,timing)}}
    bg.unlink()
    atomic_json(pending, payload)
    return finish_delivery(root, pending)



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
