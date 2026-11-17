"""Add same-voice hesitation audio to existing MOSS deliveries."""

import argparse
import json
import os
from pathlib import Path
import random
import re

import numpy as np
import soundfile as sf

from scripts.config import load_env
from scripts.sound.synthesis.dialogue_to_wav import AudioQualityError

from .config import RESOURCES
from .delivery import file_sha256, finish_delivery
from .render import FastSynth
from .storage import atomic_json


FILLERS = ("呃", "嗯", "那个，那个", "我想一下", "让我想想")
EXISTING_FILLER = re.compile(r"(?:呃+|额+|嗯+|那个[，、 ]*那个|我想一下|让我想想|等我想想)[\s，、。…—~～]*$")


def patch_case(source, destination, receipt_path, synth):
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    annotation = receipt["annotation"]
    ident = annotation["dialogue_id"]
    clean_name, noisy_name, timeline_name = (
        f"{ident}.flac", f"{ident}_babble.flac", f"{ident}.timeline.json",
    )
    if (destination / "completed" / receipt_path.name).is_file():
        return 0
    timeline = json.loads((source / "audio" / timeline_name).read_text(encoding="utf-8"))
    clean, rate = sf.read(source / "audio" / clean_name, dtype="float32", always_2d=True)
    noisy, noisy_rate = sf.read(source / "audio" / noisy_name, dtype="float32", always_2d=True)
    if noisy_rate != rate or clean.shape != noisy.shape or clean.shape != (receipt["frames"], 2):
        raise ValueError(f"Inconsistent source audio pair: {ident}")
    voices = annotation["profile"]["voice_assignments"]
    patched = 0
    for row in timeline:
        if row["kind"] != "incomplete":
            continue
        turn = annotation["turns"][row["turn_id"]]
        head = turn["parts"][0]
        voice = voices.get(row["speaker"])
        if not voice or EXISTING_FILLER.search(head):
            continue
        begin = round(row["pause_start_sec"] * rate)
        end = round(row["pause_end_sec"] * rate)
        minimum_pause = 2.5 if row.get("resolution") == "assistant_handoff" else .5
        start = begin + round(.06 * rate)
        available = end - start - round(minimum_pause * rate)
        if available < round(.16 * rate):
            continue
        if any(other is not row and other["start_sec"] < end / rate
               and other["end_sec"] > begin / rate for other in timeline):
            continue
        channel = int(row["speaker"] == "assistant")
        if np.max(np.abs(clean[begin:end, channel])) > 1 / 32768:
            continue
        for filler in random.Random(f"{ident}:{row['turn_id']}").sample(FILLERS, len(FILLERS)):
            try:
                wave = synth.line(filler, voice)
            except AudioQualityError:
                continue
            if not round(.16 * rate) <= len(wave) <= available:
                continue
            stop = start + len(wave)
            peak = float(np.max(np.abs(wave)))
            room = .98 - float(np.max(np.abs(noisy[start:stop, channel])))
            if peak <= 0 or room <= 0:
                continue
            wave = wave * min(1., room / peak)
            clean[start:stop, channel] += wave
            noisy[start:stop, channel] += wave
            turn["parts"][0] = head.rstrip("，。！？、；：…— ") + "，" + filler + "……"
            turn["text"] = row["text"] = "".join(turn["parts"])
            row["pause_start_sec"] = stop / rate
            turn["pause_ms"] = round((end - stop) * 1000 / rate)
            if row.get("resolution") == "assistant_handoff":
                row["end_sec"] = stop / rate
            receipt["credited_seconds"] += (stop - begin) / rate
            patched += 1
            break
    if not patched:
        return 0
    stage = destination / "staging" / ident
    stage.mkdir(parents=True, exist_ok=True)
    sf.write(stage / clean_name, clean, rate, subtype="PCM_16")
    sf.write(stage / noisy_name, noisy, rate, subtype="PCM_16")
    atomic_json(stage / timeline_name, timeline)
    receipt["sha256"] = {name: file_sha256(stage / name) for name in (clean_name, noisy_name, timeline_name)}
    pending = stage / "delivery.pending.json"
    atomic_json(pending, receipt)
    finish_delivery(destination, pending)
    return patched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--case", action="append", help="Patch only these dialogue IDs")
    args = parser.parse_args()
    if args.source.resolve() == args.destination.resolve():
        parser.error("Choose a separate destination for validated replacements.")
    load_env()
    os.environ.update(TTS_BACKEND="moss_onnx", TTS_STRICT="1")
    manifest = RESOURCES / "voices/AISHELL-3/manifest.json"
    if manifest.is_file():
        os.environ.setdefault("VOICE_MANIFEST", str(manifest))
    for folder in ("audio", "completed", "staging"):
        (args.destination / folder).mkdir(parents=True, exist_ok=True)
    synth = FastSynth(args.destination / "staging")
    receipts = sorted((args.source / "completed").glob("*.json"))
    if args.case:
        receipts = [path for path in receipts if path.stem in args.case]
    for receipt in receipts:
        count = patch_case(args.source, args.destination, receipt, synth)
        if count:
            print(f"{receipt.stem}: patched {count}", flush=True)


if __name__ == "__main__":
    main()
