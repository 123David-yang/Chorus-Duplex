"""Synthesize MOSS speech and place turns on the stereo timeline."""
import os
import random

import numpy as np
import soundfile as sf

from scripts.sound.synthesis.dialogue_to_wav import DialogueSynth, SAMPLE_RATE, _track

from .storage import atomic_json


class FastSynth(DialogueSynth):
    def __init__(self, folder):
        super().__init__(folder)
        self.short_cache = {}

    def line(self, content, voice, offset=0):
        key = (content, voice, offset)
        if key in self.short_cache:
            return self.short_cache[key]
        from scripts.sound.synthesis.dialogue_to_wav import AudioQualityError
        budget = max(5., len(content)*.40+2.)
        try:
            wave = self.turn_to_mono(content, voice, target_offset_db=offset, synthesis_budget=budget)
        except AudioQualityError:
            previous = os.environ.get("MOSS_SEED", "1234")
            os.environ["MOSS_SEED"] = "20261115"
            try:
                wave = self.turn_to_mono(content, voice, target_offset_db=offset, synthesis_budget=budget * 1.35)
            finally:
                os.environ["MOSS_SEED"] = previous
        if len(content) <= 12 and len(self.short_cache) < 512:
            self.short_cache[key] = wave
        return wave


def render(d, synth, clean, timeline_path):
    turns = d["turns"]
    voices = d["profile"]["voice_assignments"]
    backchannel_gain = random.Random(f"backchannel-gain:{d.get('dialogue_id', 'preview')}")
    placed, timeline = [], []
    cursor = host_end = boundary = listeners = 0
    for i, t in enumerate(turns):
        speaker, content, kind = t["speaker"], t["text"], t["kind"]
        voice = voices[speaker]
        timing = {}
        hold_frames = 0
        if kind == "backchannel":
            offset_db = round(backchannel_gain.uniform(-10.0, -3.0), 1)
            wave = synth.line(content, voice, offset_db)
            spacing = d["profile"].get("listener_spacing_ms", 120)
            start = boundary + int((.08 + spacing / 1000 * listeners) * SAMPLE_RATE)
            if listeners and d["profile"].get("listener_sequence") == "sequential":
                _, previous_start, previous_wave = placed[-1]
                start = previous_start + len(previous_wave) + round(spacing * SAMPLE_RATE / 1000)
            if start >= host_end:
                raise ValueError("附和开始时主轮已结束")
            timing.update(host_turn_id=t["responds_to"][0], anchor_boundary_sec=boundary/SAMPLE_RATE)
            t["insert_ratio"] = (start - placed[t["responds_to"][0]][1]) / (
                host_end - placed[t["responds_to"][0]][1])
            listeners += 1
        else:
            upcoming = turns[i+1] if i+1<len(turns) else {}
            if kind == "interruption":
                host_id = t["responds_to"][0]
                _, host_start, host_wave = placed[host_id]
                start = max(host_start + 1, host_start + len(host_wave) - round(.35*SAMPLE_RATE))
                timing["host_turn_id"] = host_id
            else:
                start = cursor + round(t["gap_before_ms"]*SAMPLE_RATE/1000)
            listeners = 0
            if kind == "interruption" and "backchannel_prefix" in t:
                prefix = t["backchannel_prefix"]
                reply = content[len(prefix):].lstrip(" ，,。；;、：:…")
                offset_db = round(backchannel_gain.uniform(-10.0, -3.0), 1)
                lead = synth.line(prefix, voice, offset_db)
                takeover = host_start + len(host_wave)
                start = takeover - len(lead)
                if start <= host_start:
                    raise ValueError("主轮过短，无法在打断前放置附和")
                wave = np.concatenate((lead, synth.line(reply, voice)))
                timing.update(backchannel_prefix=prefix, takeover_sec=takeover / SAMPLE_RATE)
            elif kind == "incomplete":
                head = synth.line(t["parts"][0], voice)
                pause = round(t["pause_ms"] * SAMPLE_RATE / 1000)
                if t.get("resolution") == "assistant_handoff":
                    wave = head
                    hold_frames = pause
                    timing["resolution"] = "assistant_handoff"
                else:
                    tail = synth.line(t["parts"][1], voice)
                    wave = np.concatenate((head, np.zeros(pause, dtype=np.float32), tail))
                timing.update(pause_start_sec=(start+len(head))/SAMPLE_RATE,
                              pause_end_sec=(start+len(head)+pause)/SAMPLE_RATE)
            elif upcoming.get("kind") == "backchannel":
                anchor = upcoming["anchor_text"]
                head = synth.line(anchor, voice)
                tail = synth.line(content[len(anchor):].lstrip("，。；、 "), voice)
                boundary = start + len(head)
                wave = np.concatenate((head, np.zeros(round(.08*SAMPLE_RATE),dtype=np.float32), tail))
            else:
                spoken=content
                if upcoming.get("kind")=="interruption" and not content.endswith(("——","…")):
                    spoken=content.rstrip("。！？")+"——"
                wave=synth.line(spoken, voice, -2 if kind=="side_talk" else 0)
            host_end = start + len(wave)
        end = start + len(wave)
        if kind == "interruption":
            t["gap_before_ms"] = round((start - (placed[-1][1] + len(placed[-1][2]))) * 1000 / SAMPLE_RATE)
        if any(s==speaker and start<at+len(a) and at<end for s,at,a in placed):
            raise ValueError(f"同一说话人重叠，轮次 {i}")
        cursor = max(cursor, end + hold_frames)
        placed.append((speaker,start,wave))
        timeline.append({k:t[k] for k in ("turn_id","speaker","kind","addressee","text")} |
                        {"start_sec":start/SAMPLE_RATE,"end_sec":end/SAMPLE_RATE,
                         "overlaps_prev":bool(i and start < timeline[-1]["end_sec"]*SAMPLE_RATE)} | timing)
    stereo = np.column_stack((_track(placed,cursor,lambda x:x!="assistant"),
                              _track(placed,cursor,lambda x:x=="assistant")))
    peak = float(np.max(np.abs(stereo)))
    if not np.isfinite(peak) or peak <= 0:
        raise ValueError("音频无效")
    sf.write(clean, stereo*min(1,.95/peak), SAMPLE_RATE, subtype="PCM_16")
    atomic_json(timeline_path, timeline)
    return cursor, sum(t.get("pause_ms",0) for t in turns)
