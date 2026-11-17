"""Synthesize MOSS speech and place turns on the stereo timeline."""
import os
import random

import numpy as np
import soundfile as sf

from scripts.sound.synthesis.dialogue_to_wav import AudioQualityError, DialogueSynth, SAMPLE_RATE, _track

from .storage import atomic_json


class FastSynth(DialogueSynth):
    def __init__(self, folder):
        super().__init__(folder)
        self.short_cache = {}

    def line(self, content, voice, offset=0):
        key = (content, voice, offset)
        if key in self.short_cache:
            return self.short_cache[key]
        budget = max(5., len(content)*.40+2.)
        try:
            wave = self.turn_to_mono(content, voice, target_offset_db=offset, synthesis_budget=budget)
        except AudioQualityError:
            previous = os.environ.get("MOSS_SEED", "1234")
            os.environ["MOSS_SEED"] = "20261115"
            try:
                wave = self.turn_to_mono(content, voice, target_offset_db=offset, synthesis_budget=budget * 1.35)
            except AudioQualityError as exc:
                if str(exc) != "generation_budget_exhausted":
                    raise AudioQualityError(f"{exc} (voice={voice}, chars={len(content)})") from exc
                os.environ["MOSS_SEED"] = "20261116"
                try:
                    wave = self.turn_to_mono(content, voice, target_offset_db=offset, synthesis_budget=budget * 1.35)
                except AudioQualityError as final_exc:
                    raise AudioQualityError(f"{final_exc} (voice={voice}, chars={len(content)})") from final_exc
            finally:
                os.environ["MOSS_SEED"] = previous
        if len(content) <= 12 and len(self.short_cache) < 512:
            self.short_cache[key] = wave
        return wave


def render(d, synth, clean, timeline_path):
    turns = d["turns"]
    voices = d["profile"]["voice_assignments"]
    english = d["profile"].get("language") == "en"
    backchannel_gain = random.Random(f"backchannel-gain:{d.get('dialogue_id', 'preview')}")
    placed, timeline = [], []
    cursor = host_end = boundary = listeners = 0
    for i, t in enumerate(turns):
        speaker, content, kind = t["speaker"], t["text"], t["kind"]
        voice = voices[speaker]
        bindings = d["profile"].get("voice_bindings")
        if bindings is not None:
            from scripts.text.voices import voice_bindings_snapshot
            if bindings.get(voice) != voice_bindings_snapshot([voice])[voice]:
                raise ValueError(f"Voice binding changed while rendering: {voice}")
        timing = {}
        hold_frames = 0
        if kind == "backchannel":
            offset_db = round(backchannel_gain.uniform(-7.0, -3.0), 1)
            try:
                wave = synth.line(content, voice, offset_db)
            except AudioQualityError:
                if english:
                    raise
                if len(content.rstrip("。！？.!?")) > 3:
                    raise
                content = "嗯，我听到了。" if content.startswith("嗯") else "嗯，" + content
                wave = synth.line(content, voice, offset_db)
                t["text"] = content
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
            elif t.get("overlap_with") is not None:
                other = t["overlap_with"]
                if type(other) is not int or other != i - 1 or other < 0:
                    raise ValueError("同时请求必须引用上一位说话人")
                _, other_start, other_wave = placed[other]
                if len(other_wave) < 2:
                    raise ValueError("同时请求的上一段音频太短")
                start = other_start + min(round(.2 * SAMPLE_RATE), len(other_wave) // 2)
                timing["overlap_with"] = other
            elif kind == "side_talk" and t.get("side_talk_entry") == "unsolicited":
                _, previous_start, previous_wave = placed[-1]
                start = max(previous_start+1, previous_start+len(previous_wave)-round(.18*SAMPLE_RATE))
                timing["side_talk_entry"] = "unsolicited"
            else:
                start = cursor + round(t["gap_before_ms"]*SAMPLE_RATE/1000)
            listeners = 0
            if kind == "interruption" and "backchannel_prefix" in t:
                prefix = t["backchannel_prefix"]
                reply = content[len(prefix):].lstrip(" ，,。；;、：:…")
                offset_db = round(backchannel_gain.uniform(-7.0, -3.0), 1)
                lead = synth.line(prefix, voice, offset_db)
                # Start the pivot while the host is still audible.
                overlap = min(round(.28 * SAMPLE_RATE), max(0, len(host_wave) - len(lead) - 2))
                takeover = host_start + len(host_wave) - overlap
                available = takeover - host_start - len(lead) - 1
                if available < 0:
                    raise ValueError("主轮过短，无法在打断前放置附和")
                gap_ms = t.get("backchannel_gap_ms")
                if gap_ms is None:
                    gap_rng = random.Random(f"{d.get('dialogue_id', 'preview')}:{i}:transition-gap")
                    gap_ms = gap_rng.randint(250, 900) if gap_rng.random() < .65 else 0
                if type(gap_ms) is not int or gap_ms < 0:
                    raise ValueError("backchannel_gap_ms 须为非负整数")
                gap = min(available, round(gap_ms * SAMPLE_RATE / 1000))
                start = takeover - len(lead) - gap
                wave = np.concatenate((lead, np.zeros(gap, dtype=np.float32), synth.line(reply, voice)))
                timing.update(backchannel_prefix=prefix, takeover_sec=takeover / SAMPLE_RATE,
                              backchannel_end_sec=(takeover - gap) / SAMPLE_RATE,
                              backchannel_gap_ms=round(gap * 1000 / SAMPLE_RATE))
            elif kind == "incomplete":
                first_part = t["parts"][0].rstrip()
                # The timeline inserts the pause. Trailing ellipses can exhaust MOSS's frame budget.
                head = synth.line(first_part.rstrip("，,；;：:、…—.") or first_part, voice)
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
                tail_text = content[len(anchor):].lstrip("，。；、,; ")
                if tail_text.startswith(". "):
                    tail_text = tail_text[2:].lstrip()
                tail = synth.line(tail_text, voice)
                boundary = start + len(head)
                wave = np.concatenate((head, np.zeros(round(.08*SAMPLE_RATE),dtype=np.float32), tail))
            else:
                spoken = content
                if upcoming.get("kind") == "interruption":
                    # The timeline supplies the interruption. Trailing dashes can exhaust MOSS's frame budget.
                    spoken = content.rstrip("。！？—…") or content
                wave=synth.line(spoken, voice, -2 if kind=="side_talk" else 0)
            host_end = start + len(wave)
        end = start + len(wave)
        if kind == "interruption" or timing.get("side_talk_entry") == "unsolicited" or "overlap_with" in timing:
            t["gap_before_ms"] = round((start - (placed[-1][1] + len(placed[-1][2]))) * 1000 / SAMPLE_RATE)
        if any(s==speaker and start<at+len(a) and at<end for s,at,a in placed):
            raise ValueError(f"同一说话人重叠，轮次 {i}")
        cursor = max(cursor, end + hold_frames)
        placed.append((speaker,start,wave))
        if t.get("interaction"):
            timing["interaction"] = t["interaction"]
        for key in ("choice_options", "choice_index", "choice_outcome"):
            if t.get(key) is not None:
                timing[key] = t[key]
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
