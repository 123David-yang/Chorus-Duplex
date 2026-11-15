"""对话 jsonl → 语音（MVP）：支持 双声道 / 多 speaker 独立音轨。

- 双声道模式：assistant → 左声道，user(s) → 右声道（快速试听）
- 多轨模式：每个 speaker 一个独立 wav（各自音色 + 独立时间线），可导入 DAW
- 内置 18 个音色（Junhao/Xiaoyu/Lingyu/...），不同 speaker 可用不同音色

流程：MOSS-TTS-Nano（进程内单例，模型只加载一次）逐句合成 mono 48kHz
→ 按 turn 时序、按 json 顺序严格串行拼接（等长静音对齐，无重叠）。

用法：
    # 双声道
    python -m scripts.sound.synthesis.dialogue_to_wav --input data/annotations/x.jsonl --index 0 --out output/d.wav
    # 多轨（每 speaker 独立 wav + 音色映射）
    python -m scripts.sound.synthesis.dialogue_to_wav --input x.jsonl --mode tracks --out-dir output/tracks \
        --voices "assistant=Junhao,user_01=Xiaoyu,user_02=Lingyu"
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.text.voices import fallback_voice, voices_for_dialogue  # noqa: E402

from scripts.config import MOSS_DIR, OUTPUT_DIR
from scripts.text.utils.schema import DEFAULT_GAP_MS, Dialogue, load_dialogue_lines

SAMPLE_RATE = 48000
MAX_TURN_PAUSE_SEC = 0.4   # 轮内停顿上限：超过则压缩（模型偶发自产 1~3s 长静音）
# A1 生成长度预算：防模型收尾失败时一路吐静音到默认上限（30s）
# 实测语速 p90≈0.34 s/字、非失控 max≈0.40 → 0.45 留余量；+1.5s 收尾缓冲
GEN_SEC_PER_CHAR = 0.45
GEN_PAD_SEC = 1.5
# 响度规整：每轮 RMS 归一到目标（±小抖动防呆板），带增益上限 + 峰值保护
TARGET_RMS_DB = -23.0   # 目标响度（RMS dBFS），接近自然口语水平
RMS_JITTER_DB = 1.5     # 每轮允许的随机抖动（±dB），避免所有轮响度完全一致
GAIN_CLAMP_DB = 12.0    # 单轮最大增益（防把近静音放大成噪声）
PEAK_CEIL_DB = -1.0     # 归一后峰值上限（防削波）
BACKCHANNEL_OFFSET_DB = -3.0   # 附和比被附和的主轮略轻（表"在听"而非抢话）
THIRD_PARTY_OFFSET_DB = -3.0   # 第三方（非主角团：外卖员/电话里/邻居）略轻——距离感/电话感，
                               # 也与"环境中非响应人群"的定位一致（用户 2026-09-17 要求）
FADE_MS = 8                    # 轮首尾淡入淡出时长（防硬切咔哒）


def _apply_fade(audio: np.ndarray, ms: float = FADE_MS) -> np.ndarray:
    n = int(SAMPLE_RATE * ms / 1000)
    if n <= 0 or len(audio) < 2 * n:
        return audio
    out = audio.copy()
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    out[:n] *= ramp
    out[-n:] *= ramp[::-1]
    return out


def normalize_loudness(audio: np.ndarray, target_db: float = TARGET_RMS_DB,
                       jitter_db: float = 0.0) -> np.ndarray:
    """按 RMS 把一轮音频归一到目标响度（+小抖动），超峰则整体回缩。"""
    if audio.size == 0:
        return audio
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    if rms < 1e-5:  # 近静音：不动
        return audio
    target = 10 ** ((target_db + jitter_db) / 20.0)
    gain = min(target / rms, 10 ** (GAIN_CLAMP_DB / 20.0))
    out = audio * gain
    peak = float(np.max(np.abs(out)))
    ceil = 10 ** (PEAK_CEIL_DB / 20.0)
    if peak > ceil:
        out = out * (ceil / peak)
    return out.astype(np.float32)


class AudioQualityError(ValueError):
    """A generated waveform is not suitable for automatic inclusion in a dataset."""


class DialogueSynth:
    """MOSS-TTS 进程内单例（模型只加载一次，CLAUDE.md 规范）。"""

    def __init__(self, tmp_dir: str | Path | None = None) -> None:
        from scripts.sound.synthesis.backends import create_backend
        from scripts.sound.synthesis.vad import SileroVAD
        d = Path(tmp_dir) if tmp_dir else OUTPUT_DIR
        d.mkdir(parents=True, exist_ok=True)
        self.tmp = d / f"_tmp_turn_{os.getpid()}.wav"
        self.backend = create_backend(self.tmp)
        self.runtime = getattr(self.backend, "runtime", None)  # compatibility for diagnostics
        self.vad = SileroVAD()
        self.frames_per_sec = getattr(self.backend, "fps", 12.5)
        self.max_frames_default = getattr(self.backend, "max_frames", 10000)
        # 轮首尾淡入淡出开关（2026-09-15 用户要求：默认开启；TTS_FADE=0 fallback 硬切，便于 A/B 对比）
        self.fade = os.environ.get("TTS_FADE", "1") != "0"

    def _codec_fps(self) -> float:
        """编解码帧率（帧/秒）——max_new_frames = 秒数 × 该值。"""
        try:
            cc = self.runtime.codec_meta["codec_config"]
            return int(cc["sample_rate"]) / int(cc["downsample_rate"])
        except Exception:
            return 12.5

    def max_frames_for(self, text: str) -> int:
        """A1：按字数限制单次生成长度（不超过模型默认上限）。"""
        budget = len(text) * GEN_SEC_PER_CHAR + GEN_PAD_SEC
        return max(1, min(self.max_frames_default, round(budget * self.frames_per_sec)))

    def turn_to_mono(self, text: str, voice: str = "Junhao", jitter_db: float = 0.0,
                     target_offset_db: float = 0.0, synthesis_budget: float | None = None) -> np.ndarray:
        from scripts.text.voices import reference_for_voice
        started = time.perf_counter()
        self.last_synthesis_metrics = {"status":"started", "voice":voice, "text_chars":len(text)}
        result = self.backend.synthesize(text, voice, reference_for_voice(voice),
                                         len(text)*GEN_SEC_PER_CHAR+GEN_PAD_SEC if synthesis_budget is None else synthesis_budget)
        if result.truncated:
            print(f"    ⚠ TTS generation reached its frame budget ({len(text)} chars)")
        d = np.asarray(result.waveform, dtype=np.float32)
        sr = int(result.sample_rate)
        if d.ndim > 1: d = d.mean(axis=1)
        if sr != SAMPLE_RATE:
            from scipy.signal import resample_poly
            from math import gcd
            common = gcd(sr, SAMPLE_RATE)
            d = resample_poly(d, SAMPLE_RATE//common, sr//common).astype(np.float32)
            sr = SAMPLE_RATE
        # VAD 裁剪首尾静音 + 压掉轮内超长停顿（模型自产长静音，见 devlog 04）
        trimmed = self.vad.compact(d, sr, max_gap_sec=MAX_TURN_PAUSE_SEC, pad_sec=0.06)
        # 响度规整：各轮音量拉到相近范围（±抖动），避免忽大忽小
        out = normalize_loudness(trimmed, target_db=TARGET_RMS_DB + target_offset_db,
                                 jitter_db=jitter_db)
        final = _apply_fade(out) if self.fade else out
        flags=[]
        if result.truncated: flags.append("generation_budget_exhausted")
        if not len(final) or not np.all(np.isfinite(final)): flags.append("invalid_waveform")
        elif float(np.sqrt(np.mean(final.astype(np.float64)**2))) < 1e-5: flags.append("near_silent_waveform")
        self.last_synthesis_metrics = {
            "status":"quarantine" if flags else "ok", "quality_flags":flags, "voice":voice, "text_chars":len(text),
            "truncated":bool(result.truncated), "raw_duration_s":round(len(d)/sr,6),
            "duration_s":round(len(final)/SAMPLE_RATE,6),
            "elapsed_s":round(time.perf_counter()-started,4),
            "native_sample_rate":int(result.sample_rate), "output_sample_rate":SAMPLE_RATE,
            "backend_provenance":getattr(result,"provenance",None),
        }
        if flags and os.environ.get("TTS_STRICT","1") != "0":
            raise AudioQualityError(";".join(flags))
        return final


def load_dialogues(path: str | Path) -> list:
    """读取对话文件（JSONL / 美化排版单 JSON 均可）。"""
    return [Dialogue.from_jsonl_line(line) for line in load_dialogue_lines(path)]


def _all_speakers(d: Dialogue) -> list[str]:
    """对话中出现的全部 speaker（assistant + 有发言的正式成员 + 第三方）。"""
    spoken = {t.speaker for t in d.turns}
    members = ["assistant"] + [p.id for p in d.participants if p.id in spoken]
    return members + sorted(s for s in spoken if s not in members)


def _tts_text_for_turn(d: Dialogue, index: int) -> str:
    """Mark a planned unfinished floor with punctuation, never add spoken words."""
    turn=d.turns[index]
    if turn.kind=="backchannel": return turn.text
    following=next((t for t in d.turns[index+1:] if t.kind!="backchannel"),None)
    if (following is not None and following.kind=="interruption" and following.speaker!=turn.speaker
            and turn.text and turn.text[-1] not in "。！？!?.—…"):
        return turn.text+"——"
    return turn.text


def _synthesize_with_anchors(d, index, synth, voice, jitter, offset):
    turn=d.turns[index]
    anchors=[]
    for bc in d.turns[index+1:]:
        if bc.kind!="backchannel":break
        if bc.anchor_text:anchors.append(bc.anchor_text)
    text=_tts_text_for_turn(d,index)
    if not anchors:
        return synth.turn_to_mono(text,voice,jitter_db=jitter,target_offset_db=offset),{}
    anchors=sorted(set(anchors),key=len)
    if any(not turn.text.startswith(a) or not 0<len(a)<len(turn.text) for a in anchors):
        raise ValueError("Semantic anchor is not an exact proper prefix")
    whole=getattr(synth,"synthesize_with_anchor_offsets",None)
    if whole is not None:
        aligned=whole(text,voice,anchors,jitter,offset)
        if aligned is not None:return aligned
    pieces=[];boundaries={};metrics=[];cursor=0;previous=0
    for end in [len(a) for a in anchors]+[len(text)]:
        piece=text[previous:end]
        audio=synth.turn_to_mono(piece,voice,jitter_db=jitter,target_offset_db=offset)
        metrics.append(dict(getattr(synth,"last_synthesis_metrics",{})))
        pieces.append(audio);cursor+=len(audio)
        for anchor in anchors:
            if len(anchor)==end:boundaries[anchor]=cursor
        if end<len(text):
            pause=np.zeros(round(.08*SAMPLE_RATE),dtype=np.float32)
            pieces.append(pause);cursor+=len(pause)
        previous=end
    combined=np.concatenate(pieces)
    synth.last_synthesis_metrics={**(metrics[-1] if metrics else {}),"duration_s":len(combined)/SAMPLE_RATE,
        "elapsed_s":sum(m.get("elapsed_s",0) for m in metrics),"text_chars":len(text),
        "truncated":any(m.get("truncated",False) for m in metrics),"segments":metrics,
        "quality_flags":[flag for m in metrics for flag in m.get("quality_flags",[])],"semantic_split":True,"whole_anchor_fallback":getattr(synth,"last_anchor_attempt",None)}
    return combined,boundaries


def _layout(d: Dialogue, synth: "DialogueSynth",
            voices: dict | None) -> tuple[list[tuple[str, int, np.ndarray]], int]:
    """合成全部话轮并按标注放置到统一时间线，返回 [(speaker, start, audio)] 与总长。

    位置规则（与 schema 对应）：
    - speech/interruption：start = 上一轮结束 + gap_before_ms（负值 = 提前开口，重叠）
    - backchannel：start = 上一轮起点 + insert_ratio × 上一轮时长 − 150ms（压住词尾）
    """
    from scripts.text.quality import audit
    check = audit(d, required=[])
    if not check["passed"]:
        raise ValueError("Invalid dialogue annotations; run quality audit before synthesis: " + str(check["issues"][:3]))
    if voices is None:
        # 按 dialogue_id 确定性分配音色（与提示词层的音色表共用同一函数）：
        # 对话内一致、跨对话错开 → 主角团音色多样性；assistant 固定 Junhao
        n_users = sum(1 for p in d.participants if p.id.startswith("user"))
        declared_backend=d.profile.get("tts_backend")
        selected_backend=os.environ.get("TTS_BACKEND","moss_onnx")
        if declared_backend and declared_backend!=selected_backend:
            raise ValueError("TTS backend differs from the script voice contract; regenerate or explicitly remap")
        voices = d.profile.get("voice_assignments") or voices_for_dialogue(d.dialogue_id, n_users)
    placed: list[tuple[str, int, np.ndarray]] = []
    cursor = 0        # 上一轮结束采样点
    prev_start = 0    # 上一轮起点（backchannel 定位用）
    prev_len = 0      # 上一轮时长（采样点）
    rng = random.Random(zlib.crc32(d.dialogue_id.encode()))  # 响度抖动：同对话可复现
    prev_main_jitter = 0.0
    prev_anchor_offsets = {}
    prev_main_index = None
    synth.last_layout_timing = []
    for i, t in enumerate(d.turns):
        t0 = time.time()
        kind = getattr(t, "kind", "speech")
        jitter = rng.uniform(-RMS_JITTER_DB, RMS_JITTER_DB)
        offset = 0.0
        if kind == "backchannel":
            # 附和比所附和的主轮略轻（不抢话、但听得清）；基准沿用主轮抖动 → 相对关系稳定
            jitter, offset = prev_main_jitter, BACKCHANNEL_OFFSET_DB
        else:
            prev_main_jitter = jitter
            if t.speaker.startswith("third_party"):
                offset = THIRD_PARTY_OFFSET_DB
        # 字典外 speaker 兜底走用户池音色，绝不占用 assistant 的 Junhao（见 voices.fallback_voice）
        voice=voices.get(t.speaker) or fallback_voice(t.speaker)
        timing={"turn_id":i,"placement_method":"gap"}
        if kind=="backchannel":
            audio=synth.turn_to_mono(t.text,voice,jitter_db=jitter,target_offset_db=offset)
        else:
            audio,prev_anchor_offsets=_synthesize_with_anchors(d,i,synth,voice,jitter,offset)
        if kind == "backchannel" and placed:
            # 附和：插入到上一主轮内部；不占话轮 → 不推进时间线游标
            timing["host_turn_id"]=prev_main_index
            if t.anchor_text:
                if t.anchor_text not in prev_anchor_offsets:raise ValueError("Semantic anchor was not synthesized")
                boundary=prev_start+prev_anchor_offsets[t.anchor_text]
                start=boundary+round(.05*SAMPLE_RATE)
                timing.update(placement_method="semantic_anchor",anchor_text=t.anchor_text,
                              anchor_boundary_sec=boundary/SAMPLE_RATE,listener_latency_ms=50)
                if start>=prev_start+prev_len:raise ValueError("Semantic anchor leaves no overlapping host tail")
                mark=" semantic-anchor"
            else:
                ratio = .6 if t.insert_ratio is None else min(max(float(t.insert_ratio),.2),.9)
                start = max(0, prev_start + int(ratio * prev_len) - int(.15*SAMPLE_RATE))
                timing["placement_method"]="legacy_ratio"
                mark=f" ratio={ratio:.2f}"
        else:
            gap_ms = int(getattr(t, "gap_before_ms", DEFAULT_GAP_MS) or 0)
            start = max(0, cursor + int(gap_ms / 1000 * SAMPLE_RATE))
            mark = f" ↯{abs(gap_ms)}ms重叠" if gap_ms < 0 else ""
            cursor = start + len(audio)          # 仅主轮推进游标
            prev_start, prev_len = start, len(audio)
            prev_main_index = i
        addr = t.addressee if isinstance(t.addressee, str) else "+".join(t.addressee)
        print(f"  [{i}] @{start / SAMPLE_RATE:6.2f}s {t.speaker}({voices.get(t.speaker, '?')})"
              f"→{addr} {kind}{mark}: {t.text[:24]}... ({time.time() - t0:.0f}s)")
        for previous_speaker, previous_start, previous_audio in placed:
            if previous_speaker == t.speaker and start < previous_start + len(previous_audio) and previous_start < start + len(audio):
                raise ValueError(f"Acoustic self-overlap: speaker={t.speaker}, turn={i}; regenerate timing")
        timing.update(start_sec=start/SAMPLE_RATE,end_sec=(start+len(audio))/SAMPLE_RATE)
        synth.last_layout_timing.append(timing)
        callback=getattr(synth,"on_turn_complete",None)
        if callback is not None:callback(i,t,audio,dict(getattr(synth,"last_synthesis_metrics",{})),timing)
        placed.append((t.speaker, start, audio))
    total = max((s + len(a) for _, s, a in placed), default=0)
    return placed, total


def _track(placed: list[tuple[str, int, np.ndarray]], total: int,
           keep=lambda spk: True) -> np.ndarray:
    """把放置好的片段叠加到单条轨（只保留 keep(speaker) 的片段）。"""
    buf = np.zeros(total, dtype=np.float32)
    for spk, s, a in placed:
        if keep(spk):
            buf[s:s + len(a)] += a
    return buf


def dialogue_to_stereo(d, synth: DialogueSynth, out_path: str | Path,
                       voices: dict | None = None,
                       timeline_path: str | Path | None = None) -> tuple[Path, float]:
    """双声道分轨：左 = 非 assistant（用户/第三方），右 = assistant；按标注时序放置（支持重叠）。

    timeline_path 非空时，额外写出实现时间线 JSON（每轮实际起止秒）。
    """
    placed, total = _layout(d, synth, voices)
    left = _track(placed, total, lambda s: s != "assistant")
    right = _track(placed, total, lambda s: s == "assistant")
    out_path = Path(out_path).absolute()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stereo=np.stack([left,right],axis=1)
    peak=float(np.max(np.abs(stereo))) if stereo.size else 0.
    gain=min(1.,(10**(PEAK_CEIL_DB/20))/peak) if peak else 1.
    synth.last_mix_metrics={"pre_gain_peak":peak,"mix_gain":gain}
    sf.write(str(out_path),stereo*gain,SAMPLE_RATE,subtype="PCM_16")
    if timeline_path:
        _write_timeline(d, placed, timeline_path, getattr(synth,"last_layout_timing",None))
    return out_path, total / SAMPLE_RATE


def dialogue_to_tracks(d, synth: DialogueSynth, out_dir: str | Path,
                       voices: dict | None = None) -> dict[str, Path]:
    """多轨：每个 speaker 独立 wav（同一时间线，各自只含自己的片段）。"""
    placed, total = _layout(d, synth, voices)
    out_dir = Path(out_dir).absolute()
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    for s in _all_speakers(d):
        p = out_dir / f"{s}.wav"
        sf.write(str(p), _track(placed, total, lambda spk, s=s: spk == s), SAMPLE_RATE)
        result[s] = p
    return result


def _write_timeline(d, placed: list[tuple[str, int, np.ndarray]], path: str | Path, timing: list | None = None) -> None:
    """回写实现时间线（每轮实际起止秒）——把意图（gap/ratio）落成绝对时间。"""
    rows = []
    max_end = 0  # 此前所有轮的实现结束时刻（任一轨）
    for i, ((spk, start, audio), t) in enumerate(zip(placed, d.turns)):
        rows.append({
            **(timing[i] if timing else {}),
            "turn_id": i, "speaker": spk, "kind": getattr(t, "kind", "speech"),
            "addressee": t.addressee, "text": t.text,
            "tts_text": _tts_text_for_turn(d, i),
            "start_sec": round(start / SAMPLE_RATE, 3),
            "end_sec": round((start + len(audio)) / SAMPLE_RATE, 3),
            "overlaps_prev": start < max_end,
        })
        max_end = max(max_end, start + len(audio))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def dialogue_to_mix(d, synth: DialogueSynth, out_path: str | Path,
                    voices: dict | None = None,
                    timeline_path: str | Path | None = None) -> tuple[Path, float]:
    """混音：全部 speaker 到单个 wav，按标注时序放置（负 gap 自然重叠）。

    timeline_path 非空时，额外写出实现时间线 JSON（每轮实际起止秒）。
    """
    placed, total = _layout(d, synth, voices)
    data = _track(placed, total)
    out_path = Path(out_path).absolute()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), data, SAMPLE_RATE)
    if timeline_path:
        _write_timeline(d, placed, timeline_path, getattr(synth,"last_layout_timing",None))
    return out_path, len(data) / SAMPLE_RATE


def dialogue_to_multichannel(d, synth: DialogueSynth, out_path: str | Path,
                             voices: dict | None = None) -> tuple[Path, float]:
    """多声道单文件：一个 wav，每个 speaker 一个声道（同一时间线）。"""
    placed, total = _layout(d, synth, voices)
    speakers = _all_speakers(d)
    cols = [_track(placed, total, lambda spk, s=s: spk == s) for s in speakers]
    data = np.stack(cols, axis=1)  # (N, n_speakers)
    out_path = Path(out_path).absolute()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), data, SAMPLE_RATE)
    return out_path, len(data) / SAMPLE_RATE


def _parse_voices(s: str | None) -> dict | None:
    if not s:
        return None
    out = {}
    for item in s.split(","):
        k, _, v = item.strip().partition("=")
        if k and v:
            out[k.strip()] = v.strip()
    return out or None


def main() -> None:
    ap = argparse.ArgumentParser(description="对话 jsonl → 语音合成（双声道 / 多轨）")
    ap.add_argument("--input", required=True)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--mode", choices=["stereo", "tracks", "mix", "multichannel"], default="stereo")
    ap.add_argument("--out", default="output/dialogue_stereo.wav")
    ap.add_argument("--out-dir", default="output/tracks")
    ap.add_argument("--voices", default=None, help='如 "assistant=Junhao,user_01=Xiaoyu"')
    args = ap.parse_args()

    try:
        dialogues = load_dialogues(args.input)
        if not dialogues:
            print("❌ 空数据"); sys.exit(1)
        d = dialogues[args.index]
        speakers = _all_speakers(d)
        print(f"对话 {args.index}: {d.dialogue_id} | {d.scenario} | {len(d.turns)} 轮 | {len(speakers)} 个 speaker: {speakers}")
        print("加载 MOSS-TTS 模型（首次约 15-30s）...")
        tmp_dir = Path(args.out_dir if args.mode == "tracks" else args.out).parent
        synth = DialogueSynth(tmp_dir=tmp_dir)
        voices = _parse_voices(args.voices)
        if args.mode == "stereo":
            out, dur = dialogue_to_stereo(d, synth, args.out, voices)
            print(f"✅ → {out} | {dur:.1f}s | stereo（左=assistant，右=user）")
        elif args.mode == "mix":
            out, dur = dialogue_to_mix(d, synth, args.out, voices)
            print(f"✅ → {out} | {dur:.1f}s | 单文件混音（全部 speaker，不同音色）")
        elif args.mode == "multichannel":
            out, dur = dialogue_to_multichannel(d, synth, args.out, voices)
            print(f"✅ → {out} | {dur:.1f}s | 单文件多声道（每 speaker 一个声道）")
        else:
            result = dialogue_to_tracks(d, synth, args.out_dir, voices)
            for s, p in result.items():
                print(f"  🎵 {s}: {p}")
            print(f"✅ 多轨完成 → {args.out_dir}")
    except Exception as e:
        print(f"❌ 失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
