"""VAD 语音活动检测工具（vad.py）—— 基于 silero-vad（ONNX）。

用途：
1. 检测每句 TTS 合成的 speech 区间（标注具体发音位置）
2. 裁剪前后静音 → 拼接更符合人类对话停顿时长（话轮间 ~200-400ms）

数据流转：
    wav → 重采样 16k → 512样本帧逐帧推理 → 语音概率
    → 连续语音帧合并为区间 → 输出区间(秒) + 裁剪音频

用法：
    python -m scripts.sound.synthesis.vad --wav output/xxx.wav
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

MODEL_PATH = Path(__file__).parents[1] / "models" / "silero_vad.onnx"
SR_TARGET = 16000
FRAME = 512  # silero-vad 帧长（16k 下 32ms）
SPLICE_FADE_MS = 8  # 轮内长停顿拼接缝的淡出/淡入时长（与轮首尾 fade 一致，防咔哒）


def _quietest_cut(audio: np.ndarray, lo: int, hi: int) -> int:
    """在 [lo, hi) 内找振幅最低的采样点（作为剪切点，避开残余声音）。"""
    lo, hi = max(0, lo), min(len(audio), hi)
    if hi - lo <= 1:
        return max(0, lo)
    return lo + int(np.argmin(np.abs(audio[lo:hi])))


class SileroVAD:
    def __init__(self, model_path: str | Path | None = None, threshold: float = 0.5,
                 min_speech_frames: int = 3, min_silence_frames: int = 12) -> None:
        """threshold: 语音概率阈值；min_silence_frames: 合并断裂的容错（帧数）。"""
        import os

        model_path = model_path or os.environ.get("VAD_MODEL", MODEL_PATH)
        import onnxruntime as ort  # noqa: PLC0415
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.threshold = threshold
        self.min_speech_frames = min_speech_frames
        self.min_silence_frames = min_silence_frames
        self.input_name = self.session.get_inputs()[0].name
        self.state_name = self.session.get_inputs()[1].name
        self.sr_name = self.session.get_inputs()[2].name
        self.state_out = [o.name for o in self.session.get_outputs() if o.name != "output"]
        self.zero_state = np.zeros((2, 1, 128), dtype=np.float32)

    def resample16k(self, audio: np.ndarray, src_sr: int) -> np.ndarray:
        if src_sr == SR_TARGET:
            return audio
        n = int(len(audio) * SR_TARGET / src_sr)
        return np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)

    def detect(self, audio16k: np.ndarray) -> list[tuple[float, float]]:
        """返回 speech 区间列表 [(start_sec, end_sec), ...]"""
        probs = []
        state = self.zero_state.copy()
        n_frames = len(audio16k) // FRAME
        for i in range(n_frames):
            frame = audio16k[i * FRAME:(i + 1) * FRAME]
            if len(frame) < FRAME:
                break
            frame = frame.reshape(1, -1)
            outs = self.session.run(
                ["output"] + self.state_out,
                {self.input_name: frame, self.state_name: state, self.sr_name: np.array(SR_TARGET, dtype=np.int64)},
            )
            probs.append(float(outs[0][0][0]))
            state = np.stack([outs[1], outs[2]], axis=0) if len(outs) == 3 else outs[1]
            if state.shape[1] == 1 and state.ndim == 3:
                state = state
            elif state.ndim == 2:
                state = state.reshape(2, 1, 128)

        # 帧 → 区间
        speech = [p > self.threshold for p in probs]
        intervals: list[tuple[int, int]] = []
        in_speech = False
        start = 0
        silence_run = 0
        for i, sp in enumerate(speech):
            if sp and not in_speech:
                in_speech = True
                start = i
                silence_run = 0
            elif not sp and in_speech:
                silence_run += 1
                if silence_run >= self.min_silence_frames:  # 静音足够长才断开
                    if i - start >= self.min_speech_frames:
                        intervals.append((start, i - silence_run + 1))
                    in_speech = False
            elif sp:
                silence_run = 0
        if in_speech and n_frames - start >= self.min_speech_frames:
            intervals.append((start, n_frames))

        sec_per_frame = FRAME / SR_TARGET
        return [(s * sec_per_frame, e * sec_per_frame) for s, e in intervals]

    def compact(self, audio: np.ndarray, src_sr: int, max_gap_sec: float = 0.4,
                pad_sec: float = 0.06) -> np.ndarray:
        """裁剪首尾静音，并把轮内超长停顿压到 max_gap_sec（模型偶发自产长静音）。

        长停顿的剪法（用户 2026-09-13 反馈"剪辑突兀"）：
        - 从**中间**剪，前后各留一半静音余量 → 不剪进前一句的语音尾巴/后一句的起音
        - 剪点在标称位置 ±40ms 内取**能量最低**处 → 避免正好剪在残余声音/呼吸上
        - 拼接缝两侧各做 8ms 淡出/淡入（用户 2026-09-17 要求，与轮首尾 fade 同一时长）→ 双保险防咔哒
        ≤max_gap_sec 的自然停顿保留；话轮之间的停顿不归这里管（由 gap_before_ms 控制）。
        """
        a16k = self.resample16k(audio, src_sr)
        intervals = self.detect(a16k)
        if not intervals:
            return audio
        pad = int(pad_sec * src_sr)
        max_gap = int(max_gap_sec * src_sr)
        search = max(1, int(0.04 * src_sr))
        segs: list[np.ndarray] = []
        prev_end = 0
        for idx, (s_sec, e_sec) in enumerate(intervals):
            s = max(0, int(s_sec * src_sr) - pad)
            e = min(len(audio), int(e_sec * src_sr) + pad)
            if idx == 0:
                segs.append(audio[s:e])
            else:
                gap = s - prev_end
                if gap > max_gap:
                    # 中间整段剪掉：前段尾留 keep_l、后段头留 keep_r，剪点取能量最低处
                    keep_l = max_gap // 2
                    keep_r = max_gap - keep_l
                    p1 = _quietest_cut(audio, prev_end + keep_l - search, prev_end + keep_l + search)
                    p2 = _quietest_cut(audio, s - keep_r - search, s - keep_r + search)
                    # 拼接缝淡出/淡入（防咔哒；用户 2026-09-17）
                    nf = max(1, int(SPLICE_FADE_MS / 1000 * src_sr))
                    tail = audio[prev_end:p1].copy()
                    head = audio[p2:e].copy()
                    if len(tail) >= nf:
                        tail[-nf:] *= np.linspace(1.0, 0.0, nf, dtype=np.float32)
                    if len(head) >= nf:
                        head[:nf] *= np.linspace(0.0, 1.0, nf, dtype=np.float32)
                    segs[-1] = np.concatenate([segs[-1], tail])
                    segs.append(head)
                else:
                    segs.append(audio[prev_end:e])
            prev_end = e
        return np.concatenate(segs)

    def trim(self, audio: np.ndarray, src_sr: int, pad_sec: float = 0.05) -> tuple[np.ndarray, list[tuple[float, float]]]:
        """裁剪前后静音（保留 pad_sec 余量），返回 (裁剪后音频, 原始采样率下的区间)。"""
        a16k = self.resample16k(audio, src_sr)
        intervals = self.detect(a16k)
        if not intervals:
            return audio, []
        # 合并为一个整体：从第一个起点到最后一个终点
        start16, end16 = intervals[0][0], intervals[-1][1]
        pad = int(pad_sec * src_sr)
        s = max(0, int(start16 * src_sr) - pad)
        e = min(len(audio), int(end16 * src_sr) + pad)
        return audio[s:e], intervals


def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    import soundfile as sf  # noqa: PLC0415
    d, sr = sf.read(str(path), dtype="float32")
    if d.ndim > 1:
        d = d.mean(axis=1)
    return d.astype(np.float32), sr


def main() -> None:
    ap = argparse.ArgumentParser(description="VAD 语音活动检测（silero-vad onnx）")
    ap.add_argument("--wav", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    try:
        audio, sr = load_wav(args.wav)
        vad = SileroVAD(threshold=args.threshold)
        a16k = vad.resample16k(audio, sr)
        intervals = vad.detect(a16k)
        print(f"音频: {len(audio)/sr:.2f}s @ {sr}Hz | 检测到 {len(intervals)} 个 speech 区间")
        for i, (s, e) in enumerate(intervals):
            print(f"  [{i}] {s:.2f}s - {e:.2f}s（时长 {e-s:.2f}s）")
        if intervals:
            total_speech = sum(e - s for s, e in intervals)
            print(f"语音占比: {total_speech/(len(audio)/sr)*100:.0f}%")
    except Exception as e:
        print(f"❌ 失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
