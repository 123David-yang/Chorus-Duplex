"""外部背景片段拼接（clips.py）——把短片段（如 VOiCES babble 5s/个）拼成一整条背景轨。

比"单片段循环"自然：每条对话按 dialogue_id 稳定随机选片段序列，50ms 交叉淡化消除接缝。
配套 mix.py 使用（背景只混入非 assistant 侧）。
"""
from __future__ import annotations

import argparse
import random
import sys
import zlib
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 48000
_XFADE_MS = 50
CLIP_RMS_TARGET = 0.05  # 每个片段统一到该 RMS（拼接后池内音量一致；最终整体归一化留运行余量）


def stitch_clips(clip_dir: str | Path, out_path: str | Path, duration: float,
                 seed_key: str = "clips", glob: str = "*.wav",
                 sample_rate: int = SAMPLE_RATE) -> Path:
    """从 clip_dir 随机抽片段拼到指定时长（交叉淡化），确定性由 seed_key 决定。"""
    clips = sorted(Path(clip_dir).glob(glob))
    if not clips:
        raise FileNotFoundError(f"目录中无片段: {clip_dir}/{glob}")
    rng = random.Random(zlib.crc32(f"clips-{seed_key}".encode()))
    need = int(duration * sample_rate)
    xfade = int(_XFADE_MS / 1000 * sample_rate)
    buf = np.zeros(0, dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, xfade, dtype=np.float32)

    while len(buf) < need:
        c, sr = sf.read(str(rng.choice(clips)), dtype="float32")
        if c.ndim > 1:
            c = c.mean(axis=1)
        if sr != sample_rate:  # 统一到目标采样率（如 VOiCES 16k → 48k）
            n = int(len(c) * sample_rate / sr)
            c = np.interp(np.linspace(0, len(c) - 1, n), np.arange(len(c)), c).astype(np.float32)
        # 逐片段响度规整（不同拾音位置本征音量差可达十几 dB：隔墙 vs 桌面）
        r = float(np.sqrt((c ** 2).mean()))
        if r > 1e-6:
            c = c * (CLIP_RMS_TARGET / r)
        pk = float(np.abs(c).max())
        if pk > 0.95:  # 提升后防削波
            c = c * (0.95 / pk)
        if len(buf) == 0:
            buf = c
            continue
        n = min(xfade, len(c), len(buf))
        if n > 1:  # 交叉淡化接缝
            buf[-n:] = buf[-n:] * (1 - ramp[:n]) + c[:n] * ramp[:n]
            buf = np.concatenate([buf, c[n:]])
        else:
            buf = np.concatenate([buf, c])

    buf = buf[:need]
    peak = float(np.abs(buf).max())
    if peak > 0:
        buf = buf / peak * 0.9  # 留余量，细节能量交给 mix 的 SNR 缩放
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), buf, sample_rate)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="短片段 → 长背景轨（交叉淡化拼接）")
    ap.add_argument("--clip-dir", required=True, help="片段目录（如 VOiCES babb/）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration", type=float, default=150.0)
    ap.add_argument("--seed-key", default="clips")
    args = ap.parse_args()
    try:
        out = stitch_clips(args.clip_dir, args.out, args.duration, args.seed_key)
        print(f"✅ 背景轨完成 → {out}")
    except Exception as e:
        print(f"❌ 失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
