"""背景混入（mix.py）—— 背景只混入**非 assistant 侧** + SNR 能量调整

数据流转结构：
┌─────────────────────────────────────────────────────────────┐
│ 输入                                                         │
│   dialogue.wav     对话音频（dialogue_to_wav 的 stereo 产物） │
│   background.wav   背景音频（babble.py 生成的 babble / 噪声） │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
① 预处理：保留声道结构（stereo 走 stereo）、统一采样率、背景循环/裁剪到对话等长
                               ▼
② 能量调整（核心）：按目标 SNR 缩放背景音量
     目标SNR = 10·log10(P非assistant侧 / P背景)
     背景缩放系数 k = sqrt(P非assistant侧 / (10^(SNR/10) · P背景))
     → SNR 高 = 背景小声清晰；SNR 低 = 背景嘈杂
                               ▼
③ 淡入淡出防爆音 → 叠加到非 assistant 侧（左声道；assistant 侧保持纯净）
                               ▼
┌──────────────────────────────┬──────────────────────────────┐
│ 输出                                                         │
│   mixed.wav            对话+背景 混合成品（直接试听）            │
│   background_track.wav 背景独立声轨（与对话等长，训练区分用）     │
└─────────────────────────────────────────────────────────────┘

用法：
    python -m scripts.sound.background.mix \
        --dialogue output/dialogue_mix.wav \
        --background output/ambient_babble.wav \
        --out-dir output/mixed --snr 12
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


def _to_mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1) if x.ndim > 1 else x


def _resample(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return x
    n = int(len(x) * dst_sr / src_sr)
    grid = np.linspace(0, len(x) - 1, n)
    if x.ndim > 1:
        return np.stack([np.interp(grid, np.arange(len(x)), x[:, c])
                         for c in range(x.shape[1])], axis=1).astype(np.float32)
    return np.interp(grid, np.arange(len(x)), x).astype(np.float32)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt((x ** 2).mean()))


def mix(dialogue_path: str | Path, background_path: str | Path,
        out_dir: str | Path | None = None, snr_db: float = 12.0,
        sample_rate: int = SAMPLE_RATE,
        out_mixed: str | Path | None = None,
        out_bg: str | Path | None | bool = None,
        seed_key: str | None = None) -> dict[str, Path]:
    """对话 + 背景按目标 SNR 混音，输出混合成品（+ 可选背景独立轨）。

    **背景只混入非 assistant 侧**（2026-09-15 用户定）：双声道产物（左=非assistant，
    右=assistant）叠加到左声道、右声道保持纯净；单声道旧产物（v10 及以前）整轨叠加。
    out_mixed / out_bg 显式指定路径时优先（批量编排用）；否则写入 out_dir 固定名。
    **out_bg=False → 不写背景轨**（省空间；用户 2026-09-15 定批量混音默认不留副本）。
    """
    d, sr_d = sf.read(str(dialogue_path), dtype="float32")
    b, sr_b = sf.read(str(background_path), dtype="float32")
    d = _resample(d, sr_d, sample_rate)          # 保留原声道结构（stereo 走 stereo）
    b = _to_mono(_resample(b, sr_b, sample_rate))

    # 背景对齐对话长度：轨道够长 → 按 seed_key 稳定随机截取一段（避免各对话总用同一开头）；
    # 不够长 → 循环补足
    if seed_key and len(b) >= len(d):
        off = random.Random(zlib.crc32(f"bg-{seed_key}".encode())).randint(0, len(b) - len(d))
        b = b[off:off + len(d)]
    else:
        if len(b) < len(d):
            reps = int(np.ceil(len(d) / max(len(b), 1)))
            b = np.tile(b, reps)
        b = b[: len(d)]

    # 能量调整：按目标 SNR 缩放背景；参考能量 = 背景实际混入的那侧（非 assistant 侧）
    ref = d[:, 0] if d.ndim > 1 else d
    p_d = _rms(ref) ** 2
    target_p_b = p_d / (10 ** (snr_db / 10))
    k = np.sqrt(target_p_b / max(_rms(b) ** 2, 1e-12))
    b_scaled = b * k
    print(f"  参考轨(非assistant侧) RMS={_rms(ref):.4f} | 背景原 RMS={_rms(b):.4f} → 缩放 k={k:.3f} → 调整后 RMS={_rms(b_scaled):.4f}")
    print(f"  实际 SNR = {10 * np.log10(_rms(ref) ** 2 / max(_rms(b_scaled) ** 2, 1e-12)):.1f} dB（目标 {snr_db} dB）")

    # 淡入淡出（首尾 50ms）防爆音
    fade = int(0.05 * sample_rate)
    fade = min(fade, len(b_scaled) // 2)
    if fade:
        b_scaled[:fade] *= np.linspace(0, 1, fade)
        b_scaled[-fade:] *= np.linspace(1, 0, fade)

    # 混入非 assistant 侧（双声道：左声道；单声道：整轨）；防削波：峰值压到 0.95
    if d.ndim > 1:
        mixed = d.copy()
        mixed[:, 0] = d[:, 0] + b_scaled
    else:
        mixed = d + b_scaled
    peak = np.abs(mixed).max()
    if peak > 0.95:
        mixed = mixed / peak * 0.95

    write_bg = out_bg is not False
    if out_mixed is None or (write_bg and out_bg is None):
        out_dir = Path(out_dir or "output/mixed").absolute()
        out_dir.mkdir(parents=True, exist_ok=True)
        if out_mixed is None:
            out_mixed = out_dir / "mixed.wav"
        if write_bg and out_bg is None:
            out_bg = out_dir / "background_track.wav"
    out_mix = Path(out_mixed)
    out_mix.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_mix), mixed, sample_rate)
    result = {"mixed": out_mix}
    if write_bg and out_bg:
        out_bg = Path(out_bg)
        out_bg.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_bg), b_scaled, sample_rate)
        result["background_track"] = out_bg
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="背景音混入 + 能量调整（SNR 控制）")
    ap.add_argument("--dialogue", required=True, help="对话音频（mix/tracks 产物）")
    ap.add_argument("--background", required=True, help="背景音频（babble / 噪声）")
    ap.add_argument("--out-dir", default="output/mixed")
    ap.add_argument("--snr", type=float, default=12.0,
                    help="目标信噪比 dB：15=清晰背景 / 10=电视 / 5=咖啡厅 / 0=嘈杂")
    ap.add_argument("--scene", default=None, choices=["tv", "cafe", "noisy"],
                    help="按场景模板取 SNR（scenes.py；给出时覆盖 --snr）")
    ap.add_argument("--out-mixed", default=None, help="混合成品路径（默认 out-dir/mixed.wav）")
    ap.add_argument("--out-bg", default=None, help="背景独立轨路径（默认 out-dir/background_track.wav）")
    args = ap.parse_args()

    snr = args.snr
    if args.scene:
        from scripts.sound.background.scenes import scene_params
        snr = scene_params(args.scene)["snr_db"]
        print(f"  场景={args.scene} → SNR={snr}dB")

    try:
        result = mix(args.dialogue, args.background, args.out_dir, snr,
                     out_mixed=args.out_mixed, out_bg=args.out_bg,
                     seed_key=Path(args.dialogue).stem)
        print(f"✅ 混音完成（SNR={snr}dB）")
        for k, p in result.items():
            print(f"  🎵 {k}: {p}")
    except Exception as e:
        print(f"❌ 失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
