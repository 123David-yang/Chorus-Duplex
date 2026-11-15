"""背景人声合成（babble）：模拟电视解说 / 咖啡厅 / 鸡尾酒会等嘈杂环境。

原理（鸡尾酒会效应）：多段不同说话人的语音错位叠加 → 自然产生嘈杂背景人声。
复用 MOSS-TTS（6 个中文音色）合成中文闲聊 → 随机起始时间叠加 → 音量随机
→ **场景声学处理**（低通=距离感 / 轻混响=空间感 / 能量包络=人群起伏，见 scenes.py）→ 归一化。

用法：
    python -m scripts.sound.background.babble --speakers 6 --duration 30 --scene cafe \
        --out output/bg_pool/babble_cafe_00.wav
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from scripts.sound.synthesis.dialogue_to_wav import DialogueSynth

SAMPLE_RATE = 48000

# 中文闲聊池（背景人声内容，无意义也无所谓，只要"像人声"）
CHITCHAT = [
    "今天天气不错，出去走走吗", "我听说楼下新开了一家川菜馆", "上次那个电影你看了没",
    "孩子下个月就要考试了", "这家的咖啡豆是云南产的", "周末要不要一起去爬山",
    "我昨天在超市看到打折了", "那个新来的同事人挺好的", "晚饭想吃点什么好呢",
    "最近好像要降温了", "你那个项目做到哪一步了", "我家猫昨晚又跑出去了",
    "这歌挺好听的，叫什么名字", "明天记得把报表交给我", "电梯又坏了，爬楼吧",
    "这盘棋下得真不错", "你电话响了，快接一下", "楼下广场舞又开始跳了",
]

# 中文音色（6 个）——此前混入了 Ava/Bella/Adam 三个英文音色念中文，已弃
DEFAULT_VOICES = ["Junhao", "Xiaoyu", "Lingyu", "Weiguo", "Zhiming", "Yuewen"]


# ------------------------------------------------ 声学真实感 DSP（阶段 1，numpy 手写免 scipy）
def _lowpass_fft(x: np.ndarray, sr: int, fc: float) -> np.ndarray:
    """频域一阶低通（高频衰减 → 距离感/隔着门）。"""
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    gain = 1.0 / np.sqrt(1.0 + (f / fc) ** 2)
    return np.fft.irfft(spec * gain, n=len(x)).astype(np.float32)


def _fft_convolve(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    n = len(x) + len(h) - 1
    y = np.fft.irfft(np.fft.rfft(x, n) * np.fft.rfft(h, n), n)
    return y[: len(x)].astype(np.float32)   # 保持原长（尾音裁掉，池材料循环用）


def _tiny_reverb(x: np.ndarray, sr: int, rt60: float, mix: float,
                 rng: random.Random) -> np.ndarray:
    """指数衰减噪声 IR 的简易混响（空间感；RT60 秒时衰减 -60dB）。"""
    n_ir = int(rt60 * sr)
    env = np.exp(-6.9 * np.arange(n_ir) / n_ir)
    gen = np.random.default_rng(rng.randint(0, 2**31))
    ir = (gen.normal(0, 1, n_ir) * env).astype(np.float32)
    ir[0] = 0.0                              # 干声由 (1-mix) 路径保留
    ir /= max(np.sqrt((ir ** 2).sum()), 1e-9)
    wet = _fft_convolve(x, ir)
    wet_rms = np.sqrt((wet ** 2).mean()) or 1e-9
    dry_rms = np.sqrt((x ** 2).mean()) or 1e-9
    wet *= dry_rms / wet_rms                 # 湿声与干声同能量级再按 mix 混合
    return ((1.0 - mix) * x + mix * wet).astype(np.float32)


def _envelope(x: np.ndarray, sr: int, depth: float,
              rate_range: tuple[float, float], rng: random.Random) -> np.ndarray:
    """人群声忽大忽小：0.1~0.5Hz 慢速幅度调制（3 个随机相位正弦叠加）。"""
    t = np.arange(len(x)) / sr
    mod = np.zeros(len(x))
    for _ in range(3):
        f = rng.uniform(*rate_range)
        ph = rng.uniform(0, 2 * np.pi)
        mod += np.sin(2 * np.pi * f * t + ph)
    env = (1.0 + depth * mod / 3.0).astype(np.float32)
    return x * env


def process_babble(x: np.ndarray, sr: int, scene: str, seed: int = 42) -> np.ndarray:
    """按场景模板做声学处理：低通 → 混响 → 能量包络。scene='none' 原样返回。"""
    from scripts.sound.background.scenes import scene_params
    p = scene_params(scene)
    if not p:
        return x
    rng = random.Random(seed ^ 0x5CE7E)
    y = _lowpass_fft(x, sr, p["lowpass_hz"])
    y = _tiny_reverb(y, sr, p["reverb_rt"], p["reverb_mix"], rng)
    y = _envelope(y, sr, p["envelope_depth"], p["envelope_rate"], rng)
    return y


def make_babble(synth: DialogueSynth, n_speakers: int, duration: float,
                out_path: str | Path, seed: int = 42,
                voices: list[str] | None = None, scene: str = "none") -> Path:
    """合成 n_speakers 段闲聊 → 错位叠加 → 场景声学处理 → 归一化 → 写 wav。"""
    voices = voices or DEFAULT_VOICES
    rng = random.Random(seed)
    total = np.zeros(int(duration * SAMPLE_RATE), dtype=np.float32)

    for i in range(n_speakers):
        voice = voices[i % len(voices)]
        texts = rng.sample(CHITCHAT, rng.randint(1, 2))
        for t in texts:
            audio = synth.turn_to_mono(t, voice=voice)
            start = rng.randint(0, max(0, len(total) - len(audio)))
            seg = total[start:start + len(audio)]
            seg += audio * rng.uniform(0.35, 0.85)  # 每人音量随机，层次感
        print(f"  🗣 speaker{i}({voice}): {texts[0][:15]}...")

    total = process_babble(total, SAMPLE_RATE, scene, seed)

    peak = np.abs(total).max()
    if peak > 0:
        total = total / (peak * 1.6)  # 归一化留余量
    out_path = Path(out_path).absolute()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), total, SAMPLE_RATE)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="背景人声 babble 合成（MVP）")
    ap.add_argument("--speakers", type=int, default=6, help="背景说话人数")
    ap.add_argument("--duration", type=float, default=30, help="背景时长（秒）")
    ap.add_argument("--out", default="output/ambient_babble.wav")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scene", default="none", choices=["none", "tv", "cafe", "noisy"],
                    help="场景声学模板：none 不处理 / tv / cafe / noisy（低通+混响+能量包络）")
    args = ap.parse_args()

    try:
        print(f"加载 MOSS-TTS 模型（首次约 15-30s）...")
        synth = DialogueSynth(tmp_dir=Path(args.out).parent)
        out = make_babble(synth, args.speakers, args.duration, args.out, args.seed,
                          scene=args.scene)
        print(f"✅ 背景音完成 → {out} | {args.speakers} 人闲聊，{args.duration}s | 场景={args.scene}")
    except Exception as e:
        print(f"❌ 失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
