"""背景场景模板：低通（距离感）/ 混响（空间感）/ 能量包络（人群起伏）+ SNR 参考。

生成侧参数（lowpass/reverb/envelope）由 babble.py 应用；
混音侧 SNR 供 mix.py（--scene）与编排参考。数值为初始主观设定，听感迭代可调。
"""
from __future__ import annotations

SCENES: dict[str, dict] = {
    "tv": {  # 电视/广播背景：较清晰、空间小
        "snr_db": 12.0, "lowpass_hz": 3800.0, "reverb_rt": 0.25, "reverb_mix": 0.25,
        "envelope_depth": 0.15, "envelope_rate": (0.10, 0.30),
    },
    "cafe": {  # 咖啡厅闲聊：中等距离、有空间感
        "snr_db": 5.0, "lowpass_hz": 2600.0, "reverb_rt": 0.55, "reverb_mix": 0.35,
        "envelope_depth": 0.30, "envelope_rate": (0.10, 0.50),
    },
    "noisy": {  # 嘈杂人群（大厅/排档）：近 0dB、长混响、起伏大
        "snr_db": 0.0, "lowpass_hz": 3000.0, "reverb_rt": 0.80, "reverb_mix": 0.40,
        "envelope_depth": 0.40, "envelope_rate": (0.20, 0.50),
    },
}


def scene_params(name: str) -> dict:
    if name == "none":
        return {}
    if name not in SCENES:
        raise KeyError(f"未知场景 {name!r}（可选：none/{'/'.join(SCENES)}）")
    return SCENES[name]
