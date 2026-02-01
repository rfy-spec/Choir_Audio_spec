# src/pitch_eval/reference.py
# =========================================================
# Step B2：计算并可视化“内部参考音高中心”
#
# 小白版理解：
# - 先用 Step B1 的 weights(t) 选出“值得评音准”的时间点（weights>0）
# - 在这些时间点里，把 F0 当作“候选答案”
# - 用稳健统计（中位数）找一个中心：这就是内部参考音高中心
#
# 为什么用“对数域中位数”：
# - 音高感知本质上更接近“倍频关系”（比如高八度=频率×2）
# - 在 log2 里取中位数更稳健，能减少偶发八度跳动的影响
# =========================================================

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple

import numpy as np


@dataclass
class PitchReferenceConfig:
    # 只用 weight >= 这个阈值的点来算参考中心（默认：只要进入评估就算）
    min_weight_for_reference: float = 0.001

    # 输出一些稳健范围统计（用于解释）
    # 例如 25%-75%（IQR）或 10%-90%
    p_low: float = 25.0
    p_high: float = 75.0

    # 参考A4的频率（用于log域计算与midi换算）
    a4_hz: float = 440.0


def hz_to_midi(hz: float, a4_hz: float = 440.0) -> float:
    """Hz -> MIDI（可以是小数）"""
    if not np.isfinite(hz) or hz <= 0:
        return float("nan")
    return 69.0 + 12.0 * np.log2(hz / a4_hz)


def midi_to_note_name(midi: float) -> str:
    """把最接近的MIDI整数映射成音名（仅作提示，不作为评判标准）"""
    if not np.isfinite(midi):
        return "N/A"
    midi_int = int(round(midi))
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    name = names[midi_int % 12]
    octave = midi_int // 12 - 1
    return f"{name}{octave}"


def compute_internal_pitch_reference(
    times_sec: np.ndarray,
    f0_hz: np.ndarray,
    weights: np.ndarray,
    cfg: PitchReferenceConfig = PitchReferenceConfig(),
) -> Dict[str, Any]:
    """
    输入：
    - times_sec: 时间轴
    - f0_hz: 主导F0（Hz）
    - weights: Step B1 的评估权重（0~1）
    输出：
    - ref_hz: 内部参考音高中心（Hz）
    - ref_midi/ref_note: 参考中心对应的音名提示（可解释用）
    - spread_cents: 在参考中心附近的离散程度（越小越稳定）
    - used_points_ratio: 用于参考计算的点占比
    """
    t = np.asarray(times_sec).astype(float)
    f0 = np.asarray(f0_hz).astype(float)
    w = np.asarray(weights).astype(float)

    n = min(len(t), len(f0), len(w))
    t = t[:n]
    f0 = f0[:n]
    w = w[:n]

    # 只用“进入评估”的点，并且 F0 有效
    mask = (w >= cfg.min_weight_for_reference) & np.isfinite(f0) & (f0 > 0)

    if np.sum(mask) < 5:
        # 点太少，无法稳健估计
        return {
            "ref_hz": None,
            "ref_midi": None,
            "ref_note": "N/A",
            "spread_cents_iqr": None,
            "spread_cents_p10_p90": None,
            "used_points_ratio": float(np.mean(mask)) if len(mask) else 0.0,
            "config": asdict(cfg),
        }

    f0_used = f0[mask]

    # 在 log2 域求中位数（更符合音高倍频结构）
    log2_vals = np.log2(f0_used / cfg.a4_hz)
    log2_med = float(np.median(log2_vals))
    ref_hz = float(cfg.a4_hz * (2.0 ** log2_med))

    # 用“cents偏差”描述离散程度：cents = 1200*log2(f/ref)
    cents_dev = 1200.0 * np.log2(f0_used / ref_hz)

    # IQR（25-75）与 10-90 范围，作为“稳定度”解释指标
    p25, p75 = np.percentile(cents_dev, [cfg.p_low, cfg.p_high])
    p10, p90 = np.percentile(cents_dev, [10, 90])

    ref_midi = hz_to_midi(ref_hz, cfg.a4_hz)
    ref_note = midi_to_note_name(ref_midi)

    return {
        "ref_hz": round(ref_hz, 3),
        "ref_midi": round(ref_midi, 3) if np.isfinite(ref_midi) else None,
        "ref_note": ref_note,
        "spread_cents_iqr": round(float(p75 - p25), 2),
        "spread_cents_p10_p90": round(float(p90 - p10), 2),
        "used_points_ratio": round(float(np.mean(mask)), 4),
        "config": asdict(cfg),
    }


def compute_cents_deviation(f0_hz: np.ndarray, ref_hz: float) -> np.ndarray:
    """计算每个F0点相对参考中心的 cents 偏差（无效点输出 nan）"""
    f0 = np.asarray(f0_hz).astype(float)
    out = np.full_like(f0, np.nan, dtype=float)
    if ref_hz is None or (not np.isfinite(ref_hz)) or ref_hz <= 0:
        return out
    mask = np.isfinite(f0) & (f0 > 0)
    out[mask] = 1200.0 * np.log2(f0[mask] / float(ref_hz))
    return out
