# src/choir_judge/quality_gate.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import numpy as np
import librosa


@dataclass(frozen=True)
class ChoirQualityThresholds:
    # 经验阈值（先给“能用版本”，后面你可以用数据再校准）
    rms_db_low: float = -35.0      # 太小声：<-35 dBFS 基本听不清
    rms_db_high: float = -6.0      # 太大声：>-6 dBFS 容易接近爆音
    clip_ratio_high: float = 0.002 # 削顶比例阈值（0.2%）
    zcr_high: float = 0.20         # ZCR 高通常更“噪/擦”，合唱会比独唱略高，阈值放宽
    silence_ratio_high: float = 0.60  # 超过 60% 近似静音：录得太远/无效片段太多
    flatness_high: float = 0.45       # 平坦度太高更像噪声（越接近1越像噪声）
    # F0相关（合唱：降权、放宽）
    f0_coverage_low: float = 0.15
    voiced_ratio_low: float = 0.15


@dataclass(frozen=True)
class ChoirQualityWeights:
    rms: float = 0.25
    clip: float = 0.20
    zcr: float = 0.10
    silence: float = 0.20
    flatness: float = 0.15
    f0: float = 0.10   # 合唱里给小权重


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def compute_choir_quality(
    y: np.ndarray,
    sr: int,
    f0_hz: Optional[np.ndarray] = None,
    voiced_flag: Optional[np.ndarray] = None,
    thr: ChoirQualityThresholds = ChoirQualityThresholds(),
    w: ChoirQualityWeights = ChoirQualityWeights(),
) -> Dict[str, Any]:
    """
    合唱音频质量门控（MVP）：
    输出 global_confidence (0-1) + warnings（解释原因）
    """
    eps = 1e-12

    # 1) 音量：RMS -> dBFS（相对满幅）
    rms = float(np.sqrt(np.mean(y ** 2) + eps))
    rms_db = float(20.0 * np.log10(rms + eps))

    # 2) 削顶/爆音比例
    clip_ratio = float(np.mean(np.abs(y) >= 0.999))

    # 3) 静音比例：用短时能量判定（非常直观）
    hop = 512
    frame = 2048
    rms_frames = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    # “近似静音”：比整体中位数还低很多，且绝对很小
    med = float(np.median(rms_frames) + eps)
    silence_flag = (rms_frames < max(1e-4, 0.15 * med))
    silence_ratio = float(np.mean(silence_flag))

    # 4) ZCR：噪声/擦音多时通常更高（粗略）
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame, hop_length=hop)[0]
    zcr_mean = float(np.mean(zcr))

    # 5) 谱平坦度：越高越像噪声，越低越“有音高/更像唱声”
    flat = librosa.feature.spectral_flatness(y=y, n_fft=frame, hop_length=hop)[0]
    flat_mean = float(np.mean(flat))

    # 6) F0覆盖（合唱里仅作“弱证据”）
    f0_coverage = 0.0
    voiced_ratio = 0.0
    if f0_hz is not None:
        finite = np.isfinite(f0_hz)
        f0_coverage = float(np.mean(finite))
    if voiced_flag is not None:
        voiced_ratio = float(np.mean(voiced_flag))

    # ---- 归一化到 0~1（越大越好）----
    rms_ok = _sigmoid((rms_db - thr.rms_db_low) / 3.0) * _sigmoid((thr.rms_db_high - rms_db) / 3.0)
    clip_ok = _sigmoid((thr.clip_ratio_high - clip_ratio) / 0.001)
    zcr_ok = _sigmoid((thr.zcr_high - zcr_mean) / 0.03)
    silence_ok = _sigmoid((thr.silence_ratio_high - silence_ratio) / 0.08)
    flat_ok = _sigmoid((thr.flatness_high - flat_mean) / 0.08)
    f0_ok = _sigmoid((f0_coverage - thr.f0_coverage_low) / 0.10) * _sigmoid((voiced_ratio - thr.voiced_ratio_low) / 0.10)

    conf = (
        w.rms * rms_ok +
        w.clip * clip_ok +
        w.zcr * zcr_ok +
        w.silence * silence_ok +
        w.flatness * flat_ok +
        w.f0 * f0_ok
    )
    conf = float(np.clip(conf, 0.0, 1.0))

    # ---- 生成小白可懂的 warnings ----
    warnings: List[str] = []
    if rms_db < thr.rms_db_low:
        warnings.append(f"音量偏小（RMS≈{rms_db:.1f} dBFS），可能离麦太远/录得太轻。")
    if rms_db > thr.rms_db_high:
        warnings.append(f"音量偏大（RMS≈{rms_db:.1f} dBFS），可能接近过载。")
    if clip_ratio > thr.clip_ratio_high:
        warnings.append(f"疑似爆音/削顶（削顶比例≈{clip_ratio*100:.2f}%），会严重影响评估可信度。")
    if silence_ratio > thr.silence_ratio_high:
        warnings.append(f"有效声音占比偏低（静音比例≈{silence_ratio*100:.1f}%），可能包含大量空段/非演唱段。")
    if zcr_mean > thr.zcr_high:
        warnings.append(f"噪声/擦音偏多（ZCR≈{zcr_mean:.3f}），可能环境杂音较大或录音较“糙”。")
    if flat_mean > thr.flatness_high:
        warnings.append(f"信号偏“噪声化”（谱平坦度≈{flat_mean:.3f}），清晰度可能不足。")
    # F0项只做轻提示（合唱不强制）
    if f0_hz is not None and f0_coverage < thr.f0_coverage_low:
        warnings.append(f"音高可追踪帧偏少（F0覆盖≈{f0_coverage*100:.1f}%），合唱多声部时可能正常，但会降低音准评分稳定性。")

    return {
        "rms_db": rms_db,
        "clip_ratio": clip_ratio,
        "silence_ratio": silence_ratio,
        "zcr_mean": zcr_mean,
        "spectral_flatness_mean": flat_mean,
        "f0_coverage": f0_coverage,
        "voiced_ratio": voiced_ratio,
        "global_confidence": conf,
        "warnings": warnings,
    }
