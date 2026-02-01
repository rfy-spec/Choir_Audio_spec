# src/choir_judge/pitch_confidence.py
# =========================================================
# Pitch Confidence（音高可用性置信度）
#
# 目标（给小白的人话版）：
#   不是判断“有没有声音”，而是判断：
#   “这一小段时间里，用单一F0来代表音高，靠不靠谱？”
#
# 设计要点：
# - 输出是连续值 0~1，不再是 0/1 开关
# - 合唱中：即使F0提取缺失，也可能仍有歌声结构
#   → 我们用“结构可信度”给一个低但不为0的底座
# - 图中显示的 Pitch Confidence Timeline 主要用于“可视化表达”
#   不是用于打分（打分在后面步骤才接入）
# =========================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import librosa


@dataclass
class PitchConfidenceConfig:
    """Pitch Confidence 计算配置（偏工程直觉，可解释）"""
    # 频谱上限：人声主要信息区域（Hz）
    fmax_show: float = 5000.0

    # 谱平坦度阈值：越低越像歌声
    flat_lo: float = 0.03   # <= 0.03 认为非常像歌声
    flat_hi: float = 0.15   # >= 0.15 认为更像噪声/摩擦音

    # 童声F0合理范围（宽松先验，不是“对错标准”）
    f0_good_min: float = 200.0
    f0_good_max: float = 1200.0
    f0_ok_min: float = 120.0
    f0_ok_max: float = 1500.0

    # 连续性阈值（cents）：越小越连续
    dcent_lo: float = 30.0
    dcent_hi: float = 80.0

    # 声谱支持（dB）：F0附近能量比背景高多少更可信
    support_lo_db: float = 0.0
    support_hi_db: float = 6.0

    # 如果F0缺失，给“结构底座”的比例（让合唱不至于全是0）
    # 解释：有歌声结构，但不适合用“单一F0”精细分析
    base_from_structure: float = 0.20


def _clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _score_low_better(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """
    x 越小越好：
    - x <= lo => 1
    - x >= hi => 0
    - 中间线性下降
    """
    y = (hi - x) / (hi - lo + 1e-12)
    return _clamp01(y)


def _score_in_range(x: np.ndarray, good_min: float, good_max: float, ok_min: float, ok_max: float) -> np.ndarray:
    """
    x 在 good 区间 => 1
    x 在 ok 边缘区间 => 从1线性降到0
    x 超出 ok 区间 => 0
    """
    score = np.zeros_like(x, dtype=float)

    # good区间：满分
    good = (x >= good_min) & (x <= good_max)
    score[good] = 1.0

    # ok区间：线性过渡
    left = (x >= ok_min) & (x < good_min)
    if np.any(left):
        score[left] = (x[left] - ok_min) / (good_min - ok_min + 1e-12)

    right = (x > good_max) & (x <= ok_max)
    if np.any(right):
        score[right] = (ok_max - x[right]) / (ok_max - good_max + 1e-12)

    return _clamp01(score)


def _hz_to_cents_ratio(f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
    """
    计算相邻两帧频率变化（cents）
    cents = 1200 * log2(f2/f1)
    """
    eps = 1e-12
    return 1200.0 * np.log2((f2 + eps) / (f1 + eps))


def compute_pitch_confidence(
    y: np.ndarray,
    sr: int,
    f0_hz: np.ndarray,
    times_sec: np.ndarray,
    frame_length: int,
    hop_length: int,
    cfg: PitchConfidenceConfig = PitchConfidenceConfig(),
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    输入：
    - y, sr：音频
    - f0_hz, times_sec：已经提取好的F0轨迹（可包含nan）
    - frame_length, hop_length：与F0提取一致的帧参数
    输出：
    - times_sec（对齐）
    - pitch_confidence（0~1）
    - components：子分量（便于解释/调参）
    """

    # 1) 谱平坦度：越低越像歌声（更“谐波化”）
    #    使用与F0相同hop_length保证时间对齐
    flat = librosa.feature.spectral_flatness(y=y, n_fft=frame_length, hop_length=hop_length)[0]
    # 对齐长度（保险起见）
    n = min(len(flat), len(f0_hz), len(times_sec))
    flat = flat[:n]
    f0 = f0_hz[:n]
    t = times_sec[:n]

    asc = _score_low_better(flat, cfg.flat_lo, cfg.flat_hi)  # Acoustic Structure Confidence

    # 2) F0物理合理性（童声宽松先验）
    fpc = _score_in_range(f0, cfg.f0_good_min, cfg.f0_good_max, cfg.f0_ok_min, cfg.f0_ok_max)
    fpc[np.isnan(f0)] = 0.0

    # 3) 连续性：相邻帧F0变化（cents）
    fcc = np.zeros(n, dtype=float)
    if n >= 2:
        f0_prev = f0[:-1]
        f0_next = f0[1:]
        valid = (~np.isnan(f0_prev)) & (~np.isnan(f0_next)) & (f0_prev > 0) & (f0_next > 0)
        dcent = np.full(n - 1, np.nan, dtype=float)
        dcent[valid] = np.abs(_hz_to_cents_ratio(f0_prev[valid], f0_next[valid]))
        # 连续性分数：变化越小越好
        s = np.zeros(n - 1, dtype=float)
        s[valid] = _score_low_better(dcent[valid], cfg.dcent_lo, cfg.dcent_hi)
        # 对齐到每一帧（把相邻变化分数赋给“后一帧”更直觉）
        fcc[1:] = s
        fcc[0] = fcc[1] if n > 1 else 0.0

    # 4) 声谱支持度：F0附近的能量是否“站得住脚”
    #    做法：对每帧找F0对应频点能量，与局部背景能量比
    S = np.abs(librosa.stft(y, n_fft=frame_length, hop_length=hop_length)) ** 2  # power
    freqs = librosa.fft_frequencies(sr=sr, n_fft=frame_length)
    # 只看fmax_show以内
    mask = freqs <= cfg.fmax_show
    S = S[mask, :]
    freqs = freqs[mask]

    # 对齐帧数
    n_spec = S.shape[1]
    n2 = min(n, n_spec)
    S = S[:, :n2]
    f0 = f0[:n2]
    t = t[:n2]
    asc = asc[:n2]
    fpc = fpc[:n2]
    fcc = fcc[:n2]
    n = n2

    ssc = np.zeros(n, dtype=float)
    eps = 1e-12

    # 背景能量：用每帧频率维度的中位数作背景（稳健）
    bg = np.median(S, axis=0) + eps

    for i in range(n):
        if np.isnan(f0[i]) or f0[i] <= 0:
            ssc[i] = 0.0
            continue
        # 找到最接近f0的频点
        k = int(np.argmin(np.abs(freqs - f0[i])))
        p_bin = S[k, i] + eps
        ratio_db = 10.0 * np.log10(p_bin / bg[i])
        # ratio_db 越大 => 越支持
        # 线性映射到[0,1]
        ssc[i] = _clamp01((ratio_db - cfg.support_lo_db) / (cfg.support_hi_db - cfg.support_lo_db + 1e-12))

    # 5) 组合：乘法（弱项拉低整体） + 结构底座（合唱友好）
    #    解释：
    #      - “可用性核心” = FPC * FCC * SSC
    #      - 即使核心为0，只要结构像歌声，仍给少量底座：base_from_structure * ASC
    core = fpc * fcc * ssc
    pc = asc * (cfg.base_from_structure + (1.0 - cfg.base_from_structure) * core)
    pc = _clamp01(pc)

    components = {
        "ASC_structure": asc,
        "FPC_plausibility": fpc,
        "FCC_continuity": fcc,
        "SSC_support": ssc,
        "core": core,
        "spectral_flatness": flat[:n],
        "f0_hz": f0,
    }
    return t, pc, components
