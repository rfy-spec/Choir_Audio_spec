# -*- coding: utf-8 -*-
"""
每帧音高估计（Hz）：使用自相关法 + 简单置信度。
适合先跑通“每帧Hz输出”和可视化。
"""

from __future__ import annotations
import numpy as np

def _parabolic_interpolation(y: np.ndarray, i: int) -> float:
    """
    在峰值点 i 附近做抛物线插值，让频率更平滑一点。
    返回“更精确的峰位置”（可以是小数）。
    """
    if i <= 0 or i >= len(y) - 1:
        return float(i)
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    denom = (y0 - 2 * y1 + y2)
    if denom == 0:
        return float(i)
    shift = 0.5 * (y0 - y2) / denom
    return float(i) + float(shift)

def estimate_f0_autocorr(
    frame: np.ndarray,
    sr: int,
    fmin: float = 50.0,
    fmax: float = 1000.0,
) -> tuple[float, float]:
    """
    输入：一帧音频 frame（1D numpy）
    输出：
      f0_hz: 估计出的基频（Hz），如果不可靠返回 0.0
      confidence: 置信度(0~1左右的一个值)，越大越可信
    """
    x = frame.astype(np.float32)

    # 去直流 + 加窗（减少边界影响）
    x = x - np.mean(x)
    x = x * np.hanning(len(x)).astype(np.float32)

    # 能量太小：当作静音
    energy = float(np.mean(x * x))
    if energy < 1e-6:
        return 0.0, 0.0

    # 自相关（只取非负滞后）
    corr = np.correlate(x, x, mode="full")
    corr = corr[len(corr)//2:]  # 从 lag=0 开始
    corr0 = float(corr[0])
    if corr0 <= 0:
        return 0.0, 0.0

    # 根据 fmin/fmax 转成 lag 搜索范围
    lag_min = int(sr / fmax)
    lag_max = int(sr / fmin)
    lag_max = min(lag_max, len(corr) - 1)
    if lag_max <= lag_min + 2:
        return 0.0, 0.0

    # 在范围内找最大峰
    search = corr[lag_min:lag_max]
    i_rel = int(np.argmax(search))
    i = lag_min + i_rel

    # 峰值插值（可选但很有用）
    i_refined = _parabolic_interpolation(corr, i)
    if i_refined <= 0:
        return 0.0, 0.0

    f0 = float(sr / i_refined)

    # 置信度：峰值相对 corr[0] 的比例（很粗，但够你先理解）
    peak = float(corr[int(round(i_refined))])
    confidence = peak / corr0

    # 很不可信就返回 0
    if confidence < 0.1:
        return 0.0, confidence

    return f0, confidence
# Core pitch frequency estimation (corresponds to pitch_analyzer)