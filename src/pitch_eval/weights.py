# src/pitch_eval/weights.py
# =========================================================
# Step B1：定义哪些时间段进入音准评估（基于 Pitch Confidence）
#
# 核心思想（小白版）：
# - Pitch Confidence 是“F0值有多可信”
# - 我们把它变成“评分权重 weights(0~1)”
#   * 高可信：权重≈1（正常评分）
#   * 中可信：权重≈0.5（降权评分）
#   * 低可信：权重=0（不评分、不扣分）
#
# 注意：
# - 这里不计算音准分数，只输出“阅卷范围/权重”
# =========================================================

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class PitchWeightConfig:
    # 可靠阈值：>= reliable_th 视为“可靠段”
    reliable_th: float = 0.70
    # 可用阈值：>= usable_th 视为“可用段”（但要降权）
    usable_th: float = 0.40

    # 中间段的降权系数（可用段默认 0.5）
    usable_weight: float = 0.50

    # 时间段合并/过滤：太短的段落不单独列出来（秒）
    min_segment_sec: float = 0.20

    # 平滑：让权重线更稳（秒），0 表示不平滑
    smooth_sec: float = 0.15


def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    kernel = np.ones(win, dtype=float) / float(win)
    return np.convolve(x, kernel, mode="same")


def build_pitch_weights(
    times_sec: np.ndarray,
    pitch_conf: np.ndarray,
    cfg: PitchWeightConfig = PitchWeightConfig(),
) -> Dict[str, object]:
    """
    输入：
    - times_sec: 与F0/PC对齐的时间轴
    - pitch_conf: Pitch Confidence (0~1)
    输出：
    - weights: 0~1 权重时间线
    - masks: reliable/usable/eval 三类mask
    - segments: 可靠段/可用段/不评段 的时间区间列表
    - coverage: 各类覆盖比例（便于你看“有多少能评”）
    """
    t = np.asarray(times_sec).astype(float)
    pc = np.asarray(pitch_conf).astype(float)
    n = min(len(t), len(pc))
    t = t[:n]
    pc = pc[:n]

    # 估计帧间隔
    if n >= 2:
        dt = float(np.median(np.diff(t)))
    else:
        dt = 0.0

    # 平滑Pitch Confidence（可视化更稳）
    if cfg.smooth_sec > 0 and dt > 0:
        win = int(round(cfg.smooth_sec / dt))
        win = max(1, win)
        pc_s = _moving_average(pc, win)
        pc_s = np.clip(pc_s, 0.0, 1.0)
    else:
        pc_s = pc

    # 三类mask
    reliable = pc_s >= cfg.reliable_th
    usable = (pc_s >= cfg.usable_th) & (pc_s < cfg.reliable_th)
    eval_mask = reliable | usable

    # 权重规则：可靠=1；可用=usable_weight；不评=0
    weights = np.zeros_like(pc_s, dtype=float)
    weights[reliable] = 1.0
    weights[usable] = float(cfg.usable_weight)
    weights = np.clip(weights, 0.0, 1.0)

    # 把mask转成时间段列表
    segments = {
        "reliable": _mask_to_segments(t, reliable, cfg.min_segment_sec),
        "usable": _mask_to_segments(t, usable, cfg.min_segment_sec),
        "not_evaluated": _mask_to_segments(t, ~eval_mask, cfg.min_segment_sec),
    }

    # 覆盖率统计
    total = float(len(t)) if len(t) else 1.0
    coverage = {
        "reliable_ratio": float(np.sum(reliable) / total),
        "usable_ratio": float(np.sum(usable) / total),
        "evaluated_ratio": float(np.sum(eval_mask) / total),
        "not_evaluated_ratio": float(np.sum(~eval_mask) / total),
    }

    return {
        "times_sec": t,
        "pitch_confidence": pc,
        "pitch_confidence_smooth": pc_s,
        "weights": weights,
        "masks": {
            "reliable": reliable,
            "usable": usable,
            "evaluated": eval_mask,
        },
        "segments": segments,
        "coverage": coverage,
        "config": asdict(cfg),
    }


def _mask_to_segments(
    times_sec: np.ndarray,
    mask: np.ndarray,
    min_segment_sec: float,
) -> List[Dict[str, float]]:
    """
    把 bool mask 转换为时间段列表 [{start, end, duration}, ...]
    """
    t = np.asarray(times_sec)
    m = np.asarray(mask).astype(bool)
    n = min(len(t), len(m))
    t = t[:n]
    m = m[:n]
    if n == 0:
        return []

    segments: List[Dict[str, float]] = []
    in_seg = False
    start = 0.0

    for i in range(n):
        if m[i] and not in_seg:
            in_seg = True
            start = float(t[i])
        if in_seg and (not m[i] or i == n - 1):
            # 结束点：若到末尾且仍为True，end用末尾时间
            end = float(t[i]) if (not m[i]) else float(t[i])
            # 让end至少>=start
            if end < start:
                end = start
            dur = end - start
            if dur >= float(min_segment_sec):
                segments.append({"start": round(start, 3), "end": round(end, 3), "duration": round(dur, 3)})
            in_seg = False

    return segments
