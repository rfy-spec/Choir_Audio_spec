# src/pitch_eval/scoring.py
# =========================================================
# 音准评估 - 评分与可解释诊断
#
# 我们做的事情：
# 1) 把每一帧的 F0 变成“离最近半音的偏差（cents）”
# 2) 给出可解释指标：
#    - 平均偏差（越小越准）
#    - 稳定度（偏差波动大小）
#    - 跑调严重比例（>50 cents 的比例）
#    - 漂移（随时间整体变高/变低）
# 3) 生成“哪里坏”的时间段列表（给小白看得懂）
#
# 注意：
# - 这里是“无参考音高”的评估：不知道乐谱/伴奏
# - 所以我们用“就近半音”来衡量“唱得稳不稳、是否经常偏离音级”
# - 未来如果你有乐谱或伴奏基准，可以升级为“对照目标音高”的评估
# =========================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np


@dataclass(frozen=True)
class PitchScoreConfig:
    # 分段：用于定位“哪里坏”
    window_sec: float = 1.0
    hop_sec: float = 0.5

    # 解释阈值（cents）
    good_cents: float = 15.0
    ok_cents: float = 30.0
    bad_cents: float = 50.0

    # 评分权重（可后续校准）
    w_accuracy: float = 0.65   # 平均偏差
    w_outlier: float = 0.20    # 严重跑调比例
    w_drift: float = 0.15      # 漂移


def hz_to_cents(f_hz: np.ndarray, ref_hz: np.ndarray) -> np.ndarray:
    """cents = 1200 * log2(f/ref)；ref 和 f 逐元素"""
    eps = 1e-12
    return 1200.0 * np.log2((f_hz + eps) / (ref_hz + eps))


def hz_to_midi(f_hz: np.ndarray) -> np.ndarray:
    """MIDI = 69 + 12*log2(f/440)"""
    eps = 1e-12
    return 69.0 + 12.0 * np.log2((f_hz + eps) / 440.0)


def midi_to_hz(m: np.ndarray) -> np.ndarray:
    return 440.0 * (2.0 ** ((m - 69.0) / 12.0))


def _label_by_abs_cents(abs_c: float, cfg: PitchScoreConfig) -> str:
    if abs_c <= cfg.good_cents:
        return "好（基本很准）"
    if abs_c <= cfg.ok_cents:
        return "一般（轻微偏差）"
    if abs_c <= cfg.bad_cents:
        return "差（明显跑调）"
    return "很差（严重跑调）"


def score_pitch(
    times_sec: np.ndarray,
    f0_hz: np.ndarray,
    voiced_flag: np.ndarray,
    cfg: PitchScoreConfig = PitchScoreConfig(),
) -> Dict[str, Any]:
    """
    输入：逐帧F0（Hz）与voiced
    输出：可解释评分结果（含分段“哪里坏”列表）
    """
    # 只取有效帧
    mask = voiced_flag.astype(bool) & np.isfinite(f0_hz)
    if np.sum(mask) < 10:
        return {
            "ok": False,
            "reason": "有效有声音高帧太少，无法做可靠音准评估（可能是多声部过强/噪声过大/音量过小）。"
        }

    t = times_sec[mask]
    f = f0_hz[mask]

    # 1) 计算“离最近半音”的偏差
    midi = hz_to_midi(f)
    midi_nearest = np.round(midi)          # 最近半音
    ref_hz = midi_to_hz(midi_nearest)
    cents = hz_to_cents(f, ref_hz)         # 有正负：正=偏高（sharp），负=偏低（flat）
    abs_cents = np.abs(cents)

    # 2) 全局指标
    mean_abs = float(np.mean(abs_cents))
    median_abs = float(np.median(abs_cents))
    std_abs = float(np.std(abs_cents))
    outlier_ratio = float(np.mean(abs_cents > cfg.bad_cents))  # >50c 的比例

    # 3) 漂移（对 cents 随时间拟合一条线，斜率单位：cents/秒）
    #    斜率绝对值越大，代表越唱越高/越唱越低
    if t.size >= 2:
        slope, intercept = np.polyfit(t, cents, 1)
        drift_cents_per_sec = float(slope)
    else:
        drift_cents_per_sec = 0.0

    # 4) 分段定位“哪里坏”
    segments: List[Dict[str, Any]] = []
    win = cfg.window_sec
    hop = cfg.hop_sec
    t0 = float(np.min(t))
    t1 = float(np.max(t))
    cur = t0

    while cur <= t1:
        seg_start = cur
        seg_end = cur + win
        idx = (t >= seg_start) & (t < seg_end)
        if np.sum(idx) >= 5:
            seg_c = cents[idx]
            seg_abs = np.abs(seg_c)
            med_abs = float(np.median(seg_abs))
            med_signed = float(np.median(seg_c))
            label = _label_by_abs_cents(med_abs, cfg)

            # 小白解释：偏高/偏低
            if med_signed > 5:
                direction = "整体偏高"
            elif med_signed < -5:
                direction = "整体偏低"
            else:
                direction = "整体接近准"

            segments.append({
                "start_sec": round(seg_start, 2),
                "end_sec": round(seg_end, 2),
                "median_abs_cents": round(med_abs, 1),
                "median_signed_cents": round(med_signed, 1),
                "direction": direction,
                "label": label
            })
        cur += hop

    # 取最差的若干段，作为“哪里坏”的证据
    segments_sorted = sorted(segments, key=lambda x: x["median_abs_cents"], reverse=True)
    worst_segments = segments_sorted[:8]

    # 5) 评分（0~100），并明确“怎么得出来”
    #   - accuracy：平均偏差越小越好（以 60c 作为“差到很离谱”的上限）
    #   - outlier：严重跑调比例越高越扣分
    #   - drift：漂移越大越扣分
    # 这些都是“可解释”的传统规则映射，可后续用数据校准阈值/权重
    acc_score = float(np.clip(100.0 * (1.0 - mean_abs / 60.0), 0.0, 100.0))
    outlier_score = float(np.clip(100.0 * (1.0 - outlier_ratio / 0.30), 0.0, 100.0))  # 30% 严重跑调视为很差
    drift_score = float(np.clip(100.0 * (1.0 - (abs(drift_cents_per_sec) / 8.0)), 0.0, 100.0))  # 8 cents/s 以上很差

    final_score = (
        cfg.w_accuracy * acc_score +
        cfg.w_outlier * outlier_score +
        cfg.w_drift * drift_score
    )
    final_score = float(np.clip(final_score, 0.0, 100.0))

    # 6) 总结性结论（给小白一句话）
    if final_score >= 85:
        verdict = "音准整体很好（大多数时间很准）"
    elif final_score >= 70:
        verdict = "音准中等（有一些明显偏差段落）"
    elif final_score >= 55:
        verdict = "音准偏差较多（需要重点改进）"
    else:
        verdict = "音准较差（存在较多严重跑调段）"

    explain = (
        f"我们先用传统算法提取每一帧的主导音高F0，然后把它换算成“离最近半音的偏差（cents）”。"
        f"偏差越小越准。最后用平均偏差、严重跑调比例、整体漂移三项加权得到总分。"
    )

    return {
        "ok": True,
        "verdict": verdict,
        "explain": explain,
        "score": {
            "final": round(final_score, 1),
            "breakdown": {
                "accuracy_score": round(acc_score, 1),
                "outlier_score": round(outlier_score, 1),
                "drift_score": round(drift_score, 1),
                "weights": {
                    "accuracy": cfg.w_accuracy,
                    "outlier": cfg.w_outlier,
                    "drift": cfg.w_drift
                }
            }
        },
        "metrics": {
            "mean_abs_cents": round(mean_abs, 2),
            "median_abs_cents": round(median_abs, 2),
            "std_abs_cents": round(std_abs, 2),
            "outlier_ratio_gt50c": round(outlier_ratio, 4),
            "drift_cents_per_sec": round(drift_cents_per_sec, 3)
        },
        "worst_segments": worst_segments,
        "series": {
            "times_sec": times_sec.tolist(),   # 全帧（含NaN），用于画图或进一步分析
            "f0_hz": np.where(np.isfinite(f0_hz), f0_hz, np.nan).tolist()
        }
    }
