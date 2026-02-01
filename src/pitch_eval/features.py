# src/pitch_eval/features.py
# =========================================================
# 音准评估 - 特征提取（传统声学）
# 核心：提取单条“主导音高轨迹”F0（Hz） + 有声标记
#
# 说明：
# - 合唱是多音高叠加（polyphonic），传统单F0方法会更难
# - 这里先做“可用版本”：用 pYIN 给出一条相对稳定的F0轨迹
# - 后续如果需要更强鲁棒性，可升级为深度学习多音高或多声部分析
# =========================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import librosa


@dataclass(frozen=True)
class PitchConfig:
    # 合唱/童声大致音域范围（可后续用数据再校准）
    fmin_hz: float = librosa.note_to_hz("C3")  # 童声较高，先用 C3
    fmax_hz: float = librosa.note_to_hz("C7")

    frame_length: int = 2048
    hop_length: int = 256

    # voiced_prob 太低的帧容易是错检：这里做一个简单过滤
    voiced_prob_min: float = 0.6

    # 后处理平滑：把明显跳变稍微抑制一下（单位：帧）
    median_filter_len: int = 5


def extract_pitch_track(y: np.ndarray, sr: int, cfg: PitchConfig = PitchConfig()) -> Dict[str, Any]:
    """
    提取F0轨迹（Hz）与有声标记

    返回字段：
    - f0_hz: np.ndarray, NaN 表示无声/不可用
    - voiced_flag: np.ndarray[bool]
    - voiced_prob: np.ndarray[float]
    - times_sec: np.ndarray[float]
    - cfg: PitchConfig
    """
    f0_hz, voiced_flag, voiced_prob = librosa.pyin(
        y=y,
        fmin=cfg.fmin_hz,
        fmax=cfg.fmax_hz,
        sr=sr,
        frame_length=cfg.frame_length,
        hop_length=cfg.hop_length,
    )

    times_sec = librosa.frames_to_time(
        np.arange(len(f0_hz)),
        sr=sr,
        hop_length=cfg.hop_length
    )

    # 过滤：voiced_prob 低的帧直接视为不可用
    good = (voiced_flag.astype(bool)) & np.isfinite(f0_hz) & (voiced_prob >= cfg.voiced_prob_min)
    f0_clean = f0_hz.copy()
    f0_clean[~good] = np.nan

    # 简单中值滤波，减少“毛刺跳变”
    if cfg.median_filter_len >= 3:
        f0_smooth = f0_clean.copy()
        half = cfg.median_filter_len // 2
        for i in range(len(f0_clean)):
            j0 = max(0, i - half)
            j1 = min(len(f0_clean), i + half + 1)
            win = f0_clean[j0:j1]
            win = win[np.isfinite(win)]
            if win.size > 0:
                f0_smooth[i] = float(np.median(win))
            else:
                f0_smooth[i] = np.nan
    else:
        f0_smooth = f0_clean

    voiced_final = np.isfinite(f0_smooth)

    return {
        "f0_hz": f0_smooth,
        "voiced_flag": voiced_final,
        "voiced_prob": voiced_prob,
        "times_sec": times_sec,
        "cfg": cfg,
    }
