# -*- coding: utf-8 -*-
"""
把整段音频切成“帧”，对每帧估计 f0。
输出：times, f0s, confidences
"""

from __future__ import annotations
import numpy as np
from .pitch_estimator import estimate_f0_autocorr
from .pitch_units import nearest_equal_temperament, cents_off




def frame_audio(y: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    """
    返回 shape=(num_frames, frame_size) 的帧矩阵
    """
    if len(y) < frame_size:
        # 太短就补零
        pad = frame_size - len(y)
        y = np.pad(y, (0, pad), mode="constant")

    num_frames = 1 + (len(y) - frame_size) // hop_size
    frames = np.zeros((num_frames, frame_size), dtype=np.float32)
    for k in range(num_frames):
        start = k * hop_size
        frames[k] = y[start:start + frame_size]
    return frames

def extract_f0_per_frame(
    y: np.ndarray,
    sr: int,
    frame_size: int = 2048,
    hop_size: int = 512,
    fmin: float = 50.0,
    fmax: float = 1000.0,
) -> dict:
    frames = frame_audio(y, frame_size, hop_size)
    num_frames = frames.shape[0]

    f0s = np.zeros(num_frames, dtype=np.float32)
    confs = np.zeros(num_frames, dtype=np.float32)
    times = np.arange(num_frames, dtype=np.float32) * (hop_size / sr)

    # 新增：最近标准音频率、音名、cents偏差
    f_refs = np.zeros(num_frames, dtype=np.float32)
    cents = np.full(num_frames, np.nan, dtype=np.float32)
    note_names = ["NA"] * num_frames



    for i in range(num_frames):
        f0, c = estimate_f0_autocorr(frames[i], sr=sr, fmin=fmin, fmax=fmax)
        f0s[i] = f0
        confs[i] = c

        # 只在“可信且有音高”的情况下算 cents
        if f0 > 0 and c >= 0.2:
            note, f_ref, _ = nearest_equal_temperament(float(f0))
            note_names[i] = note
            f_refs[i] = float(f_ref)
            cents[i] = float(cents_off(float(f0), float(f_ref)))


    return {
        "times": times,
        "f0_hz": f0s,
        "confidence": confs,

        "note_name": note_names,
        "f_ref_hz": f_refs,
        "cents": cents,

        "frame_size": frame_size,
        "hop_size": hop_size,
        "sr": sr,
        "fmin": fmin,
        "fmax": fmax,
    }

# Frame segmentation → frequency estimation → cents calculation → statistics (corresponds to analysis_pipeline)