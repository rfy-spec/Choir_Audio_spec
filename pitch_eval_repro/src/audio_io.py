# -*- coding: utf-8 -*-
"""
读取音频并转成单声道浮点数组（numpy），可选重采样。
尽量少依赖：优先用 soundfile；没有就退回 librosa。
"""

from __future__ import annotations
import numpy as np

def load_audio_mono(path: str, target_sr: int = 48000) -> tuple[np.ndarray, int]:
    """
    返回:
      y: np.ndarray, float32, 范围通常在[-1, 1]
      sr: 采样率
    """
    # 优先用 soundfile（常见且轻量）
    try:
        import soundfile as sf
        y, sr = sf.read(path, dtype="float32", always_2d=False)
        # 如果是多声道，取平均变单声道
        if y.ndim == 2:
            y = y.mean(axis=1).astype("float32")
    except Exception:
        # 退回 librosa
        import librosa
        y, sr = librosa.load(path, sr=None, mono=True)
        y = y.astype("float32")

    # 重采样到 target_sr（统一后面分析更稳定）
    if target_sr is not None and sr != target_sr:
        try:
            import librosa
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr).astype("float32")
            sr = target_sr
        except Exception as e:
            raise RuntimeError(
                "需要重采样但当前环境缺少 librosa。你可以：\n"
                "1) 安装 librosa；或\n"
                "2) 把 target_sr 设为 None，先不重采样。"
            ) from e

    return y, sr
# Audio I/O: read audio, resample, convert to numpy arrays