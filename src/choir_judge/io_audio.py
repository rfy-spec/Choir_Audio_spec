# src/choir_judge/io_audio.py
# ============================================
# 音频输入 / 输出与基础预处理模块
#
# 职责边界：
# 1. 统一读取音频（格式 / 采样率 / 声道）
# 2. 提供基础保存接口（用于调试与中间结果）
# 3. 不做任何评估、不引入主观阈值
# ============================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import librosa
import soundfile as sf


@dataclass(frozen=True)
class AudioConfig:
    """
    音频读取与保存的统一配置
    """
    sr: int = 22050       # 目标采样率（Hz）
    mono: bool = True     # 是否转为单声道
    normalize: bool = True  # 是否做简单幅度归一化


def load_audio(
    path: str | Path,
    cfg: AudioConfig = AudioConfig()
) -> Tuple[np.ndarray, int]:
    """
    读取音频并统一格式
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"音频文件不存在: {path}")

    y, sr = librosa.load(
        path.as_posix(),
        sr=cfg.sr,
        mono=cfg.mono
    )

    if y.size == 0:
        raise ValueError(f"读取到空音频: {path}")

    y = y.astype(np.float32)

    if cfg.normalize:
        peak = float(np.max(np.abs(y)))
        if peak > 0:
            y = y / peak

    return y, sr


def save_audio(
    path: str | Path,
    y: np.ndarray,
    sr: int
) -> None:
    """
    保存音频到磁盘（用于调试或中间结果输出）

    参数说明：
    - y: 音频波形（float32，范围 [-1, 1]）
    - sr: 采样率
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 使用 soundfile 保存，保证精度
    sf.write(
        file=path.as_posix(),
        data=y,
        samplerate=sr
    )
