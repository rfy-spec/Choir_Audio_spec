# src/choir_judge/spectrogram_vis.py
# =========================================================
# 声谱图专项可视化模块
#
# 功能：
# 1) 生成【独立声谱图（无 F0）】
# 2) 生成【独立声谱图（叠加 F0）】
#
# 设计原则：
# - 图像只用于“看结构”，不做任何评分
# - 图中文字全部使用英文（避免字体问题）
# - 代码注释全部使用中文（方便你理解）
# =========================================================

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import librosa

from pitch_eval.features import PitchConfig, extract_pitch_track


def _compute_spectrogram(
    y: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
):
    """
    计算声谱图（功率谱，dB）

    返回：
    - S_db: (freq, time) 的 dB 声谱
    - freqs: 每一行对应的频率（Hz）
    - times: 每一列对应的时间（秒）
    """
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) ** 2
    S_db = librosa.power_to_db(S, ref=np.max)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(
        np.arange(S_db.shape[1]),
        sr=sr,
        hop_length=hop_length,
    )

    return S_db, freqs, times


def save_spectrogram(
    y: np.ndarray,
    sr: int,
    out_png: str | Path,
    audio_name: str,
    pitch_cfg: Optional[PitchConfig] = None,
    overlay_f0: bool = False,
):
    """
    保存一张声谱图

    参数：
    - overlay_f0=False: 只画声谱图（看结构）
    - overlay_f0=True : 声谱图 + F0 叠加（看 F0 是否贴着谐波）
    """
    if pitch_cfg is None:
        pitch_cfg = PitchConfig()

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # 1) 计算声谱图
    S_db, freqs, times = _compute_spectrogram(
        y=y,
        sr=sr,
        n_fft=pitch_cfg.frame_length,
        hop_length=pitch_cfg.hop_length,
    )

    # 2) 频率显示上限（人声主要信息区）
    fmax_show = 5000
    freq_mask = freqs <= fmax_show

    # 3) 创建大尺寸图像（专门看细节）
    plt.figure(figsize=(14, 6))

    plt.imshow(
        S_db[freq_mask, :],
        origin="lower",
        aspect="auto",
        extent=[
            times[0],
            times[-1],
            freqs[freq_mask][0],
            freqs[freq_mask][-1],
        ],
        cmap="magma",
    )

    plt.colorbar(label="Power (dB)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")

    if not overlay_f0:
        plt.title(
            f"Spectrogram (structure view, no F0) - {audio_name}\n"
            "Use this view to inspect harmonics, noise, and overall structure"
        )
    else:
        # 4) 提取并叠加 F0
        pitch = extract_pitch_track(y, sr, pitch_cfg)
        f0 = pitch["f0_hz"]
        t_f0 = pitch["times_sec"]

        # 用非常显眼的颜色和线宽
        plt.plot(
            t_f0,
            f0,
            color="cyan",
            linewidth=2.5,
            label="Dominant F0",
        )

        plt.legend(loc="upper right")

        plt.title(
            f"Spectrogram with F0 overlay - {audio_name}\n"
            "Check whether F0 follows the harmonic structure"
        )

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
