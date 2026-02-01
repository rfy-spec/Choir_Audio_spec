# src/choir_judge/audio_sheet.py
# =========================================================
# Audio Sheet（音频试卷底稿）生成模块
#
# 输出：
# - 一张评委可读的多面板图（audio_sheet.png）【图中文字全英文，避免字体问题】
# - 一个简短概览 JSON（audio_sheet.json）【JSON 可保留中文总结，便于你理解】
#
# 设计目标：
# - 小白/评委能直观看懂“音频原貌”
# - 为后续音准/节奏/吐字/融合等评估提供可复用底稿
#
# 【本次改动】：
# - 仅将 Panel 3 从 Voicing Timeline(0/1) 替换为 Pitch Confidence Timeline(0~1)
# - 其他统计/JSON/图的其余面板逻辑尽量不变
# =========================================================

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt
import librosa

from pitch_eval.features import PitchConfig, extract_pitch_track


from choir_judge.pitch_confidence import compute_pitch_confidence, PitchConfidenceConfig


def _rms_db_over_time(y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """
    计算逐帧 RMS 并转为 dB（相对最大值）。

    说明：
    - 这里的 dB 是“相对最大值”的 dB（ref=np.max），适合看趋势与稳定性
    - 不是严格的 dBFS，但对于“评委可读的响度变化”完全够用
    """
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    return rms_db


def _make_summary_cn(stats: Dict[str, Any]) -> str:
    """
    给你（开发者/中文读者）的一句话总结：写进 JSON，方便对照理解。
    图里会用英文 summary。
    """
    voiced_ratio = float(stats["voiced_ratio"])
    rms_mean = float(stats["rms_db_mean_rel"])
    f0_ok = float(stats["f0_valid_ratio"])

    parts = []

    if voiced_ratio >= 0.6:
        parts.append("有效演唱段充足")
    else:
        parts.append("有效演唱段偏少（可能停顿多或声音太弱）")

    if rms_mean > -12:
        parts.append("整体响度偏强")
    elif rms_mean < -28:
        parts.append("整体响度偏弱")
    else:
        parts.append("整体响度适中")

    if f0_ok >= 0.5:
        parts.append("音高轨迹可用（适合后续音准分析）")
    else:
        parts.append("音高轨迹较不稳定（后续音准需谨慎）")

    return "；".join(parts) + "。"


def _make_summary_en(stats: Dict[str, Any]) -> str:
    """
    给评委/图像读者的一句话英文总结：用于 PNG 图标题。
    """
    voiced_ratio = float(stats["voiced_ratio"])
    rms_mean = float(stats["rms_db_mean_rel"])
    f0_ok = float(stats["f0_valid_ratio"])

    parts = []

    if voiced_ratio >= 0.6:
        parts.append("Enough voiced singing segments")
    else:
        parts.append("Voiced singing segments are limited")

    if rms_mean > -12:
        parts.append("Overall loudness is strong")
    elif rms_mean < -28:
        parts.append("Overall loudness is weak")
    else:
        parts.append("Overall loudness is moderate")

    if f0_ok >= 0.5:
        parts.append("Pitch track is usable for further evaluation")
    else:
        parts.append("Pitch track is unstable (use with caution)")

    return "; ".join(parts) + "."


def build_audio_sheet(
    y: np.ndarray,
    sr: int,
    pitch_cfg: PitchConfig | None = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    生成 audio_sheet 所需的数据：
    - sheet_data：用于画图（包含时间轴、波形、声谱、F0等）
    - sheet_json：用于保存的简短概览 JSON
    """
    if pitch_cfg is None:
        pitch_cfg = PitchConfig()

    duration_sec = float(len(y) / sr)

    # 1) 提取主导 F0（作为“音准批改底稿”的基础）
    pitch = extract_pitch_track(y, sr, pitch_cfg)
    f0 = pitch["f0_hz"]
    voiced = pitch["voiced_flag"]
    t_f0 = pitch["times_sec"]

    # ✅ 新增：计算 Pitch Confidence（0~1 连续值）
    # 说明：这是“F0 是否值得信任”的度量，不等价于“有没有声音”
    pc_cfg = PitchConfidenceConfig()
    t_pc, pitch_conf, _parts = compute_pitch_confidence(
        y=y,
        sr=sr,
        f0_hz=f0,
        times_sec=t_f0,
        frame_length=pitch_cfg.frame_length,
        hop_length=pitch_cfg.hop_length,
        cfg=pc_cfg,
    )

    # 2) 逐帧响度（用同 hop_length 对齐，便于多图对齐）
    rms_db = _rms_db_over_time(y, pitch_cfg.frame_length, pitch_cfg.hop_length)
    t_rms = librosa.frames_to_time(np.arange(len(rms_db)), sr=sr, hop_length=pitch_cfg.hop_length)

    # 3) 声谱图（dB）
    S = np.abs(librosa.stft(y, n_fft=pitch_cfg.frame_length, hop_length=pitch_cfg.hop_length)) ** 2
    S_db = librosa.power_to_db(S, ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=pitch_cfg.frame_length)
    t_spec = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr, hop_length=pitch_cfg.hop_length)

    # 4) 统计概览（JSON）——✅ 保持原样：仍然沿用 voiced_ratio / f0_valid_ratio
    voiced_ratio = float(np.mean(voiced)) if len(voiced) > 0 else 0.0
    f0_valid_ratio = float(np.mean(np.isfinite(f0))) if len(f0) > 0 else 0.0

    f0_valid = f0[np.isfinite(f0)]
    if f0_valid.size > 0:
        f0_min = float(np.percentile(f0_valid, 5))
        f0_max = float(np.percentile(f0_valid, 95))
        f0_med = float(np.median(f0_valid))
    else:
        f0_min = f0_max = f0_med = float("nan")

    stats = {
        "duration_sec": round(duration_sec, 3),
        "sample_rate_hz": int(sr),
        "voiced_ratio": round(voiced_ratio, 4),
        "f0_valid_ratio": round(f0_valid_ratio, 4),
        # RMS dB（相对最大值）
        "rms_db_mean_rel": round(float(np.mean(rms_db)), 2),
        "rms_db_std_rel": round(float(np.std(rms_db)), 2),
        "f0_hz_p05": round(f0_min, 2) if np.isfinite(f0_min) else None,
        "f0_hz_median": round(f0_med, 2) if np.isfinite(f0_med) else None,
        "f0_hz_p95": round(f0_max, 2) if np.isfinite(f0_max) else None,
    }

    sheet_json = {
        "meta": {
            "stage": "audio_sheet",
            "sample_rate_hz": sr,
        },
        "stats": stats,
        "pitch_config": asdict(pitch_cfg),
        # JSON 里保留中英文总结：你看中文，评委系统也可读英文
        "summary_cn": _make_summary_cn(stats),
        "summary_en": _make_summary_en(stats),
    }

    sheet_data = {
        "y": y,
        "sr": sr,
        "t_wave": np.arange(len(y)) / sr,
        "t_rms": t_rms,
        "rms_db": rms_db,
        "t_f0": t_f0,
        "f0_hz": f0,
        "voiced": voiced,  # ✅ 保留（不影响旧统计/卡片信息）
        "t_pc": t_pc,      # ✅ 新增
        "pitch_conf": pitch_conf,  # ✅ 新增
        "S_db": S_db,
        "freqs": freqs,
        "t_spec": t_spec,
    }

    return sheet_data, sheet_json


def save_audio_sheet_png(
    out_png: str | Path,
    audio_name: str,
    sheet_data: Dict[str, Any],
    sheet_json: Dict[str, Any],
) -> None:
    """
    生成评委可读版本的 Audio Sheet 图（全英文文字，避免中文字体问题）。
    """
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    y = sheet_data["y"]
    t_wave = sheet_data["t_wave"]
    t_rms = sheet_data["t_rms"]
    rms_db = sheet_data["rms_db"]
    t_f0 = sheet_data["t_f0"]
    f0 = sheet_data["f0_hz"]
    voiced = sheet_data["voiced"]  # ✅ 保留（用于旧卡片统计，不用于第三板块绘图）
    S_db = sheet_data["S_db"]
    freqs = sheet_data["freqs"]
    t_spec = sheet_data["t_spec"]

    # ✅ 新增：第三板块要画的 Pitch Confidence
    t_pc = sheet_data["t_pc"]
    pitch_conf = sheet_data["pitch_conf"]

    stats = sheet_json["stats"]
    summary_en = sheet_json["summary_en"]

    # 频率显示上限：5kHz（评委可读 + 人声关键信息区）
    fmax_show = 5000

    fig = plt.figure(figsize=(12, 10))

    # Panel 1: Waveform
    ax1 = plt.subplot(5, 1, 1)
    ax1.plot(t_wave, y, linewidth=0.6)
    ax1.set_xlim(0, stats["duration_sec"])
    ax1.set_ylabel("Amplitude")
    ax1.set_title(
        f"Audio Sheet (Audio Overview) - {audio_name}\n"
        f"Summary: {summary_en}"
    )

    # Panel 2: Loudness over time (RMS, relative dB)
    ax2 = plt.subplot(5, 1, 2, sharex=ax1)
    ax2.plot(t_rms, rms_db, linewidth=1)
    ax2.set_ylabel("Loudness (dB, relative)")
    ax2.set_title("Loudness over Time (RMS)")

    # Panel 3: Pitch Confidence timeline (0~1)  ✅【唯一替换的面板】
    ax3 = plt.subplot(5, 1, 3, sharex=ax1)
    ax3.plot(t_pc, pitch_conf, linewidth=1.2)
    ax3.set_ylim(-0.02, 1.02)
    ax3.set_ylabel("Confidence")
    ax3.set_title("Pitch Confidence Timeline (0~1)  (higher = F0 is more trustworthy)")
    # 两条参考线（不改变其他结构，只是帮助你看）
    ax3.axhline(0.7, linestyle="--", linewidth=1.0)
    ax3.axhline(0.4, linestyle="--", linewidth=1.0)

    # Panel 4: Dominant pitch track (F0)
    ax4 = plt.subplot(5, 1, 4, sharex=ax1)
    ax4.plot(t_f0, f0, linewidth=1)
    ax4.set_ylabel("F0 (Hz)")
    ax4.set_title("Dominant Pitch Track (F0)")

    # Panel 5: Spectrogram + F0 overlay
    ax5 = plt.subplot(5, 1, 5, sharex=ax1)
    fmask = freqs <= fmax_show
    ax5.imshow(
        S_db[fmask, :],
        origin="lower",
        aspect="auto",
        extent=[t_spec[0], t_spec[-1], freqs[fmask][0], freqs[fmask][-1]],
    )
    ax5.plot(t_f0, f0, linewidth=1)
    ax5.set_ylim(0, fmax_show)
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Frequency (Hz)")
    ax5.set_title("Spectrogram (dB) with F0 overlay")

    # 图角落放一个英文“概览卡片”——✅ 保持不变（仍使用 voiced_ratio / valid f0）
    card = (
        f"Duration: {stats['duration_sec']} s | "
        f"Voiced ratio: {stats['voiced_ratio']*100:.1f}% | "
        f"Valid F0: {stats['f0_valid_ratio']*100:.1f}%\n"
        f"F0 range (5–95%): {stats['f0_hz_p05']}–{stats['f0_hz_p95']} Hz | "
        f"Median F0: {stats['f0_hz_median']} Hz"
    )
    fig.text(0.02, 0.005, card)

    # 图内加一个非常简短的“How to read”（英文）
    # ✅ 这里把 “voicing=...” 改成 “pitch confidence=...”，其余不动
    how_to_read = (
        "How to read: waveform=loudness/pauses; RMS=loudness stability; "
        "pitch confidence=how trustworthy F0 is; F0=pitch movement; "
        "spectrogram=harmonic clarity vs noise."
    )
    fig.text(0.02, 0.028, how_to_read)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(out_png, dpi=200)
    plt.close(fig)


def save_audio_sheet_json(out_json: str | Path, sheet_json: Dict[str, Any]) -> None:
    """
    保存 Audio Sheet JSON（可包含中文总结，便于你对照理解）。
    """
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(sheet_json, f, ensure_ascii=False, indent=2)
