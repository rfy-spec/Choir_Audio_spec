# scripts/run_pitch_mask.py
# =========================================================
# Step B1 入口脚本：生成“音准评估时间段/权重”结果
# - 不算音准分数
# - 只输出：pitch confidence -> weights -> segments
# =========================================================

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from choir_judge.io_audio import load_audio, AudioConfig
from pitch_eval.features import PitchConfig, extract_pitch_track
from choir_judge.pitch_confidence import compute_pitch_confidence, PitchConfidenceConfig
from pitch_eval.weights import build_pitch_weights, PitchWeightConfig


def _out_dirs(audio_path: Path) -> tuple[Path, Path]:
    base = Path("outputs") / audio_path.stem
    rep = base / "reports"
    fig = base / "figures"
    rep.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)
    return rep, fig


def _plot_pitch_mask(
    out_png: Path,
    audio_name: str,
    times: np.ndarray,
    pc: np.ndarray,
    weights: np.ndarray,
    segments: dict,
):
    fig = plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(times, pc, linewidth=1.2)
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_ylabel("Pitch Confidence")
    ax1.set_title(f"Pitch Confidence & Evaluation Weights - {audio_name}")

    # 标注阈值（英文）
    ax1.axhline(0.7, linestyle="--", linewidth=1.0)
    ax1.axhline(0.4, linestyle="--", linewidth=1.0)
    ax1.text(0.01, 0.72, "0.7: reliable", transform=ax1.transAxes, fontsize=9)
    ax1.text(0.01, 0.42, "0.4: usable (down-weight)", transform=ax1.transAxes, fontsize=9)

    # 用“可靠段”做高亮（便于你一眼看哪些会被重点评）
    for seg in segments.get("reliable", []):
        ax1.axvspan(seg["start"], seg["end"], alpha=0.15)

    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    ax2.plot(times, weights, linewidth=1.2)
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_ylabel("Evaluation Weight")
    ax2.set_xlabel("Time (s)")
    ax2.set_title("Evaluation Weight Timeline (0~1)")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Step B1: build evaluation time mask/weights from Pitch Confidence.")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio")
    parser.add_argument("--sr", type=int, default=22050, help="Target sample rate for analysis (default: 22050)")
    parser.add_argument("--reliable", type=float, default=0.70, help="Reliable threshold (default: 0.70)")
    parser.add_argument("--usable", type=float, default=0.40, help="Usable threshold (default: 0.40)")
    parser.add_argument("--usable_weight", type=float, default=0.50, help="Weight for usable region (default: 0.50)")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # 1) 读音频
    y, sr = load_audio(audio_path, AudioConfig(sr=args.sr))

    # 2) 提F0（dominant）
    pitch_cfg = PitchConfig()
    pitch = extract_pitch_track(y, sr, pitch_cfg)
    f0 = pitch["f0_hz"]
    t_f0 = pitch["times_sec"]

    # 3) 计算Pitch Confidence（0~1）
    t_pc, pc, _parts = compute_pitch_confidence(
        y=y,
        sr=sr,
        f0_hz=f0,
        times_sec=t_f0,
        frame_length=pitch_cfg.frame_length,
        hop_length=pitch_cfg.hop_length,
        cfg=PitchConfidenceConfig(),
    )

    # 4) PC -> weights + segments（这就是 Step B1 的核心产物）
    wcfg = PitchWeightConfig(
        reliable_th=float(args.reliable),
        usable_th=float(args.usable),
        usable_weight=float(args.usable_weight),
    )
    result = build_pitch_weights(t_pc, pc, wcfg)

    # 5) 输出 JSON（简短、核心）
    reports_dir, figures_dir = _out_dirs(audio_path)
    out_json = reports_dir / f"{audio_path.stem}.pitch_mask.json"
    out_png = figures_dir / f"{audio_path.stem}.pitch_mask.png"

    export = {
        "meta": {
            "stage": "pitch_mask_step_b1",
            "audio": audio_path.name,
            "sample_rate_hz": sr,
        },
        "config": result["config"],
        "coverage": result["coverage"],
        "segments": result["segments"],  # 关键：可解释的区间列表
        # 不把每一帧都写进JSON，避免九千行
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False, indent=2)

    # 6) 输出图（英文标注）
    _plot_pitch_mask(
        out_png=out_png,
        audio_name=audio_path.name,
        times=result["times_sec"],
        pc=result["pitch_confidence_smooth"],
        weights=result["weights"],
        segments=result["segments"],
    )

    print("========== Step B1: Pitch Mask/Weights ==========")
    print(f"Audio              : {audio_path.name}")
    print(f"Output JSON        : {out_json}")
    print(f"Output Figure      : {out_png}")
    cov = export["coverage"]
    print("-----------------------------------------------")
    print(f"Reliable ratio     : {cov['reliable_ratio']*100:.1f}%")
    print(f"Usable ratio       : {cov['usable_ratio']*100:.1f}%")
    print(f"Evaluated ratio    : {cov['evaluated_ratio']*100:.1f}%")
    print(f"Not evaluated ratio: {cov['not_evaluated_ratio']*100:.1f}%")
    print("===============================================")


if __name__ == "__main__":
    main()
