# scripts/run_pitch_reference.py
# =========================================================
# Step B2 入口脚本：画“内部参考音高中心”
# - 不打分
# - 只输出：参考中心 + 可视化 + 简短JSON
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
from pitch_eval.reference import compute_internal_pitch_reference, compute_cents_deviation, PitchReferenceConfig


def _out_dirs(audio_path: Path) -> tuple[Path, Path]:
    base = Path("outputs") / audio_path.stem
    rep = base / "reports"
    fig = base / "figures"
    rep.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)
    return rep, fig


def _plot_reference(
    out_png: Path,
    audio_name: str,
    times: np.ndarray,
    f0_raw: np.ndarray,
    f0_dom: np.ndarray,
    weights: np.ndarray,
    ref_hz: float | None,
):
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure(figsize=(12, 6))
    ax = plt.subplot(1, 1, 1)

    # 1) Raw F0：浅灰细线（背景）
    ax.plot(times, f0_raw, linewidth=0.8, alpha=0.35, label="Raw F0 (candidate)")

    # 2) Dominant F0：深色粗线（主角）
    ax.plot(times, f0_dom, linewidth=1.6, alpha=0.95, label="Dominant F0")

    # 3) Reference center：橙色虚线（基准）
    if ref_hz is not None and np.isfinite(ref_hz) and ref_hz > 0:
        ax.axhline(ref_hz, linestyle="--", linewidth=1.6, label=f"Reference center (ref≈{ref_hz:.1f} Hz)")

    # 4) 高亮评估区域（weights>0）
    eval_mask = weights > 0
    if len(times) > 1:
        diff = np.diff(eval_mask.astype(int))
        starts = list(np.where(diff == 1)[0] + 1)
        ends = list(np.where(diff == -1)[0] + 1)
        if eval_mask[0]:
            starts = [0] + starts
        if eval_mask[-1]:
            ends = ends + [len(eval_mask) - 1]
        for s, e in zip(starts, ends):
            ax.axvspan(times[s], times[e], alpha=0.12)

    ax.set_title(f"Step B2: Internal Pitch Reference - {audio_name}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Step B2: visualize internal pitch reference center.")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio")

    # 分析采样率（保持与你现在流程一致）
    parser.add_argument("--sr", type=int, default=22050, help="Target sample rate (default: 22050)")

    # Step B1 阈值（沿用 B1）
    parser.add_argument("--reliable", type=float, default=0.70, help="Reliable threshold (default: 0.70)")
    parser.add_argument("--usable", type=float, default=0.40, help="Usable threshold (default: 0.40)")
    parser.add_argument("--usable_weight", type=float, default=0.50, help="Usable region weight (default: 0.50)")

    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # 1) 读音频
    y, sr = load_audio(audio_path, AudioConfig(sr=args.sr))

    # 2) 提F0
    pitch_cfg = PitchConfig()
    pitch = extract_pitch_track(y, sr, pitch_cfg)
    f0 = pitch["f0_hz"]
    t_f0 = pitch["times_sec"]

    # 3) Pitch Confidence
    t_pc, pc, _parts = compute_pitch_confidence(
        y=y,
        sr=sr,
        f0_hz=f0,
        times_sec=t_f0,
        frame_length=pitch_cfg.frame_length,
        hop_length=pitch_cfg.hop_length,
        cfg=PitchConfidenceConfig(),
    )

    # 4) Step B1：PC -> weights
    wcfg = PitchWeightConfig(
        reliable_th=float(args.reliable),
        usable_th=float(args.usable),
        usable_weight=float(args.usable_weight),
    )
    wres = build_pitch_weights(t_pc, pc, wcfg)
    times = wres["times_sec"]
    weights = wres["weights"]

    # 5) Step B2：算参考中心
    rcfg = PitchReferenceConfig()
    ref_res = compute_internal_pitch_reference(times, f0[:len(times)], weights, rcfg)

    # 6) 输出 JSON（短、核心）
    reports_dir, figures_dir = _out_dirs(audio_path)
    out_json = reports_dir / f"{audio_path.stem}.pitch_ref.json"
    out_png = figures_dir / f"{audio_path.stem}.pitch_ref.png"

    export = {
        "meta": {
            "stage": "pitch_reference_step_b2",
            "audio": audio_path.name,
            "sample_rate_hz": sr,
        },
        "b1_config": wres["config"],
        "b1_coverage": wres["coverage"],
        "reference": ref_res,
        "explain_cn": (
            "参考中心=在‘会被评估的时间段（权重>0）’里，主导F0的稳健统计中心（log2域中位数）。"
            "后续音准偏差会相对这个中心计算（不是相对乐谱/标准音）。"
        ),
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False, indent=2)

    # 7) 输出图
    ref_hz = ref_res["ref_hz"]
    _plot_reference(out_png, audio_path.name, times, f0[:len(times)], f0[:len(times)], weights, ref_hz)

    print("========== Step B2: Internal Pitch Reference ==========")
    print(f"Audio        : {audio_path.name}")
    print(f"Output JSON  : {out_json}")
    print(f"Output PNG   : {out_png}")
    print("------------------------------------------------------")
    if ref_hz is None:
        print("[WARN] Not enough evaluated points to estimate reference.")
    else:
        print(f"Reference F0 : {ref_res['ref_hz']} Hz  (approx note: {ref_res['ref_note']})")
        print(f"Used points  : {ref_res['used_points_ratio']*100:.1f}% (of frames)")
        print(f"Spread (IQR) : {ref_res['spread_cents_iqr']} cents  (smaller = more stable)")
    print("=======================================================")


if __name__ == "__main__":
    main()
