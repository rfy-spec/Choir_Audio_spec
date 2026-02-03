# -*- coding: utf-8 -*-
"""
命令行运行：
  python pitch_eval_repro/scripts/run_eval.py --audio /path/to/xxx.wav

输出（按音频名建子目录）：
  pitch_eval_repro/outputs/<audio_name>/f0_tracks.csv
  pitch_eval_repro/outputs/<audio_name>/f0_curve.png

例如输入：1_good.wav
输出：
  pitch_eval_repro/outputs/1_good/f0_tracks.csv
  pitch_eval_repro/outputs/1_good/f0_curve.png
"""

from __future__ import annotations

import os
import argparse
import numpy as np

import sys
import os
# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pitch_eval_repro.src.audio_io import load_audio_mono
from pitch_eval_repro.src.pipeline import extract_f0_per_frame

import json
from datetime import datetime


def save_csv(out_csv: str, times, f0, conf, note_names, f_ref_hz, cents) -> None:
    """
    保存每一帧的 time / f0 / confidence 到 CSV 文件
    """
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("time_sec,f0_hz,confidence,note_name,f_ref_hz,cents\n")
        for t, hz, c, nn, fr, ce in zip(times, f0, conf, note_names, f_ref_hz, cents):
            nn_safe = nn if nn is not None else "NA"
            ce_str = "" if (ce is None or (isinstance(ce, float) and np.isnan(ce))) else f"{ce:.6f}"
            f.write(f"{t:.6f},{hz:.6f},{c:.6f},{nn_safe},{fr:.6f},{ce_str}\n")


def plot_f0(out_png: str, times: np.ndarray, f0: np.ndarray, conf: np.ndarray) -> None:
    """
    画出“时间-音高Hz曲线”，并保存成 PNG
    """
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    import matplotlib.pyplot as plt

    # 把不可信（或者 f0<=0）的点隐藏掉，这样图更清晰
    f0_plot = f0.copy()
    f0_plot[(conf < 0.2) | (f0_plot <= 0)] = np.nan

    plt.figure()
    plt.plot(times, f0_plot)
    plt.xlabel("Time (sec)")
    plt.ylabel("F0 (Hz)")
    plt.title("Per-frame Pitch (F0) Curve")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_cents(out_png: str, times: np.ndarray, cents: np.ndarray) -> None:
    """
    画出“时间-cents偏差曲线”，并保存成 PNG（单独文件，不覆盖f0图）
    """
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    import matplotlib.pyplot as plt

    cents_plot = cents.copy()
    # 把 nan 的点隐藏掉
    # （nan 会自动让曲线断开，正好表示“这里没估到/不可信”）
    plt.figure()
    plt.plot(times, cents_plot)
    plt.axhline(0.0, linewidth=1.0)  # 0 cents 的基准线（不用指定颜色）
    plt.xlabel("Time (sec)")
    plt.ylabel("Cents (relative to nearest note)")
    plt.title("Per-frame Pitch Deviation (Cents)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_pitch_equal_temperament_colored(
    out_png: str,
    times: np.ndarray,
    f0: np.ndarray,
    cents: np.ndarray,
    confidence: np.ndarray,
) -> None:
    """
    正确的“十二平均律网格 + 上色点”画法：
    - 纵轴使用 MIDI（半音等间距），而不是 Hz
    - 只画有效点（f0>0 且 confidence>=0.2 且 cents不是nan）
    - 网格线每个半音一条（MIDI整数）
    - y轴标签只标每个八度的C（避免挤成一团）
    """
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    import matplotlib.pyplot as plt
    from pitch_eval_repro.src.pitch_units import freq_to_midi, midi_to_note_name

    # ===== 1) 过滤有效点：避免底部一排灰点（f0=0） =====
    mask = (f0 > 0) & (confidence >= 0.2) & (~np.isnan(cents))
    if np.sum(mask) < 5:
        print("有效点太少：无法生成十二平均律上色图（可能整段太安静或confidence过低）")
        return

    t = times[mask]
    f = f0[mask]
    ce = cents[mask]

    # ===== 2) 把频率(Hz)转换为 MIDI：这样每个半音在纵轴等间距 =====
    midi = np.array([freq_to_midi(float(x)) for x in f], dtype=np.float32)

    # ===== 3) 给每个点分配颜色（按 |cents| 分段）=====
    # 这里用你之前的分档：绿(<=5)、黄(<=20)、红(<=50)、其他灰
    colors = []
    for c in ce:
        ac = abs(float(c))
        if ac <= 5:
            colors.append("green")
        elif ac <= 20:
            colors.append("yellow")
        elif ac <= 50:
            colors.append("red")
        else:
            colors.append("gray")

    # ===== 4) 决定绘图的 MIDI 范围，并画“每半音一条网格线”=====
    midi_min = int(np.floor(np.min(midi))) - 2
    midi_max = int(np.ceil(np.max(midi))) + 2

    plt.figure()

    # 背景网格：每个半音（MIDI整数）一条横线
    for m in range(midi_min, midi_max + 1):
        plt.axhline(m, linewidth=0.6, alpha=0.25)

    # ===== 5) y轴标签只标每个八度的C（MIDI为12的倍数）=====
    y_ticks = []
    y_labels = []
    for m in range(midi_min, midi_max + 1):
        if m % 12 == 0:  # 每个八度的C
            y_ticks.append(m)
            y_labels.append(midi_to_note_name(m))
    plt.yticks(y_ticks, y_labels)

    # ===== 6) 画散点：横轴时间，纵轴MIDI，上色表示音准偏差 =====
    plt.scatter(t, midi, s=8, c=colors)

    plt.xlabel("Time (sec)")
    plt.ylabel("Pitch (MIDI) on Equal Temperament Grid")
    plt.title("Pitch on Equal Temperament (Colored by Cents Deviation)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def compute_stats_and_score(
    times: np.ndarray,
    f0_hz: np.ndarray,
    confidence: np.ndarray,
    cents: np.ndarray,
    conf_threshold: float = 0.2,
    sigma: float = 45.0,            # 软准确评分的“宽松程度”，越大越宽松（合唱建议30~45）
    stability_scale: float = 55.0,  # 稳定性衰减尺度，越大越宽松
    acc_weight: float = 0.85,       # 准确性权重（0~1），合唱建议0.75~0.85
) -> dict:
    """
    统计指标 + 软评分总分（满分100）
    统计只使用“可信 + 有音高 + cents有效”的帧
    """

    # 有效帧：有音高、置信度足够、cents不是nan
    mask = (f0_hz > 0) & (confidence >= conf_threshold) & (~np.isnan(cents))
    total_frames = int(len(times))
    valid_frames = int(np.sum(mask))

    if valid_frames < 5:
        return {
            "total_frames": total_frames,
            "valid_frames": valid_frames,
            "valid_ratio": float(valid_frames / max(total_frames, 1)),

            "ratio_within_20c": None,
            "ratio_within_50c": None,
            "mean_abs_cents": None,

            "soft_accuracy": None,
            "stability": None,
            "total_score": None,

            "conf_threshold": float(conf_threshold),
            "sigma": float(sigma),
            "stability_scale": float(stability_scale),
            "acc_weight": float(acc_weight),

            "note": "有效帧太少（可能音频太安静或confidence过低），统计与评分不可靠。",
        }

    cents_valid = cents[mask].astype(np.float32)
    abs_cents = np.abs(cents_valid)

    # ===== 基础统计（保留，便于解释）=====
    ratio_20 = float(np.mean(abs_cents <= 20.0))
    ratio_50 = float(np.mean(abs_cents <= 50.0))
    mean_abs = float(np.mean(abs_cents))

    # ===== 软准确率：每帧一个连续分数（0~1），偏差越小越接近1 =====
    # per_frame = exp(-(abs/sigma)^2)
    # 这样不会出现“超过20就完全不算准”的硬阈值
    soft_per_frame = np.exp(- (abs_cents / float(sigma)) ** 2)
    soft_accuracy = float(np.mean(soft_per_frame))

    # ===== 稳定性：仍然由平均绝对偏差决定（越小越接近1）=====
    stability = float(np.exp(- mean_abs / float(stability_scale)))

    # ===== 总分（满分100）=====
    w = float(acc_weight)
    if w < 0.0:
        w = 0.0
    if w > 1.0:
        w = 1.0
    total_score = 100.0 * (w * soft_accuracy + (1.0 - w) * stability)

    # 限制到 0~100
    if total_score < 0.0:
        total_score = 0.0
    if total_score > 100.0:
        total_score = 100.0

    return {
        "total_frames": total_frames,
        "valid_frames": valid_frames,
        "valid_ratio": float(valid_frames / max(total_frames, 1)),

        "ratio_within_20c": ratio_20,
        "ratio_within_50c": ratio_50,
        "mean_abs_cents": mean_abs,

        "soft_accuracy": soft_accuracy,
        "stability": stability,
        "total_score": float(total_score),

        "conf_threshold": float(conf_threshold),
        "sigma": float(sigma),
        "stability_scale": float(stability_scale),
        "acc_weight": float(acc_weight),
    }



def build_report_dict(
    audio_path: str,
    sr: int,
    frame_size: int,
    hop_size: int,
    fmin: float,
    fmax: float,
    stats: dict,
    outputs: dict,
) -> dict:
    """
    组织成一份“给人看也给程序用”的报告字典（可写入 JSON）
    """
    report = {
        "meta": {
            "audio_path": audio_path,
            "audio_name": os.path.splitext(os.path.basename(audio_path))[0],
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "params": {
            "sr": int(sr),
            "frame_size": int(frame_size),
            "hop_size": int(hop_size),
            "fmin": float(fmin),
            "fmax": float(fmax),
        },
        "stats_and_score": stats,
        "outputs": outputs,
        "explanation_for_beginner": {},
    }

        # ===== 给小白的解释（核心：分数怎么来的）=====
    s = stats
    if s.get("total_score") is None:
        report["explanation_for_beginner"] = {
            "summary": "本次音频可用于统计的有效帧太少，因此无法给出可靠的音准分数。",
            "how_to_fix": "请确认音频不是全程很小声/静音，或者适当降低 confidence 阈值再试。",
        }
        return report

    ratio_20 = s["ratio_within_20c"]
    mean_abs = s["mean_abs_cents"]
    soft_acc = s["soft_accuracy"]
    stab = s["stability"]
    score = s["total_score"]
    sigma = s["sigma"]
    stability_scale = s["stability_scale"]
    w = s["acc_weight"]

    report["explanation_for_beginner"] = {
        "summary": f"音准总分（满分100）：{score:.2f} 分。",
        "what_is_ratio_20": "±20 cents 内比例：传统意义上“基本准”的时间占比（越高越好）。",
        "what_is_mean_abs": "平均绝对偏差：整体离标准音有多远（越小越好）。",
        "what_is_soft_scoring": [
            "为了避免‘超过±20就完全算不准’太刻薄，我们采用软评分：偏差越小得分越高，偏差越大得分越低（连续变化）。",
            f"软准确率 soft_accuracy = 平均(exp(-(abs(cents)/{sigma})^2))，取值0~1，越接近1越准。",
        ],
        "how_score_is_computed": [
            f"稳定性 stability = exp(- mean_abs_cents / {stability_scale})，取值0~1，越接近1越稳定。",
            f"总分 = 100 × (acc_weight×soft_accuracy + (1-acc_weight)×stability)",
            f"其中 acc_weight = {w}。",
            f"代入本音频：soft_accuracy={soft_acc:.3f}，stability={stab:.3f} → 总分={score:.2f}",
        ],
        "how_to_read_colored_pitch_plot": [
            "彩色十二平均律图上：点越贴近某条标准音线，说明越准。",
            "绿色/黄色/红色表示偏差大小（越接近绿色越准）。",
        ],
    }


    return report











def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="音频文件路径，例如 wav/mp3")
    ap.add_argument("--sr", type=int, default=48000, help="重采样目标采样率，默认 48000")
    ap.add_argument("--frame", type=int, default=2048, help="每帧采样点数，默认 2048")
    ap.add_argument("--hop", type=int, default=512, help="帧移（步长），默认 512")
    ap.add_argument("--fmin", type=float, default=50.0, help="最低音高Hz，默认 50")
    ap.add_argument("--fmax", type=float, default=1000.0, help="最高音高Hz，默认 1000")
    args = ap.parse_args()

    # 1) 读取音频（转单声道、必要时重采样）
    y, sr = load_audio_mono(args.audio, target_sr=args.sr)

    # 2) 每帧估计 f0
    result = extract_f0_per_frame(
        y=y,
        sr=sr,
        frame_size=args.frame,
        hop_size=args.hop,
        fmin=args.fmin,
        fmax=args.fmax,
    )

    # 3) 根据输入音频文件名创建输出子目录
    audio_basename = os.path.basename(args.audio)      # 例如 1_good.wav
    audio_name, _ = os.path.splitext(audio_basename)   # 变成 1_good

    out_dir = os.path.join("pitch_eval_repro", "outputs", audio_name)
    out_csv = os.path.join(out_dir, "f0_tracks.csv")
    out_f0_png = os.path.join(out_dir, "f0_curve.png")
    out_cents_png = os.path.join(out_dir, "cents_curve.png")
    out_et_png = os.path.join(out_dir, "pitch_et_colored.png")


    # 4) 保存结果
    save_csv(
        out_csv,
        result["times"], result["f0_hz"], result["confidence"],
        result["note_name"], result["f_ref_hz"], result["cents"],
    )

    plot_f0(out_f0_png, result["times"], result["f0_hz"], result["confidence"])
    plot_cents(out_cents_png, result["times"], result["cents"])
    plot_pitch_equal_temperament_colored(
        out_et_png,
        result["times"],
        result["f0_hz"],
        result["cents"],
        result["confidence"],
    )



        # ===== 第3步：统计指标 + 打分 + 生成报告（控制台 + JSON）=====
    stats = compute_stats_and_score(
        times=result["times"],
        f0_hz=result["f0_hz"],
        confidence=result["confidence"],
        cents=result["cents"],
        conf_threshold=0.2,
    )

    outputs_map = {
        "csv_f0_tracks": out_csv,
        "png_f0_curve": out_f0_png,
        "png_cents_curve": out_cents_png,
        "png_pitch_et_colored": out_et_png,
    }

    report = build_report_dict(
        audio_path=args.audio,
        sr=sr,
        frame_size=args.frame,
        hop_size=args.hop,
        fmin=args.fmin,
        fmax=args.fmax,
        stats=stats,
        outputs=outputs_map,
    )

    # 1) 控制台打印（给人看）
    print("\n================ 音准评估报告（简版） ================")
    if stats.get("total_score") is None:
        print("有效帧太少：无法给出可靠分数。")
        print("提示：检查音频是否太小声/静音，或适当降低 confidence 阈值。")
    else:
        print(f"总分（满分100）：{stats['total_score']:.2f}")
        print(f"±20 cents 内比例：{stats['ratio_within_20c']*100:.1f}%")
        print(f"±50 cents 内比例：{stats['ratio_within_50c']*100:.1f}%")
        print(f"平均绝对偏差（cents）：{stats['mean_abs_cents']:.2f}")
        print(f"准确率（0~1，越大越准）：{stats['soft_accuracy']:.3f}")
        print(f"稳定性（0~1，越大越稳）：{stats['stability']:.3f}")
        print(f"软评分参数：sigma={stats['sigma']}, stability_scale={stats['stability_scale']}, acc_weight={stats['acc_weight']}")

    print("======================================================\n")

    # 2) 保存 JSON（给程序/留档）
    report_json_path = os.path.join(out_dir, "report.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("REPORT JSON:", report_json_path)




    # 5) 打印提示
    print("完成：已输出")
    print("CSV:", out_csv)
    print("PNG:", out_f0_png)
    print("说明：CSV 是每一帧的数值，PNG 是时间-音高曲线图。")
    print("CENTS PNG:", out_cents_png)
    print("ET PNG:", out_et_png)





if __name__ == "__main__":
    main()
# Command line entry point for pitch evaluation