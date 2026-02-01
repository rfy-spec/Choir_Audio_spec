# scripts/run_pitch_eval.py
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# 将src目录添加到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from choir_judge.io_audio import load_audio, AudioConfig
from choir_judge.quality_gate import compute_choir_quality

from pitch_eval.features import PitchConfig, extract_pitch_track
from pitch_eval.scoring import PitchScoreConfig, score_pitch, hz_to_midi, midi_to_hz, hz_to_cents
from pitch_eval.report import save_pitch_json, save_pitch_figure





# ====== 质量门控的人话解释（你已经在用的逻辑）======
def _rms_comment(rms_db: float) -> str:
    if -25.0 <= rms_db <= -15.0:
        return "非常理想的人声录音区间"
    if rms_db < -35.0:
        return "音量偏小，可能离麦太远或录得太轻"
    if rms_db > -6.0:
        return "音量偏大，可能接近过载"
    return "音量基本合适"


def _clip_comment(clip_ratio: float) -> str:
    if clip_ratio <= 1e-6:
        return "完全没有爆音"
    if clip_ratio < 0.001:
        return "基本没有爆音"
    return "存在爆音/削顶风险"


def _silence_comment(silence_ratio: float) -> str:
    if silence_ratio < 0.2:
        return "大部分时间都在有效演唱"
    if silence_ratio < 0.4:
        return "静音段略多，但仍可接受"
    return "静音段偏多，可能包含大量空段/非演唱段"


def _zcr_comment(zcr_mean: float) -> str:
    if 0.08 <= zcr_mean <= 0.15:
        return "正常人声 + 少量摩擦音"
    if zcr_mean > 0.2:
        return "噪声/擦音偏多"
    return "相对纯净/有音高"


def _flatness_comment(flatness: float) -> str:
    if flatness < 0.05:
        return "非常像歌声"
    if flatness < 0.2:
        return "整体较像歌声"
    if flatness < 0.4:
        return "可能有噪声污染"
    return "偏噪声化，清晰度可能不足"


def main():
    parser = argparse.ArgumentParser(description="Choir evaluation: quality gate + pitch (traditional)")
    parser.add_argument("--audio", type=str, required=True, help="Path to choir audio file")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")

    # 1) 读音频
    y, sr = load_audio(audio_path, AudioConfig())

    # 2) 质量门控
    q = compute_choir_quality(y=y, sr=sr, f0_hz=None, voiced_flag=None)

    rms_db = q["rms_db"]
    clip_ratio = q["clip_ratio"]
    silence_ratio = q["silence_ratio"]
    zcr_mean = q["zcr_mean"]
    flat_mean = q["spectral_flatness_mean"]
    conf = q["global_confidence"]

    decision = "ACCEPTABLE" if conf >= 0.45 else "REJECTED"

    summary_quality = (
        "这段合唱录音音量合适、不爆音、绝大多数时间在唱歌、杂音很少、声音形态非常像真正的歌声，因此非常适合进行后续的音准评估，评估结果可信度高。"
        if decision == "ACCEPTABLE"
        else
        "该音频存在明显质量风险，建议改善录音条件后再进行评估。"
    )

    print("\n========== Choir Audio Quality Gate ==========")
    print(f"Audio file        : {audio_path.name}")
    print(f"Sample rate (Hz)  : {sr}")
    print("----------------------------------------------")
    print(f"Global confidence : {conf:.3f}  (0~1)")
    print("----------------------------------------------")
    print(f"RMS level         : {rms_db:.1f} dBFS（{_rms_comment(rms_db)}）")
    print(f"Clip ratio        : {clip_ratio*100:.2f} %（{_clip_comment(clip_ratio)}）")
    print(f"Silence ratio     : {silence_ratio*100:.1f} %（{_silence_comment(silence_ratio)}）")
    print(f"ZCR mean          : {zcr_mean:.3f}（{_zcr_comment(zcr_mean)}）")
    print(f"Spectral flatness : {flat_mean:.3f}（{_flatness_comment(flat_mean)}）")
    print("----------------------------------------------")
    print(f"[RESULT] {'✅' if decision=='ACCEPTABLE' else '❌'} Audio quality is {decision}.")
    print(f"最后总结：{summary_quality}")
    print("==============================================\n")

    # 写 quality JSON（保持你之前的约定）
    quality_report = {
        "meta": {
            "audio_file": audio_path.name,
            "sample_rate_hz": sr,
            "evaluated_at": datetime.now().isoformat(),
            "stage": "quality_gate"
        },
        "overall": {
            "global_confidence": round(conf, 3),
            "decision": decision,
            "summary": summary_quality
        },
        "metrics": {
            "rms_dbfs": {"value": round(rms_db, 2), "unit": "dBFS", "interpretation": _rms_comment(rms_db)},
            "clip_ratio": {"value": round(clip_ratio * 100, 3), "unit": "percent", "interpretation": _clip_comment(clip_ratio)},
            "silence_ratio": {"value": round(silence_ratio * 100, 2), "unit": "percent", "interpretation": _silence_comment(silence_ratio)},
            "zcr_mean": {"value": round(zcr_mean, 4), "unit": "ratio", "interpretation": _zcr_comment(zcr_mean)},
            "spectral_flatness_mean": {"value": round(flat_mean, 4), "unit": "ratio", "interpretation": _flatness_comment(flat_mean)},
        },
        "warnings": q["warnings"]
    }
    out_reports = Path("outputs/reports")
    out_reports.mkdir(parents=True, exist_ok=True)
    quality_path = out_reports / f"{audio_path.stem}.quality.json"
    with open(quality_path, "w", encoding="utf-8") as f:
        json.dump(quality_report, f, ensure_ascii=False, indent=2)
    print(f"[JSON SAVED] {quality_path.as_posix()}")

    # 质量不合格：停止音准评估（避免不可信）
    if decision != "ACCEPTABLE":
        return

    # =========================
    # 3) 音准评估（传统声学规则）
    # =========================
    pitch_feat = extract_pitch_track(y, sr, PitchConfig())

    f0 = pitch_feat["f0_hz"]
    voiced = pitch_feat["voiced_flag"]
    times = pitch_feat["times_sec"]

    # 把 f0 转成 cents 偏差轨迹（全帧，NaN保持）
    cents_track = np.full_like(times, np.nan, dtype=float)
    m = voiced.astype(bool) & np.isfinite(f0)
    if np.sum(m) > 0:
        midi = hz_to_midi(f0[m])
        midi_near = np.round(midi)
        ref = midi_to_hz(midi_near)
        cents_track[m] = hz_to_cents(f0[m], ref)

    pitch_result = score_pitch(times, f0, voiced, PitchScoreConfig())

    # 保存 pitch JSON
    pitch_payload = {
        "meta": {
            "audio_file": audio_path.name,
            "evaluated_at": datetime.now().isoformat(),
            "stage": "pitch_eval_traditional",
            "method": {
                "f0_extractor": "librosa.pyin (pYIN)",
                "deviation_unit": "cents vs nearest semitone (equal temperament)"
            }
        },
        "result": pitch_result
    }

    pitch_json_path = out_reports / f"{audio_path.stem}.pitch.json"
    save_pitch_json(pitch_json_path, pitch_payload)
    print(f"[JSON SAVED] {pitch_json_path.as_posix()}")

    # 保存图：一眼看出哪里偏高/偏低/严重
    out_fig = Path("outputs/figures")
    out_fig.mkdir(parents=True, exist_ok=True)
    fig_path = out_fig / f"{audio_path.stem}.pitch.png"
    save_pitch_figure(
        fig_path,
        times_sec=times,
        cents_track=cents_track,
        title=f"Pitch deviation (cents) - {audio_path.name}"
    )
    print(f"[FIG SAVED] {fig_path.as_posix()}")

    # 终端给一段“人话结论 + 最差片段”
    if pitch_result.get("ok"):
        print("\n========== Pitch Evaluation (Traditional) ==========")
        print(f"总分: {pitch_result['score']['final']} / 100")
        print(f"结论: {pitch_result['verdict']}")
        print("怎么得分：平均偏差(accuracy) + 严重跑调比例(outlier) + 漂移(drift) 加权得到。")
        print("最需要关注的时间段（Top 5）：")
        for seg in pitch_result["worst_segments"][:5]:
            print(f" - {seg['start_sec']}s~{seg['end_sec']}s："
                  f"{seg['label']}，中位绝对偏差 {seg['median_abs_cents']}c，{seg['direction']}")
        print("===============================================\n")
    else:
        print("\n[Pitch Eval STOP] " + pitch_result.get("reason", "无法评估"))


if __name__ == "__main__":
    main()
