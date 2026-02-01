# scripts/run_audio_sheet.py
# =========================================================
# 生成 Audio Sheet（音频试卷底稿）
#
# 用法：
#   python scripts/run_audio_sheet.py --audio data/example/1_good.wav
#
# 输出：
#   outputs/{stem}/reports/{stem}.audio_sheet.json
#   outputs/{stem}/figures/{stem}.audio_sheet.png
# =========================================================

from __future__ import annotations

import argparse
from pathlib import Path
import os
import sys


# 将src目录添加到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


from choir_judge.io_audio import load_audio, AudioConfig
from choir_judge.audio_sheet import build_audio_sheet, save_audio_sheet_png, save_audio_sheet_json


def main():
    parser = argparse.ArgumentParser(description="Generate Audio Sheet (judge-friendly)")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")

    # 读取音频（与项目统一配置一致）
    y, sr = load_audio(audio_path, AudioConfig())

    # 构建底稿数据
    sheet_data, sheet_json = build_audio_sheet(y=y, sr=sr)

    # 输出路径：outputs/{stem}/reports + outputs/{stem}/figures
    base_dir = Path("outputs") / audio_path.stem
    reports_dir = base_dir / "reports"
    figures_dir = base_dir / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    out_json = reports_dir / f"{audio_path.stem}.audio_sheet.json"
    out_png = figures_dir / f"{audio_path.stem}.audio_sheet.png"

    save_audio_sheet_json(out_json, sheet_json)
    save_audio_sheet_png(out_png, audio_path.name, sheet_data, sheet_json)

    print(f"[JSON SAVED] {out_json.as_posix()}")
    print(f"[FIG SAVED ] {out_png.as_posix()}")


if __name__ == "__main__":
    main()
