# scripts/run_spectrogram_vis.py
# =========================================================
# 声谱图专项可视化入口脚本
#
# 用法：
#   python scripts/run_spectrogram_vis.py --audio data/example/1_good.wav
#
# 输出：
#   outputs/{stem}/figures/
#     - {stem}.spectrogram.png
#     - {stem}.spectrogram_f0.png
# =========================================================

from __future__ import annotations

import argparse
from pathlib import Path
import os
import sys

# 将src目录添加到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from choir_judge.io_audio import load_audio, AudioConfig
from choir_judge.spectrogram_vis import save_spectrogram


def main():
    parser = argparse.ArgumentParser(description="Spectrogram visualization (with / without F0)")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # 读取音频
    y, sr = load_audio(audio_path, AudioConfig())

    # 输出目录
    out_dir = Path("outputs") / audio_path.stem / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 无 F0 的声谱图
    out_plain = out_dir / f"{audio_path.stem}.spectrogram.png"
    save_spectrogram(
        y=y,
        sr=sr,
        out_png=out_plain,
        audio_name=audio_path.name,
        overlay_f0=False,
    )

    # 2) 叠加 F0 的声谱图
    out_f0 = out_dir / f"{audio_path.stem}.spectrogram_f0.png"
    save_spectrogram(
        y=y,
        sr=sr,
        out_png=out_f0,
        audio_name=audio_path.name,
        overlay_f0=True,
    )

    print(f"[SAVED] {out_plain}")
    print(f"[SAVED] {out_f0}")


if __name__ == "__main__":
    main()
