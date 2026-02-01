# src/pitch_eval/report.py
# =========================================================
# 音准评估 - 报告输出（JSON + 图）
# =========================================================

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt


def save_pitch_json(out_path: str | Path, payload: Dict[str, Any]) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_pitch_figure(out_path: str | Path, times_sec: np.ndarray, cents_track: np.ndarray, title: str) -> None:
    """
    画一张“音准偏差随时间变化”的图：
    - 纵轴：cents（正=偏高，负=偏低）
    - 横轴：时间
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(times_sec, cents_track, linewidth=1)
    plt.axhline(0, linewidth=1)
    plt.axhline(15, linestyle="--", linewidth=1)
    plt.axhline(-15, linestyle="--", linewidth=1)
    plt.axhline(30, linestyle="--", linewidth=1)
    plt.axhline(-30, linestyle="--", linewidth=1)
    plt.axhline(50, linestyle="--", linewidth=1)
    plt.axhline(-50, linestyle="--", linewidth=1)
    plt.xlabel("时间 (秒)")
    plt.ylabel("音准偏差 (cents)  正=偏高 / 负=偏低")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
