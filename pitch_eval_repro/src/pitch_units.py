# -*- coding: utf-8 -*-
"""
音高单位换算工具：
- 频率(Hz) ↔ MIDI
- 频率(Hz) → 最近十二平均律标准音
- 计算 cents 偏差
"""

from __future__ import annotations
import numpy as np

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def freq_to_midi(f_hz: float) -> float:
    """
    把频率(Hz)转换为 MIDI 数值（可以是小数）
    参考：A4 = 440Hz 对应 MIDI=69
    """
    if f_hz <= 0:
        return float("nan")
    return 69.0 + 12.0 * np.log2(f_hz / 440.0)

def midi_to_freq(m: float) -> float:
    """
    把 MIDI 数值转换为频率(Hz)
    """
    return 440.0 * (2.0 ** ((m - 69.0) / 12.0))

def midi_to_note_name(m_int: int) -> str:
    """
    把“整数 MIDI”转成音名，例如 69 -> A4
    """
    note = NOTE_NAMES[m_int % 12]
    octave = (m_int // 12) - 1
    return f"{note}{octave}"

def nearest_equal_temperament(f_hz: float) -> tuple[str, float, int]:
    """
    给定频率，找到最近的十二平均律标准音
    返回：
      note_name: 例如 A4
      f_ref_hz: 最近标准音的频率
      midi_int: 最近标准音对应的整数 MIDI
    """
    if f_hz <= 0:
        return "NA", 0.0, -1
    m = freq_to_midi(f_hz)
    m_int = int(np.round(m))
    f_ref = float(midi_to_freq(m_int))
    return midi_to_note_name(m_int), f_ref, m_int

def cents_off(f_hz: float, f_ref_hz: float) -> float:
    """
    计算 cents 偏差：
      >0 表示偏高
      <0 表示偏低
    """
    if f_hz <= 0 or f_ref_hz <= 0:
        return float("nan")
    return float(1200.0 * np.log2(f_hz / f_ref_hz))
