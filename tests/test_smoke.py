"""
choir_judge包的冒烟测试
"""

import pytest
import numpy as np
import sys
import os

# 为测试添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from choir_judge.config import EXAMPLE_DATA_DIR, OUTPUT_DIR
from pitch_eval.features import extract_pitch_features
from pitch_eval.scoring import calculate_pitch_score


def test_extract_pitch_features():
    """测试音高特征提取是否能处理合成音频"""
    # 创建一个简单的正弦波
    sr = 22050
    duration = 1.0  # 1秒
    frequency = 440.0  # A4音
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * frequency * t)
    
    features = extract_pitch_features(audio, sr)
    
    # 检查是否获得了预期的特征
    assert 'pitch_track' in features
    assert 'pitch_mean' in features
    assert 'pitch_std' in features
    assert 'pitch_range' in features
    assert 'voiced_frames' in features
    assert 'total_frames' in features
    
    # 检查音高是否大致正确 (在合理范围内)
    assert 400 < features['pitch_mean'] < 500  # 应该在440 Hz左右


def test_calculate_pitch_score():
    """测试音高评分函数"""
    # 模拟特征
    features = {
        'pitch_track': np.array([440, 441, 439, 440, 442]),
        'pitch_mean': 440.4,
        'pitch_std': 1.2,
        'pitch_range': 3.0,
        'voiced_frames': 5,
        'total_frames': 5
    }
    
    scores = calculate_pitch_score(features, reference_pitch=440.0)
    
    # 检查是否获得了预期的评分
    assert 'stability_score' in scores
    assert 'accuracy_score' in scores
    assert 'overall_score' in scores
    
    # 评分应该在0到1之间
    for score in scores.values():
        assert 0 <= score <= 1


def test_empty_audio():
    """测试处理空音频或静音音频"""
    # 创建静音音频
    sr = 22050
    audio = np.zeros(sr)  # 1秒静音
    
    features = extract_pitch_features(audio, sr)
    scores = calculate_pitch_score(features)
    
    # 应该能够优雅处理而不崩溃
    assert isinstance(features, dict)
    assert isinstance(scores, dict)