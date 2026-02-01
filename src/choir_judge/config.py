"""
合唱音频评估配置设置
"""

# 音频处理设置
SAMPLE_RATE = 22050
HOP_LENGTH = 512
FRAME_LENGTH = 2048

# 音高检测设置
PITCH_MIN = 80.0  # Hz
PITCH_MAX = 800.0  # Hz

# 文件路径
DATA_DIR = "data"
RAW_DATA_DIR = "data/raw"
EXAMPLE_DATA_DIR = "data/example"
OUTPUT_DIR = "outputs"
REPORTS_DIR = "outputs/reports"
FIGURES_DIR = "outputs/figures"