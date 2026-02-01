# src/choir_judge/__init__.py
# =========================================================
# choir_judge 包的对外接口（公共 API）
#
# 说明：
# - 这里只导出“稳定、通用”的函数
# - 避免把实验性代码放进来，防止 import 时出错
# =========================================================

from .io_audio import AudioConfig, load_audio, save_audio
from .quality_gate import compute_choir_quality

__all__ = [
    "AudioConfig",
    "load_audio",
    "save_audio",
    "compute_choir_quality",
]
