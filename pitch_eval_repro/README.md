# Pitch Evaluation Reproduction Module

复现师姐音准评估的独立模块

## 输入 (Input)
- 音频文件路径 (Audio file path)

## 输出 (Output)  
- 音准评估结果 (Pitch evaluation results)
- 自动生成的结果文件：json/png/csv

## 运行方式 (How to Run)
```bash
python scripts/run_eval.py --input <audio_file_path>
```

## 目录结构 (Directory Structure)

```
pitch_eval_repro/                    # 复现师姐音准评估的独立模块
├── __init__.py
├── README.md                        # 写清楚：输入是什么，输出是什么，怎么跑
├── requirements_optional.txt        # 如果需要额外依赖，写在这里（不强制装）
├── scripts/
│   └── run_eval.py                  # 命令行入口：给音频路径→输出评估结果
├── src/
│   ├── audio_io.py                  # 读音频、重采样、转成numpy数组
│   ├── pitch_estimator.py           # "估计音高频率"的核心（对应师姐 pitch_analyzer）
│   ├── pipeline.py                  # 切帧→估频率→算cents→统计（对应 analysis_pipeline）
│   ├── render.py                    # 可选：画图（对应 canvas_render）
│   └── color_map.py                 # 可选：偏差→颜色（对应 color_map）
└── outputs/                         # 自动生成结果：json/png/csv
    └── .gitkeep
```

## 模块说明 (Module Description)

### 核心模块 (Core Modules)
- **audio_io.py**: 负责音频文件的读取、重采样和转换为numpy数组
- **pitch_estimator.py**: 音高频率估计的核心算法，对应原始代码中的pitch_analyzer
- **pipeline.py**: 完整的分析流水线，包括帧切分、频率估计、cents计算和统计分析，对应analysis_pipeline

### 可选模块 (Optional Modules)  
- **render.py**: 可视化和绘图功能，对应canvas_render
- **color_map.py**: 音准偏差到颜色的映射，对应color_map

### 脚本和配置 (Scripts & Config)
- **scripts/run_eval.py**: 命令行入口点，提供简单的接口来运行音准评估
- **requirements_optional.txt**: 可选依赖列表，不强制安装