# 合唱音频音高评估系统

一个用于评估合唱音频录音中音高准确性的Python项目。

## 项目结构

- `data/raw/` - 原始音频文件存放目录
- `data/example/` - 测试用示例音频文件
- `src/choir_judge/` - 音频处理核心代码
- `src/pitch_eval/` - 音高评估功能模块
- `scripts/` - 可执行脚本文件
- `outputs/` - 生成的报告和图表
- `tests/` - 测试文件

## 安装说明

```bash
pip install -r requirements.txt
```

## 使用方法

```bash
python scripts/run_pitch_eval.py
```