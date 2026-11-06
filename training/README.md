# Ultrasound ROI Model Training

## 数据集准备

数据集使用 YOLO Detection 格式，且只允许一个类别：

```text
0: ultrasound_roi
```

目录结构：

```text
datasets/ultrasound_roi/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

每张图片必须有同名 `.txt` 标签。例如 `images/train/001.jpg` 对应 `labels/train/001.txt`。每行标签格式为：

```text
class_id x_center y_center width height
```

所有坐标和尺寸均为相对于图片宽高归一化后的 `0–1` 数值：

```text
0 0.52 0.48 0.65 0.70
```

建议按设备厂家、型号和检查类型分层划分训练集与验证集，避免同一患者或同一视频的相邻帧同时出现在两个集合中，造成数据泄漏。

训练和验证命令会在运行时生成带绝对数据集根路径的临时 YAML，避免用户机器上的 Ultralytics 全局 `datasets_dir` 设置改变相对路径解析。仓库中的 `datasets/ultrasound_roi/dataset.yaml` 保持可移植，不会被改写。

## 训练

项目默认配置位于 `config/train.yaml`：

```bash
uv run python training/train.py
```

命令行参数可以覆盖配置：

```bash
uv run python training/train.py \
  --model yolo11s.pt \
  --data datasets/ultrasound_roi/dataset.yaml \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --device auto
```

`--device auto` 会依次选择 NVIDIA CUDA、Apple Silicon MPS 或 CPU，也可以显式传入 `mps`、`cuda` 或 `cpu`。

Ultralytics 会在首次使用模型名称时下载官方预训练权重。默认输出位于：

```text
runs/train/weights/best.pt
runs/train/weights/last.pt
```

如果同名运行目录已经存在，Ultralytics 可能创建带数字后缀的新目录；脚本会打印本次实际生成的最佳模型路径。

`config/train.yaml` 中的 `model.version` 用于导出产物命名。每次改变训练数据、关键超参数或模型结构时都应更新版本号。

## 验证

```bash
uv run python training/validate.py \
  --model runs/train/weights/best.pt \
  --data datasets/ultrasound_roi/dataset.yaml
```

命令会输出 `mAP50`、`mAP50-95`、Precision 和 Recall。模型上线前还应按设备厂家和设备型号分别检查指标，并人工审核漏裁、过度裁剪和 UI 信息残留样本。

## 导出

导出 ONNX：

```bash
uv run python training/export.py \
  --model runs/train/weights/best.pt \
  --format onnx
```

保存 PyTorch 权重副本：

```bash
uv run python training/export.py \
  --model runs/train/weights/best.pt \
  --format pt
```

默认导出到 `models/trained`，文件名包含配置中的模型版本，例如 `yolo11s_ultrasound_roi_v1.onnx`。可以使用 `--version` 和 `--output-dir` 覆盖。

## 部署模型替换

保留带版本的训练产物，例如：

```text
models/trained/yolo11s_ultrasound_roi_v1.pt
```

完成独立验证后，将批准上线的权重复制为推理入口使用的稳定文件名：

```bash
cp models/trained/yolo11s_ultrasound_roi_v1.pt models/yolo11s_roi.pt
```

替换前应保留旧模型、训练配置、数据集版本和验证指标，以便回滚和复现。不要直接使用 `last.pt` 替换生产模型。

## v6 训练记录（2026-11-05）

AnyLabeling 导出使用 YOLO Detection，类别仅为 `ultrasound_roi`；保留文件名，
导出时不做随机拆分，合并和去重后统一分组。标注框顶部是期望保留区域的起点；
本项目数据准备流程将左右和底边固定到图像边界，只学习顶部裁切位置。

本次导出已包含旧数据，无需再次叠加 v5：1926 对图片和标签经 SHA256 去重后为
1820 张，旧 v5 的 1424 张全部保留，新增 396 张。
`datasets/ultrasound_roi_v6/manifest.json` 记录来源、哈希、分组、集合和顶部坐标；
`summary.json` 记录统计和划分策略。旧数据保持 v5 集合归属，新病例分组后划分，
最终为 train 1285 / val 356 / test 179。

上述保留 v5 集合归属的准备步骤使用了本次临时脚本；
通用 `tools/prepare_roi_dataset.py` 会重新划分全部分组，不能直接替代该策略。
病例图片、来源清单和临时脚本不纳入本次提交。

在数据集及 v5 权重已经准备好的环境中，训练命令为：

```bash
uv run python training/train.py \
  --model models/trained/yolo11s_ultrasound_roi_v5.pt \
  --data datasets/ultrasound_roi_v6/dataset.yaml \
  --epochs 100 --imgsz 640 --batch 8 --device mps \
  --output-dir runs --name train_1820_top_boundary_v6
```

上面的运行名是本次已完成实验；重复训练请使用新名称，避免混淆产物。
CPU 机器使用 `--device cpu`。本次后台执行使用 `caffeinate -i` 防止空闲睡眠，
其训练后验证和导出串联由临时流程完成，`training/train.py` 本身不会自动执行这些步骤。

查看本次日志：

```bash
tail -f logs/train_1820_top_boundary_v6.log
```

验证、版本化导出命令：

```bash
uv run python training/validate.py \
  --model runs/train_1820_top_boundary_v6/weights/best.pt \
  --data datasets/ultrasound_roi_v6/dataset.yaml --imgsz 640

uv run python training/export.py \
  --model runs/train_1820_top_boundary_v6/weights/best.pt \
  --format pt --version v6 --output-dir models/trained
```

本次独立验证结果保存在 `runs/val_1820_top_boundary_v6`，
指标摘要保存在 `logs/train_1820_top_boundary_v6_validation.json`。
手动验证命令默认由 Ultralytics 选择新验证目录，不保证使用上述目录名。
本次模型指标、SHA256 和回测限制见 [模型记录](../models/README.md)。

部署时可挂载 v6 到 `/app/models/yolo11s_roi.pt`，并重新创建容器；
也可以备份当前稳定权重后，再复制 v6 到稳定入口并重启服务。
仅生成 `models/trained/yolo11s_ultrasound_roi_v6.pt` 不会自动切换线上模型。
