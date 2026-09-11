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

训练和验证命令会在运行时生成带绝对数据集根路径的临时 YAML，避免用户机器上的 Ultralytics 全局 `datasets_dir` 设置改变相对路径解析。仓库中的 `training/dataset.yaml` 保持可移植，不会被改写。

## 训练

项目默认配置位于 `config/train.yaml`：

```bash
python training/train.py
```

命令行参数可以覆盖配置：

```bash
python training/train.py \
  --model yolo11s.pt \
  --data training/dataset.yaml \
  --epochs 100 \
  --imgsz 640 \
  --batch 16
```

Ultralytics 会在首次使用模型名称时下载官方预训练权重。默认输出位于：

```text
runs/train/weights/best.pt
runs/train/weights/last.pt
```

如果同名运行目录已经存在，Ultralytics 可能创建带数字后缀的新目录；脚本会打印本次实际生成的最佳模型路径。

`config/train.yaml` 中的 `model.version` 用于导出产物命名。每次改变训练数据、关键超参数或模型结构时都应更新版本号。

## 验证

```bash
python training/validate.py \
  --model runs/train/weights/best.pt \
  --data training/dataset.yaml
```

命令会输出 `mAP50`、`mAP50-95`、Precision 和 Recall。模型上线前还应按设备厂家和设备型号分别检查指标，并人工审核漏裁、过度裁剪和 UI 信息残留样本。

## 导出

导出 ONNX：

```bash
python training/export.py \
  --model runs/train/weights/best.pt \
  --format onnx
```

保存 PyTorch 权重副本：

```bash
python training/export.py \
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
