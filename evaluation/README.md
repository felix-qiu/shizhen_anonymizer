# Ultrasound ROI Evaluation

## 评估前准备

测试图片和标签分别放在 `datasets/ultrasound_roi/images/test` 与 `datasets/ultrasound_roi/labels/test`。每张测试图片必须有且只有一个合法的 `ultrasound_roi` 标注，且测试集不得与训练集、验证集或其来源视频重叠。

如需按厂家和设备型号分析，在 `datasets/ultrasound_roi/metadata.json` 中提供：

```json
[
  {
    "image": "001.jpg",
    "manufacturer": "CHISON",
    "model": "SonoEye"
  }
]
```

元数据是可选的，缺失时按 `UNKNOWN` 分组。数据进入本项目之前应按适用的数据治理流程完成授权和去标识化；本阶段不执行 OCR 或 PHI 检测。

## 数据集校验

```bash
uv run python tools/validate_dataset.py
```

输出 `reports/invalid_samples.json`，检查缺失图片、缺失标签、空标注、非法字段及越界 bbox。存在错误时命令返回非零退出码。

## 数据集统计

```bash
uv run python tools/dataset_statistics.py
```

输出 `reports/dataset_statistics.json`，包含图片和标注数量、split 分布、分辨率、厂家、设备型号、ROI 面积比例及不可读取图片。

## 模型评估

直接运行模型推理：

```bash
uv run python evaluation/evaluate_roi.py \
  --model models/trained/yolo11s_ultrasound_roi_v1.pt
```

也可评估预先保存的预测：

```bash
uv run python evaluation/evaluate_roi.py \
  --predictions predictions.json
```

预测文件中的 bbox 使用原图像素坐标。`bbox` 为 `null` 或缺少某张测试图片的记录时，该图片按 FN 处理。

## 输出

```text
reports/
├── metrics.json
├── report.md
├── images/
└── errors/
    ├── FN/
    ├── LOW_IOU/
    ├── SMALL_ROI/
    ├── OUT_OF_RANGE/
    └── LOW_CONFIDENCE/
```

可视化中绿色框为人工 GT，红色框为模型预测。

ROI 覆盖率定义为 `GT 与预测框交集面积 / GT 面积`，用于衡量有效超声区域是否被完整保留。低置信、FN 和低 IoU 样本会进入敏感信息残留风险复核列表；该列表是人工审核优先级提示，不等同于 OCR 或 PHI 检测结论。

报告摘要示例：

```markdown
## Evaluation Summary

- Model: `yolo11s_ultrasound_roi_v1.pt`
- Test samples: 1200
- Precision: 97.80%
- Recall: 96.90%
- mAP50: 98.20%
- mAP50-95: 91.40%
```

以上数值仅用于说明报告格式，不代表当前仓库的真实模型结果。
