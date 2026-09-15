# Model Artifacts

```text
models/
├── pretrained/   官方或其他初始化权重
├── trained/      带版本的训练与导出产物
└── yolo11s_roi.pt  当前推理入口使用的已批准权重
```

模型权重不会提交到仓库。每个训练模型应记录模型版本、数据集版本、训练配置、训练日期、代码版本、验证指标和审批状态。

推荐命名：

```text
yolo11s_ultrasound_roi_v1.pt
yolo11s_ultrasound_roi_v1.onnx
```

推理代码继续读取 `config/config.yaml` 的 `model.path`。默认稳定路径为 `models/yolo11s_roi.pt`。

## yolo11s_ultrasound_roi_v3

- 训练日期：2026-09-12
- 数据集：ultrasound_roi，190 张（CHISON + Samsung）
- 划分：train 135 / val 33 / test 22
- 初始化权重：yolo11s_ultrasound_roi_v2
- 训练策略：冻结主干，关闭 mosaic、平移、缩放、翻转及颜色增强
- Test Precision：0.9976
- Test Recall：1.0000
- Test mAP50：0.9950
- Test mAP50-95：0.9860
- 状态：本地部署

测试集包含 Samsung 彩色 B-mode 和 B/M 双面板样本，但同一 Samsung
序列的样本相关性较高；后续仍需补充独立设备和独立患者序列进行外部验证。

## yolo11s_ultrasound_roi_v4

- 训练日期：2026-09-14
- 数据集：ultrasound_roi_v4，1402 张
- 标注策略：只学习顶部患者信息边界，左右及底部保持完整
- 训练轮数：100
- Validation Precision：0.9998
- Validation Recall：1.0000
- Validation mAP50：0.9950
- Validation mAP50-95：0.9950
- 部署路径：`models/yolo11s_roi.pt`
- 旧模型备份：`models/yolo11s_roi.before_v4.pt`
- 状态：本地部署
