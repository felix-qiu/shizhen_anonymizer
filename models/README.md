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
