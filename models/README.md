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

## 当前版本状态

- 稳定入口 `models/yolo11s_roi.pt`：v5。
- 最新训练产物：`models/trained/yolo11s_ultrasound_roi_v6.pt`，尚未替换稳定入口。
- Docker 默认打包稳定入口；挂载权重时以容器实际挂载文件为准。
- 权重文件应通过独立制品存储或部署挂载分发，不随文档提交。

## yolo11s_ultrasound_roi_v6

- 训练日期：2026-11-05；实际完成 100 轮。
- 初始化：`models/trained/yolo11s_ultrasound_roi_v5.pt`。
- 数据：导出 1926 对图片和标签，去重后 1820 张；包含全部 v5 的 1424 张及新增 396 张。
- 划分：train 1285 / val 356 / test 179，共 497 个文件名推导分组。
- 保留旧样本的 v5 集合归属，新分组按 70% / 20% / 10% 划分；分组规则仍依赖文件命名质量。
- 参数：`imgsz=640`、`batch=8`、`device=mps`。
- 运行目录：`runs/train_1820_top_boundary_v6`。
- 独立验证目录：`runs/val_1820_top_boundary_v6`。
- 验证集 Precision / Recall：1.0000 / 1.0000。
- 验证集 mAP50 / mAP50-95：0.9950 / 0.9950。
- 以上为 356 张验证集指标，不是 179 张测试集指标。
- SHA256：`322a3e7b9ff22e4063ad474146a5698a3859abe70cc758913341c2dfdbc036d7`。

三星 72 张、日立 147 张、GE LOGIQ E9 4 张历史问题图片均成功处理，
结果顶部 22% OCR 未检出姓名、编号、医院或日期时间残留。
部分图片参与过训练，因此这 223 张属于回归检查，不能作为独立泛化验证。

另一次跳过报告目录的全量检查覆盖 8716 张图片：8680 张裁切、30 张无需裁切、
6 张处理失败；8710 张结果的顶部 OCR 标记了 137 张疑似残留，仍需人工确认。
检测指标高不代表脱敏完整，v6 不应被描述为全量无残留。
该批处理与原图备份属于项目外的临时流程，不是目录 API 的默认覆盖行为。

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
