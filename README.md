# shizhen_anonymizer

## 项目介绍

`shizhen_anonymizer` 是基于 YOLO11 的超声图片和视频 ROI 检测与标准化预处理模块。系统检测 `ultrasound_roi`，裁剪有效成像区域，并通过等比例缩放与黑边填充输出标准尺寸媒体。

业务流水线仅依赖 `ROIDetector` 抽象接口，后续可扩展其他检测器，而不需要修改裁剪和图片处理逻辑。

## 环境安装

项目使用 uv 管理 Python 3.11、虚拟环境和依赖。首次安装：

```bash
uv python install 3.11
uv sync
```

`uv sync` 会根据 `.python-version` 和 `uv.lock` 自动创建 `.venv`，无需手动运行 `python -m venv` 或 `pip install`。在 CI 和部署环境中使用 `uv sync --locked`，确保锁文件未发生漂移。

常用依赖管理命令：

```bash
# 添加运行依赖
uv add <package>

# 添加开发依赖
uv add --dev <package>

# 更新并重新锁定依赖
uv lock --upgrade

# 在项目环境中执行命令
uv run <command>
```

## 模型准备

将训练完成、且包含 `ultrasound_roi` 类别的 Ultralytics YOLO11 权重放在：

```text
models/yolo11s_roi.pt
```

仓库不附带模型权重。模型路径、运行设备、置信度阈值和输出尺寸均在 `config/config.yaml` 中配置。`device.type: auto` 会在 CUDA 可用时选择 GPU，否则使用 CPU。

## 运行示例

默认读取 `input/test.png` 并写入 `output/test_clean.png`：

```bash
uv run python examples/clean_image.py
```

也可以指定路径：

```bash
uv run python examples/clean_image.py \
  --input input/example.jpg \
  --output output/example_clean.png \
  --config config/config.yaml
```

图片成功输出示例：

```text
Input: input/test.png
ROI confidence: 0.9821
ROI bbox: [100, 100, 900, 700]
Inference time: 42.37 ms
Output: output/test_clean.png
```

### 视频处理

默认读取 `input/test.mp4` 并写入 `output/test_clean.mp4`：

```bash
uv run python examples/clean_video.py
```

也可以指定路径：

```bash
uv run python examples/clean_video.py \
  --input input/example.mp4 \
  --output output/example_clean.mp4 \
  --config config/config.yaml
```

视频以流式方式逐帧处理，不会将整个文件加载到内存。默认在第 0、30、60……帧执行 ROI 检测，其余帧复用最近一次有效 bbox。周期检测漏检时继续复用上一有效 ROI；首帧无法获得 ROI 时终止处理。

当前清洗策略只删除顶部患者信息区域：采用模型预测框的顶部坐标 `y1`，左边、右边和底边固定为原图边缘。这样不会因检测框的其他边界误差而裁掉底部测量文字或左右诊断内容。

视频成功输出示例：

```text
Input video: input/test.mp4
FPS: 30.00
Frames: 900
ROI confidence: 0.9821
Processed frames: 900
Output: output/test_clean.mp4
Total time: 12.45 s
```

失败时程序以非零状态退出，并在标准错误输出中打印以下稳定错误码之一：

- `IMAGE_NOT_FOUND`
- `MODEL_NOT_FOUND`
- `IMAGE_LOAD_FAILED`
- `ROI_NOT_FOUND`
- `VIDEO_NOT_FOUND`
- `VIDEO_OPEN_FAILED`
- `VIDEO_WRITE_FAILED`
- `FRAME_PROCESS_FAILED`

## 配置

`output.size` 为整数时输出正方形图片，例如 `640` 表示 `640 x 640`。也可以配置为 `[width, height]`，或设为 `null` 以保留裁剪后的原始尺寸。标准化过程保持宽高比，并以黑色填充空白区域。

`video.detect_interval` 控制重新执行 YOLO 检测的帧间隔。`video.output_fps: same` 保留输入帧率，也可以设置为正数。视频输出使用 OpenCV `VideoWriter` 和 `mp4v` 编码器，目前仅输出 MP4。

## 模型训练

Phase 3 增加了超声 ROI 数据集目录、YOLO11 训练、验证、版本化导出和部署模型替换说明。训练配置位于 `config/train.yaml`，数据集定义位于 `training/dataset.yaml`。

准备 YOLO Detection 格式数据后执行：

```bash
uv run python training/train.py
```

验证最佳模型：

```bash
uv run python training/validate.py \
  --model runs/train/weights/best.pt
```

导出版本化 PT 或 ONNX 模型：

```bash
uv run python training/export.py \
  --model runs/train/weights/best.pt \
  --format onnx
```

详细的数据划分、标注格式、训练参数和部署替换流程参见 `training/README.md`。

## 测试

```bash
uv run pytest -q
```

测试覆盖 bbox 裁剪、图片和视频处理、ROI 间隔检测与复用、训练配置、训练参数传递、验证指标提取、PT/ONNX 导出、模型缓存、文件管理、HTTP 接口及统一错误响应。

## HTTP API 服务

Phase 4 使用 FastAPI 提供图片和视频清洗接口。API 层只负责文件上传和响应组装，所有处理均通过 `service` 层调用。YOLO 检测器在应用启动时加载一次，并在全部请求之间复用。

启动前需要将批准上线的权重放在：

```text
models/yolo11s_roi.pt
```

启动服务：

```bash
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000
```

浏览器操作页面：

```text
http://localhost:8000/
```

页面直接进入统一的批量脱敏工作台，不要求用户区分单张图片、批量图片或视频。选择数据文件夹后，系统自动识别 JPG、JPEG、PNG、BMP 和 MP4：左侧先显示当前原始文件，右侧列表显示文件名、大小和处理状态。用户可以选择“脱敏当前文件”，也可以选择“全部脱敏”；系统按文件类型自动调用图片或视频接口，左侧切换为原始内容与脱敏结果并排核对。单个文件失败不会中断剩余任务；成功结果可以逐个下载，不生成 ZIP 文件。

文件浏览快捷键：按 `A` 选择上一个文件，按 `D` 选择下一个文件。焦点位于文件名搜索框时不会触发快捷键。

服务运行时不需要 Node.js。仅在修改前端样式时安装并重新构建 CSS：

```bash
cd web
npm install
npm run build:css
cd ..
```

服务配置位于 `config/server.yaml`，运行日志写入 `logs/app.log`。Swagger 文档地址为：

```text
http://localhost:8000/docs
```

健康检查：

```bash
curl http://localhost:8000/health
```

图片清洗：

```bash
curl -X POST \
  -F file=@test.png \
  http://localhost:8000/api/v1/image/clean
```

视频清洗：

```bash
curl -X POST \
  -F file=@test.mp4 \
  http://localhost:8000/api/v1/video/clean
```

读取清洗结果（`output` 为清洗接口返回的文件名）：

```text
GET /api/v1/output/{output}
```

上传文件使用时间戳和 UUID 生成内部文件名，保存在 `storage/input`；结果保存在 `storage/output`。统一 API 错误码包括 `FILE_NOT_FOUND`、`INVALID_FILE`、`MODEL_NOT_FOUND`、`ROI_NOT_FOUND` 和 `PROCESS_FAILED`。

## 数据集管理与模型评估

Phase 5 使用 `datasets/ultrasound_roi/dataset.yaml` 作为训练、验证和测试数据的统一入口。测试集新增 `images/test` 和 `labels/test`，可选的 `metadata.json` 用于厂家和设备型号维度分析。

校验数据集：

```bash
uv run python tools/validate_dataset.py
```

生成数据统计：

```bash
uv run python tools/dataset_statistics.py
```

使用模型执行完整评估：

```bash
uv run python evaluation/evaluate_roi.py \
  --model models/trained/yolo11s_ultrasound_roi_v1.pt
```

使用预计算预测复现评估：

```bash
uv run python evaluation/evaluate_roi.py \
  --predictions predictions.json
```

结果写入 `reports/metrics.json`、`reports/report.md`、`reports/images` 和 `reports/errors`。指标包括 IoU、Precision、Recall、mAP50、mAP50-95、ROI 覆盖率、低置信风险，以及厂家和设备型号分组指标。详细格式参见 `evaluation/README.md`。

## 当前限制

当前支持：

- JPG、PNG、BMP 图片
- YOLO11 `ultrasound_roi` 检测
- 安全裁剪与标准化输出
- CPU 或 CUDA GPU 自动选择
- MP4 视频逐帧解析和标准化输出
- 可配置 ROI 检测间隔
- ROI bbox 跨帧复用
- YOLO11 训练和验证命令
- 模型版本配置
- PT 和 ONNX 版本化导出
- FastAPI 图片和视频清洗接口
- 启动时模型加载与跨请求缓存
- 唯一上传文件命名和统一错误响应
- 请求、推理耗时、置信度及输出文件日志
- train、val、test 数据集规范与校验
- 分辨率、厂家、设备型号及 ROI 面积统计
- ROI 指标、预测可视化、错误分类和 Markdown 报告

当前不支持：

- OCR 和 PHI 检测
- 患者姓名、ID、日期或图像内部烧录文字处理
- 光流、目标跟踪器等动态 ROI 跟踪
- 音频轨道复制
- MP4 以外的视频输出格式
- DICOM
- Docker
- 用户权限和数据库
- OCR 或其他多任务模型训练
- 自动标注和 Active Learning

## 后续建议

后续可增加数据集统计报告、重复样本检测、设备维度评估、训练实验追踪和模型审批流水线。OCR、PHI、自动标注和 DICOM 仍应保持为独立范围。
