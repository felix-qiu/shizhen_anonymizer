"""Generate a human-readable Markdown ROI evaluation report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def _recommendations(payload: dict[str, Any]) -> list[str]:
    metrics = payload["metrics"]
    recommendations: list[str] = []
    if metrics["recall"] < 0.95:
        recommendations.append("优先分析 FN 样本，并补充对应厂家和设备布局的训练数据。")
    if metrics["mAP50_95"] < 0.80:
        recommendations.append("复核标注边界一致性，并增加边界形态差异较大的样本。")
    if metrics["roi_coverage"]["pass_rate"] < 0.95:
        recommendations.append("重点分析过度裁剪样本，降低有效超声区域损失。")
    if metrics["sensitive_information_residual_risk"]["sample_count"]:
        recommendations.append("人工复核低置信、漏检和低 IoU 样本后再批准模型上线。")
    for manufacturer, values in sorted(payload["by_manufacturer"].items()):
        if manufacturer != "UNKNOWN" and values["recall"] < 0.95:
            recommendations.append(
                f"厂家 {manufacturer} 的 Recall 偏低，建议补充该厂家设备样本并复核漏检。"
            )
    if not recommendations:
        recommendations.append(
            "保持独立测试集，并继续监测新厂家和新设备型号的数据漂移。"
        )
    return recommendations


def generate_report(payload: dict[str, Any], output_path: str | Path) -> Path:
    metrics = payload["metrics"]
    dataset = payload["dataset"]
    errors = metrics["errors"]
    lines = [
        "# Ultrasound ROI Model Evaluation Report",
        "",
        "## Evaluation Summary",
        "",
        f"- Model: `{payload['model']}`",
        f"- Test samples: {metrics['samples']}",
        f"- Precision: {_percent(metrics['precision'])}",
        f"- Recall: {_percent(metrics['recall'])}",
        f"- mAP50: {_percent(metrics['mAP50'])}",
        f"- mAP50-95: {_percent(metrics['mAP50_95'])}",
        f"- Mean ROI coverage: {_percent(metrics['roi_coverage']['mean'])}",
        f"- ROI coverage pass rate: {_percent(metrics['roi_coverage']['pass_rate'])}",
        "",
        "## Dataset",
        "",
        f"- Dataset root: `{dataset['root']}`",
        f"- Test images: {dataset['test_images']}",
        f"- Manufacturers with metadata: {dataset['manufacturers']}",
        f"- Device models with metadata: {dataset['models']}",
        "",
        "## Error Analysis",
        "",
        "| Category | Count | Meaning |",
        "|---|---:|---|",
        f"| FN | {len(errors['FN'])} | No ROI prediction |",
        f"| LOW_IOU | {len(errors['LOW_IOU'])} | Prediction differs from GT |",
        f"| SMALL_ROI | {len(errors['SMALL_ROI'])} | Prediction area is unusually small |",
        f"| OUT_OF_RANGE | {len(errors['OUT_OF_RANGE'])} | Prediction exceeds image bounds |",
        f"| LOW_CONFIDENCE | {len(errors['LOW_CONFIDENCE'])} | Prediction confidence is below threshold |",
        "",
        "## Manufacturer Breakdown",
        "",
        "| Manufacturer | Samples | Precision | Recall | mAP50 | ROI coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for manufacturer, values in sorted(payload["by_manufacturer"].items()):
        lines.append(
            f"| {manufacturer} | {values['samples']} | {_percent(values['precision'])} | "
            f"{_percent(values['recall'])} | {_percent(values['mAP50'])} | "
            f"{_percent(values['roi_coverage']['mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Device Model Breakdown",
            "",
            "| Device model | Samples | Precision | Recall | mAP50 | ROI coverage |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for device_model, values in sorted(payload.get("by_device_model", {}).items()):
        lines.append(
            f"| {device_model} | {values['samples']} | {_percent(values['precision'])} | "
            f"{_percent(values['recall'])} | {_percent(values['mAP50'])} | "
            f"{_percent(values['roi_coverage']['mean'])} |"
        )
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in _recommendations(payload))
    lines.extend(
        [
            "",
            "## Review Artifacts",
            "",
            "Annotated predictions are stored in `reports/images`. Categorized error samples are stored in `reports/errors`.",
            "",
        ]
    )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an ROI evaluation report.")
    parser.add_argument("--metrics", type=Path, default=Path("reports/metrics.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/report.md"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.metrics.read_text(encoding="utf-8"))
    report = generate_report(payload, args.output)
    print(f"Report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
