import json
from pathlib import Path
from typing import Any

import pandas as pd

from ai_data_quality_validator.validator import DataQualityReport


def report_to_dict(report: DataQualityReport) -> dict[str, Any]:
    if hasattr(report, "model_dump"):
        return report.model_dump()

    return report.dict()


def format_text_report(report: DataQualityReport) -> str:
    return f"""
AI DATA QUALITY REPORT
======================

Dataset Summary
---------------
Total records: {report.total_records}
Bad rows detected: {report.bad_rows}

Quality Issues
--------------
Duplicate task IDs: {report.duplicate_task_ids}
Missing text records: {report.missing_text_records}
Invalid labels: {report.invalid_labels}
Low confidence records: {report.low_confidence_records}
Invalid review statuses: {report.invalid_review_statuses}

Review Status Summary
---------------------
Approved records: {report.approved_records}
Pending records: {report.pending_records}
Rejected records: {report.rejected_records}

Overall Quality Score
---------------------
{report.quality_score}%

Recommendation
--------------
{report.recommendation}
""".strip()


def save_text_report(report: DataQualityReport, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_text_report(report), encoding="utf-8")


def save_json_report(report: DataQualityReport, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    report_data = report_to_dict(report)
    path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")


def save_bad_rows(issue_rows: pd.DataFrame, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    issue_rows.to_csv(path, index=False)