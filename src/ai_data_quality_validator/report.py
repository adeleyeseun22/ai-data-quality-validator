from pathlib import Path

from ai_data_quality_validator.validator import DataQualityReport


def format_text_report(report: DataQualityReport) -> str:
    return f"""
AI DATA QUALITY REPORT
======================

Dataset Summary
---------------
Total records: {report.total_records}

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