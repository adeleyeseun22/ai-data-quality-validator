import pandas as pd

from ai_data_quality_validator.validator import (
    generate_quality_report,
    identify_issue_rows,
)


def test_generate_quality_report_detects_issues() -> None:
    df = pd.DataFrame(
        {
            "task_id": ["T001", "T002", "T002", "T003"],
            "annotator_id": ["A001", "A002", "A003", "A004"],
            "text": ["Good response", "Bad response", "", "Average response"],
            "label": ["positive", "negative", "positive", "unclear"],
            "confidence": [0.95, 0.80, 0.50, 0.40],
            "review_status": ["approved", "approved", "pending", "rejected"],
        }
    )

    report = generate_quality_report(df)

    assert report.total_records == 4
    assert report.duplicate_task_ids == 2
    assert report.missing_text_records == 1
    assert report.invalid_labels == 1
    assert report.low_confidence_records == 2
    assert report.approved_records == 2
    assert report.pending_records == 1
    assert report.rejected_records == 1
    assert report.bad_rows == 3
    assert report.quality_score < 100


def test_generate_quality_report_for_clean_dataset() -> None:
    df = pd.DataFrame(
        {
            "task_id": ["T001", "T002", "T003"],
            "annotator_id": ["A001", "A002", "A003"],
            "text": ["Good response", "Bad response", "Average response"],
            "label": ["positive", "negative", "neutral"],
            "confidence": [0.95, 0.90, 0.85],
            "review_status": ["approved", "approved", "approved"],
        }
    )

    report = generate_quality_report(df)

    assert report.total_records == 3
    assert report.duplicate_task_ids == 0
    assert report.missing_text_records == 0
    assert report.invalid_labels == 0
    assert report.low_confidence_records == 0
    assert report.invalid_review_statuses == 0
    assert report.bad_rows == 0
    assert report.quality_score == 100


def test_identify_issue_rows_adds_issue_reasons() -> None:
    df = pd.DataFrame(
        {
            "task_id": ["T001", "T002", "T002", "T003"],
            "annotator_id": ["A001", "A002", "A003", "A004"],
            "text": ["Good response", "Bad response", "", "Average response"],
            "label": ["positive", "negative", "positive", "unclear"],
            "confidence": [0.95, 0.80, 0.50, 0.40],
            "review_status": ["approved", "approved", "pending", "rejected"],
        }
    )

    issue_rows = identify_issue_rows(df)

    assert len(issue_rows) == 3
    assert "issue_reasons" in issue_rows.columns
    assert issue_rows["issue_reasons"].str.len().gt(0).all()