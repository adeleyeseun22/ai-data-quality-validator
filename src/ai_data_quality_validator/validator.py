from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field


class ValidationConfig(BaseModel):
    required_columns: list[str] = Field(
        default_factory=lambda: [
            "task_id",
            "annotator_id",
            "text",
            "label",
            "confidence",
            "review_status",
        ]
    )
    valid_labels: set[str] = Field(default_factory=lambda: {"positive", "negative", "neutral"})
    valid_review_statuses: set[str] = Field(default_factory=lambda: {"approved", "pending", "rejected"})
    min_confidence: float = Field(default=0.70, ge=0, le=1)


class DataQualityReport(BaseModel):
    total_records: int
    duplicate_task_ids: int
    missing_text_records: int
    invalid_labels: int
    low_confidence_records: int
    invalid_review_statuses: int
    approved_records: int
    pending_records: int
    rejected_records: int
    quality_score: float
    recommendation: str
    issues: dict[str, int]


def load_dataset(file_path: str | Path) -> pd.DataFrame:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() != ".csv":
        raise ValueError("Only CSV files are currently supported.")

    return pd.read_csv(path)


def validate_required_columns(df: pd.DataFrame, config: ValidationConfig) -> None:
    missing_columns = [
        column for column in config.required_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def generate_quality_report(
    df: pd.DataFrame,
    config: ValidationConfig | None = None,
) -> DataQualityReport:
    config = config or ValidationConfig()

    validate_required_columns(df, config)

    working_df = df.copy()

    working_df["label"] = working_df["label"].fillna("").astype(str).str.lower().str.strip()
    working_df["review_status"] = (
        working_df["review_status"].fillna("").astype(str).str.lower().str.strip()
    )
    working_df["text"] = working_df["text"].fillna("").astype(str).str.strip()
    working_df["confidence"] = pd.to_numeric(working_df["confidence"], errors="coerce")

    total_records = len(working_df)

    duplicate_task_ids = int(working_df["task_id"].duplicated(keep=False).sum())

    missing_text_records = int(working_df["text"].eq("").sum())

    invalid_labels = int((~working_df["label"].isin(config.valid_labels)).sum())

    invalid_review_statuses = int(
        (~working_df["review_status"].isin(config.valid_review_statuses)).sum()
    )

    low_confidence_records = int(
        working_df["confidence"].isna().sum()
        + (working_df["confidence"] < config.min_confidence).sum()
    )

    approved_records = int((working_df["review_status"] == "approved").sum())
    pending_records = int((working_df["review_status"] == "pending").sum())
    rejected_records = int((working_df["review_status"] == "rejected").sum())

    issues = {
        "duplicate_task_ids": duplicate_task_ids,
        "missing_text_records": missing_text_records,
        "invalid_labels": invalid_labels,
        "low_confidence_records": low_confidence_records,
        "invalid_review_statuses": invalid_review_statuses,
        "pending_records": pending_records,
        "rejected_records": rejected_records,
    }

    weighted_issue_score = (
        duplicate_task_ids
        + (missing_text_records * 2)
        + (invalid_labels * 2)
        + low_confidence_records
        + (invalid_review_statuses * 2)
        + pending_records
        + rejected_records
    )

    if total_records == 0:
        quality_score = 0.0
    else:
        max_possible_penalty = total_records * 4
        quality_score = max(
            0.0,
            100 - ((weighted_issue_score / max_possible_penalty) * 100),
        )
        quality_score = round(quality_score, 2)

    recommendation = get_recommendation(quality_score)

    return DataQualityReport(
        total_records=total_records,
        duplicate_task_ids=duplicate_task_ids,
        missing_text_records=missing_text_records,
        invalid_labels=invalid_labels,
        low_confidence_records=low_confidence_records,
        invalid_review_statuses=invalid_review_statuses,
        approved_records=approved_records,
        pending_records=pending_records,
        rejected_records=rejected_records,
        quality_score=quality_score,
        recommendation=recommendation,
        issues=issues,
    )


def get_recommendation(quality_score: float) -> str:
    if quality_score >= 90:
        return "Dataset is ready for model training or evaluation."
    if quality_score >= 75:
        return "Dataset is usable but needs minor cleaning."
    if quality_score >= 60:
        return "Dataset needs cleaning before model training."
    return "Dataset is not ready. Major quality issues found."


def validate_file(file_path: str | Path) -> DataQualityReport:
    df = load_dataset(file_path)
    return generate_quality_report(df)