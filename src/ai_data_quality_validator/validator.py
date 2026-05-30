from collections import Counter
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
    valid_labels: set[str] = Field(
        default_factory=lambda: {"positive", "negative", "neutral"}
    )
    valid_review_statuses: set[str] = Field(
        default_factory=lambda: {"approved", "pending", "rejected"}
    )
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
    bad_rows: int
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


def prepare_dataset(df: pd.DataFrame, config: ValidationConfig) -> pd.DataFrame:
    validate_required_columns(df, config)

    working_df = df.copy()

    working_df["annotator_id"] = (
        working_df["annotator_id"].fillna("unknown").astype(str).str.strip()
    )
    working_df["label"] = (
        working_df["label"].fillna("").astype(str).str.lower().str.strip()
    )
    working_df["review_status"] = (
        working_df["review_status"].fillna("").astype(str).str.lower().str.strip()
    )
    working_df["text"] = working_df["text"].fillna("").astype(str).str.strip()
    working_df["confidence"] = pd.to_numeric(
        working_df["confidence"],
        errors="coerce",
    )

    return working_df


def identify_issue_rows(
    df: pd.DataFrame,
    config: ValidationConfig | None = None,
) -> pd.DataFrame:
    config = config or ValidationConfig()
    working_df = prepare_dataset(df, config)
    issue_df = df.copy()

    duplicate_mask = working_df["task_id"].duplicated(keep=False)
    missing_text_mask = working_df["text"].eq("")
    invalid_label_mask = ~working_df["label"].isin(config.valid_labels)
    low_confidence_mask = working_df["confidence"].isna() | (
        working_df["confidence"] < config.min_confidence
    )
    invalid_status_mask = ~working_df["review_status"].isin(
        config.valid_review_statuses
    )
    pending_mask = working_df["review_status"].eq("pending")
    rejected_mask = working_df["review_status"].eq("rejected")

    issue_reasons: list[str] = []

    for index in working_df.index:
        reasons: list[str] = []

        if bool(duplicate_mask.loc[index]):
            reasons.append("duplicate_task_id")

        if bool(missing_text_mask.loc[index]):
            reasons.append("missing_text")

        if bool(invalid_label_mask.loc[index]):
            reasons.append("invalid_label")

        if bool(low_confidence_mask.loc[index]):
            reasons.append("low_confidence")

        if bool(invalid_status_mask.loc[index]):
            reasons.append("invalid_review_status")

        if bool(pending_mask.loc[index]):
            reasons.append("pending_review")

        if bool(rejected_mask.loc[index]):
            reasons.append("rejected_record")

        issue_reasons.append("; ".join(reasons))

    issue_df["issue_reasons"] = issue_reasons
    issue_df = issue_df[issue_df["issue_reasons"].str.len() > 0]

    return issue_df


def summarize_issue_types(issue_rows: pd.DataFrame) -> dict[str, int]:
    issue_counter: Counter[str] = Counter()

    if issue_rows.empty or "issue_reasons" not in issue_rows.columns:
        return {}

    for issue_text in issue_rows["issue_reasons"].dropna():
        issues = [issue.strip() for issue in str(issue_text).split(";")]
        issue_counter.update(issue for issue in issues if issue)

    return dict(sorted(issue_counter.items()))


def calculate_annotator_quality(
    df: pd.DataFrame,
    config: ValidationConfig | None = None,
) -> pd.DataFrame:
    config = config or ValidationConfig()
    working_df = prepare_dataset(df, config)
    issue_rows = identify_issue_rows(df, config)

    bad_row_indexes = set(issue_rows.index)
    working_df["is_bad_row"] = working_df.index.to_series().isin(bad_row_indexes)

    rows: list[dict[str, object]] = []

    for annotator_id, group in working_df.groupby("annotator_id", dropna=False):
        total_records = int(len(group))
        bad_rows = int(group["is_bad_row"].sum())
        approved_records = int((group["review_status"] == "approved").sum())
        pending_records = int((group["review_status"] == "pending").sum())
        rejected_records = int((group["review_status"] == "rejected").sum())

        valid_confidence = group["confidence"].dropna()
        average_confidence = (
            round(float(valid_confidence.mean()), 3)
            if not valid_confidence.empty
            else 0.0
        )

        quality_score = (
            round(max(0.0, 100 - ((bad_rows / total_records) * 100)), 2)
            if total_records > 0
            else 0.0
        )

        rows.append(
            {
                "annotator_id": annotator_id,
                "total_records": total_records,
                "bad_rows": bad_rows,
                "approved_records": approved_records,
                "pending_records": pending_records,
                "rejected_records": rejected_records,
                "average_confidence": average_confidence,
                "quality_score": quality_score,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "annotator_id",
                "total_records",
                "bad_rows",
                "approved_records",
                "pending_records",
                "rejected_records",
                "average_confidence",
                "quality_score",
            ]
        )

    return (
        pd.DataFrame(rows)
        .sort_values("quality_score", ascending=True)
        .reset_index(drop=True)
    )


def generate_quality_report(
    df: pd.DataFrame,
    config: ValidationConfig | None = None,
) -> DataQualityReport:
    config = config or ValidationConfig()
    working_df = prepare_dataset(df, config)

    total_records = len(working_df)

    duplicate_task_ids = int(working_df["task_id"].duplicated(keep=False).sum())
    missing_text_records = int(working_df["text"].eq("").sum())
    invalid_labels = int((~working_df["label"].isin(config.valid_labels)).sum())

    low_confidence_records = int(
        (
            working_df["confidence"].isna()
            | (working_df["confidence"] < config.min_confidence)
        ).sum()
    )

    invalid_review_statuses = int(
        (~working_df["review_status"].isin(config.valid_review_statuses)).sum()
    )

    approved_records = int((working_df["review_status"] == "approved").sum())
    pending_records = int((working_df["review_status"] == "pending").sum())
    rejected_records = int((working_df["review_status"] == "rejected").sum())

    bad_rows = int(len(identify_issue_rows(working_df, config)))

    issues = {
        "duplicate_task_ids": duplicate_task_ids,
        "missing_text_records": missing_text_records,
        "invalid_labels": invalid_labels,
        "low_confidence_records": low_confidence_records,
        "invalid_review_statuses": invalid_review_statuses,
        "pending_records": pending_records,
        "rejected_records": rejected_records,
        "bad_rows": bad_rows,
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
        bad_rows=bad_rows,
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