![Python CI](https://github.com/adeleyeseun22/ai-data-quality-validator/actions/workflows/ci.yml/badge.svg)

# AI Data Quality Validator

A production-grade Python CLI tool for validating AI annotation datasets, detecting data quality issues, and generating readiness reports for model training and evaluation workflows.

This project demonstrates practical Python engineering for AI data workflows, including data validation, annotation quality checks, structured reporting, test coverage, and command-line tooling.

---

## Why This Project Matters

AI systems depend on high-quality training and evaluation data. Poor annotation quality can introduce noise, bias, and unreliable model performance.

This tool helps teams quickly inspect annotation datasets and identify records that require cleaning before they are used for model training, evaluation, or benchmarking.

---

## Features

- Validate AI annotation datasets from CSV files
- Detect duplicate task IDs
- Detect missing annotation text
- Detect invalid labels
- Detect low-confidence annotations
- Detect invalid review statuses
- Flag pending and rejected records
- Generate a human-readable text report
- Generate a machine-readable JSON report
- Export problematic rows into a separate CSV file
- Includes automated tests with `pytest`
- Designed with a clean `src/` Python project structure

---

## Tech Stack

- Python
- Pandas
- Pydantic
- Typer
- Rich
- Pytest
- Ruff
- Mypy

---

## Generated Reports

After running the tool, five files are generated:

```text
reports/quality_report.txt
reports/quality_report.json
reports/bad_rows.csv
reports/issue_summary.csv
reports/annotator_scores.csv

---

## Project Structure

```text
ai-data-quality-validator/
│
├── data/
│   └── sample_annotations.csv
│
├── reports/
│
├── src/
│   └── ai_data_quality_validator/
│       ├── __init__.py
│       ├── cli.py
│       ├── report.py
│       └── validator.py
│
├── tests/
│   └── test_validator.py
│
├── README.md
├── requirements.txt
├── pyproject.toml
└── .gitignore