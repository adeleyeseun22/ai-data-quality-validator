from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from ai_data_quality_validator.report import (
    format_text_report,
    save_annotator_scores,
    save_bad_rows,
    save_issue_summary,
    save_json_report,
    save_text_report,
)
from ai_data_quality_validator.validator import (
    calculate_annotator_quality,
    generate_quality_report,
    identify_issue_rows,
    load_dataset,
    summarize_issue_types,
)

app = typer.Typer(
    help="Validate AI annotation datasets and generate quality reports.",
    no_args_is_help=True,
)

console = Console()


@app.callback()
def cli() -> None:
    """
    AI Data Quality Validator command-line interface.
    """


@app.command()
def validate(
    input_file: Annotated[
        Path,
        typer.Argument(help="Path to the annotation CSV file."),
    ],
    output_file: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Path where the text report should be saved.",
        ),
    ] = Path("reports/quality_report.txt"),
    json_file: Annotated[
        Path,
        typer.Option(
            "--json",
            "-j",
            help="Path where the JSON report should be saved.",
        ),
    ] = Path("reports/quality_report.json"),
    bad_rows_file: Annotated[
        Path,
        typer.Option(
            "--bad-rows",
            "-b",
            help="Path where bad rows should be exported.",
        ),
    ] = Path("reports/bad_rows.csv"),
    issue_summary_file: Annotated[
        Path,
        typer.Option(
            "--issue-summary",
            "-s",
            help="Path where issue summary should be exported.",
        ),
    ] = Path("reports/issue_summary.csv"),
    annotator_scores_file: Annotated[
        Path,
        typer.Option(
            "--annotator-scores",
            "-a",
            help="Path where annotator quality scores should be exported.",
        ),
    ] = Path("reports/annotator_scores.csv"),
) -> None:
    """
    Validate an annotation dataset and generate quality reports.
    """
    try:
        df = load_dataset(input_file)

        report = generate_quality_report(df)
        issue_rows = identify_issue_rows(df)
        issue_summary = summarize_issue_types(issue_rows)
        annotator_scores = calculate_annotator_quality(df)

        text_report = format_text_report(report)

        console.print(
            Panel(
                text_report,
                title="AI Data Quality Validator",
                expand=False,
            )
        )

        save_text_report(report, output_file)
        save_json_report(report, json_file)
        save_bad_rows(issue_rows, bad_rows_file)
        save_issue_summary(issue_summary, issue_summary_file)
        save_annotator_scores(annotator_scores, annotator_scores_file)

        console.print(f"\nText report saved to: [bold green]{output_file}[/bold green]")
        console.print(f"JSON report saved to: [bold green]{json_file}[/bold green]")
        console.print(f"Bad rows saved to: [bold green]{bad_rows_file}[/bold green]")
        console.print(f"Issue summary saved to: [bold green]{issue_summary_file}[/bold green]")
        console.print(
            f"Annotator scores saved to: [bold green]{annotator_scores_file}[/bold green]"
        )

    except Exception as error:
        console.print(f"[bold red]Error:[/bold red] {error}")
        raise typer.Exit(code=1) from error


def main() -> None:
    app()


if __name__ == "__main__":
    main()