from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from ai_data_quality_validator.report import (
    format_text_report,
    save_bad_rows,
    save_json_report,
    save_text_report,
)
from ai_data_quality_validator.validator import (
    generate_quality_report,
    identify_issue_rows,
    load_dataset,
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
    input_file: Path = typer.Argument(
        ...,
        help="Path to the annotation CSV file.",
    ),
    output_file: Path = typer.Option(
        Path("reports/quality_report.txt"),
        "--output",
        "-o",
        help="Path where the text report should be saved.",
    ),
    json_file: Path = typer.Option(
        Path("reports/quality_report.json"),
        "--json",
        "-j",
        help="Path where the JSON report should be saved.",
    ),
    bad_rows_file: Path = typer.Option(
        Path("reports/bad_rows.csv"),
        "--bad-rows",
        "-b",
        help="Path where bad rows should be exported.",
    ),
) -> None:
    """
    Validate an annotation dataset and generate text, JSON, and bad-row reports.
    """
    try:
        df = load_dataset(input_file)
        report = generate_quality_report(df)
        issue_rows = identify_issue_rows(df)

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

        console.print(f"\nText report saved to: [bold green]{output_file}[/bold green]")
        console.print(f"JSON report saved to: [bold green]{json_file}[/bold green]")
        console.print(f"Bad rows saved to: [bold green]{bad_rows_file}[/bold green]")

    except Exception as error:
        console.print(f"[bold red]Error:[/bold red] {error}")
        raise typer.Exit(code=1) from error


def main() -> None:
    app()


if __name__ == "__main__":
    main()