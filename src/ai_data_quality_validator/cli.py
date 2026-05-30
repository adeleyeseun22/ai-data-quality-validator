from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from ai_data_quality_validator.report import format_text_report, save_text_report
from ai_data_quality_validator.validator import validate_file

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
        help="Path where the report should be saved.",
    ),
) -> None:
    """
    Validate an annotation dataset and generate a quality report.
    """
    try:
        report = validate_file(input_file)
        text_report = format_text_report(report)

        console.print(
            Panel(
                text_report,
                title="AI Data Quality Validator",
                expand=False,
            )
        )

        save_text_report(report, output_file)

        console.print(f"\nReport saved to: [bold green]{output_file}[/bold green]")

    except Exception as error:
        console.print(f"[bold red]Error:[/bold red] {error}")
        raise typer.Exit(code=1) from error


def main() -> None:
    app()


if __name__ == "__main__":
    main()