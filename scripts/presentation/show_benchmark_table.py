"""Quickly prints the pre-calculated benchmark results table to the terminal."""

from pathlib import Path

import pandas as pd
from rich import box
from rich.console import Console
from rich.table import Table


def _print_prob_metrics(root, console):
    csv_path = root / "reports" / "benchmarks" / "03_holdout_probability_metrics.csv"
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path)
    table = Table(
        title="1. Probability Metrics (ROC-AUC / PR-AUC)", box=box.ROUNDED, header_style="bold cyan"
    )
    table.add_column("Model", justify="left", style="white", no_wrap=True)
    table.add_column("ROC-AUC", justify="right", style="green")
    table.add_column("PR-AUC", justify="right", style="green")
    table.add_column("Brier Score", justify="right", style="yellow")
    table.add_column("ECE", justify="right", style="yellow")

    for _, row in df.iterrows():
        style = "bold magenta" if row["model"] == "lightgbm" else ""
        table.add_row(
            row["model"],
            f"{row['roc_auc']:.4f}",
            f"{row['pr_auc']:.4f}",
            f"{row['brier_score']:.4f}",
            f"{row['ece']:.4f}",
            style=style,
        )
    console.print(table)
    console.print()


def _print_f1_metrics(root, console):
    csv_path = root / "reports" / "benchmarks" / "05_holdout_threshold_metrics_max_f1.csv"
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path).sort_values("f1", ascending=False)
    table = Table(
        title="2. Classification Performance (Optimal F1 Threshold)",
        box=box.ROUNDED,
        header_style="bold cyan",
    )
    table.add_column("Model", justify="left", style="white", no_wrap=True)
    table.add_column("Threshold", justify="right")
    table.add_column("F1 Score", justify="right", style="green")
    table.add_column("Precision", justify="right", style="yellow")
    table.add_column("Recall", justify="right", style="yellow")
    table.add_column("Accuracy", justify="right", style="blue")

    for _, row in df.iterrows():
        style = "bold magenta" if row["model"] == "lightgbm" else ""
        table.add_row(
            row["model"],
            f"{row['threshold']:.2f}",
            f"{row['f1']:.4f}",
            f"{row['precision']:.4f}",
            f"{row['recall']:.4f}",
            f"{row['balanced_accuracy']:.4f}",
            style=style,
        )
    console.print(table)
    console.print()


def _print_confusion_rates(root, console):
    csv_path = root / "reports" / "benchmarks" / "09_confusion_matrix_rates_per_model.csv"
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path).sort_values("tpr", ascending=False)
    table = Table(
        title="3. Confusion Matrix Rates (Trade-offs)", box=box.ROUNDED, header_style="bold cyan"
    )
    table.add_column("Model", justify="left", style="white", no_wrap=True)
    table.add_column("TPR (True Pos)", justify="right", style="green")
    table.add_column("FNR (Missed)", justify="right", style="red")
    table.add_column("TNR (True Neg)", justify="right", style="blue")
    table.add_column("FPR (False Alarm)", justify="right", style="yellow")

    for _, row in df.iterrows():
        style = "bold magenta" if row["model"] == "lightgbm" else ""
        table.add_row(
            row["model"],
            f"{row['tpr']:.4f}",
            f"{row['fnr']:.4f}",
            f"{row['tnr']:.4f}",
            f"{row['fpr']:.4f}",
            style=style,
        )
    console.print(table)
    console.print()


def main():
    root = Path(__file__).resolve().parents[2]
    console = Console()
    console.print("\n[bold underline]Algorithm Performance Showcase[/bold underline]\n")

    _print_prob_metrics(root, console)
    _print_f1_metrics(root, console)
    _print_confusion_rates(root, console)


if __name__ == "__main__":
    main()
