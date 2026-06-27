"""Systematically compare dataset splitting strategies (70/30, 80/20, 80/10/10)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, precision_score, recall_score

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import BOOKING_TIME_FEATURES, LEAKAGE_COLS, RANDOM_STATE, TARGET_COL
from src.data.load import load_raw_data
from src.features.build import build_preprocessor, sort_by_arrival_date
from src.models.metrics import safe_pr_auc, safe_roc_auc
from src.models.train import train_lgbm
from src.utils.reproducibility import set_global_seed
from src.utils.validate_data import clean_raw

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def run_experiment(df_ordered: pd.DataFrame, split_name: str, train_frac: float, val_frac: float, test_frac: float) -> dict[str, Any]:
    """Run a single chronological split experiment with fixed architecture and hyperparameters."""
    n = len(df_ordered)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    train_df = df_ordered.iloc[:train_end]
    if val_frac > 0:
        val_df = df_ordered.iloc[train_end:val_end]
        test_df = df_ordered.iloc[val_end:]
    else:
        val_df = pd.DataFrame()
        test_df = df_ordered.iloc[train_end:]

    feature_cols = [col for col in BOOKING_TIME_FEATURES if col in df_ordered.columns]

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL].astype(int)
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL].astype(int)

    # Identical preprocessing pipeline
    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    # Identical LightGBM architecture & hyperparameter budget (no early stopping)
    model = train_lgbm(X_train_t, y_train, scale_pos_weight=None)
    test_probs = model.predict_proba(X_test_t)[:, 1]

    roc = safe_roc_auc(y_test.to_numpy(), test_probs)
    pr = safe_pr_auc(y_test.to_numpy(), test_probs)
    brier = float(brier_score_loss(y_test.to_numpy(), test_probs))

    # Evaluate decision metrics at standard 0.50 threshold
    test_preds = (test_probs >= 0.5).astype(int)
    f1 = float(f1_score(y_test, test_preds, zero_division=0))
    acc = float(accuracy_score(y_test, test_preds))
    prec = float(precision_score(y_test, test_preds, zero_division=0))
    rec = float(recall_score(y_test, test_preds, zero_division=0))

    return {
        "Strategy": split_name,
        "Train Rows": len(train_df),
        "Val Rows": len(val_df),
        "Test Rows": len(test_df),
        "ROC-AUC": roc,
        "PR-AUC": pr,
        "Brier Score": brier,
        "F1 Score": f1,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
    }


def main() -> None:
    set_global_seed(RANDOM_STATE)
    console = Console()
    console.print("\n[bold cyan]=== SYSTEMATIC DATASET SPLITTING STRATEGY COMPARISON ===[/bold cyan]\n")

    console.print("[dim]Loading and cleaning dataset...[/dim]")
    df = load_raw_data()
    df, _ = clean_raw(df)
    df = df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns])
    df = df.dropna(subset=[TARGET_COL])

    # Sort chronologically by arrival date to simulate real-world out-of-time serving
    df_ordered = sort_by_arrival_date(df)
    console.print(f"[dim]Total clean records available: {len(df_ordered):,}[/dim]\n")

    experiments = [
        ("70/30 (Train/Test)", 0.70, 0.0, 0.30),
        ("80/20 (Train/Test)", 0.80, 0.0, 0.20),
        ("80/10/10 (Train/Val/Test)", 0.80, 0.10, 0.10),
    ]

    results = []
    for name, tr, va, te in experiments:
        console.print(f"[yellow]Evaluating strategy: {name}...[/yellow]")
        res = run_experiment(df_ordered, name, tr, va, te)
        results.append(res)

    # Render results table
    table = Table(
        title="Dataset Splitting Strategy Comparison (Out-of-Time Test Evaluation)",
        box=box.ROUNDED,
        header_style="bold cyan",
    )
    table.add_column("Strategy", justify="left", style="white", no_wrap=True)
    table.add_column("Train / Val / Test Rows", justify="center", style="dim")
    table.add_column("ROC-AUC", justify="right", style="green")
    table.add_column("PR-AUC", justify="right", style="green")
    table.add_column("Brier Score", justify="right", style="yellow")
    table.add_column("F1 (th=0.5)", justify="right", style="magenta")
    table.add_column("Accuracy", justify="right")

    # Determine best PR-AUC
    valid_pr = [r["PR-AUC"] for r in results if not np.isnan(r["PR-AUC"])]
    best_pr = max(valid_pr) if valid_pr else -1

    for r in results:
        is_best = r["PR-AUC"] == best_pr
        row_style = "bold magenta" if is_best else ""
        counts_str = f"{r['Train Rows']:,} / {r['Val Rows']:,} / {r['Test Rows']:,}"
        table.add_row(
            r["Strategy"],
            counts_str,
            f"{r['ROC-AUC']:.4f}",
            f"{r['PR-AUC']:.4f}",
            f"{r['Brier Score']:.4f}",
            f"{r['F1 Score']:.4f}",
            f"{r['Accuracy']:.4f}",
            style=row_style,
        )

    console.print("\n")
    console.print(table)

    winner = next((r["Strategy"] for r in results if r["PR-AUC"] == best_pr), "N/A")
    console.print(f"\n[bold green]Empirical Verdict:[/bold green] [cyan]{winner}[/cyan] produced the highest out-of-time PR-AUC on its evaluation set.")
    console.print("[dim italic]Methodology: All experiments strictly used identical preprocessing, feature engineering, LightGBM model architecture, and default hyperparameters. The only experimental variable was the partition boundary.[/dim italic]\n")


if __name__ == "__main__":
    main()
