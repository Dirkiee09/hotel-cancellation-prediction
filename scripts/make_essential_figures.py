"""Build reports/figures/essential/ — the publication set for the thesis.

Curates the ~18 figures that defend the thesis narrative (one claim per
figure) into a single folder with ordered, self-describing names, and
REGENERATES four figures whose notebook versions had defects found in the
2026-06 visualization audit:

  E04  model-comparison forest   — old title claimed deltas were "vs LightGBM
       Champion" while the reference model was actually XGBoost
  E07  calibration               — old figure showed only the calibrated curve,
       so it could not defend its own claim (ECE 0.062 -> 0.031)
  E12  policy cost ladder        — 1e6 axis offset notation, weak colour logic
  E13  threshold policy bars     — legend overlapped the data

Everything else is copied verbatim from reports/figures/thesis/ (PNG + PDF).

Usage:
    python scripts/make_essential_figures.py     (or `make essential-figures`)
"""

from __future__ import annotations

import json
import shutil

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    ARTIFACTS_DIR,
    BOOKING_TIME_FEATURES,
    PROJECT_ROOT,
    REPORTS_DIR,
    TARGET_COL,
)
from src.eval.notebook_utils import setup_plotting
from src.models.metrics import expected_calibration_error
from src.utils import configure_logging

THESIS_FIG_DIR = REPORTS_DIR / "figures" / "thesis"
ESSENTIAL_DIR = REPORTS_DIR / "figures" / "essential"

# Narrative-ordered copies: (essential name, existing thesis-figure stem)
COPIED: list[tuple[str, str, str]] = [
    (
        "E01_problem_monthly_cancellation_trend",
        "fig_03_monthly_cancel_rate_trend",
        "Cancellation is large (37%) and time-varying -> chronological evaluation is mandatory",
    ),
    (
        "E02_feature_separation_by_outcome",
        "fig_04_feature_distributions",
        "Booking-time features separate cancellers (lead time ~130 vs ~45 days) -> prediction is feasible",
    ),
    (
        "E03_model_selection_rolling_origin",
        "fig_02_grouped_bar_model_selection",
        "LightGBM wins the prespecified rolling-origin validation protocol",
    ),
    (
        "E05_speed_vs_accuracy_pareto",
        "fig_7.5_speed_vs_accuracy_pareto",
        "LightGBM trains ~2.8x faster at statistically equal quality (the rational tiebreaker)",
    ),
    (
        "E06_roc_pr_curves_test",
        "fig_01_roc_pr_curves",
        "Headline discrimination: ROC-AUC 0.863 / PR-AUC 0.759 on untouched test data",
    ),
    (
        "E08_confusion_matrix_operating_point",
        "fig_03_normalized_confusion_matrix_max_f1",
        "What the deployed max-F1 operating point does to real bookings",
    ),
    (
        "E09_shap_global_importance",
        "fig_13_shap_feature_importance_bar",
        "Global drivers: deposit type, lead time, prior cancellations (H1/H3 verdict)",
    ),
    ("E10_shap_beeswarm", "fig_14_shap_beeswarm", "Direction and spread of every driver's effect"),
    (
        "E11_shap_waterfall_examples",
        "fig_16_shap_waterfall_examples",
        "Single-booking explanations — the bridge to the live app",
    ),
    (
        "E14_cost_assumption_sensitivity",
        "fig_101_cost_sensitivity",
        "The cost policy is robust to the EUR 15 FP-cost assumption",
    ),
    (
        "E15_feature_ablation_deltas",
        "fig_22_feature_ablation_deltas",
        "Dependence on the deposit-type artifact is quantified, not hidden",
    ),
    (
        "E16_temporal_stability_expanding_cv",
        "fig_21_expanding_window_cv",
        "Performance is stable across time windows (no lucky split)",
    ),
    (
        "E17_transferability_rank_slope",
        "fig_112_transferability_rank_slope",
        "Algorithm rankings transfer across markets (Spearman rho = 0.71); GBTs top both",
    ),
    (
        "E18_transferability_fold_spread",
        "fig_111_transferability_cv_spread",
        "Why PH pilot metrics are directional: fold spread at n~19 vs n~9.5k",
    ),
]

REGENERATED: dict[str, str] = {
    "E04_model_comparison_significance": "Paired bootstrap deltas vs the top test-set model, with the validation-protocol "
    "champion highlighted — the honest H2/split-decision figure",
    "E07_calibration_before_after": "Reliability diagram raw vs isotonic-calibrated with both ECE values — "
    "probabilities are trustworthy AND the calibration step earned its keep",
    "E12_policy_cost_ladder_test": "De-circularised H4 on the test set: the model beats even intervene-on-everyone by ~36%",
    "E13_threshold_policy_tradeoffs": "The three deployed threshold policies as one precision/recall/F1 trade-off",
}


def _save(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(ESSENTIAL_DIR / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[regenerated] {stem}")


# ── E04: model-comparison forest (from benchmark table 14) ─────────────────


def make_model_comparison_forest() -> None:
    sig = pd.read_csv(REPORTS_DIR / "benchmarks" / "14_paired_significance_vs_champion.csv")
    pr = sig[sig["metric"] == "pr_auc"].copy()
    champion_ref = str(pr["champion_model"].iloc[0])
    pr = pr.sort_values("observed_delta", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    y = np.arange(len(pr))
    for i, row in pr.iterrows():
        is_lgbm = row["challenger_model"] == "lightgbm"
        color = "#1f77b4" if is_lgbm else "#b22222"
        ax.errorbar(
            row["observed_delta"],
            i,
            xerr=[
                [row["observed_delta"] - row["delta_ci_lower"]],
                [row["delta_ci_upper"] - row["observed_delta"]],
            ],
            fmt="o",
            color=color,
            capsize=4,
            markersize=7,
            elinewidth=2 if is_lgbm else 1.4,
        )
        star = "*" if row["significant_at_05"] else ""
        ax.annotate(
            f' +{row["observed_delta"]:.4f} [{row["delta_ci_lower"]:+.4f}, '
            f'{row["delta_ci_upper"]:+.4f}] {star}',
            (row["delta_ci_upper"], i),
            textcoords="offset points",
            xytext=(8, -4),
            fontsize=8.5,
        )
    labels = []
    for m in pr["challenger_model"]:
        labels.append(f"{m}\n(thesis champion)" if m == "lightgbm" else m)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(f"PR-AUC advantage of {champion_ref} over challenger (95% bootstrap CI)")
    ax.set_title(
        f"Paired bootstrap deltas vs {champion_ref} — best test-set model.\n"
        "LightGBM (validation-protocol champion) trails by only 0.0036; "
        "simple baselines trail by 6-70x more.",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlim(left=-0.01)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    _save(fig, "E04_model_comparison_significance")


# ── E07: calibration before/after (recomputed from artifacts) ──────────────


def make_calibration_before_after() -> None:
    from sklearn.calibration import calibration_curve

    from src.data.load import load_raw_data
    from src.features.build import split_time_aware
    from src.utils.validate_data import clean_raw

    df, _ = clean_raw(load_raw_data())
    df = df[BOOKING_TIME_FEATURES + [TARGET_COL]].dropna(subset=[TARGET_COL])
    _, _, test_df = split_time_aware(df)
    y_test = test_df[TARGET_COL].to_numpy().astype(int)

    pipeline = joblib.load(ARTIFACTS_DIR / "best_model.pkl")
    calibrator = joblib.load(ARTIFACTS_DIR / "probability_calibrator.pkl")
    raw = pipeline.predict_proba(test_df[BOOKING_TIME_FEATURES])[:, 1]
    cal = np.clip(calibrator.predict(raw), 0.0, 1.0)

    ece_raw = expected_calibration_error(y_test, raw, 10)
    ece_cal = expected_calibration_error(y_test, cal, 10)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    for probs, label, color, style in [
        (raw, f"Raw model (ECE = {ece_raw:.3f})", "#d62728", "--"),
        (cal, f"Isotonic-calibrated (ECE = {ece_cal:.3f})", "#1f77b4", "-"),
    ]:
        frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", color=color, linestyle=style, label=label)
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=1, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed cancellation rate")
    ax.set_title(
        "Reliability diagram — before vs after calibration", fontsize=11, fontweight="bold"
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    bins = np.linspace(0, 1, 31)
    ax.hist(
        cal[y_test == 0],
        bins=bins,
        alpha=0.55,
        color="#1f77b4",
        label="Kept bookings",
        density=True,
    )
    ax.hist(
        cal[y_test == 1],
        bins=bins,
        alpha=0.55,
        color="#d62728",
        label="Cancelled bookings",
        density=True,
    )
    ax.set_xlabel("Calibrated P(cancel)")
    ax.set_ylabel("Density")
    ax.set_title("Calibrated probabilities separate the classes", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "E07_calibration_before_after")


# ── E12: policy cost ladder on the test set (from metrics.json) ────────────


def make_policy_cost_ladder() -> None:
    metrics = json.loads((REPORTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    ct = metrics["cost_thresholding_test"]
    policies = [
        ("Do nothing\n(no model)", ct["no_model_cost"], "#9a9a9a"),
        ("Threshold 0.5\n(uncalibrated default)", ct["baseline_050_cost"], "#9a9a9a"),
        ("Intervene on all\n(trivial policy)", ct["intervene_all_cost"], "#e8a33d"),
        (
            f"Cost-sensitive policy\n(thr = {ct['threshold']:.2f}, val-selected)",
            ct["total_cost"],
            "#2e8b57",
        ),
    ]

    fig, ax = plt.subplots(figsize=(9.5, 5))
    xs = np.arange(len(policies))
    for x, (label, cost, color) in zip(xs, policies):
        ax.bar(x, cost / 1e6, color=color, width=0.6)
        ax.annotate(
            f"€{cost:,.0f}",
            (x, cost / 1e6),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=10.5,
            fontweight="bold",
        )
    saving_vs_all = ct["intervene_all_cost"] - ct["total_cost"]
    pct = 100.0 * saving_vs_all / ct["intervene_all_cost"]
    ax.annotate(
        f"−€{saving_vs_all:,.0f} ({pct:.0f}%) vs the trivial policy",
        xy=(3, ct["total_cost"] / 1e6),
        xytext=(1.55, 0.85),
        fontsize=10.5,
        fontweight="bold",
        color="#2e8b57",
        arrowprops=dict(arrowstyle="->", color="#2e8b57", lw=1.4),
    )
    ax.set_xticks(xs)
    ax.set_xticklabels([p[0] for p in policies], fontsize=10)
    ax.set_ylabel("Total intervention + lost-revenue cost (test set, € millions)")
    ax.set_title(
        "Decision-policy cost on the held-out test set\n"
        "(threshold selected on validation only — no test-set tuning)",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "E12_policy_cost_ladder_test")


# ── E13: threshold-policy trade-offs (from metrics.json) ───────────────────


def make_threshold_policy_tradeoffs() -> None:
    metrics = json.loads((REPORTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    thr = json.loads((ARTIFACTS_DIR / "thresholds.json").read_text(encoding="utf-8"))
    policies = ["max_f1", "high_precision", "cost_sensitive"]
    metric_keys = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]
    colors = ["#4e79a7", "#e15759", "#59a14f"]

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    width = 0.24
    xs = np.arange(len(policies))
    for j, (key, label) in enumerate(metric_keys):
        vals = [metrics[p][key] for p in policies]
        bars = ax.bar(xs + (j - 1) * width, vals, width, label=label, color=colors[j])
        for b, v in zip(bars, vals):
            ax.annotate(
                f"{v:.2f}",
                (b.get_x() + b.get_width() / 2, v),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                fontsize=8.5,
            )
    labels = [
        f"max_f1\n(thr = {thr['max_f1']['threshold']:.2f})",
        f"high_precision\n(thr = {thr['high_precision']['threshold']:.2f})",
        f"cost_sensitive\n(thr = {thr['cost_sensitive']['threshold']:.2f})",
    ]
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score on the held-out test set")
    ax.set_title(
        "One model, three operating points — each policy serves a different business question",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "E13_threshold_policy_tradeoffs")


def _write_readme() -> None:
    lines = [
        "# Essential Figures — Publication Set",
        "",
        "The curated, defense-ready subset: one claim per figure, ordered by the",
        "thesis narrative. Regenerate with `make essential-figures`.",
        "PNG for Word, PDF for LaTeX.",
        "",
        "| Figure | Claim it defends |",
        "|---|---|",
    ]
    entries: list[tuple[str, str]] = [(n, c) for n, _, c in COPIED]
    entries += list(REGENERATED.items())
    for name, claim in sorted(entries):
        lines.append(f"| `{name}` | {claim} |")
    lines += [
        "",
        "Source material and producers: see `docs/figure_manifest.md`. Figures",
        "E04/E07/E12/E13 are regenerated directly from canonical artifacts by",
        "`scripts/make_essential_figures.py` (visualization-audit fixes: wrong",
        "champion label, missing raw-calibration curve, axis offset notation,",
        "legend overlapping data).",
    ]
    (ESSENTIAL_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[written] README.md")


def main() -> None:
    configure_logging()
    setup_plotting()
    ESSENTIAL_DIR.mkdir(parents=True, exist_ok=True)

    missing = []
    for new_name, stem, _claim in COPIED:
        for ext in ("png", "pdf"):
            src = THESIS_FIG_DIR / f"{stem}.{ext}"
            if not src.exists():
                missing.append(str(src))
                continue
            shutil.copy2(src, ESSENTIAL_DIR / f"{new_name}.{ext}")
        print(f"[copied] {new_name}  <-  {stem}")
    if missing:
        print(f"WARNING: {len(missing)} source files missing (run `make notebooks` first):")
        for m in missing:
            print("  ", m)

    make_model_comparison_forest()
    make_calibration_before_after()
    make_policy_cost_ladder()
    make_threshold_policy_tradeoffs()
    _write_readme()

    n_png = len(list(ESSENTIAL_DIR.glob("*.png")))
    print(f"\nEssential set complete: {n_png} figures in {ESSENTIAL_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
