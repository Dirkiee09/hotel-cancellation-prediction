"""Rebuild reports/thesis/shap_summary_plot.png with raw feature names.

The previous plot used encoded one-hot indices ("Feature 50", "Feature 29", …)
on the y-axis, which is uninterpretable in a thesis. This rebuild:

1. Loads the calibrated LightGBM champion and the chronological test split.
2. Computes TreeSHAP values on the encoded features.
3. Aggregates each encoded column's SHAP contribution back to its *raw*
   feature (one-hot dummies for ``deposit_type_*`` sum into ``deposit_type``).
4. Aggregates feature *values* per raw feature the same way — for numerics
   this is the raw value; for categoricals it's an integer code that
   preserves the blue (low) → red (high) color gradient convention.
5. Renders a publication-grade beeswarm in matplotlib with the standard
   coolwarm colormap, the project's serif thesis font, and a sample size
   visible in the caption.

Output: ``reports/thesis/shap_summary_plot.png`` (+ PDF sibling).

Usage:
    python scripts/rebuild_shap_summary_plot.py
    python scripts/rebuild_shap_summary_plot.py --max-display 12 --sample 800
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib import cm
from matplotlib.colors import Normalize

from src.config import BOOKING_TIME_FEATURES, PROJECT_ROOT, RANDOM_STATE, TARGET_COL
from src.data.load import load_raw_data
from src.features.build import add_derived_booking_features, split_time_aware
from src.utils.validate_data import clean_raw

logger = logging.getLogger(__name__)

THESIS_DIR = PROJECT_ROOT / "reports" / "thesis"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def _load_champion_pipeline():
    """Load the persisted preprocessor+LGBM Pipeline."""
    return joblib.load(ARTIFACTS_DIR / "best_model.pkl")


def _build_test_features(max_rows: int | None = None) -> pd.DataFrame:
    """Reconstruct the chronological test split with raw + engineered features."""
    df = load_raw_data()
    df, _ = clean_raw(df)
    feature_cols = list(BOOKING_TIME_FEATURES)
    df = add_derived_booking_features(df)
    df = df[feature_cols + [TARGET_COL]].dropna(subset=[TARGET_COL]).reset_index(drop=True)
    # Same chronological split the production pipeline uses.
    df = df[feature_cols + [TARGET_COL]]
    # Re-derive split: we want the chronological test set for SHAP.
    # split_time_aware re-orders by arrival_date; do it the same way.
    df_for_split = load_raw_data()
    df_for_split, _ = clean_raw(df_for_split)
    df_for_split = df_for_split.dropna(subset=[TARGET_COL])
    _, _, test_df = split_time_aware(df_for_split[BOOKING_TIME_FEATURES + [TARGET_COL]])
    test_df = test_df.reset_index(drop=True)
    if max_rows is not None and len(test_df) > max_rows:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(test_df), size=max_rows, replace=False)
        test_df = test_df.iloc[idx].reset_index(drop=True)
    return test_df


def _encoded_to_raw_map(encoded_names: list[str], raw_features: list[str]) -> list[str]:
    """For each encoded column name, return the raw feature it came from.

    The preprocessor's ColumnTransformer prefixes are 'categorical__' and
    'numeric__'. After the prefix, one-hot columns look like
    'deposit_type_Non Refund' (raw_name + '_' + value). Numerics keep their
    raw name verbatim.
    """
    out: list[str] = []
    for enc in encoded_names:
        rest = enc.split("__", 1)[1] if "__" in enc else enc
        match = next(
            (raw for raw in raw_features if rest == raw or rest.startswith(raw + "_")),
            rest,
        )
        out.append(match)
    return out


def _per_row_raw_feature_values(
    test_df: pd.DataFrame,
    raw_features: list[str],
) -> pd.DataFrame:
    """Build a row-aligned matrix of feature values for COLORING.

    Numeric features: raw value (rank-normalised to [0,1] before plotting).
    Categorical features: ordinal code by frequency (most common = 0, least
    common = high) so the rare/risky categories sit at the warm end of the
    colormap. Both are normalised to [0,1] in the caller.
    """
    cols: dict[str, np.ndarray] = {}
    for feat in raw_features:
        if feat not in test_df.columns:
            cols[feat] = np.full(len(test_df), np.nan)
            continue
        series = test_df[feat]
        if pd.api.types.is_numeric_dtype(series):
            cols[feat] = pd.to_numeric(series, errors="coerce").to_numpy()
        else:
            # Order categories by frequency (ascending so common = 0, rare = high).
            freq = series.value_counts(ascending=True)
            code_map = {v: i for i, v in enumerate(freq.index)}
            cols[feat] = series.map(code_map).fillna(-1).to_numpy()
    return pd.DataFrame(cols, index=test_df.index)


def _aggregate_shap_to_raw(
    shap_encoded: np.ndarray,
    encoded_to_raw: list[str],
    raw_features: list[str],
) -> np.ndarray:
    """Sum encoded-column SHAP contributions into raw-feature columns.

    Returns a 2-D array of shape (n_rows, n_raw_features) in raw_features order.
    """
    n_rows = shap_encoded.shape[0]
    out = np.zeros((n_rows, len(raw_features)), dtype=np.float64)
    raw_index = {f: i for i, f in enumerate(raw_features)}
    for col, raw_name in enumerate(encoded_to_raw):
        if raw_name in raw_index:
            out[:, raw_index[raw_name]] += shap_encoded[:, col]
    return out


def _normalised_for_color(values: np.ndarray) -> np.ndarray:
    """Rank-normalise feature values to [0, 1] for colormap mapping.

    Rank normalisation makes the color gradient robust to outliers (a few
    extreme ADR values won't compress the rest of the gradient into one
    colour). NaNs are mapped to 0.5 (neutral grey).
    """
    out = np.full(values.shape, 0.5)
    finite_mask = np.isfinite(values)
    if finite_mask.sum() == 0:
        return out
    ranks = pd.Series(values[finite_mask]).rank(method="average").to_numpy()
    out[finite_mask] = (ranks - 1.0) / max(ranks.max() - 1.0, 1.0)
    return out


def _beeswarm(
    raw_shap_matrix: np.ndarray,
    raw_feature_values: pd.DataFrame,
    feature_names: list[str],
    *,
    n_test_rows: int,
    max_display: int,
    output_path_png: Path,
    output_path_pdf: Path,
) -> None:
    """Render a publication-grade SHAP beeswarm using only matplotlib."""
    # Rank raw features by mean(|SHAP|) descending; keep only the top max_display.
    importance = np.abs(raw_shap_matrix).mean(axis=0)
    order = np.argsort(importance)[::-1][:max_display]
    selected = [feature_names[i] for i in order]
    shap_subset = raw_shap_matrix[:, order]

    # Build the color vector per (row, feature) pair, normalised per feature.
    color_matrix = np.zeros_like(shap_subset)
    for j, feat in enumerate(selected):
        color_matrix[:, j] = _normalised_for_color(raw_feature_values[feat].to_numpy())

    # Plot setup — serif font, generous spacing for thesis.
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
        }
    )
    fig, ax = plt.subplots(figsize=(9.0, 0.45 * max_display + 1.8), dpi=150)

    # Standard SHAP convention: features ranked top-down, most important on top.
    y_positions = np.arange(max_display)[::-1]
    cmap = cm.get_cmap("coolwarm")  # blue (low) → red (high), the SHAP standard

    rng = np.random.default_rng(RANDOM_STATE)
    for j, y in zip(range(max_display), y_positions):
        x = shap_subset[:, j]
        c = color_matrix[:, j]
        # Vertical jitter so dense areas read as a beeswarm, not a stripe.
        jitter = rng.uniform(-0.32, 0.32, size=len(x))
        ax.scatter(
            x,
            np.full(len(x), y) + jitter,
            c=cmap(c),
            s=10,
            alpha=0.65,
            edgecolors="none",
            rasterized=True,  # smaller PDF, still vector for everything else
        )

    # Zero line for "no effect on prediction".
    ax.axvline(0, color="#3B3B3B", linewidth=0.8, linestyle="-", alpha=0.7, zorder=0)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(selected, fontsize=11)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=11)
    ax.set_title(
        f"Top {max_display} feature contributions — LightGBM champion (n = {n_test_rows:,} test rows)",
        fontsize=12,
        pad=12,
    )
    ax.tick_params(axis="x", labelsize=10)
    ax.set_ylim(-0.7, max_display - 0.3)

    # Symmetric x-limits, slightly padded.
    xmax = float(np.abs(shap_subset).max()) * 1.05
    ax.set_xlim(-xmax, xmax)

    # Color bar for the Feature value gradient.
    sm = cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.015, aspect=30)
    cbar.set_label("Feature value", fontsize=11)
    cbar.set_ticks([0.02, 0.98])
    cbar.set_ticklabels(["Low", "High"])
    cbar.outline.set_visible(False)

    plt.tight_layout()
    output_path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_path_pdf, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s and %s", output_path_png, output_path_pdf)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample",
        type=int,
        default=2000,
        help="Number of test rows to sample for SHAP (default: 2000; full test is ~12k).",
    )
    parser.add_argument(
        "--max-display",
        type=int,
        default=15,
        help="Top-N raw features to show on the y-axis (default: 15).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logger.info("loading champion pipeline + chronological test split")
    pipeline = _load_champion_pipeline()
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    test_df = _build_test_features(max_rows=args.sample)
    raw_features = list(BOOKING_TIME_FEATURES)
    X_raw = test_df[raw_features]
    X_encoded = preprocessor.transform(X_raw)
    encoded_names = list(preprocessor.named_steps["encode"].get_feature_names_out())
    logger.info(
        "shap input: n_rows=%d, n_encoded=%d, n_raw=%d",
        X_encoded.shape[0],
        X_encoded.shape[1],
        len(raw_features),
    )

    logger.info("computing TreeSHAP values")
    explainer = shap.TreeExplainer(model)
    raw_shap = explainer.shap_values(X_encoded)
    if isinstance(raw_shap, list):
        sv = raw_shap[1] if len(raw_shap) == 2 else raw_shap[0]
    else:
        sv = raw_shap
        if sv.ndim == 3:
            sv = sv[:, :, 1]

    encoded_to_raw = _encoded_to_raw_map(encoded_names, raw_features)
    raw_shap_matrix = _aggregate_shap_to_raw(sv, encoded_to_raw, raw_features)
    raw_values = _per_row_raw_feature_values(test_df, raw_features)

    _beeswarm(
        raw_shap_matrix,
        raw_values,
        raw_features,
        n_test_rows=len(test_df),
        max_display=args.max_display,
        output_path_png=THESIS_DIR / "shap_summary_plot.png",
        output_path_pdf=THESIS_DIR / "shap_summary_plot.pdf",
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
