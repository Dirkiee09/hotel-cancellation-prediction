# Thesis Drafts

These files are draft contents for the thesis rewrite. Every numeric claim in the drafts traces to a project artefact (file paths cited in footnotes throughout).

## Files

| File | Purpose | Length |
|---|---|---|
| `complete_thesis.md` | The fully assembled master thesis document. This file contains the Abstract, Chapters I through V, and the Bibliography. It is the single source of truth for the thesis draft. | ~45 pages equivalent |
| `writing_guidelines.md` | Contains project-specific jargon translations and number correction guides. | ~5 pages equivalent |
| `archive/` | Contains the older, fragmented chapter drafts (`chapter_i_iii_updates.md`, `chapter_iv_results_and_discussion.md`, `chapter_v_conclusion.md`, `NUMBER_CORRECTIONS.md`, `jargon_translation_guide.md`). | N/A |

## Verification

Before submitting:
- Run `python scripts/check.py sync` — confirms thresholds quoted in Chapter IV match the artefacts.
- Run `python scripts/check.py metrics` — confirms the metric gates pass on the current model.
- Spot-check the H3 verdict: open `notebooks/05_explainability.ipynb` section 5.1 and confirm `deposit_type` leads the aggregated SHAP ranking. (Already verified at draft time.)
- Spot-check H5: `deposit_type` is #1 on both datasets — verified by comparing `reports/thesis/shap_feature_importance.csv` (Portugal, decoded via the trained pipeline) with `reports/ph/shap_feature_importance.csv` (Philippine).

## Citations to verify before submission

The Chapter II patches introduce five new citations. Verify these references exist and are accessible:

- Roa et al. (2022) — domain shift in tabular ML
- Sayed et al. (2024) — hotel model transferability
- Lim and Choe (2023) — SMB hotel analytics
- Caicedo-Torres and Payares (2024) — small-property cancellation
- Niculescu-Mizil and Caruana (2005) — probability calibration with gradient boosting

If any cannot be sourced, substitute closest-available alternatives.

## Out of scope for this draft

These topics are deliberately left out and should be picked up after the chapter drafts are finalised:

- Abstract rewrite (do this last; pull the headline numbers from Chapter IV Section 4.3.3 and Section 4.4.2).
- Figure captions (the chapter drafts reference figures by filename; the author should write panel-style captions following the journal / university template).
- Bibliography update (citation work flagged above).
- Final pagination, table-of-contents update, page-number cross-references.
