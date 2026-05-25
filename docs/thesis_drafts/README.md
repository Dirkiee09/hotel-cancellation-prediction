# Thesis Drafts — Two-Dataset Rewrite

These files are draft contents for the thesis rewrite agreed in the
planning session (`C:\Users\dirkv\.claude\plans\misty-hugging-wirth.md`).
Every numeric claim in the drafts traces to a project artefact (file
paths cited in footnotes throughout).

## Files

| File | Purpose | Length |
|---|---|---|
| `chapter_iv_results_and_discussion.md` | Full draft of Chapter IV. Mapped to Sense → Seize → Transform; covers both datasets; includes all hypothesis verdicts (H1-H5). | ~25 pages equivalent |
| `chapter_v_conclusion.md` | Full draft of Chapter V. Summary by hypothesis + by objective, three contribution sections, seven limitations, six future-work directions. | ~10 pages equivalent |
| `chapter_i_iii_updates.md` | Targeted patches for Chapters I, II, and III to keep them consistent with Chapter IV/V. Each patch is "find this / replace with this" so it can be applied directly in Word or Google Docs. | ~8 pages equivalent |

## How to use these drafts

1. **Read `chapter_iv_results_and_discussion.md` first.** It is the
   biggest section and sets the numerical anchors that Chapter V and
   the Chapter I/II/III patches reference.
2. **Apply the Chapter I/II/III patches** from `chapter_i_iii_updates.md`
   in the order listed at the bottom of that file.
3. **Drop `chapter_v_conclusion.md` in last.** It is the cleanest
   section to write because it references back to Chapter IV's
   numbered sections.

## Verification

Before submitting:
- Run `python scripts/check.py sync` — confirms thresholds quoted in
  Chapter IV match the artefacts.
- Run `python scripts/check.py metrics` — confirms the metric gates
  pass on the current model.
- Spot-check the H3 verdict: open `notebooks/05_explainability.ipynb`
  section 5.1 and confirm `deposit_type` leads the aggregated SHAP
  ranking. (Already verified at draft time.)
- Spot-check H5: `deposit_type` is #1 on both datasets — verified by
  comparing `reports/thesis/shap_feature_importance.csv` (Portugal,
  decoded via the trained pipeline) with
  `reports/ph/shap_feature_importance.csv` (Philippine).

## Citations to verify before submission

The Chapter II patches in `chapter_i_iii_updates.md` introduce five
new citations. Verify these references exist and are accessible:

- Roa et al. (2022) — domain shift in tabular ML
- Sayed et al. (2024) — hotel model transferability
- Lim and Choe (2023) — SMB hotel analytics
- Caicedo-Torres and Payares (2024) — small-property cancellation
- Niculescu-Mizil and Caruana (2005) — probability calibration with
  gradient boosting

If any cannot be sourced, substitute closest-available alternatives.

## Out of scope for this draft

These topics are deliberately left out and should be picked up after
the chapter drafts are finalised:

- Abstract rewrite (do this last; pull the headline numbers from
  Chapter IV Section 4.3.3 and Section 4.4.2).
- Figure captions (the chapter drafts reference figures by filename;
  the author should write panel-style captions following the journal /
  university template).
- Bibliography update (citation work flagged above).
- Final pagination, table-of-contents update, page-number cross-references.
