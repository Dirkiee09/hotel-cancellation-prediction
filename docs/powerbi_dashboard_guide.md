# Building Your 8-Page Master Power BI Dashboard

*A step-by-step guide for the thesis "A Strategic Business Intelligence Approach to Predicting Hotel Booking Cancellations".*

> **How to read this guide.** Each section is one sitting (~30–45 min).
> If you do one section per day, the dashboard is done in two weeks; if you
> block out a Saturday, you can finish in a single day. Steps are numbered
> so you can pause and resume without losing your place. Skip the
> **★ Advanced** boxes on your first pass — come back to them after the
> dashboard works end-to-end.

**Cross-reference.** The eight-page structure below comes from Section 4.4.3
of `docs/thesis_drafts/complete_thesis.md`. Every page in this guide answers
one of the eight business questions named there.

---

## Pre-flight — get everything ready (15–20 min)

### 1. Install Power BI Desktop
- Download from `https://aka.ms/pbidesktopstore` (Microsoft Store version updates itself)
- Open it once, sign in with your Microsoft account, close it again
- Confirm version is **2024-09** or newer (older versions miss the new card visual)

### 2. Make sure every CSV exists
Run this in the project root and confirm every line returns a path:

```powershell
Get-ChildItem reports/test_predictions_for_powerbi.csv,
              reports/adr_test_predictions.csv,
              reports/adr_segment_performance.csv,
              reports/segment_metrics.csv,
              reports/thesis/shap_feature_importance.csv,
              data/predictions/predictions_live.csv,
              data/predictions/drift_metrics.csv
```

If anything is missing:
- Re-train: `python scripts/train.py`
- Regenerate ADR exports: `python scripts/export_adr_predictions.py`
- Make at least one live prediction so the live CSV exists: open the Gradio
  UI at `localhost:8000/ui`, submit one booking, then run
  `python scripts/export_predictions.py`
- Compute drift: `python scripts/compute_live_drift.py`

### 3. Export `reports/metrics.json` to CSV (one-shot)
Power BI prefers tabular files. From the project root:

```powershell
python -c "import json, pandas as pd; m = json.load(open('reports/metrics.json'));
pd.DataFrame([{'metric': k, **v} if isinstance(v, dict) else {'metric': k, 'value': v}
              for k, v in m.items() if isinstance(v, (int, float, dict))]
            ).to_csv('reports/metrics_for_powerbi.csv', index=False)"
```

### 4. Create a working folder
- New folder: `D:\PythonProject1\reports\powerbi\`
- This is where the `.pbix` file will live
- The `.pbix` can be git-ignored; the source CSVs are tracked

### 5. Pick a color palette
The thesis figures use a serif-classical look. For the dashboard, switch to
a **business-deck** palette:

| Role | Color | Hex |
|---|---|---|
| Primary | Deep blue | `#1F4E79` |
| Danger / cancel | Wine red | `#A6192E` |
| Safe / no-cancel | Sea green | `#107C41` |
| Neutral | Slate | `#3B3B3B` |
| Surface | Pale grey | `#F4F4F4` |
| Accent | Amber | `#F5A623` |

Keep this list visible while you build — every visual uses one of these.

---

## Step 1 — Import data & build the model (30–40 min)

### 1.1 Load every CSV
1. **Home → Get Data → Text/CSV**
2. Point at each file from the pre-flight checklist
3. In the preview pane, click **Transform Data** (not Load) so you can clean
   types before they hit the canvas
4. Rename each query to something readable (right-click the query name on the
   left rail → Rename):

| Original file | Rename to |
|---|---|
| `test_predictions_for_powerbi.csv` | `Bookings_Baseline` |
| `predictions_live.csv` | `Bookings_Live` |
| `drift_metrics.csv` | `Drift` |
| `adr_test_predictions.csv` | `ADR_Predictions` |
| `adr_segment_performance.csv` | `ADR_Segment_RMSE` |
| `segment_metrics.csv` | `Segment_Metrics` |
| `shap_feature_importance.csv` | `Feature_Importance` |
| `metrics_for_powerbi.csv` | `Model_Headline_KPIs` |

### 1.2 Fix data types in Power Query
For `Bookings_Baseline` and `Bookings_Live`, change these columns explicitly
(Power BI guesses, but it sometimes guesses wrong):

| Column | Type |
|---|---|
| `cancel_probability` / `probability` | Decimal Number |
| `is_canceled`, `predicted_cancel_*`, `label_*` | Whole Number |
| `adr`, `revenue_at_risk` | Decimal Number |
| `lead_time` | Whole Number |
| `arrival_date_year`, `arrival_date_week_number`, `arrival_date_day_of_month` | Whole Number |
| `arrival_date_month` | Text |
| `risk_tier` | Text |
| `timestamp_utc` (live only) | Date/Time |

Click **Home → Close & Apply** when done.

### 1.3 Build a Date table
Power BI needs a calendar to make time-based slicing reliable.
**Modeling → New Table** and paste:

```dax
DateTable =
ADDCOLUMNS(
    CALENDAR(DATE(2015, 1, 1), DATE(2017, 12, 31)),
    "Year",       YEAR([Date]),
    "Month",      FORMAT([Date], "MMM YYYY"),
    "MonthNum",   FORMAT([Date], "YYYY-MM"),
    "Week",       WEEKNUM([Date]),
    "DayOfWeek",  FORMAT([Date], "ddd")
)
```

> Why a calendar table? Power BI's auto-date hierarchies are convenient but
> they create one hidden hierarchy per date column and can't be sliced
> together. A single shared `DateTable` lets every page filter consistently.

Mark it as the date table: **Table view → DateTable → Table tools → Mark as
date table → Date column = Date**.

### 1.4 Build the booking-arrival date column
In `Bookings_Baseline`, **Table tools → New Column** and paste:

```dax
ArrivalDate =
DATE(
    [arrival_date_year],
    SWITCH(TRUE(),
        [arrival_date_month] = "January",   1,
        [arrival_date_month] = "February",  2,
        [arrival_date_month] = "March",     3,
        [arrival_date_month] = "April",     4,
        [arrival_date_month] = "May",       5,
        [arrival_date_month] = "June",      6,
        [arrival_date_month] = "July",      7,
        [arrival_date_month] = "August",    8,
        [arrival_date_month] = "September", 9,
        [arrival_date_month] = "October",   10,
        [arrival_date_month] = "November",  11,
        [arrival_date_month] = "December",  12),
    [arrival_date_day_of_month]
)
```

### 1.5 Wire up relationships
**Modeling view → drag and drop**:

- `Bookings_Baseline[ArrivalDate]` → `DateTable[Date]` (one-to-many, single direction)
- `Bookings_Live[arrival_date]` → `DateTable[Date]` (same)
- `ADR_Predictions[arrival_date_month + arrival_date_year]` → `DateTable[Month]`
  (you'll need to build a `MonthKey` column in `ADR_Predictions` first using the
  same `SWITCH` pattern as 1.4)

Leave `Drift`, `Feature_Importance`, `Segment_Metrics`, `ADR_Segment_RMSE`,
and `Model_Headline_KPIs` **un-related** — they're independent reference
tables that each feed one specific page.

---

## Step 2 — Theme & global page setup (15 min)

### 2.1 Apply the theme
1. **View → Themes → Browse for themes**
2. Create a JSON file at `reports/powerbi/theme.json` with:

```json
{
  "name": "ThesisDashboard",
  "dataColors": ["#1F4E79", "#A6192E", "#107C41", "#F5A623", "#3B3B3B", "#7D8CA3"],
  "background": "#FFFFFF",
  "foreground": "#3B3B3B",
  "tableAccent": "#1F4E79",
  "textClasses": {
    "title":      { "fontFace": "Segoe UI Semibold", "fontSize": 22, "color": "#1F4E79" },
    "header":     { "fontFace": "Segoe UI Semibold", "fontSize": 14 },
    "label":      { "fontFace": "Segoe UI", "fontSize": 11 }
  }
}
```

3. Load it. Every visual now starts with the right colors and font.

### 2.2 Page canvas size
On every page: **Format page → Canvas settings → Type = 16:9, Height = 720, Width = 1280**.
This is the size your panel will project at.

### 2.3 Create a placeholder for every page
Make eight empty pages (right-click tab → New Page). Rename them:

1. `1 — Hero KPIs`
2. `2 — Cancellation Trend`
3. `3 — Segment Slicer`
4. `4 — Revenue at Risk`
5. `5 — ADR Forecasting`
6. `6 — Threshold Policy`
7. `7 — Feature Importance`
8. `8 — Drift Monitoring`

The numbered prefix gives you a predictable left-to-right tab order.

---

## Step 3 — Core measures library (30 min)

These DAX measures are referenced by multiple pages. Build them once now so
later pages just drop them onto visuals. **Modeling → New measure** for each.

Group them under a folder called `_Measures`: right-click the measure →
**Display folder → _Measures**. The underscore keeps the folder at the top.

```dax
-- Cancellation counts and rates
Total Bookings = COUNTROWS(Bookings_Baseline)
Total Cancellations = CALCULATE([Total Bookings], Bookings_Baseline[is_canceled] = 1)
Cancellation Rate = DIVIDE([Total Cancellations], [Total Bookings])

-- Revenue measures
Total Revenue at Risk =
  SUMX(
    FILTER(Bookings_Baseline,
           Bookings_Baseline[predicted_cancel_max_f1] = 1),
    Bookings_Baseline[revenue_at_risk]
  )

Total Revenue Booked = SUM(Bookings_Baseline[revenue_at_risk])

-- Model performance (read from the headline KPI table)
Model PR-AUC = MAXX(FILTER(Model_Headline_KPIs, [metric] = "pr_auc_test"), [value])
Model ROC-AUC = MAXX(FILTER(Model_Headline_KPIs, [metric] = "roc_auc_test"), [value])
Model F1 = MAXX(FILTER(Model_Headline_KPIs, [metric] = "f1_test"), [value])

-- Risk tier counts (used on Page 1 and Page 4)
Bookings High Risk =
  CALCULATE([Total Bookings], Bookings_Baseline[risk_tier] = "high")
Bookings Medium Risk =
  CALCULATE([Total Bookings], Bookings_Baseline[risk_tier] = "medium")
Bookings Low Risk =
  CALCULATE([Total Bookings], Bookings_Baseline[risk_tier] = "low")

-- ADR pricing
Mean Actual ADR = AVERAGE(ADR_Predictions[adr_actual])
Mean Predicted ADR = AVERAGE(ADR_Predictions[adr_predicted])
ADR Residual Mean = AVERAGE(ADR_Predictions[residual])
ADR RMSE Test = SQRT(AVERAGEX(ADR_Predictions, ADR_Predictions[residual] ^ 2))

-- Cost-sensitive savings (for Page 6)
Savings vs No Model =
  MAXX(FILTER(Model_Headline_KPIs, [metric] = "cost_thresholding"),
       [validation_savings_vs_no_model])
```

> **DAX tip.** `MAXX` over a single-row filter is the canonical way to read a
> "lookup" value into a measure. It returns blank cleanly if the filter
> finds nothing — useful when a metric is missing.

---

## Step 4 — Page 1: Hero KPIs (40 min)

**Business question.** *"Is anything on fire today?"* — answered in 10 seconds.

### 4.1 Layout sketch

```
+--------------------------------------------------------------+
| HOTEL CANCELLATION RISK DASHBOARD                            |
|                                                              |
| [ Cancel rate ] [ High-risk count ] [ Revenue at risk ]      |
| [ ROC-AUC    ] [ PR-AUC          ] [ F1 score          ]    |
|                                                              |
| Distribution of cancellation probability    | Risk tier mix |
| (column chart, 0-1 binned)                  | (donut)        |
+--------------------------------------------------------------+
```

### 4.2 Build the cards (×6)
For each card, **Insert → Card (new)** and drag a measure:

| Card | Measure | Conditional format |
|---|---|---|
| Cancellation Rate | `[Cancellation Rate]` | Format as % |
| High-risk count | `[Bookings High Risk]` | Background `#A6192E` if > 3000 |
| Revenue at Risk | `[Total Revenue at Risk]` | Format as €, 0 decimals |
| ROC-AUC | `[Model ROC-AUC]` | 3 decimals; green if ≥ 0.85 |
| PR-AUC | `[Model PR-AUC]` | 3 decimals; green if ≥ 0.70 |
| F1 | `[Model F1]` | 3 decimals |

Position them in a 3×2 grid at y=80 (under the title). Width 200, height 90 each.

### 4.3 Probability distribution (clustered column)
- Visual: **Clustered column chart**
- X axis: `Bookings_Baseline[cancel_probability]` (binned — set bin size 0.05)
- Y axis: `Total Bookings`
- Add a vertical reference line at x = 0.40 (max_f1 threshold) — label "Decision threshold"

### 4.4 Risk tier mix (donut)
- Visual: **Donut chart**
- Legend: `Bookings_Baseline[risk_tier]`
- Values: `Total Bookings`
- Colors: low = `#107C41`, medium = `#F5A623`, high = `#A6192E`

### 4.5 Title bar
- **Insert → Text box** at y=10 height=40
- Text: `"Hotel Cancellation Risk Dashboard — Page 1 / 8: Hero KPIs"`
- Right-align: live timestamp via **Insert → Card → measure `NOW()`**

★ Advanced — Add a **page-level filter** on `DateTable[Year]` so the cards
update by year if you later filter to "2017 only".

---

## Step 5 — Page 2: Cancellation Rate Trend (25 min)

**Business question.** *"Has cancellation behaviour spiked week-over-week?"*

### 5.1 Line + column combo
- Visual: **Line and clustered column chart**
- Shared axis: `DateTable[MonthNum]` (sorted ascending)
- Column values: `Total Bookings` (volume)
- Line values: `Cancellation Rate` (trend, secondary y-axis as %)

This dual encoding lets a viewer see if a spike in cancellation rate is real
(stable volume) or a noisy slice (low volume).

### 5.2 Rolling 4-week average
Add this measure to the `_Measures` folder:

```dax
Cancel Rate 4W Avg =
  AVERAGEX(
    DATESINPERIOD(DateTable[Date], MAX(DateTable[Date]), -28, DAY),
    [Cancellation Rate]
  )
```

Drop the measure onto the same chart as a second line; format it dashed.

### 5.3 Slicers in the top bar
- Slicer: `DateTable[Year]` (horizontal layout, multi-select)
- Slicer: `Bookings_Baseline[hotel]` (vertical, multi-select)
- Position both at y=60 above the chart

### 5.4 Annotation card
Add a text box below the chart explaining the 37 % baseline — paraphrase from
Section 4.2.1 of `complete_thesis.md`.

---

## Step 6 — Page 3: Segment Slicer (35 min)

**Business question.** *"Which slice of bookings is cancelling above average?"*

### 6.1 Four slicers (one per segment dimension)
Place these in a horizontal row at the top of the page:

| Slicer | Source | Type |
|---|---|---|
| Country | `Bookings_Baseline[country]` | Dropdown |
| Market segment | `Bookings_Baseline[market_segment]` | List |
| Customer type | `Bookings_Baseline[customer_type]` | List |
| Distribution channel | `Bookings_Baseline[distribution_channel]` | List |

### 6.2 Segment ranking matrix
- Visual: **Matrix** (the new one, not the table)
- Rows: `Bookings_Baseline[market_segment]`
- Values: `Total Bookings`, `Cancellation Rate`, `Total Revenue at Risk`
- Conditional format `Cancellation Rate`: data bars, color `#A6192E`

### 6.3 Heatmap — month × market_segment
- Visual: **Matrix** with conditional formatting on background
- Rows: `Bookings_Baseline[market_segment]`
- Columns: `Bookings_Baseline[arrival_date_month]` (sort by month number)
- Values: `Cancellation Rate`
- Format: background color, white → `#A6192E`, mid-range = `#F4F4F4`

### 6.4 Sync slicers across pages
**View → Sync slicers**. Check every slicer for Pages 2, 3, 4, 5 — that way
filtering on one page carries to the next.

---

## Step 7 — Page 4: Revenue at Risk (45 min) — *the action page*

**Business question.** *"Which specific bookings should I act on this week?"*

### 7.1 KPI strip across the top
Three cards side by side:

| Card | Measure | Note |
|---|---|---|
| Bookings flagged | `CALCULATE([Total Bookings], Bookings_Baseline[predicted_cancel_max_f1] = 1)` | "max_f1" policy |
| Revenue at risk | `[Total Revenue at Risk]` | Format as € |
| Avg lead time of flagged | `CALCULATE(AVERAGE(Bookings_Baseline[lead_time]), Bookings_Baseline[predicted_cancel_max_f1] = 1)` | Days |

### 7.2 Top-50 bookings table
- Visual: **Table** (not matrix — we want one row per booking)
- Columns in order:
  1. `risk_tier` (with conditional icon — see 7.4)
  2. `cancel_probability` (data bar, color `#A6192E`)
  3. `revenue_at_risk` (data bar, color `#1F4E79`)
  4. `hotel`
  5. `market_segment`
  6. `country`
  7. `lead_time`
  8. `arrival_date_year`, `arrival_date_month`
  9. `deposit_type`
- Filter: `predicted_cancel_max_f1 = 1`
- Sort: `revenue_at_risk` descending
- Top N filter: 50

### 7.3 Three-policy comparison bar
- Visual: **Stacked bar**
- Y axis: policy name (manual category)
- X axis: count and revenue for `max_f1`, `high_precision`, `cost_sensitive`
- Build a small disconnected table for category labels:

```dax
PolicyLabels = DATATABLE(
  "Policy", STRING,
  {{"max_f1"}, {"high_precision"}, {"cost_sensitive"}}
)
```

Then create measures:

```dax
Flagged Max F1 = CALCULATE([Total Bookings], Bookings_Baseline[predicted_cancel_max_f1] = 1)
Flagged High Precision = CALCULATE([Total Bookings], Bookings_Baseline[predicted_cancel_high_precision] = 1)
Flagged Cost Sensitive = CALCULATE([Total Bookings], Bookings_Baseline[predicted_cancel_cost_sensitive] = 1)
```

### 7.4 Risk tier icon (conditional formatting)
- Right-click `risk_tier` column → Conditional formatting → Icons
- Rule: `high` → red dot, `medium` → amber dot, `low` → green dot

### 7.5 Drill-through to single-booking detail
**Insert → New Page** → name `_BookingDetail` (the underscore hides it).
- Add: card visuals for every booking field
- Right-click the page tab → Page information → Allow drill-through = on
- Drag `cancel_probability` into the **Drill-through filters** well
- Back on Page 4, right-click any row → Drill through → `_BookingDetail`

---

## Step 8 — Page 5: ADR Forecasting (35 min)

**Business question.** *"Is the guest paying what we'd expect them to pay?"*

### 8.1 Top-row KPI cards
| Card | Measure |
|---|---|
| ADR RMSE (test) | `[ADR RMSE Test]` |
| Mean residual | `[ADR Residual Mean]` |
| Mean actual ADR | `[Mean Actual ADR]` |
| Mean predicted ADR | `[Mean Predicted ADR]` |

### 8.2 Scatter — predicted vs actual ADR
- Visual: **Scatter chart**
- X axis: `ADR_Predictions[adr_predicted]`
- Y axis: `ADR_Predictions[adr_actual]`
- Details (one dot per row): `ADR_Predictions[hotel]` (or a row ID)
- Color: `ADR_Predictions[reserved_room_type]`
- Add a reference line at `y = x` (Format → Analytics → X = Y line)

### 8.3 Residual histogram
- Visual: **Clustered column**
- Axis: `ADR_Predictions[residual]` (binned — bin size 10)
- Values: count of rows
- A residual histogram centered on zero with light tails = honest regression.

### 8.4 Monthly residual line
- Visual: **Line chart**
- Axis: `DateTable[Month]`
- Values: `ADR Residual Mean`
- Color: positive months green, negative red (conditional series color)

### 8.5 Segment RMSE heatmap
- Visual: **Matrix**
- Rows: `ADR_Segment_RMSE[hotel]`
- Columns: `ADR_Segment_RMSE[reserved_room_type]`
- Values: `ADR_Segment_RMSE[rmse]` (lower = better)
- Conditional format background: green → amber → red as RMSE grows

> **★ Honest caveat to add as a footer text box.** "The ADR regressor was
> trained on four post-booking features (see CLAUDE.md). At booking-time,
> predicted ADR is slightly less accurate than the published test RMSE of
> 44.31 €. Compare residual signs across hotels, not absolute magnitudes."

---

## Step 9 — Page 6: Threshold Policy Comparison (30 min)

**Business question.** *"What if we tightened or relaxed the decision threshold?"*

### 9.1 Threshold comparison matrix
- Visual: **Table**
- Build a disconnected `Policies` table:

```dax
Policies =
DATATABLE(
  "Policy",      STRING,
  "Threshold",   DOUBLE,
  {
    {"max_f1",          0.40},
    {"high_precision",  0.98},
    {"cost_sensitive",  0.04}
  }
)
```

- Columns: Policy, Threshold, Bookings flagged, Cancellations caught (TP),
  False positives (FP), Precision, Recall, F1, Revenue at risk, Cost savings

### 9.2 Threshold slider (★ advanced)
Add a **Numeric range slicer** bound to a what-if parameter (**Modeling →
New parameter → Numeric range**). Build measures that recompute precision /
recall / cost at the slider's value. This is the visual the revenue director
plays with during a quarterly review.

### 9.3 Confusion matrices — three side-by-side
Use **Format → Matrix** with three small matrices, one per policy. Load
each from `reports/confusion_matrix_max_f1.csv`,
`reports/confusion_matrix_high_precision.csv`, and
`reports/confusion_matrix_cost_sensitive.csv` (import these three small CSVs
during Step 1.1 if you haven't already).

### 9.4 Cost-savings annotation
Add a text box: *"Cost-sensitive policy saves €1.53 M vs the no-model
baseline at this fold (95.4 % of theoretical maximum). See Section 4.4.1."*

---

## Step 10 — Page 7: Feature Importance (25 min)

**Business question.** *"Why does the model flag what it flags?"*

### 10.1 Top-15 SHAP bar chart
- Visual: **Bar chart** (horizontal — features on Y so labels read left-to-right)
- Y axis: `Feature_Importance[feature]`
- X axis: `Feature_Importance[mean_abs_shap]`
- Sort descending; top N filter = 15
- Color: solid `#1F4E79`

### 10.2 Group labels by family
Add a calculated column on `Feature_Importance`:

```dax
FeatureGroup =
SWITCH(TRUE(),
  [feature] IN { "lead_time", "is_late_window", "month_sin", "month_cos" }, "Time",
  [feature] IN { "deposit_type", "agent", "company", "had_company" },       "Booking source",
  [feature] IN { "country", "market_segment", "distribution_channel" },     "Demand origin",
  [feature] IN { "adr", "adr_per_person", "revenue_at_risk" },              "Price",
  [feature] IN { "previous_cancellations", "previous_bookings_not_canceled","is_repeated_guest" }, "Guest history",
  "Other"
)
```

Slot `FeatureGroup` as the bar color → instant categorical visual of *which
type* of signal dominates.

### 10.3 Decision-tree mini-card (optional)
Embed the static decision-tree image saved at
`reports/figures/thesis/fig_22_decision_tree_baseline.png` for visual
panelists who prefer rules to numbers.

### 10.4 Plain-English caption
Text box at the bottom: *"Each bar shows how many percentage points of
predicted cancellation probability a single feature contributes on average.
Lead time and deposit type are the top two — exactly what hotel operations
intuition expects."*

---

## Step 11 — Page 8: Drift Monitoring (30 min)

**Business question.** *"Has the world changed since we trained the model?"*

### 11.1 PSI heatbar
- Visual: **Stacked bar** (single bar per feature)
- Y axis: `Drift[feature]`
- X axis: `Drift[psi]`
- Color: conditional — green if `psi < 0.10`, amber if 0.10–0.25, red if > 0.25
- Sort: descending by PSI

### 11.2 Zone count cards
Three cards across the top:

| Card | Measure |
|---|---|
| No drift (PSI < 0.10) | `CALCULATE(COUNTROWS(Drift), Drift[zone] = "safe")` |
| Watch (0.10–0.25) | `CALCULATE(COUNTROWS(Drift), Drift[zone] = "watch")` |
| Retrain (≥ 0.25) | `CALCULATE(COUNTROWS(Drift), Drift[zone] = "retrain")` |

### 11.3 Last-computed timestamp
- Visual: **Card** showing `MAX(Drift[computed_at_utc])`
- Subtitle: "Time since last drift run"

### 11.4 Action call-out
Text box at the bottom: *"Retrain when any feature crosses 0.25 PSI, OR when
five or more features are in the 'watch' zone simultaneously. Run
`python scripts/train.py` then `python scripts/compute_live_drift.py` to
refresh this page."*

---

## Step 12 — Polish & navigation (40 min)

### 12.1 Top page-navigator bar
On every page, place a **horizontal page navigator** (Insert → Buttons →
Navigator → Page navigator) so panelists can click between pages without
using tabs.

### 12.2 Title block on every page
Reuse the title pattern from Page 1. Replace only the *page name + number*
text. Keep position, font, and color identical across pages — visual
consistency is the cheapest professionalism win in BI design.

### 12.3 Tooltip page (★ advanced)
Create a hidden tooltip page (right-click tab → Hide) named
`_BookingTooltip` with the most relevant single-booking fields. On Page 4's
table, **Format → Tooltip → Use report page = `_BookingTooltip`**.
Hovering a row pops the detail.

### 12.4 Bookmarks for the demo flow
**View → Bookmarks → Add**. Create five bookmarks named "Demo Act 1" …
"Demo Act 5" and capture the page + filters you want at each moment of the
defense walkthrough. During Q&A, click a bookmark to jump straight to the
right page with the right filters applied.

### 12.5 Accessibility check
**View → Show → Accessibility checker**. Fix any warnings:
- Add alt text to every chart (one sentence — "Bar chart of top 15 SHAP
  features by mean absolute contribution")
- Confirm text contrast against the slate foreground

---

## Step 13 — Publish & hand off (15 min)

### 13.1 Save the .pbix
- **File → Save As → `reports/powerbi/HotelCancellationDashboard_v1.pbix`**
- Add the file to `.gitignore` (it's a binary; the CSVs are the source of truth)

### 13.2 Export to PDF for the thesis appendix
- **File → Export → Export to PDF**
- Save to `docs/thesis_drafts/paper_figures/08_powerbi_dashboard/dashboard_export.pdf`
- Embed in thesis Appendix as Figure A-1

### 13.3 Refresh recipe (give this to the IT team)
Print this on a single page and tape it next to the laptop:

```
TO REFRESH THE DASHBOARD WITH NEW PREDICTIONS

1. Make new predictions:
   - Either via the Gradio UI at localhost:8000/ui
   - Or via /predict on FastAPI

2. Export the live log to CSV:
   python scripts/export_predictions.py

3. (Optional) Re-run drift if a week+ has passed:
   python scripts/compute_live_drift.py

4. Open the .pbix in Power BI Desktop
5. Home > Refresh > wait ~10 s
6. Save the .pbix

That's it. No database, no cloud, no service account.
```

### 13.4 Defense day
Right before the demo, **File → Options → Current File → Privacy →
Always ignore Privacy Level settings**. Without this, Power BI sometimes
prompts on the first refresh of a freshly downloaded file and that pop-up
is awkward in front of a panel.

---

## Verification — how to know the dashboard is done

Run through this checklist:

1. **Every page loads in <3 seconds** after Refresh.
2. **Page 1 cards show**: ROC-AUC 0.864, PR-AUC 0.760, F1 0.735 (current
   champion). If they don't, your `metrics_for_powerbi.csv` is stale —
   regenerate from Step 3 of the pre-flight.
3. **Page 2's slicer survives navigation**: select Year = 2017 on Page 2,
   click through to Pages 3, 4, 5 — the filter should persist (you wired sync
   slicers in Step 6.4).
4. **Page 4 drill-through works**: right-click a row in the top-50 table →
   Drill through → `_BookingDetail` opens with that booking's fields.
5. **Page 6 disconnected slicer**: changing the threshold what-if value
   updates the precision/recall numbers but does NOT filter any other page.
6. **Page 7's bars stay color-grouped by FeatureGroup** (Time / Booking
   source / Demand origin / Price / Guest history / Other).
7. **Page 8's zone count** matches the per-zone tally in
   `data/predictions/drift_metrics.csv`.
8. **Accessibility checker is clean** (View → Show → Accessibility checker
   → no warnings).
9. **Bookmarks cover the five demo acts** from `demo/DEFENSE_RUNBOOK.md`.

If every box ticks, the dashboard is defensible.

---

## What this guide deliberately does *not* cover

- **Power BI Service / online publishing** — out of scope; the thesis
  architecture is local-CSV by design (see Section 4.4.3's implementation
  note). A future cloud migration would replace CSV connectors with a SQL
  gateway but the page structure stays the same.
- **Row-level security** — single-user thesis demo doesn't need it.
- **Custom visuals from AppSource** — every visual in this guide is a
  built-in. Custom visuals add dependencies that can break on a fresh
  laptop the morning of the defense.
- **The Philippine dashboard** — the PH study uses the same architecture
  with `ph_predictions_live.csv` and `reports/ph/ph_test_predictions.csv`
  as sources. Pages 1–8 can be cloned with the PH CSVs as drop-in
  replacements; total build time ~2 h once Portugal is done.

---

## Time estimate summary

| Step | Time | Cumulative |
|---|---|---|
| Pre-flight | 20 min | 0 h 20 min |
| Step 1 — Data model | 40 min | 1 h 00 min |
| Step 2 — Theme & pages | 15 min | 1 h 15 min |
| Step 3 — Measures library | 30 min | 1 h 45 min |
| Step 4 — Page 1 | 40 min | 2 h 25 min |
| Step 5 — Page 2 | 25 min | 2 h 50 min |
| Step 6 — Page 3 | 35 min | 3 h 25 min |
| Step 7 — Page 4 | 45 min | 4 h 10 min |
| Step 8 — Page 5 | 35 min | 4 h 45 min |
| Step 9 — Page 6 | 30 min | 5 h 15 min |
| Step 10 — Page 7 | 25 min | 5 h 40 min |
| Step 11 — Page 8 | 30 min | 6 h 10 min |
| Step 12 — Polish | 40 min | 6 h 50 min |
| Step 13 — Publish | 15 min | 7 h 05 min |
| **Total** | **~7 hours** | — |

Realistic with breaks and learning friction: plan for **two 4-hour
sessions**, one day apart, with a polish pass on day 3.
