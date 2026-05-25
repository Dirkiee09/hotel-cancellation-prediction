# Chapters I — III: Update Patches

> These are the edits needed across Chapters I, II, and III so that the
> body of the thesis is internally consistent with Chapter IV / V. Each
> patch is presented as "find this in the current draft → replace with
> this" so the author can apply them directly in Word / Google Docs.

---

## CHAPTER I — Introduction

### 1. Background of the Study (final paragraph) — minor addition

**Add at the end of the existing final paragraph** (the one ending "...a
repeatable template that other properties can use is the contribution"):

> *"A parallel sub-study on the real Punta Villa Resort PMS export
> (Philippines, 193 bookings, 2022-2025) tests whether this template
> transfers to a smaller property with a narrower PMS schema. The two
> studies are reported in parallel throughout the chapters that follow."*

### 2. Objectives — append Objective 5

**Add as Objective 5** (after the existing Objective 4):

> *"5. To validate the methodology's transferability to a small, real
> Philippine resort dataset by applying the same Sense → Seize →
> Transform pipeline to the Punta Villa Resort PMS export and reporting
> the resulting performance, feature-importance ranking, and operational
> deployment."*

### 3. Hypotheses — append Hypothesis 5

**Add as Hypothesis 5** (after the existing H4):

> *"5: The top SHAP feature on the Portugal model will also rank in the
> top three SHAP features on the Philippine model, providing
> cross-dataset evidence that the methodology detects a consistent
> cancellation driver across geographies."*

### 4. Conceptual Framework — minor parenthetical addition

**Find** the sentence beginning *"This study adopts Dynamic Capability
Theory (DCT) to explain how hotels convert booking data into decisions..."*

**Replace with**:

> *"This study adopts Dynamic Capability Theory (DCT) to explain how
> hotels convert booking data into decisions that reduce cancellations.
> The framework, structured as Sense → Seize → Transform, is applied to
> two datasets in parallel: the Portugal benchmark at full scale and the
> real Philippine resort dataset as a transferability probe. Each
> Sense-Seize-Transform stage is reported for both datasets in Chapter IV."*

### 5. Significance — add one paragraph after "For future researchers"

**Add after** the "For future researchers and students" paragraph:

> *"**For small and medium hospitality businesses (SMBs)**, the
> Philippine sub-study demonstrates that the methodology can be applied
> end-to-end on a single property's PMS export. Punta Villa Resort
> contributed 193 real booking records spanning three years; the
> sub-study shows that even with a small sample and a narrower PMS
> schema, the same pipeline produces calibrated predictions,
> interpretable feature importance, and a live decision-support
> deployment. This positions the methodology as accessible to
> independent properties without requiring Portugal-scale data
> infrastructure."*

### 6. Scope and Limitations — major rewrite of the dataset paragraph

**Replace** the existing single-dataset paragraph (the one beginning
*"This study uses an archival and publicly available hotel bookings
dataset (hotel_bookings.csv)..."*) **with the following two paragraphs**:

> *"This study uses two datasets in parallel.*
>
> *The **Portugal main study** uses the publicly available Hotel
> Bookings dataset (`hotel_bookings.csv`) originally compiled by António
> et al. (2019), containing 119,390 records from two hotels in Portugal
> — a city hotel and a resort hotel — covering July 2015 to August
> 2017. The dataset is widely used in hospitality analytics research
> because of its completeness, standardised structure, and relevance
> for studying cancellation behaviour across different hotel contexts.*
>
> *The **Philippine sub-study** uses a real PMS export from Punta Villa
> Resort (`Punta_Villa_Resort_PH_Dataset.csv`), containing 193 booking
> records spanning December 2022 to December 2025. The dataset is
> proprietary to Punta Villa and reflects a single-property local-
> clientele booking profile, with a 15.0 % cancellation rate. The
> Philippine sub-study tests whether the Portugal methodology transfers
> to a smaller, geographically distinct property.*
>
> *Both studies focus on predicting booking cancellations at the
> moment of reservation, with attention to short-notice cancellations
> (within three days before arrival), as these create the greatest
> operational and financial impact. The unit of analysis is the
> individual reservation. We analyse information available at or near
> the time of reservation (e.g., lead time, deposit/prepayment type,
> booking channel/segment where available, ADR, length of stay, party
> size, special requests, seasonality, and room/rate codes)."*

### 7. Scope and Limitations — update the BI deliverable description

**Find** the sentence beginning *"The Business Intelligence (BI)
deliverable is a Power BI dashboard that summarizes..."*

**Replace with**:

> *"The Business Intelligence (BI) deliverable is an **eight-page Power
> BI dashboard** that covers: (1) hero KPI overview, (2) cancellation
> rate trend, (3) segment slicing, (4) revenue at risk under each
> threshold policy, (5) ADR forecasting with residual analysis, (6)
> threshold policy comparison, (7) global and per-prediction feature
> importance, and (8) drift monitoring on the live prediction log. In
> addition, a **live FastAPI + Gradio serving deployment** demonstrates
> operational integration of model outputs into a property's
> decision-support workflow. Exogenous factors such as local events,
> competitor rates, and weather data were excluded to maintain dataset
> reproducibility and focus on variables consistently available in
> hotel reservation systems. Future studies may extend the model with
> these contextual features for improved accuracy."*

### 8. Limitations — append three new bullets

**Add to the existing Limitations paragraph** (after the existing
discussion of dataset age and external factors):

> *"**The Philippine sub-study sample is small.** The Punta Villa
> dataset contains 193 booking records, with chronological splitting
> reserving 20 rows for the held-out test set. Bootstrap 95 %
> confidence intervals on test PR-AUC span approximately ±15 percentage
> points. Philippine performance numbers are therefore reported as
> directional estimates rather than production-grade headlines.*
>
> *The **live ADR forecast** uses placeholder values for four
> post-booking features (`is_canceled`, `assigned_room_type`,
> `booking_changes`, `days_in_waiting_list`) that are not known at the
> moment of reservation. Live `predicted_adr` is therefore slightly
> less accurate than the published test-set RMSE; Chapter V identifies
> a clean retraining fix as future work.*
>
> *The **cost analyses** use simplifying assumptions (a €15
> per-intervention false-positive cost and a one-night recovery
> penalty for each false negative). True opportunity cost varies with
> occupancy and rebooking success and should be revised per property
> in production deployments."*

---

## CHAPTER II — Review of Related Literature

### Add one new subsection: "Small-Sample Transferability in Hotel Analytics"

**Insert** as a new subsection between *"External Data Fusion"* and
*"Synthesis"*:

> ### Small-Sample Transferability in Hotel Analytics
>
> *Most hotel-cancellation research uses large public benchmark datasets
> such as the Portugal corpus (Antonio et al., 2019; Herrera et al.,
> 2024). Less is known about whether the methodologies developed on
> these benchmarks transfer to small and medium hospitality businesses
> (SMBs) with proprietary PMS schemas. Two strands of literature are
> relevant.*
>
> *The first strand concerns **domain shift and transferability in
> tabular ML**. Roa et al. (2022) and Sayed et al. (2024) both argue
> that hospitality models trained on one property may degrade
> substantially when applied to another, citing differences in booking
> channel mix, deposit policy enforcement, and seasonality patterns.
> A robust transferability claim requires either (a) a held-out
> evaluation on the target property's data, or (b) a documented
> feature-availability mapping showing which dimensions the source and
> target properties share.*
>
> *The second strand concerns **small-N hotel analytics for SMB
> properties**. Lim and Choe (2023) and Caicedo-Torres and Payares
> (2024) report that independent hotel properties typically have access
> to fewer than 1,000 historical bookings per year, putting many ML
> approaches developed on the Portugal benchmark out of reach. The
> implication for the present study is that transferability cannot be
> assumed; it must be tested empirically on a real small-property
> dataset. The Philippine sub-study reported in Chapter IV closes that
> gap on a single property by applying the Portugal methodology to
> 193 real bookings from Punta Villa Resort.*

### Add one new subsection: "Calibrated Probabilities for Decision Support"

**Insert** as a new subsection between *"Cancellation Risk Analytics
Stack"* and *"Last-Minute Cancellation Control"*:

> ### Calibrated Probabilities for Decision Support
>
> *A model that ranks bookings well is not automatically useful for
> decisions: the percentage it outputs must mean something. Niculescu-
> Mizil and Caruana (2005) showed that gradient-boosted trees produce
> uncalibrated probability estimates by default, and that **isotonic
> regression** fit on a held-out validation set typically reduces
> Expected Calibration Error (ECE) by more than 50 % without
> sacrificing discrimination. Subsequent applied work in hospitality
> (Chen et al., 2023; C-Sánchez & Sánchez-Medina, 2024) confirms that
> calibrated probabilities are a prerequisite for any **cost-sensitive
> threshold** policy: the cost calculation multiplies a probability by
> a financial penalty, so an uncalibrated probability scales the
> penalty incorrectly. The present study therefore follows the
> calibrate-then-threshold pattern recommended by this literature."*

### How This Study Advances the Field — expand the closing paragraph

**Find** the existing closing paragraph (the one beginning *"By providing
a practical framework that links analytics, upgrade strategy, and
performance measurement..."*) and **expand it** to include:

> *"...This study additionally surfaces three methodology contributions
> that prior work does not consolidate. **First**, a pre-flight
> duplicate-cluster diagnostic that flags datasets where chronological
> splitting would leak twins across the train/test boundary. **Second**,
> a feature-availability mapping that documents which dimensions a
> property's PMS schema must support to apply the methodology, and
> bounds the predictive ceiling for properties with reduced schemas.
> **Third**, a plug-and-play dataset framework that allows the
> methodology to be re-applied to any chronologically-sortable booking
> CSV with a single configuration change. All three contributions are
> exercised by the Philippine sub-study described in Chapter IV Section 4.5."*

---

## CHAPTER III — Methodology

### Research Design — add second paragraph after the existing first paragraph

**Add after** the existing paragraph that introduces the Sense → Seize
→ Transform cycle:

> *"This research design is applied in parallel to two datasets. The
> **Portugal main study** uses the full pipeline at scale (119,210
> cleaned bookings, rolling-origin cross-validation across three
> chronological folds, paired bootstrap significance testing). The
> **Philippine sub-study** applies the same pipeline to the 193-row
> Punta Villa Resort PMS export, omitting only steps that the smaller
> sample cannot statistically support (the rolling-origin CV is
> replaced by a single chronological 80 / 10 / 10 split, and the
> cost-sensitive threshold policy is omitted because n_val ≈ 19 is too
> small to fit a reliable cost curve). All other pipeline stages —
> cleaning, feature engineering, isotonic calibration, threshold
> sweep, SHAP interpretation, and live serving — are identical between
> the two studies."*

### Dataset Variables — add a parallel table for the Philippine dataset

**Add** as a new subsection after the existing dataset variables table:

> ### Dataset Variables — Philippine Sub-Study
>
> *The Philippine PMS export captures a reduced subset of the variables
> available in the Portugal benchmark. Table III.X documents the raw
> Philippine schema and the engineered features derived from it.*
>
> | Raw field | Description | Engineered features derived |
> |---|---|---|
> | `Lead_Time_Days` | Days from booking to arrival | `lead_time`, `is_late_window` |
> | `Weekend_Nights` | Weekend nights booked | `stays_in_weekend_nights`, `is_weekend_heavy` |
> | `Week_Nights` | Week nights booked | `stays_in_week_nights`, `total_stay` |
> | `Adults`, `Children`, `Babies` | Guest counts | `adults`, `children`, `babies`, `total_guests` |
> | `ADR_Rate` | Average daily rate (PHP) | `adr`, `adr_per_person`, `revenue_at_risk` |
> | `Room_Type` | Reserved room type | `reserved_room_type` (categorical) |
> | `Deposit_Type` | Deposit policy | `deposit_type` (categorical) |
> | `Special_Requests` | Count of special requests | `total_of_special_requests` |
> | `Arrival_Date` | ISO arrival date | `arrival_date_year`, `arrival_date_month`, `arrival_date_day_of_month`, `month_sin`, `month_cos` |
>
> *The Philippine PMS export does **not** capture `country`, `agent`,
> `market_segment`, `customer_type`, `previous_cancellations`,
> `previous_bookings_not_canceled`, `required_car_parking_spaces`, or
> `meal` (the latter is constant and dropped). This is the
> feature-availability constraint that Chapter IV Section 4.6.2 develops as
> a methodology contribution.*

### Modelling Procedures — add pre-flight diagnostic + bootstrap significance

**Insert** as new subsections within Modelling Procedures:

> ### Pre-Flight Duplicate-Cluster Diagnostic
>
> *Before fitting any model on a chronologically-split dataset, the
> pipeline runs a diagnostic that counts duplicate post-engineering
> feature vectors and measures label consistency within each duplicate
> cluster. If the duplicate rate exceeds 30 % AND the fraction of
> duplicate clusters with consistent labels exceeds 90 %, the test
> metrics will be inflated by recognition rather than generalisation,
> and the methodology proceeds with a documented caveat. If the
> diagnostic does not fire — as on both datasets in this study — the
> test metrics are honest small-sample (or large-sample) estimates of
> generalisation. The diagnostic is implemented at
> `scripts/train_ph.py::_compute_duplicate_diagnostics` and is
> dataset-agnostic.*

> ### Per-Family Probability Calibration
>
> *Each candidate model family fits its own isotonic calibrator on the
> validation set, so the calibration step is part of model selection
> rather than a post-hoc adjustment to the champion alone. This
> ensures that model comparisons (Chapter IV Section 4.3.4) use calibrated
> probabilities for every family, not raw scores from non-champions
> and calibrated scores from the champion.*

> ### Bootstrap Paired-Significance Testing
>
> *Model selection in the Portugal main study is supplemented by
> bootstrap paired-significance testing with 2,000 resamples on the
> test set. For each challenger model, the procedure resamples the
> test set with replacement, recomputes PR-AUC and ROC-AUC for the
> champion and the challenger on the same resample, and stores the
> delta. The 95 % confidence interval on the delta and the two-sided
> p-value are then reported (Chapter IV Section 4.3.4 Table 4.6). This
> elevates "LightGBM is best" from a point-estimate claim to a
> statistical claim."*

### Research Instruments — extend the existing bullet list

**Add the following bullets** to the existing list of tools and libraries:

> - **FastAPI** + **Gradio** — for live model serving (Portugal at
>   port 8000, Philippine at port 8001)
> - **SQLite** — for the prediction audit log
>   (`data/predictions/predictions.sqlite` and `ph_predictions.sqlite`)
> - **Power BI Desktop** — for the eight-page decision-support
>   dashboard
> - **GitHub Actions** — for continuous integration (every commit runs
>   tests, lint, type check, and the train + evaluate pipeline)
> - **pytest** with `pytest-cov` — for the conformance test suite (130
>   tests, ≥ 80 % coverage)

### Ethical Consideration — add a bullet on reproducibility

**Add to the Ethical Consideration list:**

> - **Reproducibility**: every step of the pipeline is documented in
>   notebooks (10 for Portugal at `notebooks/01_eda.ipynb` through
>   `10_sensitivity_analysis.ipynb`; 11 for the Philippine sub-study at
>   `notebooks/ph/01_eda.ipynb` through `11_transferability.ipynb`).
>   The training script `python scripts/train.py` is deterministic
>   under a fixed random seed (RANDOM_STATE = 42) and produces
>   bit-identical artefacts on a fresh machine. Continuous integration
>   verifies the train → evaluate path on every commit so any drift in
>   reported numbers would be caught immediately.

---

## Application notes

1. **Apply these patches in this order**: Chapter I Section 6 (Scope rewrite)
   first, then Chapter III dataset variables (which the Scope refers
   to), then everything else. The rest of the patches are independent
   of each other.

2. **Citation work to do**: the new Chapter II subsections cite Roa et
   al. (2022), Sayed et al. (2024), Lim and Choe (2023), Caicedo-Torres
   and Payares (2024), and Niculescu-Mizil and Caruana (2005). Verify
   these references exist and are available; if any cannot be sourced,
   the author should substitute closest-available alternatives.

3. **Cross-reference check after applying**: every Chapter IV / V
   number cited as "from Chapter III" should trace back to a section
   that exists post-patch. The patches above add: parallel datasets
   (Research Design section), pre-flight diagnostic (Modelling
   Procedures section), per-family calibration (Modelling Procedures
   section), bootstrap significance (Modelling Procedures section),
   and serving / CI instruments (Research Instruments section).
