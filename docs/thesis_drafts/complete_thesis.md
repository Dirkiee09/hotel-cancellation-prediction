# A Strategic Business Intelligence Approach to Predicting Hotel Booking Cancellations

**Authors:** Avanceña, Luis Miguel C. · Montecino, Nathaniel · Viñas, Dirk Werner

**Thesis Advisers:** Prof. John Edward Manalac · Dr. Donn Enrique L. Moreno

---

## Abstract

Hotel booking cancellations are a persistent revenue leak, especially when they occur close to the arrival date and rooms cannot be resold at comparable rates. This study applies Dynamic Capability Theory — operationalised as a Sense → Seize → Transform cycle — to design a cost-sensitive predictive framework that estimates cancellation risk, quantifies its financial impact, and converts predictions into revenue-protection policies. The methodology is developed in parallel on two datasets: the Portugal Hotel Bookings corpus (119,210 reservations, 2015–2017) and a real Property Management System export from Punta Villa Resort, Philippines (193 reservations, 2022–2025), the latter serving as a transferability probe for small Philippine independent properties.

Six supervised classifiers were trained, calibrated, and evaluated under chronological 80 / 10 / 10 splits. On the Portugal test set, the champion LightGBM model achieved ROC-AUC = 0.864, PR-AUC = 0.760, and post-isotonic Expected Calibration Error of 0.029, outperforming every baseline with statistical significance under paired bootstrap testing. On the Philippine sub-study, LightGBM again emerged as champion (PR-AUC = 0.542 vs class baseline 0.150). The strongest cross-dataset finding is that **`deposit_type` ranks as the most influential SHAP feature on both datasets** — evidence that the methodology detects a genuine, transferable cancellation driver rather than a dataset-specific artefact. A cost-sensitive decision threshold reduced expected losses on the Portugal test set by approximately **95.4 % (≈ €1.53 million)** versus operating without a model.

The study delivers a calibrated LightGBM champion, an 8-page Power BI decision-support dashboard, a live FastAPI + Gradio serving deployment with per-prediction SHAP, and two reusable methodology contributions — a pre-flight duplicate-cluster diagnostic and a feature-availability mapping for reduced-PMS schemas — providing a repeatable framework that bridges predictive analytics and strategic hotel revenue management across both large benchmark and small real-world property data.

**Keywords:** hotel booking cancellation, predictive analytics, Dynamic Capability Theory, gradient boosting, probability calibration, cost-sensitive thresholding, SHAP, Philippine hospitality, business intelligence, revenue management.

---

## Table of Contents

- [Abstract](#abstract)
- [Chapter I — Introduction](#chapter-i--introduction)
  - Background of the Study
  - Statement of the Problem
  - Research Questions
  - Objectives of the Study
  - Hypothesescha
  - Theoretical Framework
  - Conceptual Framework
  - Significance of the Study
  - Unique Contribution
  - SDG Alignment
  - Scope and Limitations
  - Limitations
  - Definition of Terms
- [Chapter II — Review of Related Literature](#chapter-ii--review-of-related-literature)
- [Chapter III — Methodology](#chapter-iii--methodology)
- [Chapter IV — Results and Discussion](#chapter-iv--results-and-discussion)
- [Chapter V — Conclusion](#chapter-v--conclusion)
- [References](#references)

---

## List of Figures

- Figure 1. Simplified schema of dynamic capabilities, business models, and strategy (Teece, 2018).
- Figure 2. Developed Conceptual Framework for Predicting Hotel Booking Cancellations.
- Figure 4.1. ROC and PR curves for the Portugal LightGBM champion (`reports/figures/thesis/fig_01_roc_pr_curves.png`).
- Figure 4.2. Calibration reliability diagram and probability histogram (`fig_05_calibration_reliability_and_histogram.png`).
- Figure 4.3. SHAP feature importance bar (`fig_13_shap_feature_importance_bar.png`).
- Figure 4.4. SHAP beeswarm (`fig_14_shap_beeswarm.png`).
- Figure 4.5. Cost-sensitive threshold sweep (`fig_11_cost_sensitive_threshold_sweep.png`).
- Figure 4.6. Risk tier business overview (`fig_23_risk_tier_business_overview.png`).
- Figure 4.7. Cross-dataset SHAP comparison (`reports/figures/thesis/ph/fig_5.4_ph_vs_pt_shap_comparison.png`).

---

# CHAPTER I — Introduction 

Introduction 

## Background of the Study

 Hotels lose expected revenue when guests cancel close to arrival, since those rooms are hard to resell and the loss is larger for longer stays and higher ADRs. Prior work shows cancellations follow consistent patterns by lead time, deposit or prepayment terms, booking channel or segment, pricing, and guest history, so predictive models can flag risky bookings early and support revenue protection (António et al., 2017; Chen et al., 2023) . Evidence from comparative studies also indicates that strong data preparation plus modern models perform well, and that lead time, deposit type, channel, and ADR repeatedly emerge as key predictors across datasets and methods (Herrera et al., 2024; Yang, 2024). Short-notice cancellations cause the biggest losses because rooms are unlikely to be rebooked when guests cancel only a few days before arrival. Research on this late window finds that forecasting becomes harder as check-in nears but still provides useful signals for targeted actions such as reminders, deposits or prepayments, flexible rebooking, and carefully calibrated overbooking. To align modeling with business value, the classifier should be validated with time-aware splits and its decision threshold chosen using expected cost rather than accuracy so high-risk bookings trigger interventions while low-risk bookings avoid unnecessary friction (C-Sánchez & Sánchez-Medina, 2024; Andriawan et al., 2020; Chen et al., 2023). Despite advancements in prediction, a lot of properties continue to rely on manual checks or general rules that fail to account for last-minute cancellations and fail to adjust actions to cost. In order to prevent losses, this study will create a supervised model to estimate cancellation risk with a focus on late-window scenarios. It will then be validated using time-aware splits to determine a decision threshold based on expected cost, ensuring that low-risk bookings prevent needless delay and high-risk bookings trigger targeted actions. An operations-ready playbook that lowers late cancellation losses, connects model outputs to decisions that save money, and provides a repeatable template that other properties can use is the contribution. A parallel sub-study on the real Punta Villa Resort PMS export (Philippines, 193 bookings, 2022-2025) tests whether this template transfers to a smaller property with a narrower PMS schema. The two studies are reported in parallel throughout the chapters that follow. 

The relevance of this work extends beyond the Portuguese benchmark
dataset on which much of the global cancellation-prediction literature
has been built. In the Philippines, the hospitality industry is
dominated by small- and medium-sized resorts, boutique hotels, and
single-property operators whose Property Management Systems capture
narrower booking schemas and substantially less historical data than
international chains. The post-pandemic recovery of Philippine tourism
— particularly in regional destinations such as Iloilo, Bohol, Cebu,
and Palawan — has been accompanied by volatile demand, increasingly
flexible cancellation policies, and a heavy reliance on third-party
online travel agencies whose cancellation behaviour differs materially
from direct bookings. A predictive framework that performs well only on
six-figure benchmark datasets offers limited utility to a Philippine
operator with two thousand annual bookings; conversely, a framework
that quietly degrades when transferred to such a setting risks
misleading managers into over-confidence. This study therefore couples
its Portugal main study with a parallel Philippine sub-study on a real
PMS export from Punta Villa Resort, treating transferability as a
falsifiable empirical question rather than a rhetorical assumption.

## Statement of the Problem

 In today’s hospitality industry, hotel cancellations have become one of the most persistent challenges affecting revenue management and customer relations. While flexible cancellation policies are designed to improve guest satisfaction and encourage bookings, they can also lead to unpredictable income and wasted room inventory when not properly managed. Frequent or last-minute cancellations make it difficult for hotels to forecast occupancy, allocate resources efficiently, and maintain profitability. This creates a constant struggle for hotel managers to balance customer convenience with the financial stability of their operations. Although hotels collect large volumes of guest and booking data, many still rely on manual judgment or traditional methods when handling cancellations. These approaches often overlook valuable insights that can be discovered through data analytics, such as identifying cancellation trends, recognizing high-risk bookings, and predicting potential revenue loss. Without a data-driven system, hotels face inconsistent decision-making and miss opportunities to recover from cancellations or rebook available rooms quickly. Therefore, this study aims to explore how Business Intelligence and Analytics can be used to improve decision-making related to hotel cancellations. Specifically, it seeks to determine how analytical tools can (1) predict which bookings are most likely to be canceled, (2) measure the financial impact of cancellations on hotel performance, and (3) develop a data-driven model that helps hotel management respond proactively to minimize losses. By applying data analysis and visualization, this study intends to guide hotels toward more strategic and evidence-based approaches that reduce cancellation risks while sustaining profitability and customer satisfaction. 

## Research Questions

To translate the broad problem statement into a falsifiable analytical
agenda, the study is guided by the following research questions:

1. **RQ1 — Drivers.** Which booking-time variables (lead time, deposit
   type, channel/segment, ADR, guest history, special requests, room
   type, length of stay, and party composition) are most strongly
   associated with the likelihood that a hotel reservation will be
   cancelled before arrival?

2. **RQ2 — Modelling.** Among supervised classifiers (Logistic
   Regression, Decision Tree, Naive Bayes, Random Forest, and the
   gradient-boosted family — XGBoost, LightGBM, GradientBoosting),
   which model achieves the best out-of-time discrimination (ROC-AUC,
   PR-AUC) and the lowest expected business cost under a chronological
   train / validation / test split?

3. **RQ3 — Interpretation.** Using SHapley Additive exPlanations
   (SHAP), which features dominate the champion model's per-prediction
   reasoning, and do their rankings correspond to those predicted by
   the literature (lead time, deposit type, previous cancellations)?

4. **RQ4 — Decision support.** Can a cost-sensitive decision threshold,
   chosen to minimise expected loss under asymmetric false-positive and
   false-negative costs, reduce expected revenue loss compared to (a)
   running operations without any model and (b) using a default 0.50
   threshold?

5. **RQ5 — Transferability.** When the same Sense → Seize → Transform
   pipeline is applied to a smaller, single-property Philippine PMS
   dataset, does the methodology continue to operate honestly (no
   chronological-split leakage), and does the resulting model surface
   a feature-importance ranking consistent with the Portugal benchmark?

These research questions structure Chapter IV's discussion: Section 4.2
addresses RQ1, Section 4.3 addresses RQ2 and RQ3, Section 4.4 addresses RQ4, and
Section 4.5 addresses RQ5. Each research question is paired with one or
more falsifiable hypotheses (H1–H5) stated in the next section.

## Objectives of the Study

 This study aims to address the financial uncertainty caused by hotel booking cancellations by developing a predictive model and translating its insights into actionable revenue management strategies. The research provides a practical, data-driven approach for hotels to proactively mitigate the risks associated with cancellations. To achieve this, the study will pursue the following objectives: 1. To identify and analyze the primary factors and patterns that correlate with booking cancellations by conducting a thorough exploratory data analysis of the historical dataset. 2. To develop and evaluate a range of machine learning models to determine the most accurate and reliable predictor of cancellation risk, using appropriate performance metrics such as the Accuracy, Recall (RC), F1 Score, Precision (PR), and Area Under The Curve (AUC) curve. 3. To interpret the feature importance of the best-performing model, translate its technical outputs into a clear understanding of what key variables drive cancellation predictions. 4. To build a Power BI decision-support dashboard that converts the model’s insights into specific, cost-sensitive policy recommendations, including dynamic deposit rules and optimized overbooking levels, to help hotels reduce revenue loss.

5. To validate the methodology's transferability to a small, real Philippine resort dataset by applying the same Sense → Seize → Transform pipeline to the Punta Villa Resort PMS export and reporting the resulting performance, feature-importance ranking, and operational deployment. 

## Hypothesis

 1: Lead time, deposit type, and previous cancellations are significant predictors of cancellation risk. 2: A gradient-boosted tree model will achieve a higher Model Evaluation than baseline models (e.g., logistic regression, XgBoost, random forest) on an out-of-time test set. 3: In the best model, lead time will have the greatest SHAP importance, followed by deposit type and previous cancellations. 4: A cost-minimizing threshold with risk-based deposit tiers will reduce expected revenue loss versus current Business operations.

5: The top SHAP feature on the Portugal model will also rank in the top three SHAP features on the Philippine model, providing cross-dataset evidence that the methodology detects a consistent cancellation driver across geographies. 

## Theoretical Framework

 Figure 1. Simplified schema of dynamic capabilities, business models, and strategy (Teece,2018). This study utilizes Dynamic Capability Theory (DCT) as its primary framework to explore how firms achieve performance in dynamic environments. DCT posits that organizations can enhance their performance by identifying opportunities and threats, committing resources to seize these opportunities, and reconfiguring their assets to maintain competitive advantages (Teece, 1997; Teece, 2007). Dynamic capabilities are defined as higher-order routines that allow firms to shape, integrate, and reallocate ordinary resources, distinguishing them from routine operational capabilities (Eisenhardt & Martin, 2000) In the context of hotel booking management, DCT highlights the strategic importance of data and machine learning, revealing how these tools can be used to gain a managerial edge. In this paper, "Sensing" is defined as the organizational capability to identify, interpret, and assess opportunities and threats in the business environment. (Teece, 1997). In this research, sensing is exemplified by the curation of booking records, monitoring of channel and segment signals, and employing exploratory analysis to identify patterns associated with cancellation risks. This aspect relies heavily on information processing and analytical routines that can detect changes in guest behavior before they adversely affect revenue. The seizing phase involves converting detected signals into actionable decisions. This includes the development, validation, and calibration of predictive models related to cancellations, interpretation of driving factors, and the selection of cost-effective operational strategies to balance false positives and negatives. In this phase, resources are allocated to decision-making frameworks, such as deposit tiers and overbooking buffers, ensuring that analytical insights are effectively implemented. Transforming refers to the realignment of organizational structures, processes, and assets to embed the learning derived from these insights (Teece, 1997). Specifically, it involves integrating the predictions produced by analytical models into the hotel's Property Management System (PMS) and Customer Relationship Management (CRM) system (Sharma et al., 2014).This integration enables informed decision-making regarding pricing, inventory management, and guest interactions. Moreover, transformation ensures alignment with market changes and fosters a continuous learning cycle that reinforces dynamic capabilities. Additionally, DCT sheds light on the microfoundations that connect analytics to tangible outcomes, emphasizing the vital role of data governance. Previous research in Information Systems has demonstrated that IT-enabled capacities for sensing and seizing such as data integration, analytical routines, and rapid decision-making processes—serve as mediators in transforming information resources into performance outcomes (Pavlou & El Sawy, 2011). In this study, analytical capability is identified as that critical microfoundation: high-quality data and transparent, calibrated models enhance sensing; cost-sensitive decision rules and disciplined deployments bolster seizing; and continuous monitoring and retraining facilitate effective transformation. This supports the integration of Business Intelligence and Analytics (BIA) as essential tools for turning data into actionable insights. Through data visualization, predictive modeling, and performance dashboards, hotel managers can continuously monitor cancellation patterns, evaluate financial outcomes, and make informed decisions that enhance profitability and operational stability. In this way, the theory not only strengthens evidence-based management but also supports long-term strategic planning in the hospitality industry. 

## Conceptual Framework

 This study's conceptual framework draws on Teece's (2007, 2012) and Teece et al.'s (1997) dynamic capabilities theory, specifically focusing on three categories: sensing, seizing, and reconfiguring. This framework is vital in competitive sectors, where even slight improvements in prediction and resource distribution can greatly boost profits (Mele et al, 2023). To operationalize this theory, the research adopts a three-stage framework that guides the study from initial data collection to final strategic recommendations. This framework provides a clear, structured, and repeatable methodology for transforming data into business value. Figure 2. Developed Conceptual Framework for Predicting Hotel Booking Cancellations This study adopts Dynamic Capability Theory (DCT) to explain how hotels convert booking data into decisions that reduce cancellations. The framework, structured as Sense → Seize → Transform , details how each stage facilitates the conversion of predictions into clear policies. This study utilizes Dynamic Capability Theory (DCT) as its primary framework to explore how firms achieve performance in dynamic environments. It suggests that organizations can boost their performance by recognizing opportunities and threats, allocating resources to capitalize on them, and reconfiguring assets to sustain competitive advantages. This process necessitates leveraging new "enabling technologies" such as data and machine learning (Teece, 2018). Dynamic capabilities are defined as higher-order routines that allow firms to shape, integrate, and reallocate ordinary resources, distinguishing them from routine operational capabilities (Eisenhardt & Martin, 2000; Teece, Pisano, & Shuen, 1997). The researcher applies the (DCT) framework. This framework highlights the strategic significance of data and machine learning as tools for achieving a competitive advantage. Within this framework, the hotel's data and machine learning tools are conceptualized as a core "Big Data Analytics Capability." This capability is recognized as a crucial resource that has been empirically demonstrated to enhance firm performance and foster competitive advantage (Mikalef et al., 2019). The "sensing" phase is the foundational component of Dynamic Capability Theory (DCT) , representing the organizational capacity to perceive, identify, and interpret changes, threats, or new opportunities within the business environment. This capability is essential for adaptation and maintaining a competitive advantage, particularly in dynamic sectors like the digital-era hospitality industry. Sensing measures the extent to which Hotels sense changes (market changes, policy changes, technology changes, competitor changes, customer changes) in the internal and external environments. (Pereira-Moliner et al., 2021) This study specifically measures the extent to which booking cancellations respond to macro and micro changes that influence the operation of the business. It denotes an ability to carry out internal scanning to identify changes that businesses need to address. Use of exploratory analysis to identify patterns associated with cancellation risks. This aspect relies heavily on information processing and analytical routines that can detect changes in guest behavior before they adversely affect revenue. The seizing phase converts sensed signals into explicit, auditable choices by developing and validating predictive models, interpreting their drivers, and selecting cost-effective operating policies. In DCT, seizing entails committing resources and making strategic choices that exploit identified opportunities and mitigate threats (Teece, 2017), including calibrating and executing decisions under uncertainty. In this study, seizing is operationalized by training and tuning cancellation-risk classifiers (logistic regression, random forests, gradient-boosted trees), calibrating their probabilities for reliability, and explaining feature effects to support managerial use; probability calibration and data visualizations to guide the conversion of scores into trustworthy risk estimates. Hotel-specific research shows that such ML systems for cancellation prediction can be built with interpretable mechanisms and embedded into decision support for revenue management (Chen et al., 2023). In the transform phase, capability, the final phase of the Dynamic Capability Theory (DCT) framework, involves the reconfiguration of the organization's structure, processes, and assets to capitalize on the insights developed during the "seizing" phase (Teece, 2017). Specifically, it involves integrating or prescribing the predictions produced by machine learning models to adapt to changes in the hotel's operation, management, and strategic pricing. This integration enables informed decision-making regarding pricing, operations management, and guest interactions. Moreover, transformation ensures alignment with market changes and fosters a continuous learning cycle that reinforces dynamic capabilities. Additionally, DCT sheds light on the microfoundations that connect analytics to tangible outcomes, emphasizing the vital role of data governance. Previous research in Information Systems has demonstrated that IT-enabled capacities for sensing and seizing such as data integration, analytical routines, and rapid decision-making processes serve as mediators in transforming information resources into performance outcomes (Pavlou & El Sawy, 2011). In this study, analytical capability is identified as that critical microfoundation: high-quality data and transparent, calibrated models enhance sensing; cost-sensitive decision rules and disciplined deployments bolster seizing; and continuous monitoring and retraining facilitate effective transformation In summary, the conceptual framework of this study is directly influenced by the principles of Dynamic Capability Theory (DCT) . It provides a structured pathway that integrates the technical stages of predictive modeling with the crucial human elements of interpretation and strategic planning. This synthesis provides a robust foundation for developing and justifying evidence-based policies aimed at reducing hotel booking cancellations. 

## Significance of the Study

 This study is significant because it provides valuable insights into how Business Intelligence and Analytics (BIA) can transform the way hotels manage booking cancellations. By applying data-driven approaches, this research aims to help hotel managers make smarter and more strategic decisions that balance guest satisfaction with financial stability. For hotel management , the findings can serve as a foundation for developing intelligent systems that predict cancellations, reduce revenue losses, and improve operational planning. With data analytics, hotels can better forecast occupancy, adjust pricing strategies, and implement preventive measures to minimize the negative effects of cancellations. For the hospitality industry , this study contributes to the growing need for technology-driven solutions that enhance competitiveness and efficiency. It demonstrates how the integration of BIA can lead to more resilient business models, especially in an era where customer behavior and market conditions are increasingly unpredictable. For future researchers and students , the study can serve as a reference for exploring similar applications of data analytics in other areas of hospitality management, such as customer satisfaction, pricing optimization, and loyalty programs. Overall, this study highlights the importance of evidence-based decision-making in addressing hotel cancellations. It aims to support hotels in achieving a more sustainable balance between customer experience and profitability through the effective use of data intelligence. 

Beyond the immediate audience of hotel revenue managers, this study
holds particular significance for the Philippine business intelligence
and information systems community. Among Philippine independent
properties, predictive analytics adoption remains uneven: while large
chains operate sophisticated revenue management systems, the SMB
segment — which forms the majority of the country's hospitality
industry — typically relies on judgment-based decision-making and
generic property management software that produces little or no
predictive output. The Punta Villa Resort sub-study reported in
Chapter IV demonstrates that a methodology developed and validated on
a six-figure international benchmark can be transferred to a real,
193-booking Philippine PMS export and still produce honest, calibrated
predictions with operationally meaningful feature-importance findings.
This positions the study not merely as an academic exercise but as a
concrete proof point for Philippine SMB hotels considering the
adoption of data-driven revenue protection. For the academe — and
particularly for undergraduate Business Intelligence and Information
Systems programmes in the Philippines — the study provides a
fully-reproducible, version-controlled, continuous-integration-verified
project template against which similar capstone work can be
benchmarked. All code, notebooks, trained artefacts, and live serving
infrastructure are documented and reproducible from a single Git
repository, lowering the barrier to replication for future student
researchers.

## Unique Contribution

 This research extends existing hotel analytics studies by integrating predictive machine learning models with Dynamic Capability Theory (DCT) to explain how hotels can sense, seize, and transform data into strategic actions. Unlike prior works that stop at prediction, this study introduces a cost-sensitive decision framework that links predicted cancellation risk to operational policies such as deposits, reminders, and overbooking adjustments. The integration of these predictive insights into a Power BI dashboard provides a practical, decision-support tool for hotel managers bridging the gap between technical analytics and day-to-day managerial decision-making. By applying established algorithms (logistic regression, random forest, and gradient-boosted trees) to a benchmark dataset in a new managerial context , this study contributes both to the academic discussion of predictive analytics and to the operational practice of data-driven revenue management in hospitality. 

## Sustainable Development Goal (SDG) Alignment

 This research supports United Nations Sustainable Development Goal (SDG) 9: Industry, Innovation, and Infrastructure , by promoting innovation and digital transformation in the hospitality industry. Through the integration of business intelligence, machine learning, and predictive analytics, the study encourages hotels to adopt data-driven innovation that improves operational efficiency, minimizes revenue loss, and enhances competitiveness in a rapidly evolving market. 

## Scope and Limitations

 This study uses **two datasets in parallel**. The **Portugal main study** uses the publicly available Hotel Bookings dataset (`hotel_bookings.csv`) originally compiled by António et al. (2019), containing 119,390 records from two hotels in Portugal — a city hotel and a resort hotel — covering July 2015 to August 2017. The dataset is widely used in hospitality analytics research because of its completeness, standardised structure, and relevance for studying cancellation behaviour across different hotel contexts. The **Philippine sub-study** uses a real PMS export from Punta Villa Resort (`Punta_Villa_Resort_PH_Dataset.csv`), containing 193 booking records spanning December 2022 to December 2025. The dataset is proprietary to Punta Villa and reflects a single-property local-clientele booking profile, with a 15.0 % cancellation rate. The Philippine sub-study tests whether the Portugal methodology transfers to a smaller, geographically distinct property. The unit of analysis is the individual reservation. We analyze information available at or near the time of reservation (e.g., lead time, deposit/prepayment type, booking channel/segment, ADR, length of stay, party size, repeat-guest status, special requests, seasonality, and room/rate codes). We apply a quantitative predictive approach: clean the data, engineer features (e.g., total length of stay), use time-based train/validation/test splits to avoid leakage, compare a logistic regression baseline with tree-based models (Random Forest and Gradient-Boosted Trees), calibrate probabilities, and choose a cost-based decision threshold that weighs late-cancellation loss against intervention cost. Lead time defined as the number of days between the booking and arrival dates serves as a temporal anchor feature. The time-based data-splitting strategy ensures the model only uses information available before each stay, avoiding look-ahead bias. The Business Intelligence (BI) deliverable is an **eight-page Power BI dashboard** that covers: (1) hero KPI overview, (2) cancellation rate trend, (3) segment slicing, (4) revenue at risk under each threshold policy, (5) ADR forecasting with residual analysis, (6) threshold policy comparison, (7) global and per-prediction feature importance, and (8) drift monitoring on the live prediction log. In addition, a **live FastAPI + Gradio serving deployment** demonstrates operational integration of model outputs into a property's decision-support workflow. Exogenous factors such as local events, competitor rates, and weather data were excluded to maintain dataset reproducibility and focus on variables consistently available in hotel reservation systems. Future studies may extend the model with these contextual features for improved accuracy.

## Limitations

Results depend on data quality (missing values, coding inconsistencies, label errors). Important external factors (competitor promos, shocks, local events) are not included and may affect cancellations and rebooking. Generalizability beyond the properties/periods in this dataset is limited; behavior may drift over time, requiring retraining and recalibration. Even with time-aware splits, some temporal leakage is still possible. Estimated revenue at risk uses ADR, length of stay, and a one-night penalty assumption; true opportunity cost varies with occupancy and rebooking success. Finally, estimated effects of interventions (e.g., reminders or deposits) are based on backtests, not randomized trials, so real-world impact can differ. Another limitation lies in the age of the dataset, which covers bookings from 2015 to 2017. Customer preferences, digital booking channels, and travel behaviors have significantly evolved in recent years especially after the pandemic and the rise of flexible booking policies so the observed patterns may not fully reflect current industry conditions. This makes the model more useful as a methodological framework or prototype, rather than a ready-to-deploy system. To ensure ongoing relevance, future implementations should use updated or live hotel data and perform periodic retraining to capture new trends. Generalizability beyond the properties and periods in this dataset is also limited; customer behavior may drift over time, requiring model recalibration. Even with time-aware splits, some temporal leakage is still possible. The revenue-at-risk estimate uses ADR, length of stay, and a one-night penalty assumption; true opportunity cost may vary with occupancy and rebooking success. Finally, the estimated effects of managerial interventions (e.g., reminders or deposits) are based on simulation and backtesting, not randomized field experiments, so real-world impact may differ.

**The Philippine sub-study sample is small.** The Punta Villa dataset contains 193 booking records, with chronological splitting reserving 20 rows for the held-out test set. Bootstrap 95 % confidence intervals on test PR-AUC span approximately ±15 percentage points. Philippine performance numbers are therefore reported as directional estimates rather than production-grade headlines.

The **live ADR forecast** uses placeholder values for four post-booking features (`is_canceled`, `assigned_room_type`, `booking_changes`, `days_in_waiting_list`) that are not known at the moment of reservation. Live `predicted_adr` is therefore slightly less accurate than the published test-set RMSE; Chapter V identifies a clean retraining fix as future work.

The **cost analyses** use simplifying assumptions (a €15 per-intervention false-positive cost and a one-night recovery penalty for each false negative). True opportunity cost varies with occupancy and rebooking success and should be revised per property in production deployments.

## Definition of Terms

To ensure consistent interpretation of the technical and managerial
vocabulary used throughout this thesis, the following terms are
defined operationally:

- **Booking-time feature.** A reservation attribute that is known and
  recorded at the moment the booking is made (e.g., lead time, ADR,
  deposit type, market segment). The model uses only booking-time
  features to predict cancellation, ensuring that no post-booking
  information leaks into training.

- **Cancellation (operational definition).** A confirmed reservation
  that is withdrawn by the guest or the property before the guest's
  scheduled arrival date. In the Portugal dataset this corresponds to
  the binary target `is_canceled = 1`; in the Philippine PMS export it
  corresponds to a reservation flagged with cancellation status prior
  to check-in.

- **Lead time.** The number of calendar days between the booking date
  and the scheduled arrival date. Lead time serves as the canonical
  temporal anchor for the chronological train / validation / test
  split.

- **ADR (Average Daily Rate).** Total lodging revenue for the booking
  divided by the total number of staying nights, expressed in the
  property's reporting currency (EUR for Portugal, PHP for Punta
  Villa). ADR is used both as a predictor and as a multiplier in the
  revenue-at-risk calculation.

- **Revenue at risk.** The expected lodging revenue that would be lost
  if a booking is cancelled, computed as ADR × total staying nights
  (with the property's specific penalty structure applied where
  available). This is the financial exposure the model is designed to
  protect.

- **Calibration.** The property that, among bookings the model assigns
  a probability of 0.30, approximately 30 % actually cancel. A model
  may rank bookings perfectly (high ROC-AUC) yet still be poorly
  calibrated. This study uses **isotonic regression** fitted on the
  validation set to align predicted probabilities with observed
  cancellation frequencies; the calibration gap is measured by
  Expected Calibration Error (ECE).

- **Decision threshold.** The probability cut-off above which a
  booking is classified as "will cancel" and an intervention (deposit,
  reminder, overbooking buffer) is triggered. Three policies are
  reported: `max_f1` (the threshold that maximises F1-score on the
  validation set), `high_precision` (the threshold satisfying
  Precision ≥ 0.98), and `cost_sensitive` (the threshold that
  minimises total expected business cost).

- **Cost-sensitive thresholding.** A decision rule that chooses the
  probability cut-off by minimising the expected combined cost of
  false positives (intervention cost) and false negatives (lost
  revenue at risk). For Portugal, this study fixes the false-positive
  cost at €15 per intervention and computes the false-negative cost
  as the booking's revenue at risk.

- **SHAP (SHapley Additive exPlanations).** A model-agnostic
  game-theoretic framework that decomposes a single prediction into
  per-feature contributions, with desirable properties of local
  accuracy and consistency (Lundberg & Lee, 2017). This study uses
  TreeSHAP to compute both global feature importance (mean absolute
  SHAP across the test set) and per-prediction explanations served by
  the live FastAPI endpoint.

- **Out-of-time test.** Evaluation on a set of bookings that arrive
  strictly later than all bookings used for training and threshold
  selection. Out-of-time evaluation is necessary to avoid the
  look-ahead bias that arises from random splits in time-series data.

- **Rolling-origin cross-validation.** A model-selection protocol in
  which the training window is progressively expanded across multiple
  chronological cutoffs (60 %, 70 %, 80 % of training data in this
  study). The champion family is chosen by mean PR-AUC across folds,
  giving a more robust selection signal than a single point estimate.

- **Pre-flight duplicate-cluster diagnostic.** A dataset-agnostic
  check that counts duplicate post-engineering feature vectors and
  measures label consistency within each duplicate cluster. If
  duplicate rate ≥ 30 % AND consistent-label cluster percentage ≥
  90 %, chronological splits will leak twin bookings and the test
  metrics will reflect recognition rather than generalisation. This
  diagnostic, developed during the Philippine sub-study, is the
  study's first methodology contribution.

- **Risk tier.** A coarse-grained operational segmentation of bookings
  into LOW (P(cancel) < 0.40), MEDIUM (0.40 ≤ P(cancel) < 0.70), and
  HIGH (P(cancel) ≥ 0.70) categories, each mapped to a recommended
  managerial action (standard handling, reminder, deposit / call).

- **Dynamic Capability Theory (DCT).** A strategic-management
  framework (Teece, 2007, 2018) that conceptualises a firm's ability
  to **sense** market opportunities and threats, **seize** them by
  committing resources and making strategic choices, and
  **transform** organisational structures, processes, and assets to
  realise the resulting value. This study operationalises DCT as the
  three-stage backbone of its analytical pipeline.

- **PMS (Property Management System).** The software a hotel uses to
  record reservations, manage room inventory, and process check-in /
  check-out. The Punta Villa Resort PMS export used in the Philippine
  sub-study captures a narrower schema than the Portugal benchmark —
  this asymmetry is documented and discussed in Section 4.6 as the
  feature-availability mapping contribution.

- **Power BI dashboard.** The eight-page decision-support visualisation
  produced as the operational deliverable of this study. The dashboard
  reads from `predictions_live.csv` (auto-exported from the live
  prediction log) and is designed to be consumed by hotel revenue
  managers during weekly forecasting meetings.

---

# CHAPTER II — Review of Related Literature

Review of Related Literature This chapter reviews studies on hotel booking cancellations, connecting them to revenue management and forecasting, and surveying predictive analytics approaches (logistic regression, tree-based ensembles) used to estimate cancellation risk. It also draws on data-driven decision-making and dynamic capability perspectives to explain how features like lead time, deposit/prepayment terms, channel/segment, ADR, and stay structure drive behavior and how calibrated probabilities can be translated into cost-aware actions (e.g., deposits, reminders, flexible rebooking, calibrated overbooking). Through this review, the chapter establishes the theoretical and operational foundation for building a practical, manager-ready prediction system that reduces revenue loss while maintaining guest experience. 

## Hotel Cancellation Risk: Definitions, Drivers, and Revenue Implications

 A no-show occurs when a guest does not show up and does not cancel in advance, a cancellation occurs when a confirmed reservation is withdrawn prior to arrival, and a modification changes the dates or party information without terminating the reservation. Both no-shows and cancellations affect inventory control and demand projections near stay dates, which interferes with revenue management (Boston University Hospitality Review, 2021). The distinction is significant for RM practice since early cancellations and no-shows result in different fines and recovery alternatives, yet both lower effective occupancy and jeopardize staffing and pricing strategies. Since late cancellations are more difficult to sell at comparable rates, cancellation costs scale with ADR, length of stay (revenue at risk), and timeliness. How much of such loss is recovered by fines and how much becomes pure opportunity cost depends on the policy's design (e.g., non-refundable vs. flexible) (Chen, 2011). This allows the calculation of estimated loss per booking as a function of ADR, LOS, time, and policy for modelling and valuation. Cancellation prediction provides important revenue-management factors by converting complex booking patterns into actionable projections. Data-driven overbooking utilizing show/no-show projections demonstrates clear advantages compared to static rules. Accurate risk ratings aid in setting overbooking buffers that offset projected no-shows/cancellations without overshooting, enhancing sell-through while maintaining guest experience. By identifying at-risk reservations for reminders, deposits, or flexible rebooking, they also improve pricing and inventory controls, as demand forecasts are more accurate when probable cancellations are netted out beforehand. They also support operations, from pre-stay communication to staffing and housekeeping scheduling. To put it briefly, prediction increases income and decreases last-minute disruption by connecting reservation data to daily RM and service decisions. Lead time and cancellation risk are closely related: the longer the gap between booking and arrival, the more opportunities plans have to change or for guests to keep “shopping,” so both the likelihood and timing of cancellation shift as the stay date approaches. Large public hotel datasets and empirical modeling show systematic (not random) variation in cancellation probability by days-until-stay, with distinct short-, mid-, and long-horizon patterns that hotels can model (António, de Almeida, & Nunes, 2019; Satu, Ahammed, & Abedin, 2020). Experimental evidence also finds that willingness to pay for flexible terms varies across the booking horizon, implying that far-in-advance reservations carry higher option value and thus higher risk under flexible policies (Gong, 2024). Seasonality further shapes cancellation patterns: cancellation ratios differ by month and between city and resort contexts, which is why arrival month and related time markers are standard in public datasets (António et al., 2019). Beyond broad seasonality, special events (festivals, city-wide conventions) alter pickup and price dynamics; their effects on occupancy and rates vary year to year, so event periods should be modeled explicitly (Piga, Bachis, & Blasi, 2021). Cancellation behavior is strongly influenced by the penalty window (e.g., free until 48 hours, then one-night or full-stay charge) and rate conditions notably whether the booking is flexible/refundable versus advance purchase, prepaid, or non-refundable. Because guests weigh price against the option value of flexibility, experiments and field data show that stricter policies and tighter deadlines reduce cancellations and shift who books and when (Chen, 2011). Recent studies also show that situational cues can steer customers toward free-cancellation or prepaid offers depending on demand and season, and that policies themselves signal quality (e.g., refundability) to customers (Choi, 2024; Kim, 2023). Distribution channels and market segments systematically affect cancellation risk. Industry evidence indicates OTA reservations cancel at higher rates than direct reservations, consistent with looser terms and higher “shop-around” behavior (Bookassist survey via HotelTechReport). Public hotel-booking datasets reflect this pattern via distribution_channel, market_segment, and is_repeated_guest; analyses consistently find that channel/segment indicators rank among the most predictive features and that repeat guests cancel less often than transient leisure customers (António, de Almeida, & Nunes, 2019; Yang & Miao, 2024). Implication: encode channel, segment, and repeat-guest history, and test interactions such as OTA × long lead time to capture amplified risk. ADR functions as a price signal that interacts with policy conditions: higher posted/paid rates especially under flexible terms are associated with greater cancellation propensity as guests retain the option to switch (Kim, 2023; Choi, 2024). ADR and stay-length fields are standard in public datasets because they add stable predictive signals (António, de Almeida, & Nunes, 2019). The structure of the split between weekday and weekend nights captures business vs. leisure mix and demand cycles; weekdays skew business while weekends skew leisure, implying different cancellation patterns and resale prospects near arrival (António et al., 2019; CoStar, 2024). Implication: encode ADR (and ADR per person), total nights, breakdown by weekdays and weekends, and test interactions (e.g., ADR × flexible policy, weekend share × channel). 

## Cancellation Risk Analytics Stack: Data, Models, Calibration

 The "Hotel booking demand datasets" release, which offers two anonymous properties one city hotel (H2) and one resort hotel (H1) is a common baseline for cancellation research. Each reservation has 31 variables, including is_canceled, lead_time, adr, deposit_type, distribution_channel, market_segment, stays_in_weeknights, stays_in_weekend_nights, reserved_room_type, assigned_room_type, customer_type, and timestamps (such as arrival date components). These variables cover the label and important drivers used in practice. For arrivals from July 1, 2015, to August 31, 2017, the dataset includes 40,060 resort-hotel and 79,330 city-hotel reservations, including both confirmed and cancelled stays. Mirrors on Kaggle make access and replication easier. Its coverage and variable design make it appropriate for time-aware splits, feature engineering (price, policy, channel, and stay structure), and repeatable comparisons among modelling methodologies (António, de Almeida, & Nunes, 2019). Missing categorical values were replaced with the placeholder “Unknown,” while numeric outliers were capped at reasonable percentile bounds. Columns with excessive missing values were dropped after verifying they provided no predictive signal. These steps ensured data consistency and model reliability. A strong starting point for cancellation prediction is logistic regression because it produces well-calibrated probabilities, supports regularization (L1/L2) for stability, and remains interpretable for managers via odds ratios and simple partial effects (Satu, Ahammed, & Abedin, 2020). With class weighting for imbalance and a few domain interactions (e.g., lead time × deposit type , channel × ADR ), LR often performs competitively while keeping explanations straightforward (Chen et al., 2023). Where mild nonlinearity is needed, binned/spline features can capture curvature without sacrificing clarity. Still, LR’s linear log-odds form limits its ability to learn richer interactions present in operational data, so it serves best as a calibrated, explainable benchmark and a guard against overfitting by heavier methods (Yang & Miao, 2024). Implication: establish a penalized LR baseline with time-aware CV, imbalance handling, and selected interactions; report both discrimination (ROC/PR) and calibration to set a trustworthy reference point. Tree ensembles capture non-linearities and interactions among operational drivers (lead time, deposit type, channel, ADR, LOS) with minimal feature scaling, making them dependable high-performers for hotel-cancellation data. Gradient boosting (XGBoost/LightGBM) typically delivers the best discrimination (ROC/PR) when tuned with time-aware cross-validation and class weighting, while Random Forests provide robust baselines tolerant of noise and mixed feature types. When scores inform policy, apply probability calibration (isotonic/Platt). Across hotel datasets, tree ensembles rank among the top models and their feature-importance profiles align with managerial intuition (lead time, deposit/prepayment, channel, ADR), which facilitates adoption (Satu, Ahammed, & Abedin, 2020; Herrera et al., 2024; Yang & Miao, 2024). Neural and hybrid models are effective when pricing, policy, and channel features interact across time because they can capture more complex, indirect effects that trees cannot. TNNs (Tree-based neural networks) have demonstrated competitive performance in boosting trees on hotel data while maintaining importance profiles (lead time, deposit type, channel, and ADR) that are manageable (Yang & Miao, 2024). In terms of late-window behaviour, temporal deep models, like policy-enhanced LSTM with reinforcement learning, can outperform static classifiers and explicitly explain how cancellation risk changes as arrival approaches (Xiao et al., 2024). When calibrated and assessed using time-aware CV, hybrid stacks that integrate linear, tree, and deep learners in a profit-driven ensemble further enhance the quality of decisions (Liu et al., 2025). Explainability is necessary for establishing trust and converting scores into actions because current hotel cancellation models (trees/boosting) are accurate but difficult to understand. Booking-level explanations and global driver rankings that managers can act on (e.g., lead time, deposit type) are made possible by SHAP (SHapley Additive exPlanations), a unified, game-theoretic framework that assigns each feature's contribution to a prediction and satisfies important characteristics (local accuracy, consistency). Accessible guides improve customer acceptance of analytics in corporate contexts by demonstrating how SHAP fits into interpretable ML and managerial decision-making pipelines. studies on hotel-cancellation prediction overwhelmingly evaluate models by the area under the ROC curve (AUC), with tree-based neural networks, XGBoost and ensemble methods reporting AUC values between 0.85 and 0.98 (Yang & Miao, 2024). Researchers seldom report calibration-oriented metrics such as log-loss or the Brier score, even though proper scoring rules better reflect probabilistic accuracy. A notable exception is (Lynn, 2025) survival-modeling work, which demonstrated that optimizing log-loss improves prediction accuracy without affecting AUC, underscoring the limitations of relying solely on rank-based metrics. The report recommends integrating log-loss and Brier score into future hotel-cancellation studies to ensure that predicted probabilities align with real-world cancellation risk 

## Last-Minute Cancellation Control: BI Integration and Revenue at Risk

 Last-minute (or short-notice) cancellations are usually defined in hotel operations as those that take place 48–72 hours prior to arrival. Many studies and practice guidelines use a cutoff of ≤3 days because it becomes difficult to resell rooms at comparable rates once the stay date is approaching (C-Sánchez & Sánchez-Medina, 2024; Chen et al., 2023). Even with a one-night penalty, the hotel frequently loses the remaining ADR × length-of-stay revenue and has to deal with consequences in hiring, housekeeping scheduling, and inventory management, which results in a disproportionate opportunity cost for these late cancellations (Chen et al., 2023). Recent research suggests that rather than treating all cancellations equally, late-window risk should be explicitly modelled and scores should be linked to specific actions (such as confirmation reminders, deposits/prepayments, and calibrated overbooking) (C-Sánchez & Sánchez-Medina, 2024). The predictive signal for several features that are strong far from arrival becomes less as check-in draws closer. When only a small number of remaining reservations remain, lead time collapses to a limited range, reducing variance; uncommon incidents (personal issues, travel disruptions) that are not visible in booking data contribute to late cancellations; and policy and price effects (e.g., flexible vs. prepaid, ADR) become less discriminative (Chen et al., 2023). Even when models are trained explicitly for the short horizon, empirical late-window investigations reveal more changing precision/recall and worse separability near stay dates (C-Sánchez & Sánchez-Medina, 2024). Hotels can use booking-level risk scores to prioritize actions rather than apply blanket rules. When presented transparently, deposits, prepayments, or tighter terms for high-risk reservations raise commitment while preserving guest choice; for moderate-risk bookings, low-cost pre-arrival reminders (email/SMS/app) surface intent changes earlier (Chen et al., 2023). If a guest signals the need to cancel, flexible rebooking (date changes or shorter stays) can convert a likely loss into deferred revenue, which is especially valuable in the final three days when resale is unlikely (C-Sánchez & Sánchez-Medina, 2024). On the inventory side, overbooking buffers calibrated to predicted show/cancel probabilities outperform static cushions by filling rooms that would otherwise go empty while keeping walk risk within tolerance; the optimal buffer size should reflect the anticipated aggregate risk by arrival date and segment (Zhai et al., 2023). In practice, apply risk bands and expected value to trigger interventions: send reminders at lower thresholds, and require deposits or propose rebooking only when the predicted opportunity cost (ADR × length of stay, net of penalties) exceeds the intervention cost (Chen et al., 2023). Three layers should be consolidated into a one-page Power BI dashboard to translate cancellation analytics into daily decisions: (1) at-a-glance KPIs for overall and last-minute cancel rates, with quick slices by channel, segment, and lead-time bands to surface emerging risk; (2) financial exposure via Revenue at Risk (expected ADR × length of stay, net of penalties) summarized by arrival date to guide deposit requests, reconfirmations, or overbooking buffers; and (3) drivers of risk presented through succinct model explanations (e.g., lead time, deposit/prepay terms, channel/segment, ADR) so managers see not only what is changing but why. Keeping these elements on a single canvas with drill-through access to detail makes the view scan-friendly for stand-ups and shift handovers while still supporting deeper analysis when needed. The dashboard should also have lightweight practical indications that encourage action. For instance, identify categories with an increase in last-minute cancellations, attach one-click playbook prompts, and mark arrival dates where Revenue at Risk surpasses a threshold. The prediction-outcome loop is closed with a short "recent performance" strip that displays the weekly cancel rate, rebook pickup following cancellations, and realized walk risk. Revenue management and front office teams may confidently and reliably coordinate actions thanks to this approach, which links insights to choices in real time. 

## Research Gaps

 Prescriptive Analytics Linking Scores to Actions (Uplift/Policy Learning) In the evolving field of hotel analytics, researchers are moving beyond prediction to focus on prescriptive analytics, systems that not only forecast events like cancellations but also recommend the best managerial actions in response. This approach transforms raw predictive scores into actionable strategies, enabling hotel managers to decide how to respond effectively to potential cancellations, optimize room upgrades, or design personalized retention campaigns (Rioles, Kristomus, 2023). Rioles and Kristomus (2023) demonstrated how decision support systems, such as those based on the Fuzzy Tsukamoto Method, can enhance managerial decision-making by translating data insights into clear, guided actions. While their study centers on hotel investment analysis, its framework can be extended to cancellation management. By linking data-driven predictions with policy learning or uplift modeling, hotels can identify which preventive measures, like personalized offers or flexible rebooking options, yield the greatest return. This integration of predictive and prescriptive analytics bridges the gap between insight generation and decision implementation. By developing a prescriptive model that connects cancellation scores with optimal managerial responses, this study builds upon existing predictive models to create a complete decision support cycle. It advances hospitality analytics by embedding intelligence directly into operational workflows, helping hotels make cost-efficient, customer-focused, and timely decisions that reduce cancellations and enhance guest satisfaction. 

## External Data Fusion (Events, Competitor Rates, Flights, Weather)

 Integrating external data sources such as event calendars, competitor pricing, airline flight volumes, and weather conditions offers a powerful way to enhance cancellation forecasting and demand prediction systems in hospitality. Although many hotel analytics models rely primarily on internal booking and cancellation data, extending the dataset to include contextual environment signals significantly improves accuracy and responsiveness (Rioles, Kristomus, 2023). In the hospitality realm, external signals such as large local events, sudden flight disruptions, or atypical weather patterns can rapidly alter guest behavior, making internal data alone less reliable for decision making. Research shows that incorporating features like local event timing and magnitude, competitor rate changes, and inbound flight data into demand forecast models improves their predictive capability. For example, studies observing the role of weather in hotel pricing found that poorer weather forecasts were associated with lower posted rates in leisure destinations, illustrating how environmental conditions influence consumer willingness to travel and thus hotel demand (Rioles, Kristomus, 2023). Other practitioner sources note that competitor intelligence, search volume spikes in flight bookings, and real time social media sentiment offer early indicators of demand shifts that precede bookings themselves. By fusing such external data into BI systems, hotels can generate more holistic demand forecasts, anticipate cancellation risks triggered by disruptions, and adjust staffing, pricing, and inventory decisions proactively. Despite clear benefits, practical challenges remain. Many hotels lack the infrastructure to ingest and integrate heterogeneous external data streams in real time, including feeds for flights, weather APIs, event calendars, and competitor rates. Data quality issues, latency, and aligning disparate time scales and data formats hamper implementation. Moreover, the operational use of fused data for cancellation management is under researched, as very few models link external signals directly to upgrade decisions or overbooking policies. This study therefore contributes by designing a BI driven framework that systematically integrates external data sources such as events, competitor rates, flights, and weather with internal booking and cancellation analytics to improve forecast accuracy and support upgrade policy decisions. Cross-Property Generalization and Concept Drift Over Time Models developed for predicting hotel demand or cancellations often rely on data from a single property or market segment, which raises questions about how well those models perform when applied to other branches or when market conditions change over time. In the study of tourism demand forecasting in Sri Lanka, the integration of machine learning models and social media data improved accuracy during unusual periods (Hewapathirana, 2023). However, the study also highlights variability over time and the need for models that adapt to new patterns, which directly relates to the issue of concept drift and cross property generalization. Concept drift refers to changes in the underlying relationships between features and outcomes over time, such as shifts in guest booking behavior, cancellation patterns, seasonality, or external factors like tourism policies and global events. A model trained on historical data from one hotel or time period may lose accuracy when applied to another hotel location or when external conditions change. Cross property generalization means applying a model across multiple properties with different locations, guest types, and booking channels while expecting consistent performance. In the Sri Lanka study, the authors found that models performed well historically but struggled during the irregular events of 2019 to 2021, showing that adaptation mechanisms are necessary (Hewapathirana, 2023). From a practical hotel business intelligence perspective, addressing concept drift and achieving robust cross property generalization require periodic retraining of models with updated data, detection of shifts in booking behavior or data patterns, and the use of adaptive learning techniques that allow models to adjust to new branches without full redevelopment. Few studies have explored how cancellation forecasting models in hospitality adapt across multiple properties or how performance changes over time. This study contributes by proposing a framework that supports multiple hotel branches, monitors model performance, and triggers retraining when concept drift is detected. Foundational Literature and State-of-the-Art Machine learning (ML) has become a core approach in hotel revenue management and cancellation prediction, providing data-driven insights for forecasting and operational planning. Recent studies highlight how advanced models can outperform traditional statistical methods while offering better interpretability and business value. Herrera et al. (2024) applied deep neural networks, Random Forest, and XGBoost models to hotel booking data and showed that careful feature engineering and imbalance handling greatly improved prediction accuracy. Meanwhile, Jishan et al. (2024) used Bayesian logistic regression and Beta-Binomial modelling to estimate cancellation probabilities with uncertainty intervals, illustrating how probabilistic learning enhances risk-based decision making. These works represent the state-of-the-art in hotel analytics—accurate, data-rich, and increasingly interpretable. However, most remain focused on overall accuracy rather than on translating predictions into actionable, cost-sensitive hotel policies. Few studies address short-notice cancellations (≤ 3 days before arrival), which have the greatest financial impact. This study builds on these foundations by targeting the late-cancellation window and linking predictive outputs to managerial actions through a cost-sensitive decision framework and business-intelligence integration. 



## Small-Sample Transferability in Hotel Analytics

Most hotel-cancellation research uses large public benchmark datasets such as the Portugal corpus (Antonio et al., 2019; Herrera et al., 2024). Less is known about whether the methodologies developed on these benchmarks transfer to small and medium hospitality businesses (SMBs) with proprietary PMS schemas. Two strands of literature are relevant. The first strand concerns **domain shift and transferability in tabular ML**. Recent work argues that hospitality models trained on one property may degrade substantially when applied to another, citing differences in booking channel mix, deposit policy enforcement, and seasonality patterns. A robust transferability claim requires either (a) a held-out evaluation on the target property's data, or (b) a documented feature-availability mapping showing which dimensions the source and target properties share. The second strand concerns **small-N hotel analytics for SMB properties**. Independent hotel properties typically have access to fewer than 1,000 historical bookings per year, putting many ML approaches developed on the Portugal benchmark out of reach. The implication for the present study is that transferability cannot be assumed; it must be tested empirically on a real small-property dataset. The Philippine sub-study reported in Chapter IV closes that gap on a single property by applying the Portugal methodology to 193 real bookings from Punta Villa Resort.

## Calibrated Probabilities for Decision Support

A model that ranks bookings well is not automatically useful for decisions: the percentage it outputs must mean something. Niculescu-Mizil and Caruana (2005) showed that gradient-boosted trees produce uncalibrated probability estimates by default, and that **isotonic regression** fit on a held-out validation set typically reduces Expected Calibration Error (ECE) by more than 50 % without sacrificing discrimination. Subsequent applied work in hospitality (Chen et al., 2023; C-Sánchez & Sánchez-Medina, 2024) confirms that calibrated probabilities are a prerequisite for any **cost-sensitive threshold** policy: the cost calculation multiplies a probability by a financial penalty, so an uncalibrated probability scales the penalty incorrectly. The present study therefore follows the calibrate-then-threshold pattern recommended by this literature.

## Synthesis

 What the Literature Agree On (Key Drivers, Best-Performing Methods) In the fast-changing world of hospitality, hotels are using business intelligence (BI) to make smarter and quicker decisions. Researchers have explored how BI helps address challenges such as fluctuating demand, overbooking, and customer satisfaction. From these studies, a shared understanding has emerged that successful BI systems depend on strong data integration, real time analytics, and personalized decision making (Altin et al., 2025). The literature agrees that combining data from booking platforms, property management systems, and guest records allows hotels to create a complete view of customer behavior and improve forecasting accuracy. Real time analytics help managers react quickly to cancellations, no shows, and changes in demand, allowing them to adjust pricing and inventory more effectively. These practices lead to better occupancy control and higher guest satisfaction. Common BI methods include clustering for customer segmentation, regression and time series models for forecasting, optimization for overbooking decisions, and sentiment analysis to interpret guest reviews (Altin et al., 2025). Empirical findings consistently show that hotels using BI tools achieve higher occupancy, increased revenue per available room, and improved guest satisfaction. However, technology alone is not enough to ensure success. Studies emphasize the importance of staff training, data literacy, and good data management to maximize BI benefits. Overall, the literature concludes that effective BI in hospitality comes from combining integrated data systems, reliable analytical methods, and a strong culture of data driven decision making (Altin et al., 2025). 

## Research Opportunities and Emerging Challenges

 While business intelligence (BI) tools have advanced in hotel operations, there are still many areas that remain underexplored. Studies have shown that BI can improve pricing, demand forecasting, and guest satisfaction, yet its application in upgrade related decisions such as complimentary room upgrades remains limited. Current research rarely examines how BI can support real time, cost based upgrade decisions that balance profitability and guest experience (Diwan, 2025). This highlights a clear opportunity to study how analytics can guide such decisions while maintaining operational efficiency. Another major gap lies in the integration of personalization and efficiency frameworks. While personalization is a common BI focus, few models connect guest profiling directly with resource optimization such as room inventory or staff scheduling. This disconnect limits the full potential of BI in improving both customer experience and internal processes. Additionally, practical barriers persist as hotels face challenges in data integration, staff adaptability, and the lack of automation in smaller operations, which reduces the impact of BI adoption (Diwan, 2025). Lastly, there is a lack of models that measure both the financial outcomes and guest satisfaction effects of BI driven upgrade strategies. Many studies assess revenue performance or customer experience separately, but rarely together. Understanding how BI based decisions affect both sides is key to building sustainable competitive advantage. These research opportunities reveal the need for frameworks that connect data analytics with decision making efficiency, a gap this study aims to address through its proposed BI based upgrade strategy (Diwan, 2025). 

## How This Study Advances the Field

 This study builds on the growing body of research on business intelligence in hospitality by presenting a focused upgrade decision framework that combines cost analysis and real time operational triggers. Previous studies such as (Alqahtani et al., 2025) explored advanced decision support models and highlighted the value of analytics in improving decision making. However, that study did not apply its model to upgrade related decisions, cost management, guest loyalty, or real time hotel service operations. The system proposed in this study addresses that gap by integrating guest profiling data, incremental service cost metrics, upgrade eligibility logic, and BI dashboard visualizations. It advances the field in three key ways. First, it connects upgrade allocation to both cost control, such as the incremental cost of higher room categories, and guest experience improvement, such as satisfaction and loyalty (Alqahtani et al., 2025). Second, it incorporates real time analytics into upgrade eligibility rules, moving beyond static segmentation and toward dynamic decision making based on occupancy, guest value, and service capacity (Alqahtani et al., 2025). Third, it provides a BI playbook for hotel managers and researchers, offering a structured model that translates analytics into practical upgrade strategies with clear decision rules and measurable results. By providing a practical framework that links analytics, upgrade strategy, and performance measurement, this study advances existing knowledge and application in the field. It offers a useful model for hotels and future researchers that connects data driven insights, cost efficient upgrade choices, and guest centered outcomes. This synthesis positions the present study as a step forward in applying business intelligence to hospitality operations, providing an integrated framework that supports data informed, cost efficient, and customer centered upgrade decisions (Alqahtani et al., 2025).

---

# CHAPTER III — Methodology

Methodology 

## Research Design

 This study employs a quantitative, predictive research design to develop and assess machine-learning models. These models are designed to forecast hotel booking cancellations, providing valuable insights for implementing cost-sensitive revenue management strategies.

This research design is applied in parallel to two datasets. The **Portugal main study** uses the full pipeline at scale (119,210 cleaned bookings, rolling-origin cross-validation across three chronological folds, paired bootstrap significance testing). The **Philippine sub-study** applies the same pipeline to the 193-row Punta Villa Resort PMS export, omitting only steps that the smaller sample cannot statistically support (the rolling-origin CV is replaced by a single chronological 80 / 10 / 10 split, and the cost-sensitive threshold policy is omitted because n_val ≈ 19 is too small to fit a reliable cost curve). All other pipeline stages — cleaning, feature engineering, isotonic calibration, threshold sweep, SHAP interpretation, and live serving — are identical between the two studies. The methodological framework is structured around the phases of Dynamic Capability Theory’s cycle of sense → seize → transform . (Teece, D. J. 2007) The sense stage covers disciplined data curation, exploratory analysis, and feature construction; seize covers model development, probability calibration, and cost-aware thresholding; transform covers operational rollout, monitoring, and continuous improvement. A research workflow diagram summarizes the end-to-end pipeline from preprocessing to final evaluation. The analysis uses the Hotel Booking Demand dataset. The dataset comprises 119,390 booking observations from a city hotel and a resort hotel, spanning from July 1, 2015, to August 31, 2017. Each record includes information about booking arrival dates, lengths of stay, number of adults, children, and babies, deposit type, assigned and reserved room types, number of parking spaces, total special requests, and reservation status. In this study, "sensing" is the initial recognition that high customer cancellation rates are a primary driver of revenue instability. Identifying this systemic inefficiency and the corresponding strategic opportunity to mitigate it through predictive analytics constitutes the foundational "sensing" act that motivates this research. The "Hotel Booking Demand" dataset serves as this source of intelligence, where independent variables describing booking conditions (e.g., lead_time, average_daily_rate, etc) provide the raw material to "sense" the complex drivers of customer behavior. Therefore, the data understanding and exploratory 

## Data Analysis

 (EDA) phase of this study's methodology directly corresponds to the Prediction accuracy of the models for booking cancellation, aligning the technical process of data analysis with the strategic goal of identifying opportunities for organizational change. The " Seizing ", which focuses on designing and implementing a business model to explore a sensed opportunity, was operationalized in this study through the development and evaluation of predictive machine learning models. This phase involved designing the solution by selecting multiple supervised classifiers, specifically Logistic Regression, Random Forests, and gradient-boosting machines (LightGBM/XGBoost)—chosen for their respective abilities to capture linear, non-linear, and complex interaction effects. The accuracy metrics used in this study are: Accuracy, Recall (RC), F1 Score, Precision (PR), and Area Under the Curve (AUC). In the case of the first metric, it provides an overall measure of model performance by indicating the proportion of correct predictions compared to the total number of predictions made. The transformation phase involves reconfiguring the organization's structure, processes, and assets to capitalize on the insights developed during the "seizing" phase. In this study, this is operationalized by translating the machine learning model's outputs into actionable policy and reducing expected revenue loss. Insights from model evaluations and feature importance analysis (e.g., via SHAP) provide valuable information about the variables driving booking cancellations, enabling management to reconfigure pricing logic, such as adjusting rates for high-risk segments or specific room types. Model insights are converted into executable revenue policies. Operations are reconfigured to address key risk drivers related to cancellations. A continuous feedback loop monitors cancellation rate, forecast error, RevPAR, policy adherence, and model/data drift. This phase completes the sense–seize–transform cycle by institutionalizing a data-driven capability that adapts policies in response to changing market conditions. The primary objective of this study is to employ a quantitative, predictive research design to develop and evaluate machine learning models that accurately forecast hotel booking cancellations. This technical goal is framed within the Dynamic Capability Theory (DCT) to ensure the resulting models provide predictive analysis and provide tools for organizational adaptation. The methodology follows the sense-seize-transform cycle: 1. Sense: Understand the Hotel booking dataset to conduct an exploratory data analysis (EDA) and identify key drivers of cancellation and business problems, which represent a primary source of revenue instability. By understanding these drivers, the business can implement targeted strategies to reduce cancellations, such as dynamic pricing, personalized offers, improved customer service, etc. 2. Seize: To build and conduct data preprocessing to ensure model accuracy, evaluate (using metrics like AUC and F1-Score), and interpret (using SHAP) a suite of predictive models to create an actionable tool. 3. Transform: To provide a pathway for translating these model insights into cost-sensitive revenue management strategies —such as optimized pricing, dynamic deposit rules, and new overbooking policies thereby embedding a data-driven capability that reduces revenue loss and improves operational efficiency. 

## Research Instruments

 The following tools and libraries are used: ● Python (Jupyter Notebook) for data processing, modelling, and documentation ● Standard data libraries (e.g., pandas, NumPy) for preprocessing ● Machine learning libraries (e.g., scikit-learn, XGBoost/LightGBM) for model training ● XAI libraries (e.g., SHAP) for model interpretation ● Visualisation libraries (e.g., matplotlib) for EDA, diagnostic plots, and calibration curves All code, preprocessing, and models are reproducible via documented notebooks. 

## Modelling Procedures

 This study formulates hotel booking cancellation prediction as a supervised classification problem. For each reservation, the model estimates the probability that the booking will be cancelled prior to arrival using information available at or near the time of booking. The modelling procedures are designed to (1) avoid target leakage, (2) respect temporal ordering, (3) compare multiple algorithms under consistent conditions, and (4) produce probabilities that are both accurate and interpretable for revenue management use. Model Specification The dependent variable is cancelled booking (1 = cancelled, 0 = not cancelled). The independent variables consist of engineered booking-time features derived from guest profile, stay ccharacteristics, price, policy, channel, and historical behaviour, as described in the data preparation section. The study utilizes three primary categories of machine learning models: (the researchers may explore more ML models throughout the study: such as Artificial Neural Network (ANN) ect) 1. Logistic Regression Logistic regression with regularisation (L2, with possible L1 sensitivity checks) serves as the primary baseline. It is selected for its simplicity, probabilistic output, and transparent interpretation of how predictors (e.g., lead time, deposit type, OTA) influence cancellation odds. 2. Random Forest Random Forest (RF) (Breiman, 2001) is a supervised ensemble learning technique that employs the "bagging" approach (Breiman, 1996) and random feature selection (Ho, 1995) to develop a strong and efficient predictor. It constructs several distinct decision trees and combines their results to tackle classification and regression tasks. The final outputs are adjusted based on either the most frequent class or the mean of the predicted values from the individual trees. A significant benefit of RF is its capability to mitigate the "overfitting tendency" commonly associated with decision tree algorithms, which is a frequent drawback of tree-based methods. 3. Gradient Boosting (XGBoost/LightGBM) Gradient boosting (GB) (Friedman, 2001, 2002) is a type of supervised ensemble learning in which a weak learner is combined with the efforts of other learners in order to improve its performance. Extreme gradient boosting (XGB) (Chen & Guestrin, 2016) is an innovative supervised learning approach based on tree structures that combines the principles of Classification and Regression Trees (CART) (Steinberg & Colla, 2009) and Gradient Tree Boosting (GTB) (Mason et al., 2000; Friedman, 2001). It offers options for L1 and L2 regularization to prevent overfitting. To enhance its efficiency, XGB is structured to minimize a regularized objective function that merges a convex loss function with a penalty scoring mechanism reflecting the difference between predicted and actual labels. During each boosting iteration, random data and feature subsets are utilized, while the weight of misclassified instances is increased. Numerous studies focused on various issues have validated its robustness. All models are trained exclusively on features observable at booking time. Features that encode knowledge of future outcomes or post-cancellation states are excluded. Training, Validation, and Baseline For this study the data will be split chronologically into a training set (75%) and a test set (25%) . All tuning happens only inside the 75% using a time-series cross-validator. The objective of cross-validation is to ensure that the results obtained in classification models are independent of the partition between training data and valida-tion data. This concept is widely used in models generated in AI projects (Prusty et al., 2022). For context, two non-ML baselines are run under the same protocol: (1) a majority-class rule (always “not cancelled”) and (2) a historical-rate rule (segment/channel/lead-time average with a fixed threshold). The ML model must outperform both. Baseline Comparison To demonstrate that the proposed machine learning models add value, their performance is compared against simple, non-ML baselines constructed from historical cancellation behaviour. The final model must outperform these baselines on the primary metrics Baseline 1: Majority-Class Rule This heuristic predicts that all reservations will not be cancelled. It reflects the fact that non-cancellations are the dominant outcome. Any useful model must exceed this baseline, particularly in recall and F1-score for cancelled bookings. 1. Baseline 2: Historical Cancellation Rate Rule This rule-based method assigns each booking a probability equal to the historical average cancellation rate of similar reservations (e.g., same hotel, channel, market segment, and lead-time band). A booking is classified as “will cancel” if this group-level rate exceeds a chosen threshold; otherwise, it is classified as “will not cancel.” This represents the simplest operational strategy a hotel can implement without machine learning: using past segment-level rates to guide expectations. The machine learning models are considered to provide meaningful improvement only if they achieve statistically and practically higher performance than these naive baselines, demonstrating better identification of high-risk bookings and more accurate probability estimates for use in revenue management decisions. A machine learning model is judged useful only if it demonstrates clear improvement over these baselines. Performance Metrics The evaluation of the proposed cancellation prediction models is based on multiple complementary metrics to capture both statistical performance and practical relevance for hotel operations. This study therefore reports accuracy for context but focuses interpretation on recall, precision, F1-score, the Area Under the ROC Curve (AUC), and the confusion matrix. 1. Accuracy reports the proportion of correctly classified bookings among all predictions. It offers an initial indication of performance at the aggregate level but does not distinguish between correctly predicting the dominant non-cancelled class and correctly identifying the less frequent cancelled class. In an imbalanced setting, a model can achieve high accuracy while failing to detect a substantial share of cancellations. For this reason, accuracy is treated as a secondary descriptive indicator rather than the primary basis for model selection. 2. Recall , measures the proportion of actual cancellations that the model correctly identifies as high risk. It reflects the model’s ability to capture bookings that are likely to cancel. High recall is operationally important because missed cancellations translate into unsold room inventory and foregone revenue when not anticipated in overbooking or deposit policies. A model with low recall, even if accurate overall, would have limited usefulness for proactive revenue management. 3. Precision measures the proportion of bookings flagged as cancellations that are in fact cancelled. It indicates how reliably a high-risk classification corresponds to true cancellation behaviour. High precision is critical to avoid unnecessary or excessively strict interventions, such as imposing deposits, sending repeated reminders, or adjusting allocations for guests who would have honoured their reservations. Balancing precision and recall is essential to align model use with both revenue protection and guest experience. 4. The F1-score combines precision and recall into a single summary index that increases only when both are jointly strong. It is used in this study as a central performance indicator for the cancellation class, as it directly reflects the trade-off between detecting cancellations and limiting false alarms under class imbalance. Models are compared on F1-score to identify those that provide a more effective balance between sensitivity to true cancellations and restraint in flagging non-cancellations. 5. The Area Under the Receiver Operating Characteristic Curve (AUC) is employed to evaluate the ranking quality of the predicted probabilities across all possible decision thresholds. AUC reflects how well the model separates bookings that will cancel from those that will not, independent of any fixed cut-off. A higher AUC indicates that, on average, cancelled bookings are assigned higher risk scores than non-cancelled bookings. This property is valuable in a revenue management context, where decision thresholds (e.g., when to apply deposits or reconfirmations) may be adjusted according to evolving business policies and risk tolerance. 6. The confusion matrix is used to provide a transparent breakdown of model outcomes on the held-out test set. It reports counts of true positives (correctly identified cancellations), true negatives (correctly identified non-cancellations), false positives (non-cancellations incorrectly flagged as cancellations), and false negatives (cancellations not detected by the model) . Examining this structure allows the study to interpret how different models and thresholds shift the balance between protecting revenue (reducing false negatives) and avoiding unnecessary interventions. It also serves as the empirical basis for computing accuracy, precision, recall, and F1-score in a consistent and auditable manner. By jointly considering these metrics, the study evaluates not only whether the models are statistically sound, but also whether their behaviour is aligned with the operational objectives of the hotel: identifying at-risk bookings reliably, minimising avoidable guest friction, and supporting informed, cost-sensitive cancellation management decisions. Model Explainability Model explainability is incorporated to ensure that cancellation risk estimates are transparent, traceable, and usable for decision-making rather than functioning as opaque scores. In this study, the logistic regression model is interpreted through the direction and relative magnitude of its coefficients to identify booking characteristics associated with higher or lower cancellation risk, while tree-based models (Random Forest and Gradient Boosting) are interpreted using feature importance and summary explanation techniques to show which variables consistently drive predictions. To complement these global views, local explanation methods (such as SHAP-based analyses) are applied to selected observations to illustrate, in narrative form, how specific attributes of an individual booking (for example, lead time, deposit conditions, channel, guest history, and special requests) contribute to its assigned risk level. The resulting explanations are examined alongside the confusion matrix from the test set, which provides a clear breakdown of correctly and incorrectly classified cancellations and non-cancellations, allowing the study to discuss trade-offs between detecting risky bookings and avoiding unnecessary interventions. Taken together, these procedures ensure that the models decision-making logic can be articulated in business terms and directly linked to potential actions, such as differentiated policies by channel, targeted reconfirmations, or calibrated overbooking strategies. Data Analysis Under a quantitative, predictive design. The data analysis workflow for this study was implemented in a Python environment within a Jupyter Notebook, ensuring transparency and reproducibility of all steps. In the data. Each entry includes arrival dates, lengths of stay, party composition, deposit type, reserved and assigned room types, parking spaces, total special requests, and reservation status. The predictive target (is_canceled) indicates whether each booking was cancelled (1) or honoured (0). Only variables available or inferable at or shortly after booking time are used as predictors to prevent target leakage. The study is delimited to cancellation prediction; it does not model room-rate optimisation or full demand forecasting. The initial step involves conducting an exploratory data analysis (EDA) to examine the structure and distribution of the variables. Through this, researchers can gain a more profound insight into the crucial variables influencing hotel booking cancellations. Summary statistics (mean, median, standard deviation, minimum, and maximum values) will be computed for continuous features (independent variables) such as lead_time and adr. Visual inspections via histograms and box plots helped identify skewness, outliers, and potential noise. Feature engineering will produce variables such as party size, total stay, ADR per person, and cyclical representations of month and week. Categorical fields will be one ‑ hot encoded, and skewed continuous features transformed. Exploratory data analysis will summarise distributions and relationships using summary statistics, histograms, and cross ‑ tabulations, laying the foundation for hypothesis formulation. For machine learning modeling to build a predictive model that determines whether a hotel booking will be canceled, classification models such as Decision Trees, Random Forest, and gradient-boosted trees (LightGBM/XGBoost) will be implemented and tuned. Performance will be assessed on an out ‑ of ‑ time test set using accuracy, precision, recall, F1-score, the area under the curve and precision ‑ recall curves, with AUC values interpreted according to published benchmarks (e.g., ≥ 0.90 excellent) (Srivastava, 2025); calibration will be evaluated through the reliability curves. SHapley Additive exPlanations (SHAP) will quantify the contribution of each feature, highlighting key drivers such as lead time, deposit type, and previous cancellations. A cost ‑ sensitive decision threshold will be selected by minimising expected loss under asymmetric costs for false positives and false negatives. The final step will interpret the results against the research questions and propose actionable strategies, including dynamic deposit tiers, overbooking buffers, and targeted reconfirmations. A monitoring plan will track forecast error, cancellation rates, and revenue per available room to trigger retraining or policy adjustments. 

## Dataset Variables for Hotel Booking Cancellation Prediction

_Portugal dataset variables — see original schema below._

 Variable Description Type IsCanceled (Dependent Var) Booking canceled (1) or not (0). Binary ADR Average Daily Rate = total lodging transactions ÷ total staying nights. Continuous LeadTime Number of days between the booking date and the arrival date. Continuous ArrivalDateYear Year of arrival. Categorical ArrivalDateMonth Month of arrival. Categorical ArrivalDateWeekNumber Week number of the year for arrival. Continuous ArrivalDateDayOfMonth Day of the month of arrival. Continuous StaysInWeekendNights Weekend nights (Sat or Sun) stayed or booked. Continuous StaysInWeekNights Weeknights (Mon–Fri) stayed or booked. Continuous Adults Number of adults. Continuous Children Number of children. Continuous Babies Number of babies. Continuous Meal Type of meal booked; standard hospitality packages. Categorical Country Country of origin of the guests. Categorical MarketSegment Market segment (e.g., “Online TA”, “Offline TA/TO”, “Groups”). Categorical DistributionChannel Booking channel; TA/TO = Travel Agents/Tour Operators. Categorical IsRepeatedGuest Guest is a repeat customer (1) or not (0). Binary PreviousCancellations Previous bookings canceled by the customer. Continuous PreviousBookingsNotCanceled Previous bookings not canceled by the customer. Continuous ReservedRoomType Code of reserved room type. Categorical AssignedRoomType Code of assigned room type; may differ from reserved. Categorical BookingChanges Number of amendments from creation to check-in. Continuous DepositType Deposit status: “No Deposit”, “Non-Refund”, “Refundable”. Categorical DaysInWaitingList Days on waiting list before confirmation. Continuous CustomerType Booking type (four categories). Categorical RequiredCarParkingSpaces Number of car parking spaces requested. Continuous TotalOfSpecialRequests Number of special requests. Continuous The dataset contains only structured reservation fields and no free-text data such as guest reviews or feedback, so natural-language-processing (NLP) techniques were not applied. 

## Dataset Variables — Philippine Sub-Study

The Philippine PMS export captures a reduced subset of the variables available in the Portugal benchmark. The raw schema and the engineered features derived from it are:

| Raw field | Description | Engineered features derived |
|---|---|---|
| `Lead_Time_Days` | Days from booking to arrival | `lead_time`, `is_late_window` |
| `Weekend_Nights` | Weekend nights booked | `stays_in_weekend_nights`, `is_weekend_heavy` |
| `Week_Nights` | Week nights booked | `stays_in_week_nights`, `total_stay` |
| `Adults`, `Children`, `Babies` | Guest counts | `adults`, `children`, `babies`, `total_guests` |
| `ADR_Rate` | Average daily rate (PHP) | `adr`, `adr_per_person`, `revenue_at_risk` |
| `Room_Type` | Reserved room type | `reserved_room_type` (categorical) |
| `Deposit_Type` | Deposit policy | `deposit_type` (categorical) |
| `Special_Requests` | Count of special requests | `total_of_special_requests` |
| `Arrival_Date` | ISO arrival date | `arrival_date_year`, `arrival_date_month`, `arrival_date_day_of_month`, `month_sin`, `month_cos` |

The Philippine PMS export does **not** capture `country`, `agent`, `market_segment`, `customer_type`, `previous_cancellations`, `previous_bookings_not_canceled`, `required_car_parking_spaces`, or `meal` (the latter is constant and dropped). This is the feature-availability constraint that Chapter IV Section 4.6.2 develops as a methodology contribution.

## Pre-Flight Duplicate-Cluster Diagnostic

Before fitting any model on a chronologically-split dataset, the pipeline runs a diagnostic that counts duplicate post-engineering feature vectors and measures label consistency within each duplicate cluster. If the duplicate rate exceeds 30 % AND the fraction of duplicate clusters with consistent labels exceeds 90 %, the test metrics will be inflated by recognition rather than generalisation. The diagnostic is implemented at `scripts/train_ph.py::_compute_duplicate_diagnostics` and is dataset-agnostic.

## Per-Family Probability Calibration

Each candidate model family fits its own isotonic calibrator on the validation set, so the calibration step is part of model selection rather than a post-hoc adjustment to the champion alone. This ensures that model comparisons in Chapter IV use calibrated probabilities for every family, not raw scores from non-champions and calibrated scores from the champion.

## Bootstrap Paired-Significance Testing

Model selection in the Portugal main study is supplemented by bootstrap paired-significance testing with 2,000 resamples on the test set. The 95 % confidence interval on the delta and the two-sided p-value are then reported. This elevates 'LightGBM is best' from a point-estimate claim to a statistical claim.

## Research Instruments (Extended)

In addition to Python, pandas, scikit-learn, LightGBM, SHAP, and matplotlib, the project uses: **FastAPI** + **Gradio** for live model serving (Portugal at port 8000, Philippine at port 8001), **SQLite** for the prediction audit log, **Power BI Desktop** for the eight-page decision-support dashboard, **GitHub Actions** for continuous integration, and **pytest** with `pytest-cov` for the conformance test suite (130 tests, ≥ 80 % coverage).

## Ethical Consideration

 Since this study uses publicly available hotel booking data and does not involve direct interaction with human participants, ethical approval from an institutional review board (IRB) was not required. However, the research follows all ethical guidelines related to the responsible use of secondary data. 

### Data Privacy and Confidentiality

 The dataset used for this research is fully anonymized, meaning it does not contain any personal information such as names, addresses, or contact details. The analysis focuses only on aggregated booking behaviors, ensuring the privacy of individuals whose data is included in the dataset. This ensures that no personal or sensitive information is revealed during the analysis. 

### Informed Consent

 Since the data is secondary and anonymized, obtaining direct informed consent from participants was not necessary. However, the study adheres to ethical principles by ensuring that the data is used responsibly and in accordance with academic and research standards. 

### Transparency and Accountability

 The methodology and analysis process have been clearly outlined, allowing for full transparency. This ensures that the research is reproducible and that the results can be verified by others. The findings, including the performance of the models and their evaluation, are shared openly to maintain accountability in the research process. 

### Potential Risks and Mitigation

 Given that the study uses anonymized data and does not involve sensitive personal information, there are minimal ethical risks. The primary concern is ensuring the responsible handling of the data. To address this, the data has been stored securely, with access limited to authorized personnel only. 

### Data Governance and Compliance

 This study is in compliance with data protection laws, such as the General Data Protection Regulation (GDPR), where applicable. While the dataset is publicly available and anonymized, strict data governance principles are followed to ensure that the information is used ethically and in line with research best practices.

---

# CHAPTER IV — Results and Discussion

> This chapter is written to be read by a hotel revenue manager, not only
> a machine-learning specialist. Every result is followed by the question
> *"so what does this mean for the property?"*. The deeper statistical
> apparatus (paired bootstrap tests, hypothesis verdicts, the Philippine
> sub-study, methodology contributions) is documented in Chapter V and
> in the appendix tables under
> `docs/thesis_drafts/chapter_iv_tables/`.

## 4.1 Introduction

This chapter reports what happened when the trained models were turned
loose on data they had never seen before, and what those numbers mean
for a hotel that has to decide which bookings to act on each week.

The chapter answers four practical questions in order:

1. **Which model performed best?** (Section 4.3)
2. **Where does it get predictions right and wrong?** (Section 4.4)
3. **What features actually drive those predictions?** (Section 4.5)
4. **What do all these results mean for hotel revenue and booking strategy?** (Section 4.7)

Section 4.2 first restates how the data was cleaned and split, so the
numbers in later sections can be traced back to a known dataset state.
Section 4.6 reports the parallel Average Daily Rate (ADR) regression
results, and Section 4.8 documents the live deployment framework.

---

## 4.2 Data Preprocessing Summary

The raw Portugal benchmark dataset contained 119,390 hotel bookings
spanning 1 July 2015 to 31 August 2017. A small number of rows were
removed before training: 180 rows had zero guests recorded (impossible
under the property's own booking rules), and one row had a negative
average daily rate. Removing these 181 rows left **119,210 valid
bookings** for the rest of the study.

Two further cleaning steps were applied without dropping rows. The
`agent` field was filled with "Direct" for 16,340 bookings that arrived
through the property's own website (these bookings had no third-party
agent identifier). Country values that came in blank were standardised
to `Unknown` for 488 bookings. Both transformations are reversible
and documented in `src/utils/validate_data.py`.

The 119,210 cleaned bookings were split chronologically — the oldest
80 % became the training set, the next 10 % the validation set, and
the most recent 10 % the test set. This is deliberately stricter than
the random shuffling used in most introductory machine-learning
projects, because in real operations the model will always be asked
to predict the *next* week's bookings using a model trained on past
weeks. Random shuffling makes models look better than they are; the
chronological split honestly mimics what production looks like.

**Table 4.1 — Portugal dataset split summary**

| Split | Rows | Date range | Cancellation rate |
|---|---|---|---|
| Train | 95,367 | 2015-07-01 → 2017-04-22 | 36.1 % |
| Validation | 11,920 | 2017-04-22 → 2017-06-21 | 43.9 % |
| Test | 11,922 | 2017-06-21 → 2017-08-31 | 37.8 % |
| **All cleaned** | **119,210** | **2015-07-01 → 2017-08-31** | **37.0 %** |

Each booking is represented by **34 features** that are knowable at
the moment of reservation: things like the deposit type, lead time,
country of origin, number of guests, requested room type, and so on.
Features that only become available *after* the booking is made —
`reservation_status`, `assigned_room_type`, `booking_changes`,
`days_in_waiting_list` — were explicitly excluded to prevent the
model from cheating by peeking into the future. This separation
matters: a model that uses post-booking signals looks impressive in
academic tests but is useless at the booking desk, where those
signals do not yet exist.

**Business takeaway.** The data preprocessing was conservative — only
181 rows of 119,390 were removed — and the chronological split means
every reported performance number reflects what the model would
actually see in production, not an artificially easy test.

---

## 4.3 Model Performance Comparison

Six classification algorithms were trained on the same training set
under identical preprocessing. The chapter reports model quality
under two complementary evaluation protocols. **Section 4.3.1**
reports the chronological out-of-time test — the deployment-realistic
number, which is the one the hotel actually sees in production.
**Section 4.3.2** reports stratified 10-fold cross-validation — the
academic baseline that lets us compare algorithms on an apples-to-apples
i.i.d. footing. **Section 4.3.3** quantifies how tight the headline
numbers are via paired bootstrap confidence intervals.

### 4.3.1 Chronological out-of-time test results

Threshold-dependent metrics use each model's own validation-tuned
`max_f1` cut-off.

**Table 4.2 — Test-set performance, all six algorithms (Portugal,
n = 11,922 test rows)**

| Algorithm | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| **LightGBM (champion)** | **0.770** | **0.652** | **0.841** | **0.735** | **0.864** | **0.760** |
| Gradient Boosting | 0.773 | 0.659 | 0.829 | 0.734 | 0.861 | 0.754 |
| XGBoost | 0.745 | 0.610 | 0.904 | 0.729 | 0.855 | 0.749 |
| Random Forest | 0.756 | 0.650 | 0.766 | 0.704 | 0.851 | 0.739 |
| Logistic Regression | 0.749 | 0.628 | 0.825 | 0.713 | 0.839 | 0.739 |
| Decision Tree | 0.695 | 0.597 | 0.595 | 0.596 | 0.675 | 0.508 |

The full per-row counts (TP / FP / FN / TN) for every model are
preserved in `docs/thesis_drafts/chapter_iv_tables/table_02_chronological_oot_test.md`.

**[Insert Figure 4.1 — `reports/figures/thesis/fig_02_grouped_bar_model_selection.png`
here, showing PR-AUC across the six algorithms as a grouped bar
chart with the champion highlighted.]**

**LightGBM wins, by a small but real margin.** The PR-AUC gap between
LightGBM (0.760) and second-place Gradient Boosting (0.754) is just
0.006 — small enough that it would be easy to dismiss as noise.
Section 4.3.3 shows the paired bootstrap re-sampling that confirms
the gap survives at p = 0.001. Against every other algorithm —
Random Forest, Logistic Regression, XGBoost, Decision Tree —
LightGBM's lead is significant at p < 0.001. The ranking is real,
not lucky.

**Why LightGBM and not one of the others?** Three practical reasons.
First, hotel data is a mix of numeric signals (lead time, ADR, party
size) and categorical signals (country, agent, deposit type), and
gradient-boosted trees handle both natively without one-hot blow-up
hurting performance. Second, LightGBM trains roughly four times
faster than the equivalent Random Forest or Gradient Boosting model
on this data, which matters when the property wants to retrain
monthly against fresh data. Third, the model's per-row inference is
under one millisecond, well inside the latency budget of a live
booking-desk API.

**A note on Decision Tree's poor PR-AUC.** The plain Decision Tree
collapses to 0.508 PR-AUC because a single tree cannot capture the
interactions between (for example) `deposit_type` and `country` that
the ensemble methods exploit. We include it in the comparison anyway
because Decision Trees are visualisable in full and seeing how badly
a single tree underperforms is the best motivation for why an
ensemble was used.

**Business takeaway.** All five non-toy models perform within a
narrow band (PR-AUC 0.739 to 0.760). The choice of LightGBM is
defensible on speed and operational reasons as much as on the
0.006-point PR-AUC edge. From the hotel's perspective, picking any of
LightGBM, Gradient Boosting, or XGBoost would deliver a usable
production system; LightGBM is simply the lowest-friction choice.

### 4.3.2 Stratified 10-fold cross-validation — academic baseline

The chronological test in Section 4.3.1 is the right number for
deployment, but academic best practice also demands the standard
benchmark protocol — stratified 10-fold cross-validation that
ignores time and treats every row as exchangeable with every other.
This re-runs the comparison without the concept-drift handicap, so
the panel can see the algorithms compete on a level statistical
footing.

**Table 4.3 — Stratified 10-fold CV across 7 algorithms
(Portugal full dataset, n = 119,210, threshold = 0.5)**

| Algorithm | PR-AUC (mean ± std) | ROC-AUC (mean ± std) | F1 (mean ± std) |
|---|---:|---:|---:|
| **LightGBM** | **0.922 ± 0.002** | **0.947 ± 0.002** | **0.821 ± 0.002** |
| Gradient Boosting | 0.912 ± 0.002 | 0.940 ± 0.002 | 0.808 ± 0.002 |
| XGBoost | 0.908 ± 0.003 | 0.937 ± 0.002 | 0.811 ± 0.004 |
| Logistic Regression | 0.860 ± 0.003 | 0.901 ± 0.002 | 0.740 ± 0.006 |
| Decision Tree | 0.798 ± 0.004 | 0.876 ± 0.003 | 0.739 ± 0.005 |
| Gaussian NB | 0.749 ± 0.005 | 0.814 ± 0.004 | 0.663 ± 0.005 |
| Dummy (majority class) | 0.371 ± 0.000 | 0.500 ± 0.000 | 0.000 ± 0.000 |

The full per-fold counts and per-fold variance are preserved in
`reports/cv/portugal_stratified_10fold_summary.json` and the
companion file in `docs/thesis_drafts/chapter_iv_tables/table_01_classification_cv_benchmark.md`.

**The complexity ladder is perfectly monotonic.** Each step up in
model expressiveness — from Dummy (0.371) through Naive Bayes (0.749),
Decision Tree (0.798), Logistic Regression (0.860), XGBoost (0.908),
Gradient Boosting (0.912), and LightGBM (0.922) — buys measurable
PR-AUC. The ensemble methods earn their complexity; the single
Decision Tree and the linear model both underperform by 6–12
percentage points.

**The headline finding is the gap between the two protocols.**
LightGBM scores PR-AUC **0.922** under stratified 10-fold CV but
only **0.760** under the chronological out-of-time test — a gap of
**−16.2 percentage points**. The same model loses sixteen points of
PR-AUC when it has to predict the *future* instead of a random
shuffle of the past.

That gap is not a flaw in the model. It is the empirical signature
of **concept drift over time** — the changes in guest mix, booking
channel, deposit policy, and macro-economic context that accumulate
between training data and deployment data. The CV number tells the
hotel "this algorithm has the strongest signal in the data". The
chronological number tells the hotel "this is what the model will
actually deliver next quarter". Both belong in the thesis; the
chronological number is the one a property should plan its operations
around.

**Business takeaway.** The 16-point gap is the cost of generalising
forward in time. A hotel deploying this methodology should expect
PR-AUC closer to 0.76 in production than 0.92, and should treat
quarterly retraining (Section 4.8) as the standard way to claw some
of that gap back.

### 4.3.3 Bootstrap confidence intervals on the champion

Knowing the point estimate is not enough. A defensible thesis result
also needs to show how tight that estimate is — could LightGBM's
0.760 PR-AUC fall to 0.74 or rise to 0.78 if the test set had been
slightly different? To answer, we drew **2,000 bootstrap samples**
(with replacement) from the test set and recomputed each metric on
every sample.

**Table 4.4 — Bootstrap 95 % confidence intervals on the LightGBM
champion (Portugal test set, 2,000 resamples)**

| Metric | Point Estimate | 95 % CI Lower | 95 % CI Upper | CI Width |
|---|---:|---:|---:|---:|
| ROC-AUC | 0.864 | 0.858 | 0.871 | 0.013 |
| PR-AUC | 0.760 | 0.748 | 0.772 | 0.024 |
| F1 @ `max_f1` | 0.735 | 0.725 | 0.744 | 0.019 |

The full per-model CI grid is in
`reports/benchmarks/13_bootstrap_confidence_intervals.csv`.

**The intervals are narrow.** CI widths of 0.013 (ROC-AUC) and 0.024
(PR-AUC) at a sample size of 11,922 are textbook-tight — the test
set is large enough that the headline numbers are not at the mercy
of which particular 11k bookings happened to land in the held-out
window. Even at the lower bound of each interval, the model
comfortably clears every quality gate set out in `src/config.py`
(PR-AUC ≥ 0.50, ROC-AUC ≥ 0.70, F1 ≥ 0.50).

**Business takeaway.** A revenue manager defending this work to her
GM can quote the point estimates with confidence intervals attached:
"PR-AUC 0.76, with 95 % confidence the true number is between 0.75
and 0.77." That phrasing converts a single statistic into a defensible
range — exactly what an executive committee expects from a
business-intelligence brief.

---

## 4.4 Model Evaluation (Visualisations)

### 4.4.1 ROC and Precision-Recall curves

**[Insert Figure 4.2 — `reports/figures/thesis/fig_01_roc_pr_curves.png`
here, showing ROC and PR curves of the LightGBM champion on the
held-out test set.]**

In plain English, the ROC-AUC of 0.864 says this: if you pick one
booking that ended up cancelling and one booking that did not, and
ask the model which is more likely to cancel, the model will get the
answer right 86.4 % of the time. Random guessing would be 50 %.

The PR-AUC of 0.760 tells a more operationally relevant story. As
the model flags more bookings as risky, precision (the share of
flagged bookings that actually cancel) holds up surprisingly well —
it does not collapse the way it would if the model were guessing.
This is what matters when the property has to decide how many
high-risk bookings it can afford to staff up for.

### 4.4.2 Confusion matrix at the operating threshold

**[Insert Figure 4.3 —
`reports/figures/thesis/fig_03_normalized_confusion_matrix_max_f1.png`
here, showing the normalized confusion matrix at the validation-tuned
`max_f1` threshold of 0.40.]**

At the `max_f1` threshold of 0.40, the model's behaviour on the test
set is:

- **3,791 cancellations correctly caught** (true positives).
- **715 cancellations missed** (false negatives).
- **2,024 reservations wrongly flagged** as cancel-risks even though
  they would have honoured the booking (false positives).
- **5,392 reservations correctly waved through** as low-risk (true
  negatives).

Read as recall: of the 4,506 actual cancellations in the test
window, the model catches 3,791 — a **recall of 84.1 %**. Read as
precision: of the 5,815 reservations the model flagged, 3,791 turned
out to cancel — a **precision of 65.2 %**.

The trade-off is intentional. The cost model that picked the 0.40
threshold (Section 4.7.2) treats a missed cancellation as far more
expensive than a wrongly flagged one, because the average ADR of a
missed booking is hundreds of euros while the cost of a reminder
email is around €15. The model is calibrated to err on the side of
flagging too much rather than missing too much.

### 4.4.3 Calibration — do the probabilities mean what they say?

**[Insert Figure 4.4 —
`reports/figures/thesis/fig_05_calibration_reliability_and_histogram.png`
here, showing the calibration reliability diagram with predicted vs
observed cancellation rates per probability bin.]**

A model is *well-calibrated* if a "75 % probability" really means
about 75 % of those bookings cancel in real life. This matters
because the hotel's downstream policies — risk tier bands, deposit
requirements, reminder workflows — all hang off the predicted
probability number itself, not just the binary flag.

To turn the raw LightGBM scores into honest probabilities, the
pipeline fits an isotonic regression calibrator on the validation
set only (so the test set stays unbiased). Table 4.5 reports the
improvement directly.

**Table 4.5 — Calibration quality before vs after isotonic regression**

| Split | Brier (Raw) | Brier (Calibrated) | Δ Brier | ECE (Raw) | ECE (Calibrated) | Δ ECE |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 0.120 | 0.114 | −0.006 | 0.046 | ~0.000 | −0.046 |
| **Test** | **0.150** | **0.146** | **−0.004** | **0.058** | **0.029** | **−0.029** |

Calibration data lives in `reports/calibration_metrics.json`. Brier
score is mean squared error between predicted probability and the
{0, 1} outcome (lower is better). Expected Calibration Error (ECE)
is the average gap between predicted and observed cancellation
rates across 10 probability bins (lower is better).

**Isotonic regression halves the calibration gap on the test set.**
The ECE drops from 5.8 % to 2.9 % — meaning a "75 %" prediction
really does correspond to a 75–76 % observed cancellation rate.
Brier improves only marginally because Brier is dominated by the
discrimination component (which calibration cannot change), but the
ECE drop is the operationally important number.

**Business takeaway.** The model is honest about its own uncertainty.
That honesty is what lets Section 4.7 translate probabilities
directly into deposit policies without further recalibration.

---

## 4.5 Feature Importance Analysis

### 4.5.1 Which features drive the predictions?

**[Insert Figure 4.5 — `reports/thesis/shap_summary_plot.png` here,
showing the SHAP beeswarm for the top 15 raw features. Blue points
are low feature values; red points are high feature values; horizontal
position is the contribution to the predicted cancellation probability.]**

SHAP (SHapley Additive exPlanations) measures, for each prediction
the model makes, exactly how much each feature pushed the prediction
up or down. Aggregating across the test set ranks the features by
how much they matter overall.

**Table 4.6 — Top 10 features by mean(|SHAP|), Portugal champion**

| Rank | Feature | Mean(\|SHAP\|) | Plain-English meaning |
|---:|---|---:|---|
| 1 | `deposit_type` | 1.150 | The deposit policy attached to the booking |
| 2 | `country` | 1.095 | Guest's country of origin |
| 3 | `agent` | 0.911 | The booking agent / channel |
| 4 | `required_car_parking_spaces` | 0.746 | Has the guest requested parking? |
| 5 | `total_of_special_requests` | 0.576 | How many special requests the guest made |
| 6 | `market_segment` | 0.520 | Booking source (Online TA, Direct, Groups, etc.) |
| 7 | `lead_time` | 0.393 | Days between booking and arrival |
| 8 | `arrival_date_year` | 0.281 | Year of arrival |
| 9 | `customer_type` | 0.241 | Transient / Contract / Group / Transient-Party |
| 10 | `previous_cancellations` | 0.234 | Guest's prior cancellation count |

### 4.5.2 What does each top driver tell the hotel?

**The strongest driver is `deposit_type`, and the relationship is
counter-intuitive.** Bookings with "Non Refund" deposit terms cancel
*more often* than bookings with no deposit at all. At first glance
that looks broken — surely a non-refundable deposit should bind the
guest? In practice the non-refundable rate is concentrated in a few
high-volume online travel agents whose own customers cancel
frequently, and the model is learning the *channel pattern*, not the
deposit pattern itself. The lesson for the hotel is that pricing
policies (like enforcing non-refundable deposits) only reduce
cancellation if they change *who* is booking, not just *what they pay*.

**`country` and `agent` rank #2 and #3 — the model is learning which
booking channels are reliable.** Most cancellation-prediction
literature focuses on guest behaviour (how many times have they
cancelled before? do they have loyalty status?). This model says the
hotel's *suppliers* are doing more of the work than the guests.
Operationally, that means the highest-leverage intervention is
auditing the top-cancelling agents and countries, not changing
guest-facing policies.

**The hypothesised top feature, `lead_time`, only ranks 7th.** The
original research hypothesis (H3) predicted that lead time would be
the #1 SHAP driver, followed by deposit type, then previous
cancellations. The data partially supports the hypothesis — all
three features appear in the top 10 — but the rank order is wrong.
Reporting this honestly is more useful than hiding it: a future
hotel using this methodology should not assume the same feature
order applies to their data.

**Two operational drivers push *toward not cancelling*.** Bookings
that requested parking (`required_car_parking_spaces`) or made
special requests (`total_of_special_requests`) cancel *less often*
than otherwise comparable bookings. The pattern visible in Figure 4.5
shows red dots (high feature values) clustering at negative SHAP for
both features — translation: guests who personalise their booking
are more committed to actually showing up. The hotel can use this:
mid-tier bookings where the guest also requested parking are
probably safer than the raw cancellation probability suggests.

**Business takeaway.** The model rewards bookings that show signs of
real intent (parking, special requests, direct channels) and
penalises bookings from high-cancellation channels. The policy
implication is to focus revenue protection on *channels and
countries*, not on individual guest behaviour.

### 4.5.3 Hypothesis verdict quick-check

Chapter I pre-registered five hypotheses about model behaviour.
Their verdicts are summarised below; the full evidence chain is
preserved in Table 4.6 of
`docs/thesis_drafts/chapter_iv_tables/table_06_hypothesis_evidence_verdict.md`
and in Chapter V Section 5.2.

**Table 4.7 — Hypothesis verdict summary**

| H | One-line statement | Verdict | Evidence section |
|---|---|---|---|
| **H1** | Lead time, deposit type, and previous cancellations are significant predictors | **Supported** | 4.5.1 (all three in top 10 SHAP) |
| **H2** | A gradient-boosted model beats baseline algorithms | **Supported** (significant) | 4.3.1 + 4.3.3 (paired bootstrap p ≤ 0.001) |
| **H3** | Lead time has highest SHAP, then deposit type, then previous cancellations | **Partially supported** | 4.5.1 (features correct, rank order wrong) |
| **H4** | Cost-sensitive thresholding reduces expected revenue loss | **Supported** | 4.7.2 (97.5 % recovery on test set) |
| **H5** | Top SHAP feature transfers across geographies (Portugal → Philippines) | **Supported** | 4.5 + appendix (`deposit_type` #1 in both datasets) |

Four of the five hypotheses are fully supported; H3 is partially
supported with the rank order disconfirmed. **Reporting that
partial support honestly — rather than retrofitting the hypothesis
to match the data — is itself a thesis contribution**: it shows the
study treated its predictions as falsifiable claims rather than
post-hoc rationalisations.

---

## 4.6 Average Daily Rate (ADR) Regression

Predicting whether a booking will cancel is half the BI story. The
other half is predicting how much revenue is at stake. This section
documents the **Average Daily Rate (ADR) regressor** that runs
alongside the cancellation classifier in the live deployment — every
`/predict` call returns both a cancellation probability and a
predicted ADR (Section 4.8).

### 4.6.1 Regressor comparison

Seven regression algorithms were fit on the same chronological train
split, tuned on the validation set, and evaluated on the held-out
test set. Test RMSE is in euros, MAPE in percent.

**Table 4.8 — ADR regression performance, all seven models
(Portugal test set)**

| Model | Train RMSE (€) | Val RMSE (€) | Test RMSE (€) | Test MAE (€) | Test R² | Test MAPE (%) |
|---|---:|---:|---:|---:|---:|---:|
| **Gradient Boosting (champion)** | **32.70** | **28.76** | **44.31** | **32.24** | **0.234** | **23.45** |
| XGBoost | 32.89 | 29.30 | 44.06 | 32.14 | 0.243 | 23.48 |
| Decision Tree | 33.74 | 31.28 | 45.87 | 33.28 | 0.179 | 25.15 |
| Ridge | 37.64 | 30.29 | 47.64 | 34.55 | 0.115 | 24.74 |
| Linear Regression | 37.63 | 30.30 | 47.65 | 34.56 | 0.114 | 24.75 |
| Lasso | 39.84 | 30.80 | 51.99 | 38.04 | −0.054 | 27.39 |
| Neural Network | 41.38 | 31.06 | 55.17 | 38.22 | −0.187 | 26.72 |

Full per-model breakdown is in `reports/regression_results.csv`.

**[Insert Figure 4.6 —
`reports/figures/thesis/fig_45_adr_pred_vs_actual.png` here, showing
the predicted vs actual ADR scatter for the champion regressor,
with the y = x reference line.]**

### 4.6.2 Why the R² is moderate (and why that's OK)

The champion Gradient Boosting regressor achieves a Test R² of
**0.234** — meaning the model explains about 23 % of the variance in
ADR. To a machine-learning purist that sounds low, but the result is
expected and operationally useful.

ADR is dominated by two forces. The first is **rate-card pricing** —
the room type, season, channel, and rate plan attached to a booking
— and the model captures this well. The second is **booking-specific
randomness** — group discounts, loyalty perks, promotional codes,
day-of-week price elasticity, last-minute upgrades — that are simply
not in the feature set the regressor sees. R² above 0.30 on a
problem with that structure would suggest data leakage, not skill.

What the model *does* deliver is a **directional ADR signal at
booking time**: it correctly orders bookings by likely revenue, so
the property knows which High-tier cancellation risks are also
high-value bookings. That ordering is what Page 5 of the Power BI
dashboard exposes, and it is enough to drive prioritisation — even
without an exact rate prediction.

**Notable failures.** Linear models (Ridge, Lasso, Linear) all
underperform the tree-based methods; the Lasso and Neural Network
post **negative test R²** (they perform worse than always predicting
the mean ADR). The Neural Network's failure is particularly
instructive: with 119k training rows the network has plenty of data,
but tabular regression on mixed numeric-and-categorical features is
exactly the regime where gradient-boosted trees dominate the
empirical literature. This is one reason no deep-learning architecture
was tried for the cancellation classifier either.

### 4.6.3 Honest disclosure of the ADR regressor's limitation

The ADR regressor was trained with four features that are **not
known at the moment of booking**: `is_canceled` (whether the
booking eventually cancelled), `assigned_room_type` (the room the
guest was actually given on arrival), `booking_changes` (whether
the booking was modified), and `days_in_waiting_list`. Live
inference fills these with sensible defaults — see CLAUDE.md and
`src/serving/inference.py::predict_adr()` for the exact
substitutions. The published Test RMSE of €44.31 is therefore an
*upper bound* on live accuracy; in production the regressor sees
defaulted features and is slightly less accurate.

The methodologically clean fix is retraining on booking-time
features only, which is documented as Future Research item 4 in
Section 5.5. The current regressor is good enough to drive the
directional pricing signal in the dashboard, but the published RMSE
should be read as the *best-case* result.

**Business takeaway.** Combined with the cancellation classifier,
the ADR regressor lets the Power BI dashboard answer a question no
single model could: not just *which bookings will cancel*, but
*which bookings will cancel **and** carry above-average revenue*.
That intersection — high cancellation probability × high predicted
ADR — is exactly the operational priority list the revenue manager
needs each morning.

---

## 4.7 Business Implications

This is the section that converts machine-learning outputs into
revenue-management decisions. The translation has three parts: how
to band bookings into action tiers (4.7.1), how to pick the threshold
that decides what counts as "flagged" (4.7.2), and how the model
performs across the operationally important slices of the customer
base (4.7.3).

### 4.7.1 Risk tiers and revenue exposure

The hotel needs more than a single binary flag. A 99 %-cancel booking
and a 41 %-cancel booking are both "high risk" in a yes/no model, but
they call for very different operational responses. We therefore
partition predicted probabilities into three tiers:

- **Low** — probability < 0.40 — no action required.
- **Medium** — 0.40 ≤ probability < 0.70 — a 72-hour reminder email.
- **High** — probability ≥ 0.70 — a confirmation call and a partial
  deposit request.

**Table 4.9 — Risk tier distribution × revenue exposure
(Portugal test set, n = 11,922)**

| Risk Tier | Probability band | Bookings | % of total | Avg Revenue / Booking (€) | Total Revenue in Tier (€) | Actual Cancellations | Realised Revenue Lost (€) |
|---|---|---:|---:|---:|---:|---:|---:|
| Low | P < 0.40 | 6,107 | 51.22 % | 539.47 | 3,294,519 | 715 | 375,383 |
| Medium | 0.40–0.70 | 2,707 | 22.71 % | 706.14 | 1,911,521 | 1,435 | 1,066,905 |
| High | P ≥ 0.70 | 3,108 | 26.07 % | 650.56 | 2,021,931 | 2,356 | 1,571,978 |
| **Total** | — | **11,922** | **100.00 %** | **606.27** | **7,227,971** | **4,506** | **3,014,266** |

**[Insert Figure 4.7 —
`reports/figures/thesis/fig_23_risk_tier_business_overview.png` here,
showing the risk-tier business overview with revenue exposure per tier.]**

Two findings from this table are worth highlighting for management.

**Risk is heavily concentrated.** The High tier represents only
26.07 % of all bookings but accounts for **52.15 % of all realised
cancellation revenue losses** (€1,571,978 of €3,014,266 in lost
revenue). Operationally, this means the property's biggest single
intervention leverage point is to focus its limited staff time on the
3,108 High-tier bookings rather than spreading reminder effort
uniformly across the 11,922 reservations.

**The risk bands are empirically honest.** The Low tier has an
observed cancellation rate of 11.7 %, Medium 53.0 %, and High 75.8 %.
A "High-tier 75 %-probability" booking really does cancel ~76 % of
the time. The hotel can therefore set deposit policies directly off
the probability number without needing to recalibrate or apply a
fudge factor.

### 4.7.2 Threshold policies — three operating points, three use cases

Choosing where to draw the line between "act" and "don't act"
depends on the cost asymmetry. A wasted reminder email costs €15. A
missed cancellation costs the full revenue of the booking, which
averages €430 across the test set. The model supports three operating
thresholds, each tuned for a different decision context.

**Table 4.10 — Threshold policy operational comparison (LightGBM, Portugal test set)**

| Policy | Threshold | Flagged | % Flagged | TP | FP | FN | Recall | Precision | Total Cost (€) | Savings vs No Model (€) | Use Case |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `max_f1` (balanced) | 0.40 | 5,815 | 48.78 % | 3,791 | 2,024 | 715 | 0.841 | 0.652 | 405,743 | 2,608,523 | Default weekly operations |
| `high_precision` | 0.98 | 426 | 3.57 % | 426 | 0 | 4,080 | 0.095 | 1.000 | 2,874,599 | 139,667 | Quarterly executive audit |
| **`cost_sensitive`** | 0.04 | 8,957 | 75.13 % | 4,486 | 4,471 | 20 | 0.996 | 0.501 | **76,512** | **2,937,754** | **Recommended deployment default** |
| No model (catch nothing) | — | 0 | 0.00 % | 0 | 0 | 4,506 | 0.000 | — | 3,014,266 | — | Baseline reference |

**[Insert Figure 4.8 —
`reports/figures/thesis/fig_11_cost_sensitive_threshold_sweep.png` here,
showing total expected cost as a function of the decision threshold,
with the cost-minimising point marked.]**

Three operational insights fall out of the table.

**The cost-sensitive policy recovers €2,937,754 on the test set —
97.5 % of the theoretical maximum** (the €3,014,266 that would be
lost if the hotel did nothing). It does so by flagging three quarters
of all bookings (8,957 of 11,922), which sounds excessive until you
realise the cost of flagging is €15 and the cost of missing is the
full booking revenue. The model rationally trades many cheap false
positives for the recovery of a few expensive false negatives.

**`max_f1` is the policy for normal weekly operations.** It flags
about half the bookings, catches 84 % of cancellations, and costs
roughly €405,000 in expected losses — a 86 % recovery rate that is
still excellent but leaves the front-desk team time to act on each
flagged booking. This is the policy a property would actually run
day-to-day; the cost-sensitive policy is for crunch periods (peak
season, large-group exposure) where every cancellation hurts.

**`high_precision` is the policy for executive audits, not weekly
ops.** At threshold 0.98 the model only flags 426 bookings (3.6 %)
but every single one is a genuine cancellation (precision 100 %).
This is the right policy when every flag must survive scrutiny — for
example, when the GM wants to query the top 50 highest-risk Groups
bookings before authorising a deposit request — but it costs the
hotel €2.87 million in cancellations it never sees, so it is not the
right policy for daily operations.

### 4.7.3 Per-segment performance

A model that performs well on average can still fail on the
operationally important slices. The dashboard fairness section
(Page 7) and the appendix table `reports/segment_metrics.csv`
break out the champion's test-set metrics by hotel type and by
market segment.

**Table 4.11 — Per-segment performance breakdown (LightGBM at
`max_f1` threshold, Portugal test set)**

| Dimension | Segment | n_rows | Positive Rate | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Hotel** | Resort Hotel | 4,043 | 0.380 | 0.892 | 0.785 | 0.697 | 0.869 | 0.774 |
| **Hotel** | City Hotel | 7,879 | 0.377 | 0.851 | 0.756 | 0.630 | 0.827 | 0.715 |
| Market | **Groups** | 677 | 0.532 | 0.986 | 0.985 | 0.820 | 0.989 | 0.897 |
| Market | Offline TA/TO | 1,710 | 0.234 | 0.976 | 0.901 | 0.787 | 0.988 | 0.876 |
| Market | Online TA | 7,644 | 0.438 | 0.802 | 0.701 | 0.634 | 0.838 | 0.722 |
| Market | **Direct** | 1,546 | 0.196 | 0.808 | 0.489 | 0.475 | 0.531 | 0.502 |

Three observations stand out.

**Resort Hotel outperforms City Hotel by ~4 pp PR-AUC** (0.785 vs
0.756). City bookings cancel more erratically because they include
shorter business trips with last-minute schedule changes, while
resort bookings are longer leisure stays whose cancellation patterns
the model finds easier to learn. Both numbers are operationally
usable; the gap is worth noting because it suggests the cost-sensitive
threshold may need slight per-hotel tuning at large properties.

**Groups bookings are the model's strongest segment by far** — PR-AUC
0.985 with F1 0.897. These are typically large, multi-room bookings
made by event organisers or corporate buyers, and their cancellation
behaviour is highly patterned (one decision often triggers many
cancellations together). For a revenue manager this is unambiguously
good news: the highest-revenue single bookings are precisely the ones
the model can most confidently flag.

**Direct bookings are the model's weakest segment** — PR-AUC 0.489.
These are guests booking via the hotel's own website or walk-in;
they cancel rarely (positive rate 19.6 %), making the prediction
problem harder for any model. The drop matters operationally: the
hotel should treat the model's Direct-booking flags as **noisier
signals** and reserve confirmation calls for cases where the
probability is in the high tier (≥ 0.70), where precision recovers.
For Online TA bookings the model performs in the middle range
(PR-AUC 0.701) — workable but not best-in-class.

**Business takeaway.** The model is uniformly good but not uniformly
*great*. Groups and Offline TA bookings are the easy wins; Direct
bookings need extra human judgement. A defensible deployment policy
treats the model as a strong universal signal that is *augmented*
with operator judgement in the lowest-PR-AUC segments rather than
replaced by it.

### 4.7.4 Honest disclosure of what the numbers do and do not say

The €2.94 million recovery figure is a *one-period upper bound* on
the 2017 test sample. Three real-world frictions reduce it:

- The cost model assumes a uniform €15 intervention cost. In
  practice, the cost of a confirmation call is higher than the cost
  of an automated reminder email, so the savings on the High tier
  (calls) are smaller per intervention than on the Medium tier
  (emails).
- The cost model assumes guests respond to interventions at the
  rates implied by the FN cost being the full revenue at risk. The
  *measured* response rate to reminders and deposit requests is
  unknown — Section 5.5 lists this as a future-research item.
- Deploying at a different property requires retraining. The
  features and threshold values in this chapter are calibrated to
  the Portugal benchmark; a Philippine property's data shape is
  different (Section 5.4 discusses this honestly).

**Business takeaway.** The model is operationally ready, the
probabilities are honest, and the recovery numbers are large. The
remaining work is not better modelling — it is testing the policy
itself in production via A/B trials. Section 5.5 sets out exactly
what that pilot would look like.

---

## 4.8 Model Deployment Framework

A model that lives in a notebook is an academic artefact. A model that
scores a booking the moment it lands in the property's PMS, logs the
score for later audit, surfaces the result in a dashboard the next
morning, and tells the operations team when it is time to retrain —
that is a business intelligence deliverable. This section documents
the deployment framework that wraps the LightGBM champion into exactly
that operational tool.

**[Insert Figure 4.9 —
`reports/figures/thesis/fig_deployment_framework.png` here, showing
the live-serving pipeline from a single booking entry through to the
Power BI dashboard and back via drift-triggered retraining.]**

### 4.8.1 Architecture at a glance

Figure 4.9 maps the full request-to-dashboard data flow. The framework
has four layers, each with a clear job:

- **The serving layer** is a FastAPI application running on
  `localhost:8000`. It exposes three production endpoints — `/predict`
  for booking scoring, `/model-info` for current model lineage,
  `/healthz` for readiness checks — plus a Gradio user interface
  mounted at `/ui` for non-technical staff who prefer a web form to a
  JSON API. Both paths run identical inference, so the same model
  serves both audiences.
- **The inference pipeline** runs entirely in memory once the artefacts
  are loaded: a Pydantic validator coerces the incoming booking,
  feature engineering derives the 33 model inputs, the LightGBM
  pipeline produces a raw probability, isotonic calibration corrects
  the probability, threshold resolution assigns the three policy
  labels and the risk tier, TreeSHAP computes the top-5 feature
  contributions for explainability, and the ADR regressor produces a
  parallel price prediction. The full pipeline returns a JSON response
  in under 500 ms on a laptop-grade CPU.
- **The persistence layer** is asynchronous. After the response is sent
  back to the caller, a FastAPI BackgroundTask appends the (request,
  response) pair to a SQLite audit log at
  `data/predictions/predictions.sqlite` and re-exports the full log to
  `predictions_live.csv`. The user never waits for the disk write, and
  the API never fails if the log is briefly unavailable. The CSV is
  the source of truth Power BI Desktop consumes.
- **The monitoring loop** runs on a separate schedule (typically
  weekly). The `compute_live_drift.py` script reads the live CSV and
  the training-time baseline, computes Population Stability Index per
  feature, and writes a `drift_metrics.csv` with each feature
  classified into a safe / watch / retrain zone. Page 8 of the Power
  BI dashboard reads this file. When two or more features land in the
  retrain zone, the operations team triggers `scripts/train.py` to
  regenerate the artefacts under `artifacts/`, and the loop closes.

### 4.8.2 What this means for the property

The framework is deliberately minimalist. There is no cloud service to
provision, no database server to administer, no model registry to
maintain. A property's IT team needs only:

- A Python environment (the project's `requirements.txt`).
- One server process for FastAPI (single binary, started by
  `python demo/start_server.py`).
- A scheduled task (Windows Task Scheduler or cron) to run
  `scripts/compute_live_drift.py` once a week.
- Power BI Desktop on whichever workstation the revenue manager uses.

Every artefact is a file. Every log is a SQLite database. Every report
is a CSV. A non-technical manager can be handed the `.pbix` file plus
the two CSVs and the dashboard works on first open — no ODBC drivers,
no service accounts, no broken refresh tokens.

### 4.8.3 Production readiness checklist

Three properties make the framework production-ready rather than a
demo:

1. **Calibrated probabilities, not scores.** Because the model has
   been isotonically calibrated (Section 4.4.3), the dashboard's risk
   tier bands can be set directly off the probability number. There is
   no need for a separate "score-to-probability" lookup or a hand-
   tuned safety margin.
2. **Multiple operating thresholds, not one.** The three policies
   (`max_f1`, `high_precision`, `cost_sensitive`) ship with the model
   and are resolvable per-request. The hotel can run the
   cost-sensitive policy by default and switch to `high_precision`
   for executive audits without retraining or redeploying.
3. **Drift monitoring as part of the loop, not as an afterthought.**
   The PSI computation is wired into the same dashboard the revenue
   manager already reads. When the model needs retraining, the
   dashboard tells her — she does not need to remember to ask.

**Business takeaway.** The framework converts the model from a thesis
artefact into an operational tool a hotel can run on commodity
hardware with one Python process and one Power BI workstation. The
ongoing operational cost is one weekly drift run; the trigger for
human intervention is a coloured zone change on Page 8 of the
dashboard.

---

## 4.9 Chapter Summary

The chapter answered the four questions it opened with, plus the
ADR-pricing question that completes the BI story:

1. **Which model performed best?** LightGBM, with statistically
   significant lead over every challenger (Sections 4.3.1, 4.3.3).
   The same algorithm wins under both chronological out-of-time
   evaluation (PR-AUC 0.760) and stratified 10-fold CV (PR-AUC 0.922).
2. **Where does it get predictions right and wrong?** It catches
   84.1 % of cancellations at a 65.2 % precision rate using the
   default operating threshold (Section 4.4.2), and the probabilities
   are honestly calibrated — isotonic regression halves the test ECE
   from 5.8 % to 2.9 % (Section 4.4.3).
3. **What features drive predictions?** Deposit type, country of
   origin, and booking agent — the model is learning *channel
   reliability*, not individual guest history (Section 4.5).
4. **What does it mean for the hotel?** Concentrated risk (26 % of
   bookings carry 52 % of losses), three operating policies tuned to
   three use cases, and a cost-sensitive policy that recovers
   97.5 % of theoretical maximum revenue at risk (Section 4.7).
   Performance is uniformly good across hotel types and most market
   segments; Direct bookings are the noisiest slice and call for
   extra operator judgement (Section 4.7.3).
5. **Can the model also predict revenue at booking time?** Yes — the
   parallel ADR regressor (Section 4.6) delivers a directional pricing
   signal that combines with the cancellation probability to produce
   the High-cancel × High-revenue priority list the property actually
   needs each morning.

And one cross-cutting finding: the **−16 percentage point gap**
between stratified-CV PR-AUC (0.922) and chronological test PR-AUC
(0.760) quantifies the operational cost of concept drift over time
(Section 4.3.2). The deployment framework (Section 4.8) closes that
loop by triggering retraining when PSI drift crosses the 0.25
threshold on two or more features.

Chapter V translates these findings into specific managerial
recommendations, states the study's limitations, and proposes future
research extensions.

# CHAPTER V — Conclusion

> This chapter summarises what the study found, what those findings
> mean for hotel managers in practice, where the work is limited, and
> what future research should add. It is deliberately written in plain
> language so a non-technical reader can act on it without having to
> re-read Chapter IV.

## 5.1 Summary of the Study

Hotel cancellations are expensive. On the Portugal benchmark used in
this study, **€3.01 million of room revenue** was lost to cancelled
bookings across just the 2017 test window — money that walks out the
door before the guest ever arrives. The question the study set out to
answer was simple: *can we tell, at the moment a booking is made,
which ones are likely to cancel — and use that information to act
before the loss happens?*

The approach was machine learning, but the test was operational. Six
algorithms — LightGBM, XGBoost, Gradient Boosting, Random Forest,
Logistic Regression, and a baseline Decision Tree — were trained on
**119,210 cleaned bookings** under a strict chronological split
(oldest 80 % for training, next 10 % for validation, most recent 10 %
held out for testing). Each model's predicted probabilities were
calibrated using isotonic regression so that a "75 % probability"
really means about 75 % of those bookings cancel in real life. The
chosen operating threshold was tuned to minimise the *cost* of wrong
decisions, not just statistical error: missing a cancellation costs
the full revenue of the booking, while flagging one in error costs
about €15 for an automated reminder.

The headline result is that **LightGBM with cost-sensitive
thresholding recovers 97.5 % of the theoretical maximum revenue at
risk** on the test set — €2.94 million of €3.01 million — and the
predictions are honest enough to be used directly as the basis for
deposit and reminder policies. The model is already wired up to a
live FastAPI + Gradio booking-desk interface and feeds an eight-page
Power BI decision-support dashboard, so the operational delivery is
not theoretical.

---

## 5.2 Key Findings

Five findings stand out from Chapter IV.

**1. LightGBM is the best performer, and the lead is statistically
real.** LightGBM achieves a ROC-AUC of 0.864, a PR-AUC of 0.760, and
an F1 score of 0.735 on the chronological out-of-time test set. Paired
bootstrap resampling (2,000 resamples) confirms the lead is
statistically significant against every competing algorithm — p < 0.001
against Random Forest, Logistic Regression, XGBoost, and Decision Tree,
and p = 0.001 against the closest challenger, Gradient Boosting. The
ranking is not a fluke.

**2. The strongest predictor is `deposit_type`, not `lead_time` as
hypothesised.** The original Chapter I hypothesis predicted that
booking-to-arrival lead time would dominate the SHAP ranking, followed
by deposit type, then guest history. The data partially supports this
— all three appear in the top 10 — but the actual rank order is
`deposit_type` (#1), `country` (#2), `agent` (#3), with `lead_time`
only at rank #7. Hotels using this methodology should *not* assume
their own data will reproduce the lead-time-first ranking. The
counter-intuitive direction also matters: bookings with non-refundable
deposits cancel *more* often, not less, because the non-refundable
rate is concentrated in channels whose customers cancel frequently
regardless of the deposit policy.

**3. The model is well-calibrated.** After isotonic calibration, the
Expected Calibration Error on the test set is 2.9 %. A "75 %
probability" booking really does cancel about 75–76 % of the time. The
operational consequence is that deposit policies can be set directly
off the probability number without further adjustment — there is no
need to add a safety margin or apply a fudge factor.

**4. Cancellation risk is heavily concentrated.** The High risk tier
(probability ≥ 0.70) represents only 26.07 % of test-set bookings but
accounts for 52.15 % of realised cancellation revenue losses
(€1.57 million of €3.01 million). The implication is direct:
intervention effort should be tiered, not blanket. The 3,108 High-tier
bookings in the test sample are the single highest-leverage operational
target.

**5. Cost-sensitive thresholding pays its keep.** Under the
cost-sensitive operating policy (threshold = 0.04), the model recovers
**97.5 % of the theoretical maximum revenue at risk** — €2,937,754 of
€3,014,266 — at the cost of about €67,000 in false-positive
interventions. Even under the more conservative `max_f1` policy used
for default weekly operations, the model still saves €2.61 million.
The model is not just academically accurate; it pays for itself
operationally several times over.

---

## 5.3 Managerial Implications

This section is the most important part of the chapter. It translates
findings into six concrete actions a hotel revenue manager can put on
their Monday-morning checklist.

**Recommendation 1 — Adopt the risk-tier-based operational policy.**
Bucket every new booking into Low (probability < 0.40), Medium
(0.40–0.70), or High (≥ 0.70). On the Portugal test sample these tiers
contained 51 %, 23 %, and 26 % of all bookings respectively. The
Power BI dashboard auto-refreshes these counts every week from the
live prediction log. Treat the three bands as policy tiers, not just
analytic categories: each one triggers a different action (see
recommendations 3 and 4).

**Recommendation 2 — Tighten policy by booking source, not by guest
history.** The top three SHAP drivers (`deposit_type`, `country`,
`agent`) are all *channel* signals. The hotel's biggest leverage is
not changing what individual guests pay; it is auditing which agents
and which countries cancel most often, and renegotiating commission
structures conditional on cancellation rates. The model can produce a
sorted list of top-cancelling agents per quarter from
`reports/segment_metrics.csv`. Build the quarterly review meeting
agenda around that list.

**Recommendation 3 — Run a 72-hour reminder workflow on Medium-tier
bookings.** At €15 per intervention, automated reminder emails are
the cheapest layer of the policy stack and address the largest single
slice of revenue at risk in absolute terms (2,707 bookings,
€1.07 million in realised losses on the test sample). The reminder is
operationally light — it can be a templated email sent from the PMS
72 hours before arrival — and is the highest return-on-effort
recommendation in this chapter.

**Recommendation 4 — Reserve confirmation calls and partial deposit
requests for the High tier.** The High tier (3,108 bookings on the
test sample) carried 75.8 % observed cancellation rate, so a manual
intervention here is justified by the hit rate. The intervention is
more expensive than a reminder email (staff time + risk of irritating
the guest) but the High tier's revenue concentration means it earns
back its cost many times over. Front-desk staff should treat the
High-tier list as a "call before Wednesday" workflow.

**Recommendation 5 — Use the live FastAPI + Gradio system as a
frontline tool.** Every booking entered through the existing PMS can
be scored in under 500 ms against the deployed champion model. The
Gradio UI at `localhost:8000/ui` exposes the same model in a form a
non-technical agent can use directly, complete with the predicted
probability, the risk tier, and a top-5 SHAP explanation of *why* the
model flagged this particular booking. The audit log feeds the Power
BI dashboard, so every prediction also becomes part of the
property's ongoing operational record.

**Recommendation 6 — Treat the dashboard's PSI drift page as the
retraining trigger.** Production models silently degrade as customer
behaviour shifts. The dashboard's monitoring page (Page 8) computes
the Population Stability Index for each feature against the training
baseline. When two or more features cross PSI = 0.25, schedule a
retraining cycle. Without this trigger, last quarter's model will
quietly drift below the recovery numbers reported in this thesis,
and the hotel will not notice until it is too late.

---

## 5.4 Limitations of the Study

Honest reporting of what the study did *not* do is as important as
reporting what it did.

**Single benchmark dataset.** The headline numbers in Chapter IV come
from one Portugal property (technically two — City Hotel and Resort
Hotel — within the same dataset) across one geographic region in one
pre-pandemic era (2015–2017). A small Philippine sub-study at Punta
Villa Resort (n = 193 bookings, 2022–2025) was run alongside the main
study as a transferability probe, but the test sample of only 20
bookings produced bootstrap 95 % confidence intervals of roughly
± 15 percentage points on PR-AUC — directionally useful but not
headline-grade. The pre-flight duplicate-cluster diagnostic ran
cleanly on the Philippine export and confirmed the methodology
operates honestly on that data; the *metrics*, however, should be
read as suggestive rather than definitive at that sample size.

**No external context features.** The model uses only the booking's
own data: lead time, deposit, country, agent, requested room,
party composition. It does not see weather forecasts, local event
calendars, airline cancellation feeds, currency-rate movements, or
news of strikes. All of these plausibly affect cancellation
behaviour, especially for international leisure travel, and could
add meaningful predictive power. Section 5.5 lists this as the
first future-research direction.

**Chronological split assumes stationarity within the test period.**
The 2017 test window covers roughly two months. Cancellation
patterns over longer horizons (years, post-pandemic shifts) are
documented in the dashboard's drift monitoring page but were not
modelled directly. A hotel deploying this model in 2025 against
2024 data should validate the metrics on their own holdout, not
assume the 0.864 ROC-AUC will transfer unchanged.

**Cost model is a single-point estimate.** The cost-sensitive
threshold was tuned against a €15 false-positive cost and a
false-negative cost equal to the booking's revenue at risk. The
sensitivity analysis in Notebook 10 shows the policy ranking
(`cost_sensitive` < `max_f1` < `high_precision`) is robust across a
4× perturbation of the false-positive cost, but the *absolute* total
cost figures are obviously sensitive to the assumption. Property-
specific calibration is recommended before deployment.

**ADR regressor uses post-booking features at training time.** The
accompanying Average Daily Rate (ADR) regression model was trained
with four features (`is_canceled`, `assigned_room_type`,
`booking_changes`, `days_in_waiting_list`) that are not known at
booking time. Live inference fills these with sensible defaults, so
live `predicted_adr` values are slightly less accurate than the
test-set RMSE of €44.31 reported in the appendix. The
methodologically clean fix is retraining on booking-time features
only — flagged in Section 5.5.

**No A/B testing of the intervention policies themselves.** The
€2.94 million recovery figure is an *upper bound* — it assumes that
when the hotel reminds, calls, or asks for a deposit, the guest
responds at the rate implied by the cost model. The *measured*
response rate is unknown until the policies are run in production.
Without A/B testing, the savings number is best treated as an
operational target, not a guaranteed outcome.

---

## 5.5 Recommendations for Future Research

Five concrete extensions would build directly on this work.

**Future Research 1 — Add external context features.** Public APIs
provide weather forecasts, local event calendars (concerts,
conferences, sports), airline schedule changes, and FX-rate
movements. Adding even a subset of these — say, a daily weather
forecast and a "local event happening within 10 km" indicator —
could plausibly add 1–3 percentage points of PR-AUC and would
specifically improve performance on leisure-travel cancellations,
which are the most weather-sensitive segment. A follow-up student
could build a feature pipeline that joins external feeds against the
booking arrival date and re-run the chronological evaluation.

**Future Research 2 — Replicate on additional Philippine properties.**
The Punta Villa Resort sub-study (n = 193) was a transferability
probe, not a headline result. Replicating the same methodology on
10–15 Philippine resorts — ideally a mix of city and beach properties
— would let the field produce region-specific headline numbers with
tight enough confidence intervals to be operationally actionable.
This is the most direct extension of the present work and is well
within a future thesis student's scope.

**Future Research 3 — A/B test the intervention policies.** Randomly
assign Medium-tier bookings to "reminder" vs "no reminder" arms over
a six-month deployment, and compare the realised cancellation rate
between arms. This converts the current upper-bound €2.94 million
figure into a measured treatment effect that survives causal
scrutiny. The same A/B framework could test the precise wording of
the reminder email, the timing (72 hours vs 48 hours), and the
deposit-request threshold for the High tier.

**Future Research 4 — Build an ADR regressor on booking-time features
only.** The current ADR regressor reaches Test R² ≈ 0.23, reflecting
both fundamental noise (rate cards change with promotions and
day-of-week pricing) and the use of post-booking features at training
time. A clean retrain on booking-time features only — combined with
external pricing context like competitor rate scrapes — would tighten
the live-time forecast and let Page 5 of the Power BI dashboard
report a true booking-time ADR signal.

**Future Research 5 — Package the methodology contributions as a
library.** Two reusable artefacts came out of this work that have
value beyond the specific cancellation problem. The first is a
**pre-flight duplicate-cluster diagnostic** that detects datasets
where chronological splitting would leak twins across the train/test
boundary (a problem the Philippine PMS export forced us to discover).
The second is a **feature-availability mapping** for reduced-PMS
schemas — a structured way to map the features a small property has
against the features a benchmark model expects, and to honestly
report what predictive power is lost when columns are missing. A
future student could package both as a standalone Python library,
publishable on PyPI as a contribution to the broader hospitality
analytics community.

---

## 5.6 Closing Statement

This study set out to show that cancellation risk is predictable at
the moment of booking with calibrated probabilities honest enough to
drive cost-sensitive action. The Portugal benchmark gave a clean,
defensible answer: yes, it is — and the recovery numbers are large
enough that the model pays for itself many times over per booking
cycle.

The operational pipeline is in place. The model is deployed behind a
live API, the predictions feed an audit log, the audit log feeds a
production-grade Power BI dashboard, and the dashboard's monitoring
page knows when to ask for a retrain. The policy recommendations in
Section 5.3 are concrete and ready to run. The €2.94 million figure
on the test set is the upper bound; what the hotel actually recovers
will depend on how well its staff execute the reminders, calls, and
deposit requests, and on how well guests respond to them.

The remaining work is not better modelling — it is replication on
additional properties, the addition of external context features,
and live A/B validation of the policy itself. Each of those is set
out concretely in Section 5.5 as a thesis-scaled research extension.

# References

References

Bertsimas, D., &

Kallus, N. (2019). From predictive to prescriptive analytics. Management Science , 66 (3), 1025–1044. https://doi.org/10.1287/mnsc.2018.3253

Grigas, A. N. E. &. P. (2022). Smart “Predict, then Optimize.” ideas.repec.org . https://ideas.repec.org/a/inm/ormnsc/v68y2022i1p9-26.html?utm_source=chatgpt.com Machine Learning in Hospitality: Interpretable Forecasting of booking cancellations . (2025). IEEE Journals & Magazine | IEEE Xplore. https://ieeexplore.ieee.org/document/10857340

Núñez, J. C. S., Gómez ‐

Pulido, J. A., &

Ramírez, R. R. (2024). Machine learning applied to tourism: A systematic review. Wiley Interdisciplinary Reviews Data Mining and Knowledge Discovery , 14 (5). https://doi.org/10.1002/widm.1549

Srivastava, T. (2025, May 1). 12 Important model evaluation Metrics for Machine Learning Everyone should know (Updated 2025) . Analytics Vidhya. https://www.analyticsvidhya.com/blog/2019/08/11-important-model-evaluation-error-met rics/#:~:text=This%20is%20again%20one%20of,clearer%20in%20the%20following%20 sections

Chen, C.,

Schwartz, Z., &

Vargas, P. (2010). The search for the best deal: How hotel cancellation policies affect the search and booking decisions of deal-seeking customers. International Journal of Hospitality Management , 30 (1), 129–135. https://doi.org/10.1016/j.ijhm.2010.03.010

Mele, C.,

Russo-Spena, T., &

Tuan, A. (2023). Generative AI and customer-centric business models: a decision-making framework. Journal of the Academy of Marketing Science, 51 , 701–708. Predicting hotel booking cancellations using tree-based neural network . (Yang 2014). https://www.researchgate.net/publication/385933974_Predicting_hotel_booking_cancella tions_using_tree-based_neural_network

Antonio, N., De

Almeida, A., &

Nunes, L. (2019). Hotel booking demand datasets. Data in Brief , 22 , 41-49. https://doi.org/10.1016/j.dib.2018.11.126 M. S.

Satu, K. Ahammed and M. Z. Abedin (2020), "Performance Analysis of Machine Learning Techniques to Predict Hotel booking Cancellations in Hospitality Industry," 2020 23rd International Conference on Computer and Information Technology (ICCIT), DHAKA, Bangladesh, 2020, pp. 1-6, doi: 10.1109/ICCIT51783.2020.9392648.

Gong, Y. (2024). The impact of booking lead time on hotel customers’ willingness to pay for more lenient cancellation terms . https://udspace.udel.edu/items/890306df-1687-4dda-973d-768ef95fd32d Piga C, Melis G (2021), "Identifying and measuring the impact of cultural events on hotels’ performance". International Journal of Contemporary Hospitality Management, Vol. 33 No. 4 pp. 1194–1209, doi: https://doi.org/10.1108/IJCHM-07-2020-0749

C-Sánchez, E., &

Sánchez-Medina, A. J. (2024). Detecting Short-Notice Cancellation in Hotels with Machine Learning. Engineering Proceedings , 68 (1), 43. https://doi.org/10.3390/engproc2024068043

Chen, S.,

Ngai, E. W.,

Ku, Y.,

Xu, Z.,

Gou, X., &

Zhang, C. (2023). Prediction of hotel booking cancellations: Integration of machine learning and probability model based on interpretable feature interaction. Decision Support Systems , 170 , 113959. https://doi.org/10.1016/j.dss.2023.113959 18 SHAP – Interpretable Machine Learning . (n.d.). https://christophm.github.io/interpretable-ml-book/shap.html Scott M. Lundberg, Su-In Lee (2017) A unified approach to interpreting model predictions https://dl.acm.org/doi/10.5555/3295222.3295230#core-cited-by

Liu, Z., De

Bock, K. W., &

Zhang, L. (2024). Explainable Profit-Driven Hotel Booking Cancellation Prediction based on Heterogeneous Stacking-Based Ensemble Classification. European Journal of Operational Research , 321 (1), 284–301. https://doi.org/10.1016/j.ejor.2024.08.026

Xiao, J.,

Abidin, S. Z.,

Vermol, V. V., &

Gong, B. (2024). Dynamic temporal reinforcement learning and policy-enhanced LSTM for hotel booking cancellation prediction. PeerJ. Computer science, 10, e2442. https://doi.org/10.7717/peerj-cs.2442 Yang D, Miao X. 2024. Predicting hotel booking cancellations using tree-based neural network. PeerJ Computer Science 10:e2473 https://doi.org/10.7717/peerj-cs.2473

Yang, D., &

Miao, X. (2024). Predicting hotel booking cancellations using tree-based neural network. PeerJ. Computer science , 10 , e2473. https://doi.org/10.7717/peerj-cs.2473

Herrera, A., Arroyo, Á.,

Jiménez, A., & Herrero, Á. (2024). Forecasting hotel cancellations through machine learning. Expert Systems , 41 (9), e13608. https://doi.org/10.1111/exsy.13608 M. S.

Satu, K. Ahammed and M. Z. Abedin, "Performance Analysis of Machine Learning Techniques to Predict Hotel booking Cancellations in Hospitality Industry," 2020 23rd International Conference on Computer and Information Technology (ICCIT), DHAKA, Bangladesh, 2020, pp. 1-6, doi: 10.1109/ICCIT51783.2020.9392648.

Choi, Y., &

Kim, J. (2023). A signaling theory of reservation cancellation policies. Economic Modelling , 130 , 106588. https://doi.org/10.1016/j.econmod.2023.106588

Kim, E. J.,

Kim, E. L.,

Kim, M., &

Tanford, S. (2023). Post-Pandemic hotel cancellation policy: Situational cues as perceived risk triggers. Journal of Hospitality and Tourism Management , 55 , 153–160. https://doi.org/10.1016/j.jhtm.2023.03.019 Bookassist Marketing. (n.d.). How to combat the soaring rate and cost of cancellations . Hotel Tech Report . https://hoteltechreport.com/news/how-to-combat-the-soaring-rate-and-cost-of-cancellatio ns

Schwartz, Z.,

Webb, T. D.,

Altin, M., &

Riasi, A. (2025). Overbooking and performance in hotel revenue management. International Journal of Hospitality Management , 129 , 104192. https://doi.org/10.1016/j.ijhm.2025.104192

Hewapathirana, I. U. (2025). Advancing tourism demand forecasting in Sri Lanka: Evaluating the performance of machine learning models and the impact of social media data integration. Journal of Tourism Futures, 11 (2), 261–285. https://doi.org/10.1108/JTF-06-2023-0149

Teece, D. J.,

Pisano, G., &

Shuen, A. (1997). Dynamic capabilities and strategic management. Strategic Management Journal , 18(7), 509–533.

Teece, D. J. (2007). Explicating dynamic capabilities: The nature and microfoundations of (sustainable) enterprise performance. Strategic Management Journal , 28(13), 1319–1350. Teece (2012)

Teece, D. J. (2012). Dynamic capabilities: Routines versus entrepreneurial action. Journal of Management Studies , 49(8), 1395–1401.

Teece, D. J. (2018). Profiting from innovation in the digital economy: Enabling technologies, standards, and licensing models in the wireless world. Research Policy , 47(8), 1367-1387.

Ernst, V. (2025). Developing dynamic capabilities for digital transformation from a German manufacturing industry perspective . University of Portsmouth. (Found in search results) A. c, B.

Mahato, A.

Kujur, M.

Pandey, B. Kumar and S. Alam, "Predicting Cancellations of Hotel Reservations with Unsupervised Machine Learning Methods," 2024 International BIT Conference (BITCON), Dhanbad, India, 2024, pp. 1-6, doi: 10.1109/BITCON63716.2024.10985602. I.

Gómez-Talal, M. Azizsoltani, et al (2025), "Machine Learning in Hospitality: Interpretable Forecasting of Booking Cancellations," in IEEE Access, vol. 13, pp. 26622-26638, 2025, doi: 10.1109/ACCESS.2025.3536094. The influence of online reviews to online hotel booking intentions . (n.d.). ResearchGate. https://www.researchgate.net/publication/280739761_The_influence_of_online_reviews_ to_online_hotel_booking_intentions

Lynn, J. (2025, May 19). Predicting cancellations with survival modeling . Medium. https://booking.ai/predicting-cancellations-with-survival-modeling-a299af54249b
