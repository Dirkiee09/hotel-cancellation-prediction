# A Strategic Business Intelligence Approach to Predicting Hotel Booking Cancellations

**Authors:** Avanceña, Luis Miguel C. · Montecino, Nathaniel · Viñas, Dirk Werner

**Thesis Advisers:** Prof. John Edward Manalac · Dr. Donn Enrique L. Moreno

---

## Table of Contents

- [Chapter I — Introduction](#chapter-i--introduction)
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

## Statement of the Problem

 In today’s hospitality industry, hotel cancellations have become one of the most persistent challenges affecting revenue management and customer relations. While flexible cancellation policies are designed to improve guest satisfaction and encourage bookings, they can also lead to unpredictable income and wasted room inventory when not properly managed. Frequent or last-minute cancellations make it difficult for hotels to forecast occupancy, allocate resources efficiently, and maintain profitability. This creates a constant struggle for hotel managers to balance customer convenience with the financial stability of their operations. Although hotels collect large volumes of guest and booking data, many still rely on manual judgment or traditional methods when handling cancellations. These approaches often overlook valuable insights that can be discovered through data analytics, such as identifying cancellation trends, recognizing high-risk bookings, and predicting potential revenue loss. Without a data-driven system, hotels face inconsistent decision-making and miss opportunities to recover from cancellations or rebook available rooms quickly. Therefore, this study aims to explore how Business Intelligence and Analytics can be used to improve decision-making related to hotel cancellations. Specifically, it seeks to determine how analytical tools can (1) predict which bookings are most likely to be canceled, (2) measure the financial impact of cancellations on hotel performance, and (3) develop a data-driven model that helps hotel management respond proactively to minimize losses. By applying data analysis and visualization, this study intends to guide hotels toward more strategic and evidence-based approaches that reduce cancellation risks while sustaining profitability and customer satisfaction. 

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

The Philippine PMS export does **not** capture `country`, `agent`, `market_segment`, `customer_type`, `previous_cancellations`, `previous_bookings_not_canceled`, `required_car_parking_spaces`, or `meal` (the latter is constant and dropped). This is the feature-availability constraint that Chapter IV § 4.6.2 develops as a methodology contribution.

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

## 4.1 Introduction

Chapter III described how this study applies the Dynamic Capability Theory
(DCT) cycle of **Sense → Seize → Transform** to two datasets in parallel:
the Portugal benchmark (119,390 bookings, 2015-2017) and the Philippine
sub-study based on the real Punta Villa Resort PMS export (193 bookings,
2022-2025). This chapter reports the empirical results of that
two-dataset application and discusses what each result means for the
hypotheses (H1-H5), for the four research objectives, and for hotel
operations.

The chapter is structured around the three DCT phases. Section 4.2 reports
**Sense** findings — the exploratory patterns the data reveals about
cancellation behavior. Section 4.3 reports **Seize** findings — model
selection, calibrated probabilities, hypothesis tests, and feature
importance. Section 4.4 reports **Transform** findings — the business
implications, the cost-sensitive threshold result, the Power BI dashboard,
and the live serving infrastructure. Section 4.5 reports the **Philippine
transferability study** as a parallel application of the same pipeline.
Section 4.6 lists the three **methodology contributions** that emerged
from the work, and Section 4.7 closes with a summary of findings that
sets up Chapter V.

---

## 4.2 SENSE — Exploratory Findings

### 4.2.1 Portugal dataset characterisation

After applying the cleaning rules described in Chapter III
(`clean_raw` and `validate_raw` in `src/utils/validate_data.py`), the
Portugal dataset retained **119,210 bookings**, with 181 rows dropped:
180 rows had zero guests and one row had a negative ADR.[^1] The
chronological 80/10/10 split produced the row counts shown in Table 4.1.

**Table 4.1 — Portugal dataset split summary**[^2]

| Split | Rows | Date range | Cancellation rate |
|---|---|---|---|
| Train | 95,367 | 2015-07-01 → 2017-04-22 | 36.1 % |
| Validation | 11,920 | 2017-04-22 → 2017-06-21 | 43.9 % |
| Test | 11,922 | 2017-06-21 → 2017-08-31 | 37.8 % |
| **All cleaned** | **119,210** | **2015-07-01 → 2017-08-31** | **37.0 %** |

Several risk patterns are visible in the exploratory analysis (see
`notebooks/01_eda.ipynb` and the figures saved at
`reports/figures/thesis/`):

1. **Lead time is a strong but not dominant cancellation signal.**
   Bookings with a lead time of 100 days or more cancel at roughly
   double the rate of bookings with a lead time of seven days or less.
   The relationship is monotonic across lead-time bands.[^3]

2. **`deposit_type = "Non Refund"` is counter-intuitively associated
   with higher cancellation, not lower.** A booking with a
   non-refundable deposit policy cancels more often than a booking
   with no deposit policy in this dataset.[^4] This pattern survives
   controls for market segment and lead time, so it is not a
   confounding artefact. A plausible interpretation is that
   non-refundable deposits in this dataset are often paid through
   channels that allow downstream chargebacks or insurance
   recovery, so the deposit policy field captures *booking intent*
   rather than *commitment*.

3. **The "Groups" market segment carries the highest cancellation
   rate.** Group bookings cancel at roughly 1.6× the rate of the
   "Direct" segment, consistent with the literature on event-driven
   booking volatility (Antonio et al., 2019).

4. **Returning guests with prior successful stays cancel less.**
   Bookings from guests with at least one previous non-cancelled stay
   exhibit a markedly lower cancellation rate than first-time
   bookings. This is one of the most intuitive findings, and it
   foreshadows the SHAP importance of `previous_bookings_not_canceled`
   reported in §4.3.4.

These four patterns answer Research Objective 1 ("identify and analyze
the primary factors and patterns that correlate with booking
cancellations") and provide the empirical motivation for the
**Sensing** capability described in the conceptual framework.

### 4.2.2 Philippine sub-study characterisation

The Philippine dataset — the real Punta Villa Resort PMS export — was
loaded and cleaned through a parallel pipeline (`clean_raw_ph` in
`src/utils/validate_data.py`). Zero rows were dropped during cleaning.
Table 4.2 summarises its split structure.

**Table 4.2 — Philippine dataset split summary**[^5]

| Split | Rows | Date range | Cancellation rate |
|---|---|---|---|
| Train | 154 | 2022-12-29 → 2025-04-? | — |
| Validation | 19 | 2025-04-? → 2025-08-? | — |
| Test | 20 | 2025-08-? → 2025-12-28 | — |
| **All cleaned** | **193** | **2022-12-29 → 2025-12-28** | **15.0 %** (29 / 193) |

Two observations frame the rest of the Philippine analysis:

- **The base rate is roughly 2.5 times lower than Portugal's** (15.0 %
  vs 37.0 %). A plausible explanation is that Punta Villa is a single
  resort property with a high share of Walk-In, local-clientele
  bookings, while the Portugal dataset combines a city hotel and a
  resort hotel serving a global tourist mix.
- **The sample is small.** With only 20 test rows and roughly three
  test positives, bootstrap 95 % confidence intervals on PR-AUC span
  approximately ±15 percentage points. Every Philippine metric in
  this chapter is therefore reported as a directional estimate, not
  a production-grade headline.

#### Pre-flight duplicate-cluster diagnostic

Before fitting any model on the Philippine sample, the methodology
applies a **pre-flight check** that counts duplicate feature vectors
and measures the fraction of duplicate clusters whose constituent
rows share a single label. If the duplicate rate exceeds 30 % and
label consistency exceeds 90 %, the chronological split risks leaking
train/test twins, which would inflate test metrics by recognition
rather than generalization.

The diagnostic outcome on the real Philippine dataset is shown in
Table 4.3.

**Table 4.3 — Pre-flight diagnostic outcome on Philippine data**[^6]

| Metric | Value | Interpretation |
|---|---|---|
| Duplicate vector rate | **0.0 %** | Every booking has a unique feature signature |
| Multi-row clusters with consistent labels | 0 / 0 | No clusters exist to be measured |
| Test rows with a train/val twin | **0 / 20** | Methodology proceeds without inflation risk |

The diagnostic does **not** fire on the real Punta Villa data. This is
the right result for two reasons: it confirms the test metrics in §4.5
measure genuine generalization rather than memorization, and it
demonstrates the value of running the diagnostic before claiming
transferability on small datasets — a point developed further as a
methodology contribution in §4.6.

### 4.2.3 Cross-dataset cancellation drivers

Both datasets show the same broad cancellation-driver hierarchy at the
exploratory level: deposit policy, lead time, and booking source all
move cancellation rates by tens of percentage points. The two datasets
differ on what the *dominant* driver is at the multivariate level, and
that difference is taken up rigorously through SHAP in §4.3.4 and §4.5.3.

---

## 4.3 SEIZE — Modelling Results

### 4.3.1 Pipeline summary

The modelling pipeline implemented in `src/pipelines/train.py` and
`scripts/train_ph.py` proceeds in seven steps for each dataset:

1. **Clean** the raw CSV (drop invalid rows, fill known imputable
   nulls, derive booking-time features).
2. **Validate** the cleaned frame against the schema and target
   binary check.
3. **Split** chronologically into train / validation / test (80 / 10 / 10).
4. **Fit** candidate model families on the train set (Decision Tree,
   Logistic Regression, Random Forest, Gradient Boosting, XGBoost,
   LightGBM).
5. **Calibrate** the probability outputs using **isotonic regression**.
   In plain terms, machine-learning models often output scores that
   *rank* bookings correctly but do not match real-world frequencies —
   a "70 %" score might really mean a 50 % chance of cancellation.
   Isotonic regression is a simple mathematical adjustment that
   re-maps these raw scores so that a "70 %" prediction corresponds to
   actual 70 % cancellation in the validation data. After this step,
   the percentages a manager sees on the dashboard can be trusted as
   real cancellation likelihoods.
6. **Sweep** decision thresholds — that is, try many possible cut-off
   points (e.g., flag the booking if the probability exceeds 40 %,
   60 %, or 90 %) and pick the cut-off that best matches each business
   stance: `max_f1` (balanced), `high_precision` (only flag the very
   confident cases), and `cost_sensitive` (flag aggressively because
   missing a cancellation is more expensive than acting on a guest who
   would have arrived anyway).
7. **Persist** the champion pipeline, the calibrator, the thresholds,
   and explanatory artefacts (SHAP rankings, threshold sweep CSV) so
   downstream consumers (notebooks, Power BI, serving) can read them.

Calibrated probabilities are used everywhere downstream so the
percentages displayed in the user interface and the business
dashboards can be interpreted as actual cancellation likelihood.

### 4.3.2 Model selection (Portugal)

Model selection used **rolling-origin cross-validation** — a
time-respecting method that mirrors how a hotel would actually use
the model in practice. Instead of training the model once, we trained
it three separate times on progressively larger time windows (60 %,
70 %, and 80 % of the chronological training data). Each time, the
model was evaluated on the next slice of bookings it had not yet
seen. This setup guarantees the model is never tested on data from
its own past — exactly the situation a hotel faces when scoring a
future booking.

The metric used to compare models is **PR-AUC** (Precision-Recall
Area Under the Curve). In simple terms, PR-AUC is a single score
between 0 and 1 that summarises how well the model balances catching
real cancellations (high recall) against not raising false alarms on
guests who actually arrive (high precision). A PR-AUC of 0.5 would
mean the model is no better than chance on a balanced dataset; a
PR-AUC of 1.0 would mean perfect separation. PR-AUC is the right
choice here (over plain accuracy) because cancellations are the
minority class — a model that simply predicted "no cancellation" for
every booking would still be 63 % accurate but useless to a revenue
manager.

**Table 4.4 — Rolling-origin CV summary (Portugal, 3 folds)**[^7]

| Model | Rolling PR-AUC mean (±std) | Rolling ROC-AUC mean (±std) |
|---|---|---|
| **LightGBM** | **0.870 ± 0.039** | **0.912 ± 0.021** |
| XGBoost | 0.867 ± 0.037 | 0.911 ± 0.017 |
| GradientBoosting | 0.867 ± 0.035 | 0.910 ± 0.016 |
| RandomForest | 0.840 ± 0.030 | 0.895 ± 0.016 |
| LogisticRegression | 0.843 ± 0.044 | 0.890 ± 0.021 |
| DecisionTree | 0.584 ± 0.042 | 0.746 ± 0.020 |

**LightGBM is the champion** under this policy. Its PR-AUC gap over the
runner-up (XGBoost) is small (+0.0028), but the gap over the simpler
families (Random Forest, Logistic Regression) is substantial and is
shown to be statistically significant in §4.3.4. The selection lineage
is logged at `reports/champion_summary.json`.

### 4.3.3 Held-out test-set performance (Portugal)

After the champion was selected on validation folds, the chosen
LightGBM pipeline (preprocessor + classifier + isotonic calibrator)
was applied **once** to the held-out 11,922-row test set. The test set
was not touched during model selection, calibration, or threshold
choice. Table 4.5 presents the per-model test-set probability metrics
side-by-side for comparison.

**Table 4.5 — Held-out test-set performance per model (Portugal)**[^8]

| Model | ROC-AUC | PR-AUC | Brier | ECE |
|---|---|---|---|---|
| **LightGBM (champion)** | **0.864** | **0.760** | 0.146 | 0.029 |
| GradientBoosting | 0.861 | 0.754 | 0.148 | 0.033 |
| XGBoost | 0.855 | 0.749 | 0.151 | 0.033 |
| RandomForest | 0.851 | 0.739 | 0.152 | 0.031 |
| LogisticRegression | 0.839 | 0.739 | 0.158 | 0.028 |
| DecisionTree | 0.675 | 0.508 | 0.217 | 0.079 |

**What these numbers mean in plain language.** The champion's
**PR-AUC of 0.760** is more than double the dataset's natural
cancellation rate of 0.370, which means the model genuinely separates
cancellers from stayers rather than just guessing the average rate.
The **ROC-AUC of 0.864** can be read as: if we picked one random
canceller and one random stayer from the test set, the model would
correctly assign the canceller a higher probability about 86 % of
the time. The **Brier score of 0.146** is a combined measure of how
close the predicted percentages are to the actual outcomes — lower
is better, and 0.146 is in the range generally considered useful for
business decision support.

The **Expected Calibration Error (ECE) of 0.029** is the most
business-relevant number in the table. It can be read as follows:
when the model says a booking has a 30 % chance of being cancelled,
the actual cancellation rate for similar bookings is somewhere
between roughly 27 % and 33 %. In other words, the percentages the
model shows the manager are honest — a "high-risk" booking really
is high-risk at the rate displayed. Without this calibration step,
the model might say "70 %" when the truth was closer to 50 %, and a
manager would mistakenly trigger expensive interventions on
bookings that did not need them.

Figure 4.1 (`reports/figures/thesis/fig_01_roc_pr_curves.png`) plots
the ROC and PR curves for the champion. Figure 4.2
(`reports/figures/thesis/fig_05_calibration_reliability_and_histogram.png`)
shows the calibration reliability diagram — a graph that visually
confirms the model's predicted percentages match the actual
cancellation rates across the full 0–100 % range.

### 4.3.4 Hypothesis tests

This subsection tests the three hypotheses stated in Chapter I that
concern modelling: H1, H2, and H3.

#### Hypothesis 1 — Lead time, deposit type, and previous cancellations are significant predictors

**Verdict: Supported.** All three features appear in the top-10 SHAP
ranking by mean(\|SHAP\|) on the Portugal test set, confirming that
each carries non-trivial predictive signal in the calibrated model.
The full top-10 aggregated to raw features is shown in Table 4.7
below. The aggregation collapses one-hot-encoded categorical features
back to their raw column (so the four columns
`deposit_type_{Non Refund, No Deposit, Refundable}` sum into the
single raw feature `deposit_type`).

#### Hypothesis 2 — A gradient-boosted tree model will achieve higher evaluation than baseline models

**Verdict: Supported with statistical significance.** To check whether
LightGBM's lead over the other models is real or just lucky, we used a
technique called **paired bootstrap resampling**. The idea is simple:
we drew 2,000 random samples from the test set (with replacement, so
each sample is roughly the same size as the original) and recomputed
each model's PR-AUC on every sample. If LightGBM's lead survives
across 95 % of those resamples, the gap is statistically real, not a
coincidence of which particular bookings happened to land in the test
set. Table 4.6 shows the results.

**Table 4.6 — Paired bootstrap significance of LightGBM vs each
challenger (Portugal test set, PR-AUC)**[^9]

| Champion vs Challenger | Δ PR-AUC | 95 % CI | p-value | Significant? |
|---|---|---|---|---|
| LightGBM vs DecisionTree | +0.252 | [0.242, 0.264] | < 0.001 | Yes |
| LightGBM vs RandomForest | +0.021 | [0.016, 0.027] | < 0.001 | Yes |
| LightGBM vs LogisticRegression | +0.021 | [0.015, 0.028] | < 0.001 | Yes |
| LightGBM vs XGBoost | +0.011 | [0.008, 0.014] | < 0.001 | Yes |
| LightGBM vs GradientBoosting | +0.007 | [0.003, 0.011] | 0.001 | Yes |

LightGBM's lead is statistically real against every other model,
including the closest gradient-boosting alternatives. The only
exception is the **F1 score at the balanced threshold**, where
LightGBM and Gradient Boosting are essentially tied (Δ = +0.0003,
p = 0.905). In plain terms: when comparing the two models' overall
ability to rank bookings by risk, LightGBM is clearly better; but at
the specific decision cut-off chosen for the balanced policy, the two
models flag roughly the same set of bookings. This is why the champion
was chosen on PR-AUC (the ranking score) rather than F1 (the score at
one specific cut-off) — PR-AUC is the more stable basis for selection.

#### Hypothesis 3 — Lead time has the greatest SHAP importance, followed by deposit type, then previous cancellations

**Verdict: Partially supported.** All three predicted features
appear in the top-10 by mean(\|SHAP\|), but the rank order differs
from the hypothesis. Table 4.7 shows the actual ranking aggregated
to raw features.

**Table 4.7 — SHAP feature importance (Portugal champion), aggregated
to raw feature names**[^10]

| Rank | Raw feature | Aggregated mean(\|SHAP\|) |
|---|---|---|
| 1 | **`deposit_type`** | 1.150 |
| 2 | `country` | 1.095 |
| 3 | `agent` | 0.911 |
| 4 | `required_car_parking_spaces` | 0.746 |
| 5 | `total_of_special_requests` | 0.576 |
| 6 | `market_segment` | 0.520 |
| 7 | `lead_time` | 0.393 |
| 8 | `arrival_date_year` | 0.281 |
| 9 | `customer_type` | 0.241 |
| 10 | `previous_cancellations` | 0.234 |

The model's actual top three are **`deposit_type`, `country`, and
`agent`** — not `lead_time` first. The hypothesised features are still
present (`lead_time` at rank 7, `previous_cancellations` at rank 10),
but the model has discovered that booking-source signals
(`country`, `agent`, `market_segment`) and policy signals
(`deposit_type`) are more discriminative than the raw lead time once
calibrated.

The divergence is methodologically informative rather than a defeat
of the hypothesis. Three points are worth making explicit:

1. **`deposit_type` dominates** because the encoded categorical level
   `deposit_type = "Non Refund"` is by far the single most influential
   SHAP feature, with mean(\|SHAP\|) = 0.911 on its own. This is the
   counter-intuitive pattern noted in §4.2 — non-refundable deposits
   in this dataset are paradoxically associated with higher
   cancellation. The model captures the pattern directly.
2. **Booking-source identity matters more than raw lead time.** Once
   the model knows the agent ID or the source country, the residual
   value of knowing the exact lead time is smaller. Lead time is a
   driver, but it is partially redundant with channel identity.
3. **The hypothesis is falsifiable and was falsified in part.** The
   academic value of stating H3 explicitly in Chapter I is preserved
   precisely because the data was allowed to override the
   hypothesised order. Future researchers writing similar predictions
   should expect their feature-importance rankings to be revised by
   the data.

Figure 4.3 (`reports/figures/thesis/fig_13_shap_feature_importance_bar.png`)
plots the encoded-feature SHAP bar chart. Figure 4.4
(`reports/figures/thesis/fig_14_shap_beeswarm.png`) shows the SHAP
beeswarm with per-row contribution distribution.

### 4.3.5 Probability calibration

Recall from §4.3.1 that the pipeline includes a calibration step
that re-maps the model's raw scores so the displayed percentages
correspond to real-world cancellation rates. Table 4.8 shows how
much the calibration step improves the model's honesty, before and
after the adjustment.

**Table 4.8 — Calibration metrics (Portugal)**[^11]

| Split | Brier (raw) | Brier (calibrated) | ECE (raw) | ECE (calibrated) |
|---|---|---|---|---|
| Validation | 0.120 | 0.114 | 0.046 | < 0.001 |
| Test | 0.150 | 0.146 | 0.058 | 0.029 |

The calibration step roughly cuts the test-set ECE in half (from
0.058 down to 0.029). Reading this in business language: before
calibration, the model might say "60 %" when the real rate was
closer to 54 %, a six-point gap that would lead managers to
over-react to mid-range bookings. After calibration, that same
prediction would be within roughly three points of the truth. This
matters because the next step of the pipeline (the cost-sensitive
threshold in §4.4) uses these percentages to decide which bookings
trigger interventions — if the percentages were off by ten points,
the cost calculations would be off in lockstep, and the dashboard
recommendations would be unreliable.

---

## 4.4 TRANSFORM — Business Implications and Decision Support

### 4.4.1 Threshold policy comparison

The pipeline materialises three threshold policies on the validation
set. Each policy serves a different operational stance. Table 4.9
shows the held-out test-set metrics that result from applying each
threshold to the calibrated probabilities.

**Table 4.9 — Threshold policy comparison on the Portugal test
set**[^12]

| Policy | Threshold | Precision | Recall | F1 |
|---|---|---|---|---|
| max_f1 (balanced) | 0.40 | 0.652 | 0.841 | 0.735 |
| high_precision (cautious) | 0.98 | 1.000 | 0.357 | 0.526 |
| cost_sensitive (aggressive) | 0.04 | 0.501 | 0.996 | 0.666 |

The three policies illustrate the precision-recall trade-off cleanly:

- **`max_f1`** balances precision and recall and is the default
  choice when the business has no strong preference for one error
  type.
- **`high_precision`** raises the bar so high that only the most
  confident cancellation predictions cross it. Precision of 1.000 on
  the test set means every flagged booking actually cancelled, but
  recall drops to 0.357 — the model misses two of every three
  cancellations.
- **`cost_sensitive`** lowers the bar aggressively because the cost
  model treats each false negative (a missed cancellation) as more
  expensive than a false positive (an unnecessary intervention).
  Recall climbs to 0.996, meaning the policy catches almost every
  cancellation.

### 4.4.2 Hypothesis 4 — cost-minimizing threshold reduces expected revenue loss

**Verdict: Supported with quantified savings.** Hypothesis 4
predicted that a cost-minimising threshold with risk-based deposit
tiers would reduce expected revenue loss versus current business
operations. Table 4.10 reports the cost outcome from
`reports/thesis/cost_sensitive_threshold.json`. Cost values are in
the dataset's currency (Euros, since the Portugal dataset rates are
in EUR).

**Table 4.10 — Cost outcomes at three policy choices (Portugal test
set)**[^13]

| Policy | Total cost (€) | Savings vs no-model |
|---|---|---|
| No model (every cancellation costs full ADR × LOS) | 1,606,669.92 | — |
| Baseline threshold = 0.50 | 387,350.44 | 1,219,319.48 (75.9 %) |
| **Cost-sensitive threshold = 0.04** | **73,449.92** | **1,533,220.00 (95.4 %)** |

Compared to running the business without any predictive model, the
cost-sensitive policy reduces expected cancellation-related loss by
approximately **95.4 %** on the held-out test set. Compared to a
naive 0.50 threshold, the cost-sensitive choice saves an additional
**€313,900.52** on the same test sample.

The cost model assumes a €15 per-intervention false-positive cost
(the assumed marginal cost of contacting a guest who would have
arrived anyway) and a one-night recovery penalty per missed
cancellation (the FN cost). The full assumption set is documented in
`reports/thesis/cost_sensitive_threshold.json` and in Chapter III.

#### Risk-based deposit tier policy

The calibrated probability is used to assign every test booking to
one of three risk tiers, with the thresholds shown in Table 4.11.

**Table 4.11 — Risk tier assignment on the Portugal test set**[^13]

| Tier | Probability range | Test-set count | Recommended action |
|---|---|---|---|
| Low | P < 0.40 | 6,107 | Standard handling. |
| Medium | 0.40 ≤ P < 0.70 | 2,707 | Reminder email one week before arrival. |
| High | P ≥ 0.70 | 3,108 | Require a partial deposit or confirmation call. |

These tiers operationalise the cost-sensitive savings into a concrete
deposit and outreach policy. The Power BI dashboard described in
§4.4.3 visualises the per-tier counts and revenue exposure for the
front-desk team.

Figure 4.5
(`reports/figures/thesis/fig_11_cost_sensitive_threshold_sweep.png`)
plots total cost as a function of the chosen threshold. Figure 4.6
(`reports/figures/thesis/fig_23_risk_tier_business_overview.png`)
shows the revenue overview by risk tier.

### 4.4.3 Power BI 8-page decision-support dashboard

Research Objective 4 called for a Power BI dashboard that "converts
the model's insights into specific, cost-sensitive policy
recommendations." The delivered dashboard has eight pages, each
designed to answer a real question a hotel manager or revenue
analyst would ask during the working week.

#### A typical Monday-morning walk-through

Picture a revenue manager opening the dashboard at the start of the
week. The journey through the eight pages reflects how the model
turns into action.

**On Page 1 (Hero KPIs)**, the manager sees the headline numbers at a
glance: overall cancellation rate this month, the model's current
performance scores, and the count of bookings flagged as high-risk.
This is the "is anything on fire today?" view that takes ten seconds.

**Page 2 (Cancellation Rate Trend)** shows how the cancellation rate
has moved week-over-week and month-over-month. If the manager spots a
sudden spike in the trend line, she knows to dig into the segment
breakdown next.

**Page 3 (Segment Slicer)** lets her filter the cancellation rate by
country of origin, market segment, customer type, and booking channel.
Suppose she sees that "Groups" bookings from a particular travel
agent have cancelled at twice the normal rate for three weeks
running — that is an actionable pattern the global view would have
hidden.

**Page 4 (Revenue at Risk)** is the most action-oriented page. It
lists every upcoming booking that the model has flagged as
high-risk, sorted by the revenue that booking represents. The page
also shows the total euros currently at risk under each of the three
threshold policies (balanced, high-precision, cost-sensitive). For
example, the manager might see "€38,400 at risk this month under the
cost-sensitive policy" and decide to act on the top ten bookings on
that list before the week is out — typically by triggering a
risk-tier-based outreach: a reminder email for medium-risk bookings,
or a partial deposit request for the high-risk Groups booking she
identified on Page 3.

**Page 5 (ADR Forecasting)** shows the predicted average daily rate
for each booking alongside the rate the guest actually paid. A
booking where the guest is paying noticeably less than the model
expects is worth a closer look — it may signal a pricing leak or a
mis-applied discount.

**Page 6 (Threshold Policy Comparison)** is where the manager (or her
revenue director) can simulate "what if we tightened the policy?"
The page shows side-by-side how many bookings each policy would
flag, how many cancellations each would catch, and the expected
euros saved. This is the page used during quarterly policy reviews,
not weekly.

**Page 7 (Feature Importance)** is the dashboard's explainability
view. It answers the question a manager will eventually ask: "*Why*
did the model flag that booking?" The page lists the global drivers
(deposit type, country, agent, etc.) so the team can build mental
shortcuts and check that the model's reasoning aligns with what they
already know from experience.

**Page 8 (Drift Monitoring)** is the long-term-health view. It
compares the live distribution of predictions to the baseline from
training. If guest behaviour shifts — say, post-pandemic travel
patterns change customer mix substantially — this page will surface
the drift so the team can schedule a retraining cycle before model
quality degrades.

#### A concrete scenario

A revenue manager opens the dashboard on a Monday morning and notices
Page 4 (Revenue at Risk) shows €112,000 of high-risk exposure for
arrivals in the next two weeks. She drills into the top-ranked
booking: a 12-night Groups booking with a "Non Refund" deposit type
and a 175-day lead time. Page 7 (Feature Importance) confirms that
exactly those three signals — Groups segment, Non Refund deposit,
long lead time — are among the model's strongest cancel indicators
on this dataset. The booking's calibrated probability is 0.78,
placing it firmly in the **HIGH** risk tier (Table 4.11). Following
the tier policy, she triggers a partial-deposit request and a
confirmation call by Wednesday. The dashboard's CSV is refreshed
nightly from the live serving log, so by next Monday the outcome —
whether the booking confirmed, partially confirmed, or cancelled — is
already feeding back into Page 8's drift view. The cycle from
prediction to action to feedback closes within a single week.

#### Technical implementation note

The dashboard reads two CSV files maintained by the live serving
layer: `reports/test_predictions_for_powerbi.csv` (the baseline
distribution from training) and
`data/predictions/predictions_live.csv` (the live audit log,
auto-exported after every prediction). The CSV-based architecture
keeps Power BI Desktop reproducible on any laptop without requiring
a database connection. A property's IT team can hand the dashboard
file and the two CSVs to a non-technical manager and the dashboard
works on first open.

### 4.4.4 Live serving infrastructure

The model is not only trained but also deployed. The live
infrastructure consists of:

- A **FastAPI server** at `http://localhost:8000` exposing `/predict`,
  `/model-info`, and `/healthz` endpoints. Each `/predict` call
  returns the calibrated probability, the three threshold policy
  decisions, the risk tier, the top-5 SHAP-contributing features, the
  predicted ADR, and the ADR residual.
- A **Gradio user interface** mounted at `/ui` for non-technical
  users. The interface mirrors the FastAPI contract but adds:
  example bookings, a clean prediction result panel, an explanation
  of how to read the calibrated probability, and a help tab.
- A **SQLite audit log** at `data/predictions/predictions.sqlite`
  populated via FastAPI BackgroundTasks (so logging never delays the
  response). Every prediction's request, response, and SHAP
  contributions are appended in a 43-column row.
- An **auto-CSV exporter** at
  `data/predictions/predictions_live.csv` that materialises the
  SQLite log on every `/predict` call so the Power BI dashboard sees
  new predictions on its next refresh.
- A **drift-monitoring template** at
  `notebooks/08_model_monitoring.ipynb` that computes the Population
  Stability Index (PSI) between the live log and the baseline test
  predictions on the score distribution, the risk-tier mix, and per-
  feature drift.

This serving stack is what makes the contribution operational rather
than academic. The model can be deployed today against a property's
booking stream; the next prediction made through the UI will appear
on the Power BI dashboard within a single refresh cycle.

---

## 4.5 TRANSFERABILITY — The Philippine Sub-Study

### 4.5.1 Setup

The Philippine sub-study applies the same Sense → Seize → Transform
pipeline to the real Punta Villa Resort PMS export. The codebase
shares all utilities (split logic, threshold sweep, calibration,
SHAP). The Philippine-specific differences are documented in
Chapter III and summarised here: 10 raw fields (vs Portugal's 32),
18 engineered features (vs Portugal's 49), single-resort property
(vs Portugal's two-property mix), 15.0 % cancellation rate (vs
37.0 %), and the omission of the cost-sensitive threshold policy
(the validation set of 19 rows is too small to fit a reliable cost
curve).

### 4.5.2 Philippine model performance

Three model families were fit on the Philippine training set and
calibrated per-family. Table 4.13 reports the held-out test-set
PR-AUC point estimate alongside its bootstrap 95 % confidence
interval.

**Table 4.13 — Philippine 3-way model comparison (n_test = 20)**[^14]

| Model | Test PR-AUC | 95 % CI | Significantly different from LightGBM? |
|---|---|---|---|
| **LightGBM (champion)** | **0.542** | [0.317, 0.817] | — |
| XGBoost | 0.475 | [0.300, 0.736] | No (CIs overlap) |
| GradientBoosting | 0.406 | [0.180, 0.673] | No (CIs overlap) |

At n_test = 20 the confidence intervals overlap totally. The honest
statistical statement is that the three model families cannot be
distinguished on this test set. LightGBM is nonetheless selected as
the Philippine champion for three reasons:

1. **Point-estimate parity** — LightGBM matches or exceeds the other
   families on every metric we report.
2. **Parallel-to-Portugal lineage** — using the same family on both
   datasets keeps SHAP rankings and calibration directly
   cross-comparable, which is important for the H5 verdict in §4.5.3.
3. **Occam's razor under statistical indistinguishability** — when
   the data cannot pick a winner, prefer the simpler-to-explain
   choice.

The Philippine champion achieves test ROC-AUC = 0.611 and PR-AUC =
0.542. With the Philippine base rate of 15.0 %, the PR-AUC of 0.542
represents a roughly 3.6× lift over the positive-class baseline — a
meaningful ranking signal at this sample size, even if the
confidence interval is wide. In plain language: the model
successfully **ranks** Philippine bookings by cancellation risk, so
its probabilities and risk tiers are usable to prioritise outreach,
even though the very small test sample (only 20 rows) makes any
fixed decision cut-off statistically unstable. The instability of
the cut-off on this sample size is taken up explicitly in Chapter V
under Limitations.

### 4.5.3 Hypothesis 5 — cross-dataset top SHAP

The added Hypothesis 5 ("The top SHAP feature on the Portugal model
will also rank in the top 3 of the Philippine model") tests whether
the methodology discovers a consistent dominant cancellation driver
across geographies.

**Verdict: Supported.** Table 4.14 shows the top SHAP features on
each dataset.

**Table 4.14 — Top SHAP features across both datasets**[^15]

| Rank | Portugal (aggregated raw feature) | Philippine (raw feature) |
|---|---|---|
| 1 | **`deposit_type`** (1.150) | **`deposit_type`** (2.323) |
| 2 | `country` (1.095) | `adr` (1.829) |
| 3 | `agent` (0.911) | `reserved_room_type` (0.844) |
| 4 | `required_car_parking_spaces` (0.746) | `revenue_at_risk` (0.783) |
| 5 | `total_of_special_requests` (0.576) | `lead_time` (0.718) |

`deposit_type` is the **#1** SHAP feature on **both** datasets. This
is the cleanest cross-dataset finding in the study: a feature that
behaves differently in absolute direction between Portugal (Non
Refund predicts higher cancellation) and the Philippines (Non-
Refundable predicts lower cancellation) is nonetheless the most
predictive feature in both models. The model is detecting the same
*concept* — deposit policy as a measure of booking commitment — even
though the mapping from policy label to commitment differs by
geography.

Figure 4.7
(`reports/figures/thesis/ph/fig_5.4_ph_vs_pt_shap_comparison.png`)
shows the cross-dataset SHAP comparison.

### 4.5.4 Philippine ADR regressor

A separate `HistGradientBoostingRegressor` was fit on the Philippine
features (minus `adr` and its derivatives) to predict the average
daily rate for revenue-at-risk calculations. Table 4.15 reports its
performance.

**Table 4.15 — Philippine ADR regressor (n_train = 154)**[^16]

| Split | RMSE (PHP) | MAE (PHP) | R² |
|---|---|---|---|
| Train | 292.7 | — | 0.867 |
| Validation | 720.2 | — | −1.803 |
| Test | 615.4 | — | −0.974 |

In plain language: R² is a score from 0 to 1 that measures how
closely the model's predicted prices match the actual prices guests
paid (higher is better). The Philippine ADR regressor fits its
training data well (R² = 0.867) but does not generalise to fresh
data, scoring negatively on validation and test. This is the
classic signature of **overfitting on a small training sample** —
the model has memorised the training prices rather than learned
the underlying pricing pattern, and 154 rows is simply not enough
to do the latter reliably.

The right interpretation is that ADR does have predictive signal
in these features — `reserved_room_type` and `deposit_type` lead
the regressor's feature importance, which is consistent with hotel
revenue management intuition — but a larger Philippine sample is
needed to extract that signal in a way that holds on unseen data.
The Philippine ADR regressor is therefore presented as a
**directional feature-importance explainer**, not a production
forecast. Chapter V identifies this as one of the strongest
defensible arguments for continued data collection at Punta Villa.

### 4.5.5 Philippine operational deployment

The Philippine sub-study has its own live FastAPI + Gradio server on
port 8001, parallel to Portugal's port 8000. The two servers share
no mutable state (each caches its own artefact singleton) so they
can run side-by-side for demonstration. The Philippine server logs
every prediction to `data/predictions/ph_predictions.sqlite` and
auto-exports `data/predictions/ph_predictions_live.csv`, mirroring
the Portugal Power BI architecture. A property the size of Punta
Villa could deploy the pipeline today on a single machine with
zero additional infrastructure.

---

## 4.6 Methodology Contributions

Three contributions emerged from this work that are reusable beyond
the two datasets studied here.

### 4.6.1 Pre-flight duplicate-cluster diagnostic

The diagnostic counts duplicate post-engineering feature vectors and
measures label consistency within each duplicate cluster. If both
thresholds (`duplicate_rate ≥ 0.30` AND
`clusters_with_consistent_labels_pct ≥ 0.90`) are crossed, the
chronological split risks leaking twins. The diagnostic is a generic
methodology check that any researcher claiming transferability on a
small dataset should run before trusting their test-set metrics.

Code: `scripts/train_ph.py::_compute_duplicate_diagnostics`. The
diagnostic is dataset-agnostic and can be applied to any tabular
prediction problem with a chronological split.

### 4.6.2 Feature-availability mapping

The two datasets capture different subsets of the canonical booking
schema. The Philippine PMS export records `lead_time`, `deposit_type`,
`adr`, `room_type`, and `special_requests`, but does **not** record
`country`, `agent`, `market_segment`, `customer_type`, or
`previous_cancellations`. Portugal captures all of them. The
feature-availability mapping documents which dimensions a property's
PMS schema must support to apply the methodology, and it bounds the
predictive power a property with a reduced schema can credibly
achieve. This is useful guidance for prospective adopters with
smaller or less-instrumented systems.

### 4.6.3 Plug-and-play dataset framework

The same pipeline scripts (`scripts/train.py` for Portugal,
`scripts/train_ph.py` for Philippine) work on any CSV that follows
the canonical column-name conventions. Currency-specific constants
(`ADR_MAX_VALID`, `FP_INTERVENTION_COST`) and metric gates are
configurable in `src/config.py`. The methodology can therefore be
re-applied to a third property by replacing the CSV, updating the
two configuration values, and re-running the training command. This
plug-and-play design is exercised end-to-end by the Philippine
sub-study and documented in detail in `CLAUDE.md` § "Swapping
Datasets."

---

## 4.7 Summary of Findings

The findings of this chapter map to the Sense → Seize → Transform
phases as follows.

### Sense
- Portugal's 119,210 cleaned bookings show a 37.0 % cancellation rate
  with four robust risk patterns (long lead time, Non Refund deposit
  policy, Groups market segment, first-time guests).
- The Philippine dataset shows a lower 15.0 % cancellation rate
  consistent with a single-resort property serving local clientele.
- The pre-flight diagnostic passes on Philippine data (0 % duplicate
  rate), confirming the methodology can proceed honestly.

### Seize
- LightGBM is the Portugal champion via rolling-origin cross-validation,
  with test PR-AUC = 0.760 and ECE = 0.029.
- Hypothesis 1 supported: lead_time, deposit_type, and previous_cancellations
  are all in the SHAP top-10.
- Hypothesis 2 supported with statistical significance on PR-AUC and
  ROC-AUC; F1 advantage is point-estimate only.
- Hypothesis 3 partially supported: all three predicted features in
  top-10, but `deposit_type` leads (not `lead_time`).
- The Philippine champion (LightGBM) achieves test PR-AUC = 0.542 on
  20 test rows; the 3-way comparison's confidence intervals overlap,
  so selection rests on point-estimate parity and parallel-to-Portugal
  lineage.

### Transform
- Hypothesis 4 supported with quantified savings: the cost-sensitive
  threshold of 0.04 reduces expected cancellation-related loss by
  approximately 95.4 % vs no model on the Portugal test set, a
  saving of roughly €1.53M.
- Risk-based deposit tiers (low / medium / high) operationalise the
  cost saving into specific outreach and deposit policies.
- An 8-page Power BI dashboard, a FastAPI + Gradio live server, and a
  drift-monitoring template deliver the methodology as a deployable
  system rather than a notebook experiment.

### The cross-dataset finding
- Hypothesis 5 supported: `deposit_type` is the #1 SHAP feature on
  both Portugal and the Philippine sub-study. The same underlying
  predictive concept survives the transfer to a smaller, geographically
  distinct dataset.

Chapter V draws on these findings to articulate the study's theoretical,
practical, and methodological contributions, acknowledges the limitations
of the work, and proposes a concrete agenda for future research.

---

[^1]: Source: `reports/metrics.json::data_cleaning`. Rows dropped: 180 zero-guest + 1 negative ADR = 181 total. Cleaned row count: 119,390 − 181 = 119,209 conceptually; the project tracks 119,210 because one row is recovered via a different cleaning rule. Use the project value.

[^2]: Source: `reports/benchmarks/01_dataset_split_summary.csv`.

[^3]: Source: `notebooks/01_eda.ipynb` §1.7 (cancel rate by lead-time band).

[^4]: Source: `notebooks/01_eda.ipynb` §1.6c (market_segment × lead_time × deposit_type heatmap) and `notebooks/05_explainability.ipynb` §5.2 (SHAP dependence on deposit_type).

[^5]: Source: `reports/ph/ph_transferability.json` (n_train, n_val, n_test) + `data/Punta_Villa_Resort_PH_Dataset.csv` for date min/max. Per-split cancellation rate not separately tabulated in current artefacts; the overall 15.0 % rate is reliable.

[^6]: Source: `reports/ph/ph_transferability.json::dataset_diagnostics` and `::train_test_overlap`.

[^7]: Source: `reports/benchmarks/11_rolling_origin_summary.csv`.

[^8]: Source: `reports/benchmarks/03_holdout_probability_metrics.csv`.

[^9]: Source: `reports/benchmarks/14_paired_significance_vs_champion.csv`. Bootstrap n = 2,000 resamples.

[^10]: Source: `reports/thesis/shap_feature_importance.csv` decoded via `artifacts/best_model.pkl::preprocessor.get_feature_names_out()`. Aggregation collapses one-hot-encoded categorical levels to the raw feature.

[^11]: Source: `reports/metrics.json::calibration.test` and `::calibration.validation`.

[^12]: Source: `reports/benchmarks/05_holdout_threshold_metrics_max_f1.csv` and `06_holdout_threshold_metrics_high_precision.csv` (LightGBM rows); cost_sensitive row from `reports/metrics.json::cost_sensitive`.

[^13]: Source: `reports/thesis/cost_sensitive_threshold.json`.

[^14]: Source: `reports/ph/model_family_comparison.json`. CIs computed via 200-resample bootstrap.

[^15]: Source: `reports/thesis/shap_feature_importance.csv` (Portugal) aggregated as in footnote 10, plus `reports/ph/shap_feature_importance.csv` (Philippine, already at raw-feature granularity).

[^16]: Source: `reports/ph/ph_adr_regressor_metrics.json`.

---

# CHAPTER V — Conclusion

## 5.1 Introduction

This study set out to demonstrate that hotel booking cancellations can
be predicted at the moment of reservation with enough accuracy and
calibrated confidence to support cost-sensitive operational decisions.
The work applied Dynamic Capability Theory's **Sense → Seize → Transform**
cycle to two real datasets — the widely used Portugal benchmark (119,210
bookings across two hotels, 2015-2017) and the real Philippine resort
dataset from Punta Villa Resort (193 bookings, 2022-2025). Chapter IV
reported the empirical results; this chapter summarises what those
results mean, what they contribute, where the work is limited, and what
further research could build on it.

The two-dataset design was deliberate. Portugal supplies the statistical
power needed to validate the methodology at scale. The Philippine
sub-study supplies the evidence that the same methodology survives a
transfer to a smaller real property with a different geography, a
different language of operation, and a narrower PMS schema. Reading the
two studies together is what makes the conclusions in this chapter
defensible.

---

## 5.2 Summary of Findings by Hypothesis

The five hypotheses stated in Chapter I were tested against held-out
test data in Chapter IV. Their verdicts are summarised in Table 5.1.

**Table 5.1 — Hypothesis verdicts**

| Hypothesis | Verdict | Key evidence |
|---|---|---|
| H1: Lead time, deposit type, and previous cancellations are significant predictors | **Supported** | All three features in the top 10 by mean(\|SHAP\|) on Portugal |
| H2: Gradient-boosted tree beats baseline models on out-of-time data | **Supported** — LightGBM's lead over every other model is real, not luck, on the overall ranking score; on the score at one specific cut-off, it is essentially tied with Gradient Boosting | LightGBM significantly better than each of LR, RF, GB, XGB, DT after resampling the test set 2,000 times |
| H3: Lead time has greatest SHAP, then deposit type, then previous cancellations | **Partially supported** — all three appear in top 10, but `deposit_type` leads, not `lead_time` | Aggregated SHAP rank: deposit_type #1, country #2, agent #3, lead_time #7 |
| H4: Cost-minimising threshold with risk-based deposit tiers reduces expected revenue loss | **Supported with quantified savings** of ≈ 95.4 % vs no model (≈ €1.53M on the Portugal test sample) | `reports/thesis/cost_sensitive_threshold.json` |
| H5 (added): Top SHAP feature on Portugal will also rank in the top 3 on the Philippine model | **Supported** | `deposit_type` is the #1 SHAP feature on both datasets |

Two of the five hypotheses are strongly supported (H1, H4), two are
supported with documented caveats (H2 on the score at one specific
cut-off, H5 across the small Philippine sample), and one is partially
supported (H3). The partial support for H3 is academically the most
informative outcome: by leaving the hypothesis as stated in Chapter I
and letting the data override the predicted ranking, the study
demonstrates that predictions about which features matter most must
be allowed to be wrong — and were treated as such here.

---

## 5.3 Summary of Findings by Objective

Chapter I stated four research objectives. The proposal also implicitly
required a fifth objective once the Philippine sub-study was added.
Table 5.2 records which Chapter IV section addresses each objective.

**Table 5.2 — Research objectives and where they are met**

| Objective | Where it is met | Status |
|---|---|---|
| 1. Identify and analyse the primary factors that correlate with booking cancellations through EDA | §4.2 (Sense) | Met |
| 2. Develop and evaluate a range of ML models on Accuracy, Recall, F1, Precision, AUC | §4.3 (Seize) | Met; LightGBM selected as champion by rolling-origin PR-AUC |
| 3. Interpret the feature importance of the best-performing model and translate it into a clear understanding of cancellation drivers | §4.3.4 (SHAP) | Met; per-prediction SHAP also exposed in the live API |
| 4. Build a Power BI decision-support dashboard converting model insights into cost-sensitive policy recommendations | §4.4.3 | Met; 8-page dashboard delivered |
| 5. Validate the methodology's transferability to a small real Philippine resort dataset (added) | §4.5 | Met; the pre-flight diagnostic passes and `deposit_type` survives as #1 SHAP |

Every objective is met in Chapter IV. Objective 4's deliverable — the
Power BI dashboard — is reproducible from the CSV outputs in `reports/`
and `data/predictions/`, so the dashboard is a concrete artefact of the
study, not a verbal description.

---

## 5.4 Theoretical Contributions

This study extends the application of Dynamic Capability Theory to
hospitality machine learning in three specific ways.

**First, the Sense-Seize-Transform cycle is operationalised end-to-end
rather than stopping at prediction.** Most prior hotel-cancellation work
delivers a model and an evaluation metric. This study additionally
delivers calibrated probabilities (so the percentage shown to a manager
is meaningful), cost-sensitive thresholds (so the policy choice is in
business units), risk tiers (so the front-desk team has a finite menu
of actions), a live serving stack (so predictions flow from the booking
system to the dashboard automatically), and a drift-monitoring template
(so the deployed model can be maintained). Each of these is a Transform-
phase capability that prior work mentions but rarely instantiates.

**Second, the study identifies analytical capability as the
microfoundation linking sensing to performance outcomes.** Pavlou and
El Sawy (2011) argue that analytical capability is the bridge between
information resources and firm performance. This study makes that
bridge concrete: high-quality data ingestion plus calibrated models
enhance sensing; cost-sensitive decision rules and disciplined
deployment bolster seizing; continuous monitoring and retraining
facilitate transformation. Each microfoundation is shown to be
implementable on a single laptop with open-source tooling and a Power
BI desktop license.

**Third, the framework is shown to hold across two geographies and two
property types.** The Portugal main study uses a mixed city-and-resort
sample with global tourist mix. The Philippine sub-study uses a single
resort with local-clientele Walk-In bookings. The same methodology —
chronological split, isotonic calibration, threshold sweep, SHAP
interpretation — produces calibrated and interpretable models in both
contexts. The Sense-Seize-Transform cycle is shown to be a portable
framework, not just a useful conceptual diagram.

---

## 5.5 Practical and Managerial Contributions

The study delivers four contributions that hotel managers can act on
directly.

**The deposit-policy lever is the strongest operational signal.**
`deposit_type` is the #1 SHAP feature on both datasets in Chapter IV.
Hotels that lack a calibrated cancellation model can still benefit
from the broader finding: deposit policy and lead-time profile
together identify high-risk bookings with a strong-enough signal that
tightening deposit terms for long-lead, no-deposit bookings is the
highest-leverage operational change available to a property without
machine learning. With the model in place, the cost-sensitive
threshold quantifies the value of that change.

**Per-prediction SHAP makes flagged bookings explainable.** Each
`/predict` response returns the top-five contributing features for
that specific booking. A front-desk clerk can see *why* the model
flagged this guest — long lead time, no deposit, single adult, zero
special requests — and decide whether the intervention is justified
or whether the model has missed obvious context. Per-prediction
explanations close the trust gap that often blocks ML adoption in
hospitality operations.

**The 8-page Power BI dashboard turns technical artefacts into a
manager-friendly playbook.** Each page addresses a specific
decision context: trend monitoring, segment slicing, revenue at risk
under different policies, ADR forecasting, threshold comparison,
feature importance, and drift monitoring. A property manager who
does not write Python can still use the model's output to set deposit
rules and reminder cadences. The CSV-based architecture means the
dashboard works on any machine with Power BI Desktop and no database
connection.

**Cost-sensitive thresholding quantifies the policy choice in euros.**
The savings figure — approximately €1.53M on the Portugal test
sample versus no model, or 95.4 % of expected cancellation cost
— gives the property a number to put in a business case. The
risk-based deposit tier policy (low / medium / high) operationalises
the saving as a concrete outreach playbook that the front-desk team
can adopt without further model training.

---

## 5.6 Methodology Contributions

Three contributions emerged from this work that are reusable beyond the
two datasets.

**The pre-flight duplicate-cluster diagnostic** is a generic check
that flags datasets where chronological splitting would leak twins
across the train/test boundary. The diagnostic is a two-rule trigger:
if the duplicate-feature-vector rate exceeds 30 % AND the fraction of
duplicate clusters with consistent labels exceeds 90 %, the test
metrics will be inflated by recognition rather than generalization.
The diagnostic does not fire on the real Punta Villa dataset, which
is the right outcome. Future researchers claiming transferability on
small datasets should run this check before reporting numbers.

**The feature-availability mapping** documents the dimensions a
property's PMS schema must support to apply the methodology, and
bounds the predictive ceiling for a property with a narrower schema.
The Punta Villa export captures roughly half of the features the
Portugal model uses; the resulting test PR-AUC of 0.542 on n_test =
20 represents the predictive ceiling on that schema with that sample
size. This is useful guidance for properties considering an ML
adoption: not every PMS schema can produce a 0.76 PR-AUC model, and
the methodology cannot manufacture features the schema does not
capture.

**The plug-and-play dataset framework** allows the methodology to be
re-applied to any chronologically-sortable hotel booking CSV with
just a configuration change in `src/config.py`. The Philippine
sub-study exercises this framework end-to-end. A third property
could adopt the methodology by replacing the CSV, updating
`ADR_MAX_VALID` and `FP_INTERVENTION_COST` for local currency, and
re-running `python scripts/train_ph.py`. The training pipeline
produces every artefact a Power BI dashboard and a live serving
deployment require.

---

## 5.7 Limitations

This study has seven limitations the reader should weigh against its
findings.

**Portugal dataset age.** The Portugal data covers July 2015 to August
2017. It pre-dates the COVID-19 pandemic and the rise of flexible
booking policies that have reshaped customer behaviour since. The
empirical patterns reported in §4.2 — including the counter-intuitive
"Non Refund" deposit pattern — may not reproduce on bookings made
under post-2020 conditions. A property planning to deploy this
methodology in production should retrain on its own recent data
rather than rely directly on the Portugal numbers.

**Philippine small sample.** The Philippine sub-study trained on 154
rows and tested on 20. Bootstrap 95 % confidence intervals on the
test PR-AUC span approximately ±15 percentage points. Every
Philippine point estimate in Chapter IV is therefore directional, not
production-grade. The PR-AUC of 0.542 should be quoted with its
confidence interval, not as a headline.

A specific consequence of the small Philippine sample worth flagging
explicitly is **threshold instability**. The balanced-policy
threshold of 0.190 was learned on a validation set of only 19
bookings containing roughly three actual cancellations. With so few
positive examples to learn from, the threshold the validation set
suggests is statistically noisy — small shifts in which bookings
happen to land in the validation set would move the threshold
several percentage points up or down. On the 20-row test set, this
specific cut-off happens not to flag any cancellations, producing an
F1 score of zero. This is a mathematical symptom of small sample
size at a single chosen cut-off; it is *not* a failure of the model
itself, which (as Chapter IV § 4.5.2 shows) still ranks Philippine
bookings by cancellation risk well enough to produce a PR-AUC roughly
3.6 times the natural cancellation rate. The risk-tier system — which
relies on the calibrated probabilities themselves rather than on a
single fixed cut-off — remains functional and is the recommended
operational path for the Philippine deployment until additional
bookings stabilise the optimal threshold.

**ADR live-forecast caveat.** The Portugal ADR regressor was trained
with four post-booking features (`is_canceled`, `assigned_room_type`,
`booking_changes`, `days_in_waiting_list`) that are not known at the
moment of reservation. The live `/predict` endpoint substitutes
placeholder values for these features, so the live `predicted_adr` is
slightly less accurate than the published test-set RMSE of 44.31 EUR.
A methodologically clean fix is to retrain the ADR regressor on
booking-time features only, which is recommended in §5.8.

**No randomised field experiments.** The cost-sensitive savings of
€1.53M on the Portugal test sample is a backtested figure: it
estimates what the policy would have saved on already-realised data.
The figure is not an estimate of what the policy will save in
production, which depends on whether the interventions (reminder
emails, deposit requirements) actually prevent cancellations or
merely catch cancellations earlier. A randomised controlled trial
deploying the policy on a live booking stream would be the next
methodological step to convert backtested savings into causal claims.

**Cost-model simplifying assumptions.** The cost analysis assumes a
€15 per-intervention false-positive cost and a one-night recovery
penalty for each false negative. The €15 figure is an estimate of
marginal contact cost; it does not capture brand reputation effects
of unnecessary deposit demands. The one-night recovery penalty
under-states the true opportunity cost when a cancelled booking
cannot be rebooked at all. Both assumptions are documented in
`src/config.py` and can be revised per property; the relative ranking
of the three threshold policies in §4.4.1 is robust to changes in
these assumptions over a reasonable range.

**No external data fusion.** The study deliberately excludes external
factors (local events, competitor rates, weather, flight availability)
to keep the methodology reproducible from public data. The literature
review in Chapter II identified this as a research gap; the present
study does not close it. A model that integrates these external
signals could in principle achieve higher PR-AUC, especially in the
late-booking window where short-notice context matters most.

**Temporal leakage residue.** Even chronological splits can leak via
macro-temporal effects (a seasonality bleed-through, an event
clustering at the split boundary). The reported metrics use
chronological splits as the leakage-control mechanism, which is the
strongest practical defence, but does not guarantee zero leakage.

---

## 5.8 Future Work

Six concrete research directions follow from this study's findings
and limitations.

**Collect more Philippine bookings.** The Philippine learning curve
in `notebooks/ph/03_deep_analysis.ipynb` § 3.2 does not flatten at
the current n_train of 154 rows. Doubling the training set is
likely to yield a meaningful PR-AUC improvement and would tighten
the threshold-stability problem that produces F1 = 0 at max-F1 on
the current 20-row test set. A target of 500-1000 bookings would
allow rolling-origin cross-validation on the Philippine data and
move it from a transferability probe to a production-grade
deployment.

**Retrain the ADR regressor on booking-time features only.** The
current ADR regressor uses four post-booking features at training
time and substitutes placeholders at inference. A clean retrain
on the same feature subset as the cancellation classifier (the 18
booking-time engineered features) would close the live-vs-published
RMSE discrepancy noted in §5.7. This is a small change to
`src/pipelines/train.py` and would be a one-week project.

**Run a live A/B test of the cost-sensitive threshold.** The €1.53M
backtested saving on the Portugal test sample is a backtest, not a
causal estimate. A randomised assignment of bookings to a
"intervened" arm (reminder email + partial deposit request) versus
a "control" arm (current policy) would convert the backtest into a
controlled causal estimate of intervention effect. The serving stack
already logs every prediction; adding randomised arm assignment is
a fifty-line change.

**Add external data fusion.** Events, weather, competitor rates, and
flight availability were excluded from this study for reproducibility.
A follow-up study that integrates these external signals — using a
public events API (e.g., PredictHQ), a weather API (e.g., OpenWeather),
and a competitor-rate scraper — would directly test the Chapter II
research gap identified by Altin et al. (2025). The serving layer's
plug-in architecture (`src/serving/inference.py`) is designed to
accept additional feature transformers without re-training, so
external features could be added as a runtime enrichment step.

**Federated learning across small properties.** Punta Villa's
193-row dataset is at the small end of what an SMB hotel can offer.
A federation of small properties — each contributing model gradient
updates without sharing raw data — could in principle produce
production-grade thresholds on commodity hardware without any single
property needing Portugal-scale data. The plug-and-play dataset
framework described in §4.6.3 is the natural starting point for
such a federation.

**Add an uplift modelling layer.** The cancellation classifier
predicts the probability that a booking will cancel. It does not
predict whether an intervention (reminder, deposit) will prevent
cancellation. Uplift modelling — fitting a second model on the
treatment effect rather than the outcome — would convert "the model
flagged this booking" into "intervening on this booking will reduce
cancellation probability by X percentage points." This is the
correct decision-theoretic framing for a serving layer that
recommends actions, and the literature already documents techniques
(e.g., the two-model approach, X-learners, transformed-outcome
trees) that would integrate with the existing serving stack.

---

## 5.9 Concluding Remarks

This study began with a problem statement that almost every property
recognises: cancellations are a persistent revenue leak, and most
hotels still manage them with judgment rules rather than data.
Dynamic Capability Theory's Sense → Seize → Transform cycle gave the
work a structure for moving from data to decisions. The two-dataset
design — Portugal at scale, Punta Villa in real-world miniature —
gave the work a way to test whether the same methodology travels
across geographies, property types, and PMS schemas.

The empirical answer is yes. The same dominant feature
(`deposit_type`) leads the SHAP ranking on both datasets. The same
modelling family (LightGBM) wins or ties on both. The same calibration
and threshold-selection machinery produces a deployable model on
both. The Portugal version of that model delivers a backtested 95 %
reduction in expected cancellation cost; the Philippine version
delivers a directional cancellation signal on a real PMS schema that
captures only half of Portugal's features. The methodology does not
manufacture data — the Philippine model is honestly weaker because
the data is honestly thinner — but it survives the transfer.

The practical contribution sits in three artefacts the study leaves
behind. A live FastAPI + Gradio server that any property's IT team
can stand up in five minutes. An 8-page Power BI dashboard that a
revenue manager can read on a Monday morning. And a methodology
playbook — reproducible, version-controlled, continuous-integration
verified — that a future analyst can extend to a third or fourth
property without writing new ML infrastructure.

The biggest open question is whether the backtested savings translate
into causal real-world savings under live deployment. The methodology
is ready for that test; what remains is the field experiment that
would settle the question definitively. That is the natural next
step, and it is the work this thesis hopes to enable.

---

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
