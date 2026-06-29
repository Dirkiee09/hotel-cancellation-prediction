# Mathematical Foundations of the Modelling Methodology

*Draft section for the Methodology chapter. Every equation below is matched to the
study's actual implementation (`src/models/`, `src/utils/thresholds.py`,
`src/models/metrics.py`) and to the hyper-parameters recorded in `src/config.py`,
so the formal description and the code are one and the same. Notation is fixed once
in §1 and reused throughout.*

---

## 1. Problem Formalisation and Notation

Each booking is a feature vector $\mathbf{x}_i \in \mathbb{R}^d$ (after one-hot
encoding the $d$ engineered booking-time features) with a binary label
$y_i \in \{0, 1\}$, where $y_i = 1$ denotes a cancelled booking. The dataset is
$\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{n}$ with $n \approx 119{,}000$ and a
positive (cancellation) rate $\pi = \tfrac{1}{n}\sum_i y_i \approx 0.37$.

Every model learns a scoring function $f$ that is converted to a **calibrated
probability** $\hat{p}_i = P(y_i = 1 \mid \mathbf{x}_i)$, and only then to a hard
decision $\hat{y}_i = \mathbb{1}[\hat{p}_i \ge \tau]$ using a decision threshold
$\tau$ chosen by an explicit business-cost rule (§7). This two-stage design —
*estimate a probability, then choose a threshold* — is why probability quality
(calibration), not just ranking quality, matters here.

| Symbol | Meaning |
|---|---|
| $\mathbf{x}_i,\ y_i$ | feature vector and label of booking $i$ |
| $\hat{p}_i$ | predicted (calibrated) probability of cancellation |
| $\tau$ | decision threshold |
| $\sigma(z) = \dfrac{1}{1+e^{-z}}$ | logistic (sigmoid) link |
| $\mathcal{L}$ | loss function being minimised |
| $\eta$ | learning rate (shrinkage) |

The chronological 80/10/10 split (train / validation / test) means all sums above
are taken over time-ordered subsets; no future booking informs a past prediction.

---

## 2. The Loss Function Shared by the Probabilistic Models

All probabilistic classifiers in this study (logistic regression and every
gradient-boosted model) minimise the **binary cross-entropy**, also called
**log-loss**:

$$
\mathcal{L}_{\log} = -\frac{1}{n}\sum_{i=1}^{n}
\Big[\, y_i \ln \hat{p}_i + (1 - y_i)\ln(1 - \hat{p}_i) \,\Big].
$$

This is not an arbitrary choice: minimising log-loss is equivalent to **maximum-
likelihood estimation** under a Bernoulli model for $y_i$, so the resulting scores
are interpretable as probabilities. This is the single objective that ties the
linear baseline and the tree ensembles together — they differ only in the *form of
$\hat{p}_i$* and in *how* the minimisation is carried out.

---

## 3. Baseline Ladder

The baselines form a *complexity ladder*: each rung adds one modelling assumption,
so the performance gained at each step quantifies the value of that assumption.

### 3.1 Majority-Class (Dummy) Classifier

The naive floor. It ignores $\mathbf{x}$ entirely and predicts the training prior:

$$
\hat{p}_i = \pi \quad\text{for all } i, \qquad
\hat{y}_i = \arg\max_{c \in \{0,1\}} \sum_{j} \mathbb{1}[y_j = c].
$$

Implemented as `DummyClassifier(strategy="most_frequent")`. Its only role is to
establish the score any useful model must beat.

### 3.2 Logistic Regression (Linear Baseline)

A linear-in-the-features log-odds model:

$$
\hat{p}_i = \sigma\!\left(\mathbf{w}^\top \mathbf{x}_i + b\right)
          = \frac{1}{1 + \exp\!\big(-(\mathbf{w}^\top \mathbf{x}_i + b)\big)}.
$$

The weights $(\mathbf{w}, b)$ are found by minimising $\mathcal{L}_{\log}$ (§2) with
L2 regularisation, solved by the `lbfgs` quasi-Newton optimiser (`max_iter=2000`).
Because the decision boundary is a hyperplane, this model captures only *additive,
monotone* effects — it cannot represent feature interactions. It is the most
important baseline in this study: the champion's advantage over it is **statistically
tested** (§8), and that advantage turns out to be small (p = 0.177), which is why the
study's contribution is framed as the *system*, not the algorithm.

### 3.3 Decision Tree (Non-linear, Interpretable Baseline)

A single tree recursively partitions the feature space. At each node it picks the
feature $j$ and split point $s$ that most reduce the **Gini impurity**. For a node
with class proportions $p_0, p_1$:

$$
G = 1 - \sum_{c \in \{0,1\}} p_c^{\,2} = 2\,p_0 p_1 .
$$

A candidate split into left/right children $L, R$ is scored by the impurity
*decrease*

$$
\Delta G = G_{\text{parent}}
         - \frac{n_L}{n}\,G_L - \frac{n_R}{n}\,G_R ,
$$

and the split maximising $\Delta G$ is chosen greedily. The tree is deliberately
shallow (`max_depth=5`, `min_samples_leaf=50`) so it can be drawn in full in the
appendix. Class imbalance is handled with `class_weight="balanced"`, which reweights
each class $c$ by $w_c = \tfrac{n}{2\,n_c}$, so the rarer cancellation class
contributes proportionally more to the impurity calculation.

### 3.4 Gaussian Naïve Bayes (Probabilistic, Zero-Interaction Baseline)

Applies Bayes' theorem under the (deliberately wrong) assumption that features are
**conditionally independent** given the class:

$$
P(y = 1 \mid \mathbf{x}) \;\propto\;
P(y = 1) \prod_{j=1}^{d} P(x_j \mid y = 1),
$$

with each continuous feature modelled as Gaussian within a class,
$P(x_j \mid y=c) = \mathcal{N}(x_j;\, \mu_{jc}, \sigma_{jc}^2)$. The independence
assumption is false for structured booking data (e.g. `lead_time` and
`deposit_type` interact), so the gap between this model and the tree ensembles is a
direct, quantified measure of how much **feature interaction** matters for the
problem.

---

## 4. Random Forest — Bagging Ensemble

A Random Forest is an average of $T$ de-correlated decision trees. Two randomisation
devices break the correlation between trees:

1. **Bootstrap aggregation (bagging):** tree $t$ is grown on a bootstrap resample
   $\mathcal{D}_t$ drawn with replacement from $\mathcal{D}$.
2. **Feature subsampling:** at each split only a random subset of features is
   considered.

The ensemble probability is the average of the per-tree leaf frequencies:

$$
\hat{p}_i = \frac{1}{T}\sum_{t=1}^{T} h_t(\mathbf{x}_i),
$$

where $h_t(\mathbf{x}_i)$ is the cancellation frequency in the leaf of tree $t$ that
$\mathbf{x}_i$ falls into. Averaging reduces variance roughly by a factor of $T$ for
the independent component of the error, which is why a forest of high-variance deep
trees generalises far better than any single tree. In this study Random Forest is the
**bagged-ensemble contrast** to the boosted models (it appears in the chronological
test, Tables 4.2/4.3) — bagging reduces variance, whereas boosting (§5) reduces bias.

---

## 5. Gradient Boosting — The Champion Family

Gradient boosting builds an **additive model** as a sum of $M$ regression trees,
fitting each new tree to the errors of the current ensemble. This is the lineage that
produces the study's champion, so it is developed in full.

### 5.1 The Additive Model and Functional Gradient Descent

The model after $m$ rounds is

$$
F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta\, h_m(\mathbf{x}),
$$

where $h_m$ is the $m$-th regression tree, $\eta$ is the learning rate (shrinkage),
and the raw score $F_M(\mathbf{x})$ is mapped to a probability by the logistic link,
$\hat{p} = \sigma(F_M(\mathbf{x}))$.

Rather than fit $h_m$ to the raw residuals, gradient boosting fits it to the
**negative gradient** of the loss with respect to the current scores — i.e. it
performs gradient descent in *function space*. For log-loss the gradient (the
"pseudo-residual") for observation $i$ at round $m$ is elegantly simple:

$$
r_{im} = -\left[\frac{\partial \mathcal{L}_{\log}}{\partial F(\mathbf{x}_i)}\right]_{F = F_{m-1}}
       = y_i - \sigma\!\big(F_{m-1}(\mathbf{x}_i)\big)
       = y_i - \hat{p}_i^{(m-1)} .
$$

So each tree is trained to predict *how wrong the ensemble currently is* — the signed
probability error. The shrinkage $\eta$ (set to $0.1$ for the sklearn
`GradientBoostingClassifier` reference, `n_estimators=100`, `max_depth=5`) damps each
step to prevent overfitting; smaller $\eta$ needs more trees but generalises better.

### 5.2 XGBoost — Second-Order Boosting with Explicit Regularisation

XGBoost sharpens the recipe above with two mathematical changes that an examiner will
expect you to name precisely.

**(a) Second-order Taylor expansion of the loss.** Instead of using only the gradient,
XGBoost approximates the loss after adding tree $h_m$ with a second-order Taylor
expansion, using both the gradient $g_i$ and the Hessian $\hbar_i$:

$$
\mathcal{L}^{(m)} \approx \sum_{i=1}^{n}
\Big[\, g_i\, h_m(\mathbf{x}_i) + \tfrac{1}{2}\,\hbar_i\, h_m(\mathbf{x}_i)^2 \,\Big]
+ \Omega(h_m),
\qquad
g_i = \frac{\partial \mathcal{L}}{\partial F}, \quad
\hbar_i = \frac{\partial^2 \mathcal{L}}{\partial F^2}.
$$

For log-loss, $g_i = \hat{p}_i - y_i$ and $\hbar_i = \hat{p}_i(1 - \hat{p}_i)$ — the
curvature term tells the model how confident it already is, letting it take better-
scaled steps than first-order boosting.

**(b) An explicit complexity penalty.** XGBoost regularises the tree directly. For a
tree with $T$ leaves and leaf weights $w_j$:

$$
\Omega(h_m) = \gamma\, T + \tfrac{1}{2}\lambda \sum_{j=1}^{T} w_j^{2}.
$$

Substituting the optimal leaf weight $w_j^\star$ that minimises the regularised
objective gives a closed form,

$$
w_j^\star = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} \hbar_i + \lambda},
$$

where $I_j$ is the set of observations in leaf $j$. The corresponding **gain** of a
candidate split (used in place of Gini for boosted trees) is

$$
\text{Gain} = \tfrac{1}{2}\!\left[
\frac{(\sum_{i\in I_L} g_i)^2}{\sum_{i\in I_L}\hbar_i + \lambda}
+ \frac{(\sum_{i\in I_R} g_i)^2}{\sum_{i\in I_R}\hbar_i + \lambda}
- \frac{(\sum_{i\in I} g_i)^2}{\sum_{i\in I}\hbar_i + \lambda}
\right] - \gamma .
$$

A split is kept only if its gain exceeds the per-leaf penalty $\gamma$ — regularisation
is built into the structure search itself. Configured at `n_estimators=300`,
`max_depth=7`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`.

### 5.3 LightGBM — The Selected Champion

LightGBM keeps the second-order objective of §5.2 but changes **how splits are found**
and **how the tree grows**, which is what makes it the efficiency winner that this
study selects. Three mechanisms:

**(a) Histogram-based split finding.** Continuous features are bucketed into a fixed
number of discrete bins (default 255). Gains are then evaluated over
$\mathcal{O}(\#\text{bins})$ candidate splits instead of $\mathcal{O}(\#\text{unique
values})$, reducing split-search cost from $\mathcal{O}(n \cdot d)$ toward
$\mathcal{O}(\#\text{bins}\cdot d)$ and giving the ~3× training-speed advantage over
XGBoost reported in the results.

**(b) Leaf-wise (best-first) growth.** Where most implementations grow level-by-level,
LightGBM splits the single leaf with the largest gain:

$$
\text{leaf}^\star = \arg\max_{\text{leaf}} \ \text{Gain}(\text{leaf}),
$$

producing deeper, lower-loss trees for the same leaf count — controlled here by
`max_depth=7` to cap overfitting on the leaf-wise growth.

**(c) Gradient-based One-Side Sampling (GOSS).** Observations with small gradients are
already well-fit; LightGBM keeps all large-gradient (high-error) instances and randomly
subsamples the small-gradient ones, rescaling the latter's contribution by a constant
$\tfrac{1-a}{b}$ so the gain estimate stays unbiased. This focuses each round's
computation on the hard cancellations.

LightGBM (`objective="binary"`, same 300/7/0.05/0.8 capacity as XGBoost) was chosen as
champion by the pre-registered rolling-origin PR-AUC criterion (§6), **not** on the
test set.

---

## 6. Model Selection — Rolling-Origin (Champion/Challenger)

Because the data are time-ordered, model selection uses an **expanding-window,
rolling-origin** protocol rather than random cross-validation. At cut-off fractions
$c \in \{0.60, 0.70, 0.80\}$ of the training period, the model is fit on bookings up to
$c$ and scored on the next validation slice. The selection metric averaged over the
folds is the area under the precision–recall curve (§8):

$$
\text{score}(\text{model}) = \frac{1}{|C|}\sum_{c \in C} \text{PR-AUC}_c .
$$

The model with the highest mean rolling PR-AUC is promoted to champion. This protocol
respects temporal ordering (no look-ahead) and is fixed *before* the test set is
touched, so the final test metrics are unbiased.

---

## 7. Probability Calibration and the Decision Layer

### 7.1 Isotonic Calibration

Raw boosted-tree scores rank well but are not well-calibrated probabilities. The study
applies **isotonic regression** — a non-parametric, monotonic map $g$ learned on the
*validation* set (never the test set, to avoid leakage):

$$
g^\star = \arg\min_{g \,\uparrow}\ \sum_{i \in \text{val}} \big(g(\hat{s}_i) - y_i\big)^2,
\qquad \hat{p}_i = g(\hat{s}_i),
$$

where "$g \uparrow$" restricts $g$ to non-decreasing functions and $\hat{s}_i$ is the
raw model score. The minimiser is computed by the **Pool-Adjacent-Violators
Algorithm (PAVA)**: sort by score, then repeatedly merge ("pool") adjacent bins that
violate monotonicity into their weighted mean until the sequence is non-decreasing.
The effect is measured by the calibration error falling from ECE $= 0.062$ to $0.031$
(§8).

### 7.2 Cost-Sensitive Threshold Selection

The calibrated probability is turned into an action by a threshold chosen to minimise
**expected business cost**, not to maximise accuracy. With a fixed false-positive cost
$c_{\text{FP}} = €15$ (intervention) and a per-booking false-negative cost
$c_{\text{FN},i} = \text{ADR}_i \times \text{nights}$ (lost revenue), the total cost at
threshold $\tau$ is

$$
\text{Cost}(\tau) = c_{\text{FP}}\!\!\sum_{i:\,\hat{y}_i=1,\,y_i=0}\!\!1
\;+\!\! \sum_{i:\,\hat{y}_i=0,\,y_i=1}\!\! c_{\text{FN},i},
\qquad
\tau^\star = \arg\min_{\tau \in \mathcal{T}} \text{Cost}(\tau),
$$

swept over the grid $\mathcal{T} = \{0.00, 0.01, \dots, 0.99\}$. The asymmetry
$c_{\text{FN},i} \gg c_{\text{FP}}$ drives $\tau^\star$ low ($\approx 0.06$), reflecting
that missing a cancellation costs far more than an unnecessary reminder. A parallel
**max-F1** threshold (the $\tau$ maximising $F_1$, §8) is reported as the
operating point for routine use, and a **high-precision** threshold for the
scarce-capacity regime. *(All three thresholds are selected on validation data and
evaluated once on the test set.)*

### 7.3 Risk Tiers

For the deployed interface the calibrated probability is bucketed into three
managerial tiers using fixed bands:

$$
\text{tier}(\hat{p}) =
\begin{cases}
\text{Low} & \hat{p} < 0.40,\\
\text{Medium} & 0.40 \le \hat{p} < 0.70,\\
\text{High} & \hat{p} \ge 0.70.
\end{cases}
$$

---

## 8. Evaluation Metrics

All test-set claims rest on the following definitions. Let TP, FP, TN, FN be the
confusion-matrix counts at the operating threshold.

**Precision, Recall, $F_1$.**

$$
\text{Precision} = \frac{\text{TP}}{\text{TP}+\text{FP}}, \quad
\text{Recall} = \frac{\text{TP}}{\text{TP}+\text{FN}}, \quad
F_1 = \frac{2\,\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}.
$$

$F_1$ is the harmonic mean of precision and recall; the harmonic mean punishes
imbalance between the two, so a high $F_1$ requires *both* to be good.

**ROC-AUC.** The area under the curve of true-positive rate vs false-positive rate
across all thresholds; equivalently, the probability that a randomly chosen positive is
ranked above a randomly chosen negative:

$$
\text{ROC-AUC} = P\big(\hat{p}_{i^+} > \hat{p}_{j^-}\big).
$$

**PR-AUC (primary metric).** The area under the precision–recall curve, computed as the
**average precision**

$$
\text{PR-AUC} = \sum_{k} \big(R_k - R_{k-1}\big)\, P_k,
$$

where $(P_k, R_k)$ are precision/recall at successive thresholds. PR-AUC is the study's
headline metric because, unlike ROC-AUC, it ignores the large pool of easy true
negatives and focuses on the minority cancellation class — the class the business
actually cares about. The champion attains PR-AUC $\approx 0.759$ (0.70–0.76 across the
duplicate-sensitivity range).

**Expected Calibration Error (ECE).** Probabilities are partitioned into $B = 10$
equal-width bins; ECE is the support-weighted gap between confidence and accuracy:

$$
\text{ECE} = \sum_{b=1}^{B} \frac{|S_b|}{n}\,
\big|\, \text{acc}(S_b) - \text{conf}(S_b) \,\big|,
$$

where $\text{acc}(S_b)$ is the observed cancellation rate in bin $b$ and
$\text{conf}(S_b)$ is the mean predicted probability there. ECE $= 0$ means "when the
model says 70%, exactly 70% cancel"; isotonic calibration roughly halves it
($0.062 \to 0.031$).

**Paired significance test (champion vs baseline).** To test whether the champion's
PR-AUC advantage over logistic regression is real, the study uses a **paired bootstrap**:
resample the test set with replacement $B = 2000$ times, recompute the PR-AUC difference
$\delta_b = \text{PR-AUC}^{\text{champ}}_b - \text{PR-AUC}^{\text{LR}}_b$ on each
resample, and report the two-sided p-value

$$
p = \min\!\Big(1,\; 2 \cdot \min\!\big(
\tfrac{1}{B}\textstyle\sum_b \mathbb{1}[\delta_b \le 0],\;
\tfrac{1}{B}\textstyle\sum_b \mathbb{1}[\delta_b \ge 0]
\big)\Big).
$$

Pairing on identical resamples controls for test-set difficulty. The result
($p = 0.177$) is reported honestly: the advantage is not statistically significant, so
LightGBM is retained for calibration, recall, and efficiency rather than a decisive
ranking margin.

---

## 9. Why This Is Not a Black Box — Summary

| Stage | Governing principle | Equation |
|---|---|---|
| Objective | Maximum-likelihood / cross-entropy | §2 |
| Linear baseline | Logistic link + MLE | §3.2 |
| Tree splits | Gini impurity decrease | §3.3 |
| Bagging | Variance reduction by averaging | §4 |
| Boosting | Functional gradient descent | §5.1 |
| XGBoost | 2nd-order Taylor + leaf regularisation | §5.2 |
| LightGBM | Histogram + leaf-wise + GOSS | §5.3 |
| Calibration | Monotonic isotonic regression (PAVA) | §7.1 |
| Decision | Expected-cost minimisation | §7.2 |
| Evaluation | PR-AUC, ECE, paired bootstrap | §8 |

Each library call in the pipeline corresponds to one of the equations above; the
software is an implementation of this mathematics, not a substitute for understanding
it.
