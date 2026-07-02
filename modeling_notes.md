# Modeling Notes: Startup Failure Duration

Design doc: `docs/superpowers/specs/2026-07-02-startup-failure-modeling-design.md`

## Regression leaderboard (target: duration_years)

| model | train_R2 | test_R2 | test_MAE | test_RMSE | R2_gap |
|---|---|---|---|---|---|
| DecisionTree | 0.4098 | 0.3774 | 2.2335 | 2.7221 | 0.0324 |
| XGBoost | 0.7073 | 0.3661 | 2.2305 | 2.7466 | 0.3411 |
| LassoCV | 0.4429 | 0.3621 | 2.2350 | 2.7553 | 0.0807 |
| Dummy | 0.0000 | -0.0013 | 2.8973 | 3.4521 | 0.0013 |

Baseline: DummyRegressor(strategy="mean"), test R2 ≈ 0 (-0.0013).

DecisionTree reaches test R2=0.3774, a notable improvement over baseline (+0.3788). Train-test R2 gap of 0.0324 indicates low overfitting.

XGBoost reaches test R2=0.3661, a notable improvement over baseline (+0.3675). Train-test R2 gap of 0.3411 indicates high overfitting.

LassoCV reaches test R2=0.3621, a notable improvement over baseline (+0.3635). Train-test R2 gap of 0.0807 indicates low-to-moderate overfitting.

See `modeling_output/regression_predicted_vs_actual.png` and
`modeling_output/regression_tree_validation_curve.png`.

## Classification leaderboard (target: duration_class)

| model | train_accuracy | test_accuracy | precision_macro | recall_macro | f1_macro | roc_auc_macro_ovr |
|---|---|---|---|---|---|---|
| XGBoost | 0.9878 | 0.5976 | 0.6049 | 0.5818 | 0.5879 | 0.7463 |
| DecisionTree | 0.6697 | 0.5488 | 0.5717 | 0.5498 | 0.5587 | 0.7450 |
| LogisticRegressionCV | 0.6147 | 0.5366 | 0.5693 | 0.5300 | 0.5415 | 0.6957 |
| Dummy | 0.4037 | 0.4024 | 0.1341 | 0.3333 | 0.1913 | 0.5000 |

Baseline: DummyClassifier(strategy="most_frequent"), f1_macro=0.1913, roc_auc_macro_ovr=0.5000 (chance level).

XGBoost reaches f1_macro=0.5879 (roc_auc_macro_ovr=0.7463), a notable improvement over baseline (+0.3966 f1, +0.2463 AUC). Train-test accuracy gap of 0.3902 (98.8% train vs 59.8% test) indicates high overfitting.

DecisionTree reaches f1_macro=0.5587 (roc_auc_macro_ovr=0.7450), a notable improvement over baseline (+0.3674 f1, +0.2450 AUC). Its f1 is now meaningfully below XGBoost's (0.5587 vs 0.5879, a ~0.029 gap), though its AUC is essentially tied with XGBoost's (0.7450 vs 0.7463). Train-test accuracy gap of 0.1209 (66.97% train vs 54.88% test) indicates low-to-moderate overfitting.

LogisticRegressionCV reaches f1_macro=0.5415 (roc_auc_macro_ovr=0.6957), a notable improvement over baseline (+0.3502 f1, +0.1957 AUC). Train-test accuracy gap of 0.0781 indicates low-to-moderate overfitting.

Internally, `duration_class` is encoded as integers (`early=0, typical=1, long_run=2`) for XGBoost compatibility, applied consistently across all four models — every metric and plot in this section is reported using the class names, not the codes.

See `modeling_output/classification_confusion_matrices.png` and
`modeling_output/classification_tree_validation_curve.png`.

## Which features matter

**Regression.** Ranking `regression_lasso_coefficients.csv` by absolute coefficient and `regression_xgb_permutation_importance.csv` by importance score, the two agree strongly on the top two drivers: `decade_started` is the single largest signal in both (Lasso: `decade_started_2010s_plus` = -2.801, the largest-magnitude coefficient of all; `decade_started_2000s` = +0.790; XGBoost: `decade_started` permutation importance = 0.452, by far the top feature), and `raised_musd`/`log_raised_musd` is the second-largest in both (Lasso: `log_raised_musd` = +1.106, second-largest magnitude; XGBoost: `raised_musd` = 0.236, second rank). `Sector` also agrees as a secondary factor (Lasso has `Sector_Manufacturing` = -0.530; XGBoost ranks `Sector` third at 0.037). Among failure-reason flags, `No Budget` and `Poor Market Fit` show consistent secondary importance in both models (Lasso: -0.249 and -0.702; XGBoost ranks them 4th and 5th at 0.014 and 0.013).

Where the two models disagree: Lasso assigns large coefficients to `single_cause_failure` (-0.990, third-largest overall) and `Giants` (+0.536, sixth-largest), but XGBoost's permutation importance treats both as negligible or even slightly negative (-0.0041 and -0.0260 respectively — permuting these features does not hurt, and mildly helps, test-set performance). `n_flags_sq` shows the same pattern (Lasso: -0.425, a meaningful eighth-ranked coefficient; XGBoost: 0.0005, essentially zero). These three features look important to the linear model but are not confirmed as robust nonlinear predictors — plausibly linear-model artifacts of collinearity with `decade_started` and `raised_musd` rather than independent signal.

**Classification.** The same top-two pattern holds: `raised_musd`/`log_raised_musd` and `decade_started` dominate both `classification_logreg_coefficients.csv` and `classification_xgb_permutation_importance.csv`. LogReg's single largest-magnitude coefficient anywhere in the table is `decade_started_2010s_plus` on the `long_run` class (-2.871), and `log_raised_musd` has large, opposite-signed coefficients on `early` (-0.806) and `long_run` (+1.264). XGBoost ranks `raised_musd` first (0.150) and `decade_started` second (0.103). `big_tech_pressure` also agrees across both: a sizeable LogReg coefficient on `long_run` (+1.030) and XGBoost's fourth-ranked feature (0.025). `No Budget` and `Poor Market Fit` again show consistent, moderate importance in both (LogReg: -0.536 on `typical`, +0.596/-0.536 on `early`/`long_run`; XGBoost ranks 6th and 10th).

Disagreement mirrors the regression case: `single_cause_failure` has the third-largest LogReg coefficient anywhere in the table (-1.467 on `long_run`) but ranks near the bottom of XGBoost's permutation importance (0.0030, 13th of 16). `Giants` shows a similar split (LogReg: -0.514 on `early`; XGBoost: -0.0036, second-to-last, i.e. negative). `Niche Limits` disagrees in the other direction — small in LogReg (max |coef| 0.184) but XGBoost ranks it 7th (0.012), moderately important to the tree model despite near-zero linear weight.

## Honest evaluation summary

XGBoost does not clearly beat the simpler models on this data, and in the regression track it actually loses on the metric that matters: DecisionTree's test R2 (0.3774) edges out XGBoost's (0.3661) outright, while DecisionTree's train-test gap (0.0324) is roughly a tenth of XGBoost's (0.3411) — XGBoost memorizes the 327-row training split (train R2=0.707) without that showing up as better held-out performance. In the classification track XGBoost does post the best test-set numbers (f1_macro=0.5879, roc_auc_macro_ovr=0.7463), and here the margin over DecisionTree (f1_macro=0.5587, roc_auc_macro_ovr=0.7450) is real on f1 (+0.0292) but essentially tied on AUC (+0.0013) — bought at the cost of substantially more overfitting (train accuracy 98.8% vs test accuracy 59.8%, a 0.3902 gap) versus DecisionTree's gap of 0.1209 (66.97% train vs 54.88% test), roughly a 3.2x difference (0.3902 vs 0.1209) rather than negligible. At n=409 (327 train / 82 test), a +0.029 f1 edge from a model whose train accuracy is 40 points higher than its test accuracy is a modest but real advantage, not the noise-level edge it once appeared to be — though it still comes with meaningfully more overfitting risk than DecisionTree. Given the regression-track numbers, we still recommend **DecisionTree** for the regression track: it matches or beats XGBoost on held-out performance while being dramatically more stable (low variance between train and test) and easier to inspect and explain to a non-technical audience. For the classification track the recommendation is a closer call: XGBoost's f1 edge (+0.029) is real, and its AUC is essentially identical to DecisionTree's, but it carries roughly 3.2x DecisionTree's train-test overfitting gap and is far harder to explain to a non-technical audience. For a business audience prioritizing stability and interpretability, DecisionTree remains the more defensible choice; a team optimizing purely for held-out f1 would have legitimate grounds to prefer XGBoost instead.

## Business interpretation

**What the results mean:** These models explain variation in *observed*
duration among startups that already failed. They are not a forecast of how
long a currently-operating company will last -- the failure-reason flags used
as predictors are hindsight labels, assigned only after each company's
outcome was already known (see `data_notes.md`). Repositioning this as a
forward-looking survival predictor would misrepresent what it was trained on.

**Which features matter for the business question:** Across both the duration-in-years regression and the early/typical/long_run classification, and across both the linear (Lasso/LogisticRegressionCV) and nonlinear (XGBoost permutation importance) models, two features consistently dominate: how much a company raised (`raised_musd`/`log_raised_musd`) and when it was founded (`decade_started`) -- companies founded in the 2010s-plus and those that raised less tend to show shorter observed durations. Among the failure-reason flags, `No Budget`, `Poor Market Fit`, and `big_tech_pressure` are the ones that show up as important in *both* the linear and tree-based models, so these are the more trustworthy behavioral signals. By contrast, `single_cause_failure` and the `Giants` flag carry large coefficients in the linear models but wash out to near-zero (or negative) importance under XGBoost's permutation test, so we would not lean on those two for a business narrative -- they likely reflect collinearity with funding/decade rather than independent effects.

**What we would not trust this model to do:**
1. Predict the remaining runway of an active, non-failed company (hindsight-label issue above).
2. Generalize to sectors thin on data -- Food/services has only 26 rows total in this dataset, and Sector x duration_class cells can be very sparse.
3. Extrapolate past the observed funding range (max observed raise in this dataset is $3.5B).
4. Support causal claims -- e.g. "Poor Market Fit causes shorter duration" is not established; only "companies labeled this way tended to have shorter observed duration in this sample."
