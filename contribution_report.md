# Contribution Report

MLE-AI-201 Machine Learning — Final Project · bina.az price prediction
Repository: <https://github.com/nezrinibrahimova/ML_Final_Project> (tag `v1.0-final`)

All work was done on feature branches and merged through pull requests
reviewed by another team member. The commits and PRs listed below can be
checked in the Git history.

## Team
| Member (Surname, Name) | GitHub handle | Main responsibilities |
|------------------------|---------------|-----------------------|
| Ibrahimova, Nezrin | `nezrinibrahimova` | **A — data and metrics:** loading, cleaning, leakage removal, features, shared split and tier label, EDA, evaluation metrics; bonuses P8 (Ridge) and P10 (PCA + k-means) |
| Gurbanova, Sevda | `qurbanovasevda` | **B — decision tree:** from-scratch `DecisionTree`, tests, hyperparameter sweeps and learning curves, comparison with scikit-learn, CV, error analysis |
| Askerova, Melek | `askerovamelek` | **C — SVM and integration:** from-scratch Pegasos SVM with random Fourier features, SVM experiments and baselines, `run_all`, pinned requirements, README, report write-up |

## Per-member summary

**Ibrahimova, Nezrin —** Set up the repository and implemented the whole
data pipeline in `src/data_prep.py`: residential filtering, de-duplication
by listing, parsing of area/floor/land, outlier rules, removal of the
leakage columns `unit_price` and `total_price`, one-hot features, the
stratified 70/15/15 split, the train-median tier label and train-only
standardisation (PR #1, #4, tests in PR #5). Wrote all NumPy metrics in
`src/evaluate.py`, including a tie-aware ROC-AUC and the constant-target
case for R² (PR #2, #6), and the EDA figures in `src/eda.py` (PR #3).
Implemented both bonuses: closed-form Ridge regression with tests
(`src/ridge.py`, PR #7) and PCA + k-means market segmentation
(`src/unsupervised.py`, PR #8), and the cross-model analysis by cluster
(`src/final_analysis.py`, PR #17). Reviewed and merged most of the team's
pull requests.

**Gurbanova, Sevda —** Implemented the from-scratch decision tree in
`src/decision_tree.py` (PR #9): Gini, entropy and MSE criteria, a
vectorised split search based on sorting and cumulative counts, the four
hyperparameters, `predict_proba` and `feature_importances_`. Wrote the tree
tests against scikit-learn and found and fixed a float-rounding bug that
could create empty children (PR #10, #11, #12). Built the tree experiments:
depth and leaf sweeps, Gini vs entropy, learning curves (PR #14), the
matched comparison with scikit-learn (PR #15), the error analysis by price
segment, district and tier border (PR #16), and moved all tree scripts to
the shared split with joint depth × leaf selection and 5-fold CV (PR #18,
#21). Reviewed the data, metrics and bonus PRs (#1, #2, #5–#8, #17) and
the report PR (#23).

**Askerova, Melek —** Implemented the from-scratch Pegasos SVM in
`src/svm.py` (PR #13): mini-batch hinge sub-gradient with an unregularised
bias, learning-rate schedules including a stable `t0 = 1/λ` offset,
objective tracking, ‖w‖, margin and support vectors, and random Fourier
features for an approximate RBF kernel; 11 unit tests in
`tests/test_svm.py`. Wrote `src/svm_experiment.py`: λ and γ×λ sweeps
chosen on validation, the D curve, a learning-rate schedule comparison, a
feature-scaling ablation, and matched baselines (`LinearSVC`,
`SGDClassifier`, RBF `SVC`) with CV and timing. Built the single
`python -m src.run_all` entry point, pinned `requirements.txt` and checked
reproduction from a fresh clone (PR #13, #22). Wrote the README (PR #25)
and the IEEE report (PR #23). Reviewed and merged the tree PRs #14 and #15.
Parts of this work were done with an AI coding assistant, as disclosed in
the report.

## Rough effort split

| Member | Approx. % |
|--------|-----------|
| Ibrahimova, Nezrin | 34 |
| Gurbanova, Sevda   | 33 |
| Askerova, Melek    | 33 |

The work was split evenly by role. A had the most separate pieces (data,
metrics, EDA and both bonuses), while B and C each owned one
from-scratch model plus its experiments. Commit counts are 16 (A), 14 (B)
and 11 (C).
