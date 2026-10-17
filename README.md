# Pricing Baku Real Estate from Scratch

**MLE-AI-201 Machine Learning — Final Project (AI Academy)**
Melek Askerova · Sevda Gurbanova · Nezrin Ibrahimova

We predict apartment and house prices from **58,007 bina.az sale listings**
with a **decision tree** and a **soft-margin SVM (Pegasos)**, both written
from scratch in NumPy. scikit-learn is used only for the comparison
baselines and the bonus analyses.

- **Task A — regression:** predict the asking `price` (AZN).
- **Task B — classification:** predict the price tier: *premium* if the price
  is above the **training-set median** (208,000 AZN), otherwise *standard*.

📄 Full write-up: [`report/report.pdf`](report/report.pdf) (IEEE format, source in `report/report.tex`)

---

## Results (test set, 8,701 listings)

**Task A — price**

| Model | RMSE (AZN) | MAE (AZN) | R² |
|---|---:|---:|---:|
| Train-mean baseline | 222,308 | 131,332 | 0.000 |
| Ridge, NumPy (bonus) | 124,377 | 57,792 | 0.687 |
| **Decision tree (ours, log target)** | **119,931** | **43,889** | **0.709** |
| sklearn `DecisionTreeRegressor` | 119,699 | 43,852 | 0.710 |

**Task B — price tier**

| Model | F1 | AUC | 5-fold CV F1 |
|---|---:|---:|---:|
| **Decision tree, entropy (ours)** | **0.926** | 0.930 | 0.920 ± 0.003 |
| Decision tree, Gini (ours) | 0.923 | 0.954 | 0.920 ± 0.004 |
| sklearn tree, entropy | 0.925 | 0.929 | 0.920 ± 0.004 |
| **Pegasos SVM + random Fourier features (ours)** | 0.912 | **0.974** | – |
| Pegasos SVM, linear (ours) | 0.905 | 0.964 | 0.904 ± 0.005 |
| sklearn `LinearSVC` | 0.906 | 0.964 | 0.905 ± 0.005 |
| sklearn `SVC` RBF (10k subsample) | 0.912 | 0.973 | – |

Our implementations match scikit-learn (tree R² 0.709 vs 0.710; linear SVM
reaches the same objective value as `LinearSVC`). Trees are best at the hard
premium/standard label, the RBF-approximated SVM ranks listings best.

---

## Quick start

Requires **Python 3.11**.

```bash
git clone https://github.com/nezrinibrahimova/ML_Final_Project.git
cd ML_Final_Project
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Data** (not committed): download the Kaggle dataset
[binaaz-sale-project](https://www.kaggle.com/datasets/sehriyarmemmedli/binaaz-sale-project)
and save the CSV as **`data/house_sale.csv`** (see [`data/README.md`](data/README.md)).

```bash
python -m src.run_all              # reproduces every number, table and figure (~25 min)
python -m pytest tests -q          # 49 unit tests
```

Run a single step with `python -m src.run_all svm` (names below).
All randomness uses seed 42; two clean runs produce identical tables.

---

## What `run_all` does

| Step | Module | Produces |
|---|---|---|
| `eda` | `src/eda.py` | price, area, category, district, class-balance figures |
| `tree` | `src/tree_experiments.py` | tree sweeps + learning curves, sklearn comparison, 5-fold CV, error analysis (Task A + B) |
| `svm` | `src/svm_experiment.py` | Pegasos λ sweep, RFF γ×λ and D sweeps, learning-rate schedules, scaling ablation, sklearn SVM baselines |
| `ridge` | `src/ridge_experiment.py` | bonus: closed-form Ridge vs sklearn `Ridge` (Task A) |
| `clusters` | `src/unsupervised_experiment.py` | bonus: PCA + k-means market segments |
| `final` | `src/final_analysis.py` | matched Ridge vs tree comparison, model errors per cluster |

Figures go to `report/figures/`, tables to `report/tables/`. The console ends
with the headline Task A and Task B tables.

**Evaluation protocol (shared by every model):** one stratified 70/15/15
split on price quartiles (40,605 / 8,701 / 8,701 rows, seed 42); tier
threshold and feature scaling computed on train only; every hyperparameter
chosen on validation; the test set is scored once.

---

## Repository layout

```
├── data/                      # dataset goes here (not committed)
├── src/
│   ├── data_prep.py           # loading, cleaning, leakage removal, features, split, tier label, scaling
│   ├── eda.py                 # exploratory figures
│   ├── evaluate.py            # NumPy metrics: RMSE, MAE, R², confusion matrix, P/R/F1, ROC-AUC
│   ├── decision_tree.py       # DecisionTree from scratch (Gini / entropy / MSE)
│   ├── tree_*.py              # tree experiments, sklearn comparison, CV, error analysis
│   ├── svm.py                 # PegasosSVM + RandomFourierFeatures from scratch
│   ├── svm_experiment.py      # SVM experiments and sklearn baselines
│   ├── ridge.py, ridge_experiment.py              # bonus: Ridge regression
│   ├── unsupervised.py, unsupervised_experiment.py # bonus: PCA + k-means
│   ├── final_analysis.py      # cross-model analysis by cluster
│   └── run_all.py             # one command for everything
├── tests/                     # pytest unit tests
├── report/                    # report.tex, report.pdf, figures/, tables/
├── requirements.txt           # pinned versions
└── contribution_report.md     # who did what
```

---

## Data and preprocessing (short)

- 100,775 raw rows → residential listings only → latest record per listing →
  outlier rules (area 15–1000 m², price/m² 300–10,000 AZN, valid
  coordinates) → **58,007 listings**.
- **Leakage removed:** `unit_price` and `total_price` are derived from the
  target and are dropped; owner/shop names and raw address are not used.
- **148 features:** 10 numeric (area, rooms, floor, total floors, land,
  latitude, longitude, repair/deed/mortgage flags) + one-hot category and
  district.

---

## Team

| Member | Role | Main work |
|---|---|---|
| Nezrin Ibrahimova | A — data & metrics | cleaning, leakage, features, split, EDA, metrics; bonuses: Ridge, PCA + k-means |
| Sevda Gurbanova | B — decision tree | from-scratch tree, sweeps, learning curves, sklearn comparison, CV, error analysis |
| Melek Askerova | C — SVM & integration | from-scratch Pegasos SVM + RFF, SVM baselines, `run_all`, packaging |

Details in [`contribution_report.md`](contribution_report.md).
