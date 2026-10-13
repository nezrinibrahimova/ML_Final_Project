"""
run_all.py — ONE command reproduces every headline number and figure.

Run from the repo root:
    python -m src.run_all            # everything
    python -m src.run_all svm ridge  # only some steps (names below)

Each step is an experiment module with a ``main()`` that loads the data,
selects hyperparameters on VALIDATION, scores TEST once, saves its figures
to report/figures/ and its tables to report/tables/. Steps that produce
headline test numbers write them to report/tables/<name>_task_{a,b}_results.csv
(rows = models, columns = metrics); this script prints them together at the end.

All randomness is seeded with data_prep.SEED.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data_prep import DATA_PATH, SEED

TABLES = Path("report/tables")

# (step name, module, what it produces)
STEPS = [
    ("eda", "src.eda", "EDA figures"),
    ("tree", "src.tree_experiment", "Decision tree: Task A + Task B, sklearn baselines"),
    ("svm", "src.svm_experiment", "Pegasos SVM (+RFF): Task B, sklearn baselines"),
    ("ridge", "src.ridge_experiment", "Bonus P8: Ridge regression, Task A"),
    ("clusters", "src.unsupervised_experiment", "Bonus P10: PCA + k-means segments"),
]


def _banner(text: str) -> None:
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78, flush=True)


def _print_results(task: str, title: str) -> None:
    files = sorted(TABLES.glob(f"*_{task}_results.csv"))
    if not files:
        return
    table = pd.concat([pd.read_csv(f, index_col=0) for f in files])
    _banner(title)
    from .evaluate import compare
    compare({name: row.dropna().to_dict() for name, row in table.iterrows()})


def main(argv: list[str] | None = None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles: "±", Azerbaijani names
    np.random.seed(SEED)
    wanted = set(argv if argv is not None else sys.argv[1:])
    unknown = wanted - {name for name, _, _ in STEPS}
    if unknown:
        raise SystemExit(f"unknown step(s) {sorted(unknown)}; choose from {[s[0] for s in STEPS]}")
    if not Path(DATA_PATH).exists():
        raise SystemExit(f"dataset not found at {DATA_PATH} — see data/README.md")

    t_all = time.perf_counter()
    timings, missing = [], []
    for name, module, what in STEPS:
        if wanted and name not in wanted:
            continue
        _banner(f"[{name}] {what}  ({module})")
        if importlib.util.find_spec(module) is None:
            print(f"!! SKIPPED: {module.replace('.', '/')}.py is not in the repo yet")
            missing.append(name)
            continue
        t = time.perf_counter()
        importlib.import_module(module).main()
        timings.append((name, time.perf_counter() - t))

    _print_results("task_a", "HEADLINE — Task A: price regression (test split)")
    _print_results("task_b", "HEADLINE — Task B: price tier classification (test split)")

    _banner("Done")
    for name, secs in timings:
        print(f"  {name:<10} {secs:7.1f}s")
    print(f"  {'total':<10} {time.perf_counter() - t_all:7.1f}s")
    print("Figures: report/figures/   Tables: report/tables/")
    if missing:
        raise SystemExit(f"INCOMPLETE: skipped step(s) {missing} - module(s) missing")


if __name__ == "__main__":
    main()
