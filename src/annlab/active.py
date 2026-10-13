"""Uncertainty sampling vs random sampling, replayed on already-labeled data.

The pool is the set of items the human labeled; "querying" an item just reveals
the label we already have, so both strategies see exactly the same data.
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from . import config
from .model import make_model


def _initial(y, n_init, rng):
    """Random seed set that is guaranteed to contain both classes."""
    y = np.asarray(y)
    idx = [int(rng.choice(np.flatnonzero(y == c))) for c in (0, 1)]
    rest = [i for i in rng.permutation(len(y)) if i not in idx]
    return idx + rest[: n_init - 2]


def run_strategy(texts, y, test_idx, pool_idx, strategy, n_init=10, batch=5, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    texts = np.asarray(texts, dtype=object)
    pool = list(pool_idx)
    labeled = _initial(y[pool], n_init, rng)
    labeled = [pool[i] for i in labeled]
    unlabeled = [i for i in pool if i not in set(labeled)]
    curve = []
    while True:
        model = make_model().fit(texts[labeled], y[labeled])
        pred = model.predict(texts[test_idx])
        curve.append(
            {
                "n_labeled": len(labeled),
                "accuracy": accuracy_score(y[test_idx], pred),
                "macro_f1": f1_score(y[test_idx], pred, average="macro"),
            }
        )
        if not unlabeled:
            return curve
        k = min(batch, len(unlabeled))
        if strategy == "random":
            pick = rng.choice(len(unlabeled), size=k, replace=False)
        elif strategy == "uncertainty":
            proba = model.predict_proba(texts[unlabeled])[:, 1]
            pick = np.argsort(np.abs(proba - 0.5))[:k]
        else:
            raise ValueError(strategy)
        chosen = [unlabeled[i] for i in pick]
        labeled += chosen
        unlabeled = [i for i in unlabeled if i not in set(chosen)]


def compare(texts, y, repeats=20, n_init=10, batch=5, test_size=0.3, seed=config.SEED):
    """Mean learning curves over repeated random train/test splits."""
    results = {"random": [], "uncertainty": []}
    all_idx = np.arange(len(y))
    for r in range(repeats):
        pool_idx, test_idx = train_test_split(
            all_idx, test_size=test_size, random_state=seed + r, stratify=y
        )
        for strat in results:
            results[strat].append(
                run_strategy(texts, y, test_idx, pool_idx, strat, n_init, batch, seed + r)
            )
    return results


def summarize(results):
    """Average each strategy's curves into rows of n_labeled, mean, std."""
    rows = []
    for strat, runs in results.items():
        for step in range(min(len(r) for r in runs)):
            pts = [r[step] for r in runs]
            rows.append(
                {
                    "strategy": strat,
                    "n_labeled": pts[0]["n_labeled"],
                    "accuracy_mean": float(np.mean([p["accuracy"] for p in pts])),
                    "accuracy_std": float(np.std([p["accuracy"] for p in pts])),
                    "macro_f1_mean": float(np.mean([p["macro_f1"] for p in pts])),
                    "macro_f1_std": float(np.std([p["macro_f1"] for p in pts])),
                }
            )
    return rows
