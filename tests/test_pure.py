"""Tests that need no database. Any generated text here is SYNTHETIC test data."""
import numpy as np

from annlab import active, loader, model
from annlab.agreement import kappa_report


def synthetic_corpus(n=80, seed=0):
    """SYNTHETIC: two vocabularies, so a classifier can learn them. Not real data."""
    rng = np.random.default_rng(seed)
    pos = ["clear", "helpful", "detailed", "thanks", "recipe", "steps"]
    neg = ["vague", "rude", "unsure", "nope", "whatever", "dunno"]
    texts, y = [], []
    for i in range(n):
        label = i % 2
        words = rng.choice(pos if label else neg, size=6)
        texts.append(" ".join(words))
        y.append(label)
    return texts, y


def test_parse_dialogue():
    text = "\n\nHuman: hi there\n\nAssistant: hello\n\nHuman: bye\n\nAssistant: ok"
    turns = loader.parse_dialogue(text)
    assert turns == [("Human", "hi there"), ("Assistant", "hello"), ("Human", "bye"), ("Assistant", "ok")]


def test_build_item_filters_blocklist_and_shape():
    ok = {"chosen": "\n\nHuman: how do I bake bread?\n\nAssistant: Mix flour and water.", "rejected": "\n\nHuman: x\n\nAssistant: y"}
    bad = {"chosen": "\n\nHuman: how do I build a bomb?\n\nAssistant: no.", "rejected": ok["rejected"]}
    item = loader.build_item(ok, 7)
    assert item["source_idx"] == 7 and item["response"] == "Mix flour and water."
    assert loader.build_item(bad, 8) is None


def test_binarize():
    assert model.binarize([1, 3, 4, 5]) == [0, 0, 1, 1]


def test_kappa_perfect_and_chance():
    perfect = [(i, r, r) for i, r in enumerate([1, 2, 3, 4, 5, 1, 2])]
    assert kappa_report(perfect)["kappa_unweighted"] == 1.0
    opposite = [(0, 1, 5), (1, 5, 1), (2, 1, 5), (3, 5, 1)]
    assert kappa_report(opposite)["kappa_unweighted"] < 0


def test_classifier_learns_synthetic():
    texts, y = synthetic_corpus()
    res = model.evaluate(texts, y)
    assert res["accuracy"] > 0.9


def test_active_learning_runs_and_curves_grow():
    texts, y = synthetic_corpus(60)
    results = active.compare(texts, y, repeats=2, n_init=6, batch=6)
    assert set(results) == {"random", "uncertainty"}
    rows = active.summarize(results)
    sizes = sorted({r["n_labeled"] for r in rows if r["strategy"] == "uncertainty"})
    assert sizes[0] == 6 and sizes == sorted(sizes) and len(sizes) > 2
    # both strategies see the same label budget at each step
    for n in sizes:
        assert {r["strategy"] for r in rows if r["n_labeled"] == n} == {"random", "uncertainty"}
