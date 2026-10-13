"""Database tests. Items and ratings here are SYNTHETIC test fixtures."""
import pytest
from pymongo.errors import DuplicateKeyError

from annlab import config, queries
from annlab.agreement import paired_ratings
from annlab.labeling import kappa_subset_ids, next_items, save_annotation


def add_items(db, n=5):
    ids = []
    for i in range(n):
        res = db[config.ITEMS].insert_one(
            {"source_idx": i, "context": [{"role": "Human", "text": f"q{i}"}], "response": f"a{i}", "n_turns": 2 + i % 2, "text": f"q{i} a{i}"}
        )
        ids.append(res.inserted_id)
    return ids


def test_unique_annotation_per_annotator(db):
    (i,) = add_items(db, 1)
    db[config.ANNOTATIONS].insert_one({"item_id": i, "annotator": "a", "rating": 3})
    with pytest.raises(DuplicateKeyError):
        db[config.ANNOTATIONS].insert_one({"item_id": i, "annotator": "a", "rating": 4})


def test_revision_history(db):
    (i,) = add_items(db, 1)
    save_annotation(db, i, "a", 2, "meh")
    save_annotation(db, i, "a", 4, "better on reflection")
    doc = db[config.ANNOTATIONS].find_one({"item_id": i})
    assert doc["rating"] == 4 and [r["rating"] for r in doc["revisions"]] == [2]
    assert queries.revised_annotations(db)[0]["n_revisions"] == 1


def test_next_items_skips_done_and_second_mode(db):
    ids = add_items(db, 6)
    for i in ids[:4]:
        save_annotation(db, i, config.PRIMARY_ANNOTATOR, 3, "x")
    assert len(next_items(db, config.PRIMARY_ANNOTATOR)) == 2
    subset = kappa_subset_ids(db, size=3)
    assert len(subset) == 3 and set(subset) <= set(ids[:4])
    second = next_items(db, "second", second_mode=True)
    assert {d["_id"] for d in second} == set(kappa_subset_ids(db))


def test_pairs_and_disagreements(db):
    ids = add_items(db, 3)
    for i, (a, b) in zip(ids, [(1, 5), (3, 3), (4, 2)]):
        save_annotation(db, i, "p", a, "r")
        save_annotation(db, i, "s", b, "r")
    pairs = paired_ratings(db, "p", "s")
    assert sorted((a, b) for _, a, b in pairs) == [(1, 5), (3, 3), (4, 2)]
    gaps = queries.disagreements(db, min_gap=2)
    assert [g["gap"] for g in gaps] == [4, 2]
