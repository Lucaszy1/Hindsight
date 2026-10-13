"""Interactive CLI: rate each item 1-5 and give a one-sentence rationale."""
import argparse
import random
from datetime import datetime, timezone

from . import config
from .db import get_db

KAPPA_SUBSET_SIZE = 40


def save_annotation(db, item_id, annotator, rating, rationale):
    """Insert a new annotation, or revise an existing one keeping history."""
    now = datetime.now(timezone.utc)
    col = db[config.ANNOTATIONS]
    existing = col.find_one({"item_id": item_id, "annotator": annotator})
    if existing is None:
        col.insert_one(
            {
                "item_id": item_id,
                "annotator": annotator,
                "rating": rating,
                "rationale": rationale,
                "created_at": now,
                "revisions": [],
            }
        )
        return
    col.update_one(
        {"_id": existing["_id"]},
        {
            "$set": {"rating": rating, "rationale": rationale, "updated_at": now},
            "$push": {
                "revisions": {
                    "rating": existing["rating"],
                    "rationale": existing["rationale"],
                    "at": existing.get("updated_at", existing["created_at"]),
                }
            },
        },
    )


def kappa_subset_ids(db, size=KAPPA_SUBSET_SIZE, seed=config.SEED):
    """Deterministic subset of items the primary annotator already rated."""
    ids = sorted(
        d["item_id"]
        for d in db[config.ANNOTATIONS].find(
            {"annotator": config.PRIMARY_ANNOTATOR}, {"item_id": 1}
        )
    )
    random.Random(seed).shuffle(ids)
    return ids[:size]


def next_items(db, annotator, second_mode=False, limit=None):
    done = {d["item_id"] for d in db[config.ANNOTATIONS].find({"annotator": annotator}, {"item_id": 1})}
    if second_mode:
        ids = [i for i in kappa_subset_ids(db) if i not in done]
        items = list(db[config.ITEMS].find({"_id": {"$in": ids}}))
    else:
        items = list(db[config.ITEMS].find({"_id": {"$nin": list(done)}}).sort("source_idx", 1))
    return items[:limit] if limit else items


def show(item):
    print("\n" + "=" * 70)
    for turn in item["context"]:
        print(f"[{turn['role']}] {turn['text']}")
    print(f"\n>>> Assistant reply to rate: {item['response']}\n")


def ask_rating():
    while True:
        raw = input("Helpfulness of the reply, 1 (useless) to 5 (excellent), q to quit: ").strip().lower()
        if raw == "q":
            return None
        if raw in {"1", "2", "3", "4", "5"}:
            return int(raw)
        print("Enter 1-5 or q.")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotator", default=config.PRIMARY_ANNOTATOR)
    p.add_argument(
        "--second",
        action="store_true",
        help="second-annotator mode: only the shared kappa subset, first ratings hidden",
    )
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)
    if args.second and args.annotator == config.PRIMARY_ANNOTATOR:
        p.error("--second needs --annotator set to a different name")
    db = get_db()
    items = next_items(db, args.annotator, args.second, args.limit)
    print(f"{len(items)} items to label as '{args.annotator}'")
    for item in items:
        show(item)
        rating = ask_rating()
        if rating is None:
            break
        rationale = input("One-sentence rationale: ").strip()
        save_annotation(db, item["_id"], args.annotator, rating, rationale)
    print("saved. bye.")


if __name__ == "__main__":
    main()
