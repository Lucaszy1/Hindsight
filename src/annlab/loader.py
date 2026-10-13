"""Download a small, filtered, seeded subset of HH-RLHF helpful-base into MongoDB.

HH-RLHF is MIT licensed (checked on the Hugging Face dataset card).
"""
import argparse
import gzip
import json
import random
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .db import ensure_indexes, get_db

# Crude keyword screen for disturbing content. Deliberately over-inclusive.
BLOCKLIST = re.compile(
    r"\b(kill|murder|suicide|rape|molest|abuse|torture|bomb|terroris\w*|"
    r"gun|shoot\w*|weapon\w*|porn\w*|sex\w*|nazi|racist|slur|drug\w*|"
    r"cocaine|heroin|meth|self[- ]harm|stab\w*|poison\w*|kidnap\w*)\b",
    re.IGNORECASE,
)
TURN = re.compile(r"\n\n(Human|Assistant): ")


def parse_dialogue(text):
    """Split an HH transcript into [(role, utterance), ...]."""
    parts = TURN.split("\n\n" + text.lstrip("\n"))
    return [(parts[i], parts[i + 1].strip()) for i in range(1, len(parts) - 1, 2)]


def is_safe(text):
    return BLOCKLIST.search(text) is None


def build_item(record, idx):
    """Return an item document, or None if the record should be skipped."""
    chosen, rejected = record["chosen"], record["rejected"]
    if not (is_safe(chosen) and is_safe(rejected)):
        return None
    turns = parse_dialogue(chosen)
    if len(turns) < 2 or turns[-1][0] != "Assistant" or len(chosen) > 1000:
        return None
    return {
        "source": "Anthropic/hh-rlhf:helpful-base:train",
        "source_idx": idx,
        "context": [{"role": r, "text": t} for r, t in turns[:-1]],
        "response": turns[-1][1],
        "n_turns": len(turns),
        "text": " ".join(t for _, t in turns),  # model input
    }


def download(path):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(config.DATA_URL, path)
    return path


def sample_items(path, n, seed=config.SEED):
    with gzip.open(path, "rt") as f:
        records = [json.loads(line) for line in f]
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    items = []
    for idx in order:
        item = build_item(records[idx], idx)
        if item:
            items.append(item)
        if len(items) >= n:
            break
    return items


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-n", type=int, default=300, help="items to load")
    args = p.parse_args(argv)
    items = sample_items(download(config.RAW_PATH), args.n)
    db = get_db()
    ensure_indexes(db)
    now = datetime.now(timezone.utc)
    added = 0
    for it in items:
        res = db[config.ITEMS].update_one(
            {"source_idx": it["source_idx"]},
            {"$setOnInsert": {**it, "loaded_at": now}},
            upsert=True,
        )
        added += res.upserted_id is not None
    print(f"loaded {added} new items ({len(items)} sampled)")


if __name__ == "__main__":
    main()
