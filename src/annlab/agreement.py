"""Inter-annotator agreement (Cohen's kappa) between two annotators."""
from sklearn.metrics import cohen_kappa_score

from . import config


def paired_ratings(db, a, b):
    pipeline = [
        {"$match": {"annotator": {"$in": [a, b]}}},
        {"$group": {"_id": "$item_id", "r": {"$push": {"who": "$annotator", "rating": "$rating"}}}},
        {"$match": {"r.1": {"$exists": True}}},
        {"$sort": {"_id": 1}},
    ]
    out = []
    for doc in db[config.ANNOTATIONS].aggregate(pipeline):
        by = {x["who"]: x["rating"] for x in doc["r"]}
        out.append((doc["_id"], by[a], by[b]))
    return out


def kappa_report(pairs):
    if len(pairs) < 2:
        raise ValueError("need at least 2 doubly-annotated items")
    ra = [p[1] for p in pairs]
    rb = [p[2] for p in pairs]
    labels = [1, 2, 3, 4, 5]
    return {
        "n": len(pairs),
        "kappa_unweighted": cohen_kappa_score(ra, rb, labels=labels),
        "kappa_quadratic": cohen_kappa_score(ra, rb, labels=labels, weights="quadratic"),
        "exact_agreement": sum(x == y for x, y in zip(ra, rb)) / len(pairs),
    }
