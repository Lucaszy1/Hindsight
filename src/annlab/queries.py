"""MongoDB aggregation queries over the annotation data."""
from . import config


def rating_distribution(db):
    return list(
        db[config.ANNOTATIONS].aggregate(
            [
                {"$group": {"_id": {"annotator": "$annotator", "rating": "$rating"}, "n": {"$sum": 1}}},
                {"$sort": {"_id.annotator": 1, "_id.rating": 1}},
            ]
        )
    )


def mean_rating_by_turns(db):
    return list(
        db[config.ANNOTATIONS].aggregate(
            [
                {"$lookup": {"from": config.ITEMS, "localField": "item_id", "foreignField": "_id", "as": "item"}},
                {"$unwind": "$item"},
                {"$group": {"_id": "$item.n_turns", "mean_rating": {"$avg": "$rating"}, "n": {"$sum": 1}}},
                {"$sort": {"_id": 1}},
            ]
        )
    )


def revised_annotations(db):
    """Annotations changed after first save (uses the revisions array)."""
    return list(
        db[config.ANNOTATIONS].aggregate(
            [
                {"$match": {"revisions.0": {"$exists": True}}},
                {"$project": {"item_id": 1, "annotator": 1, "n_revisions": {"$size": "$revisions"}}},
                {"$sort": {"n_revisions": -1}},
            ]
        )
    )


def disagreements(db, min_gap=2):
    """Items where two annotators differ by at least min_gap points."""
    return list(
        db[config.ANNOTATIONS].aggregate(
            [
                {"$group": {"_id": "$item_id", "lo": {"$min": "$rating"}, "hi": {"$max": "$rating"}, "n": {"$sum": 1}}},
                {"$match": {"n": {"$gte": 2}}},
                {"$addFields": {"gap": {"$subtract": ["$hi", "$lo"]}}},
                {"$match": {"gap": {"$gte": min_gap}}},
                {"$sort": {"gap": -1}},
            ]
        )
    )
