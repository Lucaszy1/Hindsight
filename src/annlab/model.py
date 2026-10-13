"""TF-IDF + logistic regression on the primary annotator's labels."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from . import config


def make_model():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )


def binarize(ratings):
    return [int(r >= config.POSITIVE_RATING) for r in ratings]


def load_labeled(db, annotator=config.PRIMARY_ANNOTATOR):
    """Join annotations to item text with an aggregation $lookup."""
    pipeline = [
        {"$match": {"annotator": annotator}},
        {"$lookup": {"from": config.ITEMS, "localField": "item_id", "foreignField": "_id", "as": "item"}},
        {"$unwind": "$item"},
        {"$project": {"_id": 0, "item_id": 1, "rating": 1, "text": "$item.text"}},
        {"$sort": {"item_id": 1}},
    ]
    return list(db[config.ANNOTATIONS].aggregate(pipeline))


def evaluate(texts, y, test_size=0.3, seed=config.SEED):
    Xtr, Xte, ytr, yte = train_test_split(texts, y, test_size=test_size, random_state=seed, stratify=y)
    model = make_model().fit(Xtr, ytr)
    pred = model.predict(Xte)
    return {
        "n_train": len(Xtr),
        "n_test": len(Xte),
        "accuracy": accuracy_score(yte, pred),
        "macro_f1": f1_score(yte, pred, average="macro"),
        "majority_baseline": max(sum(yte), len(yte) - sum(yte)) / len(yte),
    }
