from pymongo import ASCENDING, MongoClient

from . import config


def get_db(uri=None, name=None):
    client = MongoClient(uri or config.MONGO_URI, serverSelectionTimeoutMS=5000)
    return client[name or config.DB_NAME]


def ensure_indexes(db):
    db[config.ITEMS].create_index([("source_idx", ASCENDING)], unique=True)
    db[config.ITEMS].create_index([("n_turns", ASCENDING)])
    # one live annotation per annotator per item; edits go to `revisions`
    db[config.ANNOTATIONS].create_index(
        [("item_id", ASCENDING), ("annotator", ASCENDING)], unique=True
    )
    db[config.ANNOTATIONS].create_index([("annotator", ASCENDING), ("rating", ASCENDING)])
