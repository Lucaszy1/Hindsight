import os

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.environ.get("MONGO_DB", "annlab")
ITEMS = "items"
ANNOTATIONS = "annotations"
PRIMARY_ANNOTATOR = os.environ.get("PRIMARY_ANNOTATOR", "primary")
DATA_URL = (
    "https://huggingface.co/datasets/Anthropic/hh-rlhf/resolve/main/"
    "helpful-base/train.jsonl.gz"
)
RAW_PATH = os.environ.get("RAW_PATH", "data/raw/helpful-base-train.jsonl.gz")
SEED = 42
POSITIVE_RATING = 4  # rating >= this counts as the "helpful" class
