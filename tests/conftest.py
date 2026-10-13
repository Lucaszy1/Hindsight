import uuid

import pytest
from pymongo.errors import PyMongoError

from annlab.db import ensure_indexes, get_db


@pytest.fixture
def db():
    """Throwaway database. Skips if MongoDB is not running."""
    name = f"annlab_test_{uuid.uuid4().hex[:8]}"
    database = get_db(name=name)
    try:
        database.client.admin.command("ping")
    except PyMongoError:
        pytest.skip("MongoDB not available")
    ensure_indexes(database)
    yield database
    database.client.drop_database(name)
