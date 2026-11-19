"""The base install works without the [import] extra (#122, fix 3): the routes load, and a request is served,
with pyarrow and openpyxl unimportable. Run in a fresh interpreter, so no module this suite already loaded can
hide a top-level import of either."""
import subprocess
import sys
import textwrap

SCRIPT = textwrap.dedent(
    """
    import sys

    for name in ("pyarrow", "openpyxl"):
        sys.modules[name] = None  # importing it now raises ImportError, as on a host without the [import] extra

    from flask import Flask
    from flask_sqlalchemy import SQLAlchemy

    from dataset_library import init_dataset_library
    from dataset_library.models import LibraryBase


    class Cache:
        def __init__(self):
            self.values = {}

        def get(self, key):
            return self.values.get(key)

        def set(self, key, value, ttl):
            self.values[key] = value

        def add(self, key, value, ttl):
            if key in self.values:
                return False
            self.values[key] = value
            return True

        def delete(self, key):
            self.values.pop(key, None)


    app = Flask("base")
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite://"
    db = SQLAlchemy()
    db.init_app(app)
    init_dataset_library(
        app,
        db=db,
        login_required=lambda view: view,
        current_owner=lambda: "owner",
        task_types=["classification"],
        cache=Cache(),
        community_datasets=lambda: [],
        community_record=lambda record_id: None,
        community_download=lambda record_id, key, max_bytes: b"",
        community_base_url=lambda: "https://zenodo.org",
    )
    with app.app_context():
        LibraryBase.metadata.create_all(db.engine)

    client = app.test_client()
    assert client.get("/api/library/options").status_code == 200
    # Community is not switched on, so its search is refused without asking the Hub or reading any file.
    assert client.get("/api/library/search?source=community").status_code == 400
    assert client.get("/api/library/datasets/community/101").status_code == 404
    assert sys.modules["pyarrow"] is None and sys.modules["openpyxl"] is None
    print("ok")
    """
)


def test_the_routes_and_a_request_load_without_pyarrow_or_openpyxl():
    result = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, timeout=120)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
