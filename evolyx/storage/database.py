"""Database connection and initialization utilities."""

import sqlite3
from pathlib import Path


class DatabaseManager:
    """Manages SQLite database connections and schema migrations."""

    def __init__(self, db_path: str = "artifacts/evolyx.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
