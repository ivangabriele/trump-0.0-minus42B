from os import path
import sqlite3
from typing import List

from _types.database_types import DatabasePost
from constants import SQLITE_DB_FILE_PATH


class _Database:
    _connection: sqlite3.Connection

    @property
    def connection(self) -> sqlite3.Connection:
        if not hasattr(self, "_connection"):
            raise RuntimeError("Database connection has not been initialized.")

        return self._connection

    @property
    def cursor(self) -> sqlite3.Cursor:
        if not hasattr(self, "_connection"):
            raise RuntimeError("Database connection has not been initialized.")

        return self._connection.cursor()

    def __init__(self):
        self._connect()

    def __del__(self):
        if not hasattr(self, "_connection"):
            raise RuntimeError("Database connection has not been initialized.")

        self._connection.close()

    def _connect(self) -> None:
        database_file_path = path.join(path.dirname(__file__), "..", SQLITE_DB_FILE_PATH)

        self._connection = sqlite3.connect(database_file_path)

        cursor = self._connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS posts (
                id TEXT PRIMARY KEY,
                date TEXT NOT NULL,
                raw_text TEXT NOT NULL,
                clean_text TEXT
            )
            """
        )
        self._connection.commit()

    def get_posts(self) -> List[DatabasePost]:
        cursor = self._connection.cursor()

        cursor.execute("SELECT * FROM posts ORDER BY date")

        rows = cursor.fetchall()
        print(f"Retrieved {len(rows)} posts from the database.")
        cursor.close()

        return [DatabasePost(id=row[0], date=row[1], raw_text=row[2], clean_text=row[3]) for row in rows]

    def insert_posts(self, posts: List[DatabasePost]) -> None:
        cursor = self._connection.cursor()

        cursor.executemany(
            "INSERT OR IGNORE INTO posts (id, date, raw_text, clean_text) VALUES (?, ?, ?, ?)",
            [(post.id, post.date, post.raw_text, post.clean_text) for post in posts],
        )

        self._connection.commit()
        cursor.close()

    def has_post_with_raw_text(self, raw_text: str) -> bool:
        cursor = self._connection.cursor()

        cursor.execute("SELECT 1 FROM posts WHERE raw_text = ?", (raw_text,))
        exists = cursor.fetchone() is not None
        cursor.close()

        return exists


database = _Database()
