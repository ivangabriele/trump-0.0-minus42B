from os import path
import sqlite3

from constants import SQLITE_DB_FILE_PATH


def initialize_database() -> sqlite3.Connection:
    database_file_path = path.join(path.dirname(__file__), "..", SQLITE_DB_FILE_PATH)

    db_connection = sqlite3.connect(database_file_path)

    cursor = db_connection.cursor()
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
    db_connection.commit()

    return db_connection
