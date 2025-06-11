import json
from os import path
import sqlite3
from pathlib import Path

from pydantic import ValidationError

from data_types import ResponseBody
import libs
import utils

_POSTS_DATA_DIR_NAME = "data/posts"
_SQLITE_DB_PATH = "data/posts.db"

DEV_FILE_LIMIT = 2


def initialize_database() -> sqlite3.Connection:
    db_connection = sqlite3.connect(_SQLITE_DB_PATH)

    cursor = db_connection.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS posts (
            id TEXT PRIMARY KEY,
            date TEXT NOT NULL,
            raw_text TEXT
            clean_text TEXT
        )
        """
    )
    db_connection.commit()

    return db_connection


def clean_page(page: int, conn: sqlite3.Connection):
    data_file_path = path.join(path.dirname(__file__), _POSTS_DATA_DIR_NAME, str(page).rjust(4, "0") + ".json")

    print("\n╔" + "═" * 118 + "╗")
    print(f"║ {str(page).rjust(4, '0')}" + " " * (120 - 7) + "║")
    print("╚" + "═" * 118 + "╝")

    try:
        with open(data_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            response_body = ResponseBody.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as e:
        utils.print_error_and_exit(f"Error processing `{data_file_path}`.", e)
        return

    cursor = conn.cursor()
    posts_to_insert = []
    for post in response_body.data:
        raw_text = post.text if post.text is not None else ""
        if not raw_text.strip():
            continue

        post_id = utils.generate_post_id(post.date, raw_text)
        cursor.execute("SELECT id FROM posts WHERE id = ?", (post_id,))
        if cursor.fetchone():
            continue

        clean_text = libs.clean_post_text_with_llm(raw_text)

        print("\n┏" + "━" * 118 + "┓")
        print(f"┃ ID:\t\t{post_id}" + " " * 39 + "┃")
        print(f"┃ Raw date:\t{post.date}" + " " * (120 - 17 - len(post.date)) + "┃")
        print(f"┃ Clean Date:\t{post.date}" + " " * (120 - 20 - len(post.date)) + "┃")
        print("┠" + "─" * 118 + "┨")
        utils.print_boxed_text(raw_text, 120, "┃")
        print("┠" + "─" * 118 + "┨")
        print(f"┃ {clean_text}")
        print("┗" + "━" * 118 + "┛")

        posts_to_insert.append((post_id, post.date, raw_text, clean_text))

    # if posts_to_insert:
    #     cursor.executemany(
    #         "INSERT INTO posts (id, date, raw_text, clean_text) VALUES (?, ?, ?)",
    #         posts_to_insert
    #     )
    #     conn.commit()
    #     print(f"  > Inserted {len(posts_to_insert)} new posts.")


def main():
    print("Info: Initializing database...")
    db_connection = initialize_database()

    page = 0
    total_pages = len(list(Path(path.join(path.dirname(__file__), _POSTS_DATA_DIR_NAME)).glob("*.json")))
    print(f"Info: Found {total_pages} pages.")
    while page < total_pages:
        page += 1

        clean_page(page, db_connection)

    db_connection.close()


if __name__ == "__main__":
    main()
