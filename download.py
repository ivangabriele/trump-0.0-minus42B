from functools import reduce
from os import path
import os
import re
import time
from typing import List, Optional
import pendulum
import requests
import validators
from constants import DOWNLOAD_API_URL, POSTS_DATA_DIR_PATH
from _types.database_types import DatabasePost
from _types.json_data_types import JsonPost, RequestParams, ResponseBody, SortOrder
import libs
import utils


def _has_page(page: int) -> bool:
    data_file_path = path.join(path.dirname(__file__), POSTS_DATA_DIR_PATH, str(page).rjust(4, "0") + ".json")

    return path.exists(data_file_path)


def _get_page(page: int) -> ResponseBody:
    params: RequestParams = RequestParams(page=page, sort="date", sort_order=SortOrder.ASC)
    response = requests.get(DOWNLOAD_API_URL, params=params)

    return ResponseBody.model_validate(response.json())


def _save_page(page: int, body: ResponseBody) -> None:
    data_file_path = path.join(path.dirname(__file__), POSTS_DATA_DIR_PATH, str(page).rjust(4, "0") + ".json")
    with open(data_file_path, "w") as file:
        file.write(body.model_dump_json(indent=2, by_alias=True))


def _download_posts() -> int:
    os.makedirs(POSTS_DATA_DIR_PATH, exist_ok=True)

    page = 1
    print("Info: Fetching page 0001 / ????...")
    body = _get_page(page)
    if not _has_page(page):
        _save_page(1, body)
    total_pages = body.meta.page_count

    while page < total_pages:
        page += 1
        if _has_page(page) and _has_page(page + 1):  # if there is a next page, we assume the current page is full
            print(f"Info: Page {str(page).rjust(4, '0')} / {total_pages} already fetched and full, skipping...")
            continue
        print(f"Info: Fetching page {str(page).rjust(4, '0')} / {total_pages}...")

        body = _get_page(page)
        _save_page(page, body)

        time.sleep(2)

    return total_pages


def _get_posts(total_pages: int) -> List[JsonPost]:
    posts = []
    page = 1
    while page <= total_pages:
        data_file_path = path.join(path.dirname(__file__), POSTS_DATA_DIR_PATH, str(page).rjust(4, "0") + ".json")

        with open(data_file_path, "r", encoding="utf-8") as file:
            body = ResponseBody.model_validate_json(file.read())
            posts.extend(body.data)

        page += 1

    return posts


def _to_database_post(post: JsonPost) -> Optional[DatabasePost]:
    if post.text is None or not post.text.strip():
        return None

    post_id = utils.generate_post_id(post.date, post.text)
    post_date = pendulum.parse(post.date).in_timezone("UTC").to_iso8601_string()  # type: ignore
    post_raw_text = post.text

    return DatabasePost(id=post_id, date=post_date, raw_text=post_raw_text)


def _nornalize_punctuation_chars(post: JsonPost) -> JsonPost:
    post_raw_text = post.text
    if post_raw_text is None:
        return post

    post_raw_text = post_raw_text.replace("…", "...")
    post_raw_text = post_raw_text.replace("’", "'")
    post_raw_text = re.sub(r"[“”]", '"', post_raw_text)
    # Figure dash, em dash, en dash, horizontal bar => en dash
    post_raw_text = re.sub(r"[‒—–―]", "–", post_raw_text)
    post_raw_text = re.sub(r"\\n", " ", post_raw_text)
    post_raw_text = re.sub(r"\s+", " ", post_raw_text)
    post_raw_text = re.sub(r"^[\s\-–]+", "", post_raw_text)

    post.text = post_raw_text

    return post


def _remove_line_breaks(post: JsonPost) -> JsonPost:
    post_raw_text = post.text
    if post_raw_text is None:
        return post

    # Remove all line breaks and space-like characters
    post_raw_text = re.sub(r"\s+", " ", post_raw_text)
    post_raw_text = post_raw_text.strip()

    post.text = post_raw_text

    return post


def _is_media(post_raw_text: str) -> bool:
    if post_raw_text.endswith("."):
        post_raw_text = post_raw_text[:-1]

    # Return true for `pic.twitter.com/xYz123`
    if re.match(r"^pic\.twitter\.com/[a-zA-Z0-9]+$", post_raw_text):
        return True

    return post_raw_text == "[Image]" or post_raw_text == "[Video]" or post_raw_text == "[QuickTime Video]"


def _is_only_handle_repost(post_raw_text: str) -> bool:
    # `RT @akaPR0B0SS`
    return re.match(r"^RT\s+@([a-zA-Z0-9_]+)\.?$", post_raw_text) is not None


def _is_url(post_raw_text: str) -> bool:
    if not post_raw_text.startswith("http://") and not post_raw_text.startswith("https://"):
        return False

    if post_raw_text.endswith("."):
        post_raw_text = post_raw_text[:-1]

    # Remove all spaces and space-like characters
    post_raw_text = re.sub(r"\s+", "", post_raw_text)

    return validators.url(post_raw_text) == True  # noqa: E712


def _merge_posts(merged_posts: List[JsonPost], post: JsonPost) -> List[JsonPost]:
    if not merged_posts:
        return [post]

    last_post = merged_posts[-1]
    if last_post.text is None or post.text is None:
        return merged_posts

    if post.text.startswith("...") or post.text.startswith("...."):
        post_text = re.sub(r"^\.+\s*", "", post.text)
        last_post.text += " " + post_text

        return merged_posts

    merged_posts.append(post)

    return merged_posts


def _insert_posts(database_posts: List[DatabasePost]) -> None:
    db_connection = libs.initialize_database()
    cursor = db_connection.cursor()

    cursor.execute("DELETE FROM posts")
    db_connection.commit()
    cursor.executemany(
        """
        INSERT INTO posts (id, date, raw_text)
        VALUES (?, ?, ?)
        """,
        [(database_post.id, database_post.date, database_post.raw_text) for database_post in database_posts],
    )
    db_connection.commit()

    db_connection.close()


def main():
    total_pages = _download_posts()
    posts = _get_posts(total_pages)
    formatted_posts = [
        _nornalize_punctuation_chars(_remove_line_breaks(post)) for post in posts if post.text is not None
    ]
    text_posts = [
        formatted_post
        for formatted_post in formatted_posts
        if formatted_post.text is not None
        and not _is_media(formatted_post.text)
        and not _is_only_handle_repost(formatted_post.text)
        and not _is_url(formatted_post.text)
    ]
    merged_posts = reduce(_merge_posts, text_posts, [])

    optional_database_posts = [_to_database_post(merged_post) for merged_post in merged_posts]
    database_posts = [database_post for database_post in optional_database_posts if database_post is not None]
    # Remove duplicate IDs
    database_posts = list({post.id: post for post in database_posts}.values())
    database_posts.sort(key=lambda post: post.date)

    _insert_posts(database_posts)

    print(f"Info: Downloaded and inserted {len(database_posts)} posts.")


if __name__ == "__main__":
    main()
