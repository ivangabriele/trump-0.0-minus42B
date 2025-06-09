import sqlite3
import time
from typing import List

from bs4 import BeautifulSoup
import pendulum
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.remote.webdriver import WebDriver

import utils
import utils.print_error_and_exit

_SQLITE_DB_PATH = "posts.db"
_TARGET_URL = "https://rollcall.com/factbase-twitter/?platform=all&sort=date&sort_order=asc&page=1"


def initialize_database() -> sqlite3.Connection:
    conn = sqlite3.connect(_SQLITE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS posts (
            id TEXT PRIMARY KEY,
            date TEXT NOT NULL,
            text TEXT NOT NULL
        )
        """
    )
    conn.commit()

    return conn


def configure_driver(headless: bool = True) -> WebDriver:
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(options=chrome_options)

    return driver


def click_consent_button(driver: WebDriver) -> None:
    try:
        consent_button = driver.find_element("xpath", "//p[contains(@class, 'fc-button-label') and text()='Consent']")
        consent_button.click()
    except Exception as e:
        utils.print_error_and_exit("Consent button not found or could not be clicked.", e)

    time.sleep(2)  # Wait for the consent action to complete

    # Click on that close icon if it exists
    try:
        close_icon = driver.find_element(
            "xpath", "//div[contains(@id, 'close-icon-') and contains(@class, 'cursor-pointer')]"
        )
        close_icon.click()
    except Exception:
        return


def scroll_to_bottom(driver: WebDriver) -> None:
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)


def extract_posts(html: str, db_connection: sqlite3.Connection) -> List[tuple[str, str, str]]:
    soup = BeautifulSoup(html, "html.parser")

    post_wrapper_anchors = soup.find_all(lambda tag: tag.name == "span" and tag.get("x-text") == "`@${item.handle}`")
    post_wrappers = [
        post_wrapper_anchor.find_parent("div", class_="flex-1") for post_wrapper_anchor in post_wrapper_anchors
    ]
    print(f"Info: Found {len(post_wrappers)} posts.")

    posts: List[tuple[str, str, str]] = []
    for post_wrapper in post_wrappers:
        raw_date = post_wrapper.find(  # type: ignore
            lambda tag: tag.name == "span"
            and tag.has_attr("x-text")
            and tag.get("x-text").startswith("new Date(item.date)")  # type: ignore
        ).get_text()  # type: ignore
        if not raw_date:
            # print("    Warning: No date found for this post.")
            continue
        post_container = post_wrapper.find_all("template")[-1].find_next_sibling("div")  # type: ignore
        if not post_container:
            # print("    Warning: Empty post.")
            continue
        raw_post = post_container.decode_contents()  # type: ignore
        if not raw_post.strip():
            # print("    Warning: Empty post.")
            continue

        id = utils.generate_post_id(raw_date, raw_post)
        cursor = db_connection.cursor()
        cursor.execute("SELECT id FROM posts WHERE id = ?", (id,))
        existing_post = cursor.fetchone()
        if existing_post:
            # print("    Warning: Existing post.")
            continue
        print("\n┏" + "━" * 118 + "┓")
        print(f"│ ID:\t\t{id}")

        try:
            dt_et = pendulum.from_format(raw_date, "MMMM D, YYYY @ h:mm A [E][T]", tz="America/New_York")
            dt_utc = dt_et.in_timezone("UTC")
            clean_date = dt_utc.to_iso8601_string()
            print(f"│ Raw date:\t{raw_date}")
            print(f"│ Clean Date:\t{clean_date}")
        except Exception as e:
            utils.print_error_and_exit(f"Error parsing date: '{raw_date}'", e)

        print("┠" + "─" * 118 + "┨")
        print(f"│ {raw_post}")
        clean_post = utils.clean_post_text(raw_post)
        print("┠" + "─" * 118 + "┨")
        print(f"│ {clean_post}")
        print("┗" + "━" * 118 + "┛")

        posts.append((id, clean_date, clean_post))

    return posts


def save_posts(db_connection: sqlite3.Connection, posts: List[tuple[str, str, str]]) -> None:
    cursor = db_connection.cursor()
    cursor.executemany("INSERT OR IGNORE INTO posts (id, date, text) VALUES (?, ?, ?)", posts)
    db_connection.commit()


def main() -> None:
    """
    Main routine to scrape posts and store them in SQLite.

    Args:
        url (str): URL of the posts page.
        db_path (str): Path to the SQLite database file.
    """
    db_connection = initialize_database()
    driver = configure_driver(headless=False)

    try:
        driver.get(_TARGET_URL)
        click_consent_button(driver)
        time.sleep(2)

        page = 1
        while page <= 20:
            print("\n" + "=" * 120)
            print(f"PAGE {page}")
            print("=" * 120)

            page_html = driver.page_source
            posts = extract_posts(page_html, db_connection)
            save_posts(db_connection, posts)

            scroll_to_bottom(driver)

            page += 1
    except WebDriverException as e:
        print(f"WebDriver error: {e}")
    finally:
        driver.quit()
        db_connection.close()


if __name__ == "__main__":
    main()
