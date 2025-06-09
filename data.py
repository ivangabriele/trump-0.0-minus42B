from os import path
import os
import time
import requests
from data_types import RequestParams, ResponseBody, SortOrder


_API_URL = "https://api.factsquared.com/json/factba.se-trump-social.php"
_DATA_DIR_NAME = "data"


def has_page(page: int) -> bool:
    data_file_path = path.join(path.dirname(__file__), _DATA_DIR_NAME, str(page).rjust(4, "0") + ".json")

    return path.exists(data_file_path)


def get_page(page: int) -> ResponseBody:
    params: RequestParams = RequestParams(page=page, sort="date", sort_order=SortOrder.ASC)
    response = requests.get(_API_URL, params=params)

    return ResponseBody.model_validate(response.json())


def save_page(page: int, body: ResponseBody) -> None:
    data_file_path = path.join(path.dirname(__file__), _DATA_DIR_NAME, str(page).rjust(4, "0") + ".json")
    with open(data_file_path, "w") as file:
        file.write(body.model_dump_json(indent=2))


def main():
    os.makedirs(_DATA_DIR_NAME, exist_ok=True)

    page = 1
    print("Info: Fetching page 0001 / ????...")
    body = get_page(page)
    if not has_page(page):
        save_page(1, body)
    total_pages = body.meta.page_count

    while page < total_pages:
        page += 1
        if has_page(page):
            print(f"Info: Page {str(page).rjust(4, '0')} / {total_pages} already fetched, skipping...")
            continue
        print(f"Info: Fetching page {str(page).rjust(4, '0')} / {total_pages}...")

        body = get_page(page)
        save_page(page, body)

        time.sleep(2)


if __name__ == "__main__":
    main()
