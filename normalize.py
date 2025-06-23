import argparse

from _types.database_types import DatabasePost
from libs import database
from libs.post_normalizer import PostNormalizer
import utils


def _clean_post(
    post_normalizer: PostNormalizer, database_post: DatabasePost, is_forced: bool, index: int, total: int
) -> None:
    utils.print_horizontal_line("═", f"{str(index + 1).rjust(len(str(total)), '0')} / {total}")
    if not is_forced and database_post.clean_text:
        print("Info: Already normalized. Skipping.")

        return

    utils.print_horizontal_line("─", "Before")
    print(database_post.raw_text)
    clean_text = post_normalizer.normalize(database_post.raw_text)
    utils.print_horizontal_line("─", "After")
    print(clean_text)

    updated_post = DatabasePost(
        id=database_post.id, date=database_post.date, raw_text=database_post.raw_text, clean_text=clean_text
    )
    database.update_post(updated_post)


def main():
    args_parser = argparse.ArgumentParser(description="Normalize posts in the database.")
    args_parser.add_argument("-f", "--force", help="Force normalization even if clean text already exists.")
    args = args_parser.parse_args()

    database_posts = database.get_posts()
    post_normalizer = PostNormalizer()
    total = len(database_posts)
    [
        _clean_post(post_normalizer, database_post, args.force, index, total)
        for index, database_post in enumerate(database_posts)
    ]


if __name__ == "__main__":
    main()
