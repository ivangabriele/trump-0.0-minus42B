from _types.database_types import DatabasePost
from libs import database
from libs import clean_post_text_with_llm
import utils


def _clean_post(database_post: DatabasePost, index: int, total: int) -> DatabasePost:
    print("\n╔" + "═" * 118 + "╗")
    utils.print_boxed_text(f"{str(index + 1).rjust(len(str(total)), '0')} / total", 120, "║")
    print("╚" + "═" * 118 + "╝")

    clean_text = clean_post_text_with_llm(database_post.raw_text)

    print("\n┏" + "━" * 118 + "┓")
    utils.print_boxed_text(f"ID:   {database_post.id}", 120, "┃")
    utils.print_boxed_text(f"Date: {database_post.date}", 120, "┃")
    print("┠" + "─" * 118 + "┨")
    utils.print_boxed_text(database_post.raw_text, 120, "┃")
    print("┠" + "─" * 118 + "┨")
    utils.print_boxed_text(clean_text, 120, "┃")
    print("┗" + "━" * 118 + "┛")

    return DatabasePost(
        id=database_post.id, date=database_post.date, raw_text=database_post.raw_text, clean_text=clean_text
    )


def main():
    print("Info: Initializing database...")

    database_posts = database.get_posts()
    total = len(database_posts)
    for index, database_post in enumerate(database_posts):
        _clean_post(database_post, index, total)


if __name__ == "__main__":
    main()
