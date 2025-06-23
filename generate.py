from _types.database_types import DatabasePost
from libs import database
from libs.post_normalizer import PostNormalizer


def _clean_post(post_normalizer: PostNormalizer, database_post: DatabasePost, index: int, total: int) -> DatabasePost:
    print(f"Normalizing post {str(index + 1).rjust(len(str(total)), '0')} / {total}...")

    clean_text = post_normalizer.normalize(database_post.raw_text)

    return DatabasePost(
        id=database_post.id, date=database_post.date, raw_text=database_post.raw_text, clean_text=clean_text
    )


def main():
    print("Info: Initializing database...")

    database_posts = database.get_posts()
    post_normalizer = PostNormalizer()
    total = len(database_posts)
    updated_posts = [
        _clean_post(post_normalizer, database_post, index, total) for index, database_post in enumerate(database_posts)
    ]
    database.update_posts(updated_posts)


if __name__ == "__main__":
    main()
