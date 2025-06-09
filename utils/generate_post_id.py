import hashlib


def generate_post_id(post_date: str, post_text: str) -> str:
    return hashlib.sha256(f"{post_date}{post_text}".encode("utf-8")).hexdigest()
