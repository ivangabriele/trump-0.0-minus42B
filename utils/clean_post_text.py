import re


def clean_post_text(post_text: str) -> str:
    post_text = re.sub(r"(\s*:\s*)?https?://\S+(?=\.\.\.)", "\\1", post_text)
    post_text = re.sub(r"(\s*:\s*)?https?://\S+", "", post_text)
    post_text = post_text.replace("\n", " ")
    post_text = re.sub(r"\s*(\.\.\.|\,|\;|\.|\!|\?)[\s\,\;\.\!\?]*", "\\1 ", post_text, flags=re.MULTILINE)
    post_text = re.sub(r"\s+", " ", post_text)

    post_text = re.sub(r"(\d+[\,\.])\s(\d+)", "\\1\\2", post_text, flags=re.IGNORECASE)
    post_text = re.sub(r"([aApP])\. ([mM])", "\\1.\\2", post_text, flags=re.IGNORECASE)

    return post_text.strip()
