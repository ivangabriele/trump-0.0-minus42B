from .clean_post_text_with_llm import clean_post_text_with_llm


def test_clean_post_text_with_llm_when_url():
    assert (
        clean_post_text_with_llm("The Blue Monster at Trump National Doral. http://fb.me/1mtpR0yqs ")
        == "The Blue Monster at Trump National Doral."
    )


def test_clean_post_text_with_llm_when_url_and_missing_punctuation():
    assert (
        clean_post_text_with_llm(
            "Trump Towers, Istanbul, Sisli will be one of the country’s top landmarks http://bit.ly/UD6fK0 "
        )
        == "Trump Towers, Istanbul, Sisli will be one of the country’s top landmarks."
    )
