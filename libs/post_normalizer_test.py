from .post_normalizer import PostNormalizer


post_normalizer = PostNormalizer()


def test_post_normalizer_normalize_when_url():
    assert (
        post_normalizer.normalize("The Blue Monster at Trump National Doral. http://fb.me/1mtpR0yqs")
        == "The Blue Monster at Trump National Doral."
    )


def test_post_normalizer_normalize_when_url_and_missing_punctuation():
    assert (
        post_normalizer.normalize(
            "Trump Towers, Istanbul, Sisli will be one of the country’s top landmarks http://bit.ly/UD6fK0"
        )
        == "Trump Towers, Istanbul, Sisli will be one of the country’s top landmarks."
    )
