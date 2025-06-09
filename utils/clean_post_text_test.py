from utils import clean_post_text


def test_clean_post_text_with_url():
    assert (
        clean_post_text(
            "Donald Trump reads Top Ten Financial Tips on Late Show with David Letterman: http://tinyurl.com/ooafwn - Very funny!"
        )
        == "Donald Trump reads Top Ten Financial Tips on Late Show with David Letterman - Very funny!"
    )


def test_clean_post_text_with_wrong_punctuation_1():
    assert (
        clean_post_text(
            "Could this be my newest apprentice? http://www.youtube.com/user/MattressSerta ...Enter the contest .. . http://www.facebook.com/sertamattress"
        )
        == "Could this be my newest apprentice? Enter the contest."
    )


def test_clean_post_text_with_wrong_punctuation_2():
    assert (
        clean_post_text(
            "Enter the contest.... http://www.facebook.com/sertamattress...and stay at Trump International Hotel Las Vegas"
        )
        == "Enter the contest... and stay at Trump International Hotel Las Vegas"
    )


def test_clean_post_text_afterfix_for_number():
    assert clean_post_text("raised $700,000 on Celebrity Apprentice") == "raised $700,000 on Celebrity Apprentice"


def test_clean_post_text_afterfix_for_time():
    assert clean_post_text("9-11 p.m. on NBC, ET") == "9-11 p.m. on NBC, ET"
