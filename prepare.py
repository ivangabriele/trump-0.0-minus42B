import json
from os import path
from pathlib import Path
from typing import List, Optional, Tuple
import random

from pydantic import ValidationError
from download_types import ResponseBody, DataItem
import libs
from prepare_types import FeedbackData, FeedbackDataComparisonPair, FeedbackDataStfPair
import utils

# Configuration
_POSTS_DATA_DIR_PATH = "data/posts"
_RLHF_DATA_DIR_PATH = "data/rlhf"
_SAMPLE_SIZE = 10
_MAX_ATTEMPTS = 3


def load_posts() -> List[DataItem]:
    posts: List[DataItem] = []
    for json_file in Path(_POSTS_DATA_DIR_PATH).glob("*.json"):
        with open(json_file, "r") as f:
            try:
                data = ResponseBody.model_validate_json(f.read())
            except ValidationError as e:
                print(f"Error: Invalid JSON format in file `{json_file}`.")
                print(repr(e.errors()[0]))
                continue
            posts.extend(data.data)

    return posts


def collect_feedback(post: DataItem, sample_index: int) -> Optional[Tuple[str, List[str]]]:
    if not post.text:
        return None

    rejected_texts: List[str] = []

    print("")
    print("╔" + "═" * 118 + "╗")
    utils.print_boxed_text(
        f"POST {str(sample_index + 1).rjust(len(str(_SAMPLE_SIZE)), '0')} / {_SAMPLE_SIZE}", 120, "║"
    )
    print("╟" + "─" * 118 + "╢")
    utils.print_boxed_text(f"Date:     {post.date}", 120, "║")
    utils.print_boxed_text(f"Platform: {post.platform}", 120, "║")
    print("╟" + "─" * 118 + "╢")
    utils.print_boxed_text(post.text, 120, "║")
    print("╚" + "═" * 118 + "╝")

    for attempt in range(_MAX_ATTEMPTS):
        if not post.text:
            return None

        proposed_text = libs.clean_post_text_with_llm(post.text, attempt=attempt)
        print("")
        print("┏" + "━" * 118 + "┓")
        utils.print_boxed_text(f"PROPOSAL {attempt + 1}", 120, "┃")
        print("┠" + "─" * 118 + "┨")
        utils.print_boxed_text(proposed_text, 120, "┃")
        print("┠" + "─" * 118 + "┨")

        print("┃ Accept? Press [y] (yes), [n] (no) or [s] (skip)... ", end="", flush=True)
        choice = utils.get_input_key()
        print("\r" + " " * 120, end="", flush=True)
        print("\r", end="", flush=True)

        if choice == "y":
            utils.print_boxed_text("Human Feedback: Accepted.", 120, "┃")
            print("┗" + "━" * 118 + "┛")

            return proposed_text, rejected_texts
        elif choice == "n":
            utils.print_boxed_text("Human Feedback: Rejected.", 120, "┃")

            rejected_texts.append(proposed_text)
        else:
            utils.print_boxed_text("Human Feedback: Skipped.", 120, "┃")
            print("┗" + "━" * 118 + "┛")

            return None

    print("")
    print("┃ Please write the expected output for this post:")
    expected_text = input("┃ > ").strip()
    print("┗" + "━" * 118 + "┛")

    return expected_text, rejected_texts


def save_rlhf_data(feedback_data: FeedbackData, batch_page: int):
    output_path = path.join(path.dirname(__file__), _RLHF_DATA_DIR_PATH, f"batch_{str(batch_page).rjust(4, '0')}.json")

    with open(output_path, "w") as file:
        json.dump(feedback_data, file, indent=2)

    print(f"Info: RLHF data saved to `{output_path}`.")


def main():
    all_posts = load_posts()
    sampled_posts = random.sample(all_posts, _SAMPLE_SIZE)
    print(f"Info: Loaded a ramdom sample of {len(sampled_posts)} posts.")

    feedback_data = FeedbackData(sft_pairs=[], comparison_pairs=[])
    for sample_index, post in enumerate(sampled_posts):
        if not post.text:
            continue

        result = collect_feedback(post, sample_index)
        if result is None:
            continue

        accepted_text, rejected_texts = result
        post_id = utils.generate_post_id(post.date, post.text)
        if accepted_text:
            feedback_data.sft_pairs.append(FeedbackDataStfPair(id=post_id, input=post.text, output=accepted_text))

            if rejected_texts:
                feedback_data.comparison_pairs.append(
                    FeedbackDataComparisonPair(
                        id=post_id, input=post.text, accepted=accepted_text, rejected=rejected_texts
                    )
                )

    print("")
    save_rlhf_data(feedback_data, batch_page=1)


if __name__ == "__main__":
    main()
