import json
from os import path
from typing import List, Optional, Tuple
import random

from constants import PREFERENCE_DATASET_PATH
from _types.database_types import DatabasePost
import libs
from _types.generator_types import PreferenceDataset, PreferenceDatasetComparisonPair, PreferenceDatasetStfPair
import utils

# Configuration
_SAMPLE_SIZE = 10
_MAX_ATTEMPTS = 3


def _load_posts() -> List[DatabasePost]:
    db_connection = libs.initialize_database()

    cursor = db_connection.cursor()
    cursor.execute("SELECT * FROM posts")

    rows = cursor.fetchall()

    cursor.close()
    db_connection.close()

    return [DatabasePost.model_validate(row) for row in rows]


def _collect_human_feedback(post: DatabasePost, sample_index: int) -> Optional[Tuple[str, List[str]]]:
    rejected_texts: List[str] = []

    print("")
    print("╔" + "═" * 118 + "╗")
    utils.print_boxed_text(
        f"POST {str(sample_index + 1).rjust(len(str(_SAMPLE_SIZE)), '0')} / {_SAMPLE_SIZE}", 120, "║"
    )
    print("╟" + "─" * 118 + "╢")
    utils.print_boxed_text(f"ID:   {post.id}", 120, "║")
    utils.print_boxed_text(f"Date: {post.date}", 120, "║")
    print("╟" + "─" * 118 + "╢")
    utils.print_boxed_text(post.raw_text, 120, "║")
    print("╚" + "═" * 118 + "╝")

    for attempt in range(_MAX_ATTEMPTS):
        if not post.raw_text:
            return None

        proposed_text = libs.clean_post_text_with_llm(post.raw_text, attempt=attempt)
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


def _load_preference_dataset() -> PreferenceDataset:
    preference_dataset_path = path.join(path.dirname(__file__), PREFERENCE_DATASET_PATH)

    if not path.exists(preference_dataset_path):
        print(f"Warning: Preference Dataset file `{preference_dataset_path}` does not exist.")

        return PreferenceDataset(sft_pairs=[], comparison_pairs=[])

    with open(preference_dataset_path, "r") as file:
        data = json.load(file)

    return PreferenceDataset.model_validate(data)


def _save_preference_dataset(feedback_data: PreferenceDataset, batch_page: int):
    preference_dataset_path = path.join(path.dirname(__file__), PREFERENCE_DATASET_PATH)

    with open(preference_dataset_path, "w") as file:
        json.dump(feedback_data, file, indent=2)

    print("")
    print(f"Info: Preference Dataset saved to `{preference_dataset_path}`.")


def main():
    preference_dataset = _load_preference_dataset()
    print(
        f"Info: Loaded the existing Preference Dataset with {len(preference_dataset.sft_pairs)} SFT pairs and {len(preference_dataset.comparison_pairs)} comparison pairs."
    )
    preference_dataset_ids = map(lambda x: x.id, preference_dataset.comparison_pairs)

    all_posts = _load_posts()
    filtered_posts = [post for post in all_posts if post.id not in preference_dataset_ids]
    sampled_posts = random.sample(filtered_posts, _SAMPLE_SIZE)
    print(f"Info: Loaded a ramdom sample of {len(sampled_posts)} posts.")

    for sample_index, post in enumerate(sampled_posts):
        result = _collect_human_feedback(post, sample_index)
        if result is None:
            continue

        accepted_text, rejected_texts = result
        post_id = utils.generate_post_id(post.date, post.raw_text)
        if accepted_text:
            preference_dataset.sft_pairs.append(
                PreferenceDatasetStfPair(id=post_id, input=post.raw_text, output=accepted_text)
            )

            if rejected_texts:
                preference_dataset.comparison_pairs.append(
                    PreferenceDatasetComparisonPair(
                        id=post_id, input=post.raw_text, accepted=accepted_text, rejected=rejected_texts
                    )
                )

    _save_preference_dataset(preference_dataset, batch_page=1)


if __name__ == "__main__":
    main()
