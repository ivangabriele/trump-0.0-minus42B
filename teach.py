import argparse
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
_MAX_ATTEMPTS = 3


def _load_preference_dataset() -> PreferenceDataset:
    preference_dataset_path = path.join(path.dirname(__file__), PREFERENCE_DATASET_PATH)

    if not path.exists(preference_dataset_path):
        print(f"Warning: Preference Dataset file `{preference_dataset_path}` does not exist.")

        return PreferenceDataset(sft_pairs=[], comparison_pairs=[])

    with open(preference_dataset_path, "r") as file:
        data = json.load(file)

    return PreferenceDataset.model_validate(data)


def _load_posts() -> List[DatabasePost]:
    db_connection = libs.initialize_database()

    cursor = db_connection.cursor()
    cursor.execute("SELECT * FROM posts")

    rows = cursor.fetchall()
    print(rows[0])

    cursor.close()
    db_connection.close()

    # TODO Use SQLAlchemy?
    return [DatabasePost(id=row[0], date=row[1], raw_text=row[2], clean_text=row[3]) for row in rows]


def _collect_human_feedback(post: DatabasePost, sample_size: int, sample_index: int) -> Optional[Tuple[str, List[str]]]:
    rejected_texts: List[str] = []

    print("")
    print("╔" + "═" * 118 + "╗")
    utils.print_boxed_text(
        f"SAMPLE {str(sample_index + 1).rjust(len(str(sample_size)), '0')} / {sample_size}", 120, "║"
    )
    print("╟" + "─" * 118 + "╢")
    utils.print_boxed_text(f"ID:   {post.id}", 120, "║")
    utils.print_boxed_text(f"Date: {post.date}", 120, "║")
    print("╚" + "═" * 118 + "╝")
    print("")

    utils.print_horizontal_line("═", "ORIGINAL TEXT")
    print(post.raw_text)

    for attempt in range(_MAX_ATTEMPTS):
        if not post.raw_text:
            return None

        proposed_text = libs.clean_post_text_with_llm(post.raw_text, attempt=attempt)
        utils.print_horizontal_line("━", f"GENERATOR LLM PROPOSAL {attempt + 1}")
        print(proposed_text)

        print("")
        print("> Accept? Press [y] (yes), [n] (no) or [s] (skip)... ", end="", flush=True)
        choice = utils.get_input_key()
        print("\r" + " " * 80, end="", flush=True)
        print("\r", end="", flush=True)

        if choice == "y":
            print("✔️ Accepted.")
            utils.print_horizontal_line("═")

            return proposed_text, rejected_texts
        elif choice == "n":
            print("❌ Rejected.")

            rejected_texts.append(proposed_text)
        else:
            print("⭕ Skipped.")
            utils.print_horizontal_line("═")

            return None

    utils.print_horizontal_line("━", "HUMAN PROPOSAL")
    expected_text = input("> ").strip()
    utils.print_horizontal_line("═")

    return expected_text, rejected_texts


def _save_preference_dataset(preference_dataset: PreferenceDataset, batch_page: int):
    preference_dataset_path = path.join(path.dirname(__file__), PREFERENCE_DATASET_PATH)

    with open(preference_dataset_path, "w") as file:
        file.write(preference_dataset.model_dump_json(indent=2, by_alias=True))

    print("")
    print(f"Info: Preference Dataset saved to `{preference_dataset_path}`.")


def main():
    parser = argparse.ArgumentParser("simple_example")
    parser.add_argument("sample_size", help="The number of posts to sample", type=int, nargs="?")
    args = parser.parse_args()
    if args.sample_size is None:
        utils.print_error_and_exit("Please provide the number of posts to sample: `python teach.py <sample_size>`.")

    preference_dataset = _load_preference_dataset()
    print(
        f"Info: Loaded the existing Preference Dataset with {len(preference_dataset.sft_pairs)} SFT pairs and {len(preference_dataset.comparison_pairs)} comparison pairs."
    )
    preference_dataset_ids = map(lambda x: x.id, preference_dataset.comparison_pairs)

    all_posts = _load_posts()
    filtered_posts = [post for post in all_posts if post.id not in preference_dataset_ids]
    sampled_posts = random.sample(filtered_posts, args.sample_size)
    print(f"Info: Loaded a ramdom sample of {len(sampled_posts)} posts.")

    for sample_index, post in enumerate(sampled_posts):
        result = _collect_human_feedback(post, args.sample_size, sample_index)
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
