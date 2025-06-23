import argparse
import readline  # noqa: F401 - We import readline to enable line editing features in the terminal.
import random
from typing import List, Optional, Tuple


from _types.database_types import DatabasePost
from libs import database, PostNormalizer, preference_dataset_manager
from _types.normalizer_types import PreferenceDatasetComparisonPair
import utils


def _collect_human_feedback(
    post_normalizer: PostNormalizer, post: DatabasePost, sample_size: int, sample_index: int
) -> Optional[Tuple[str, List[str]]]:
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

    proposed_text = post_normalizer.normalize(post.raw_text)
    utils.print_horizontal_line("━", "UNTRAINED MODEL PROPOSAL")
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

        if not rejected_texts or proposed_text != rejected_texts[-1]:
            rejected_texts.append(proposed_text)
    else:
        print("⭕ Skipped.")
        utils.print_horizontal_line("═")

        return None

    utils.print_horizontal_line("━", "HUMAN PROPOSAL")
    expected_text = input("> ").strip()
    utils.print_horizontal_line("═")

    return expected_text, rejected_texts


def main():
    parser = argparse.ArgumentParser("simple_example")
    parser.add_argument("sample_size", help="The number of posts to sample", type=int, nargs="?")
    args = parser.parse_args()
    if args.sample_size is None:
        utils.print_error_and_exit("Please provide the number of posts to sample: `python teach.py <sample_size>`.")

    preference_dataset = preference_dataset_manager.read()
    print(
        f"Info: Loaded the existing Preference Dataset with {len(preference_dataset.comparison_pairs)} comparison pairs."
    )
    preference_dataset_ids = map(lambda x: x.id, preference_dataset.comparison_pairs)

    all_posts = database.get_posts()
    filtered_posts = [post for post in all_posts if post.id not in preference_dataset_ids]
    sampled_posts = random.sample(filtered_posts, args.sample_size)
    print(f"Info: Loaded a ramdom sample of {len(sampled_posts)} posts.")

    post_normalizer = PostNormalizer(with_base_model=True)

    for sample_index, post in enumerate(sampled_posts):
        result = _collect_human_feedback(post_normalizer, post, args.sample_size, sample_index)
        if result is None:
            continue

        accepted_text, rejected_texts = result
        post_id = utils.generate_post_id(post.date, post.raw_text)
        if rejected_texts:
            preference_dataset.comparison_pairs.append(
                PreferenceDatasetComparisonPair(
                    id=post_id, input=post.raw_text, accepted=accepted_text, rejected=rejected_texts
                )
            )

    preference_dataset_manager.write(preference_dataset)


if __name__ == "__main__":
    main()
