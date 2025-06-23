from datasets import Dataset
from dotenv import load_dotenv
import os
import warnings
from peft import get_peft_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import RewardTrainer, RewardConfig  # Hugging Face TRL for reward modeling
from typing import Any

from libs import preference_dataset_manager
from constants import PREFERENCE_DATASET_PATH
from prepare_common import LORA_CONFIG, QUANTIZATION_CONFIG
import utils


# Filter out specific torch warnings that can be safely ignored.
warnings.filterwarnings("ignore", message=".*does not support bfloat16 compilation natively.*")
warnings.filterwarnings("ignore", message=".*'pin_memory' argument is set as true but no accelerator is found.*")


load_dotenv()
REWARD_MODEL_BASE = os.getenv("REWARD_MODEL_BASE")
if not REWARD_MODEL_BASE:
    raise ValueError("Missing `REWARD_MODEL_BASE` env var. Please set it in your .env file.")
REWARD_MODEL_PATH = os.getenv("REWARD_MODEL_PATH")
if not REWARD_MODEL_PATH:
    raise ValueError("Missing `REWARD_MODEL_PATH` env var. Please set it in your .env file.")


def load_preference_dataset() -> Dataset:
    utils.print_horizontal_line("═", "Preference Dataset")

    preference_dataset = preference_dataset_manager.read()
    num_pairs = len(preference_dataset.comparison_pairs)
    print(f"Info: Loaded Preference Dataset with {num_pairs} comparison pairs from `{PREFERENCE_DATASET_PATH}`.")

    print("Info: Converting preference data to Reward Model training format...")
    pairs = []
    for item in preference_dataset.comparison_pairs:
        if not item.accepted or not item.rejected:
            continue  # skip if any required field is missing (should not happen in well-formed data)
        pairs.append(
            {
                "chosen": item.accepted,  # the human-approved (accepted) output
                "rejected": item.rejected[0],  # the first rejected output for comparison
                # Note: 'input' (original prompt) is not included, as RewardTrainer expects only chosen/rejected columns:contentReference[oaicite:10]{index=10}.
            }
        )
    reward_dataset = Dataset.from_list(pairs)
    print(f"Info: Prepared dataset with {len(reward_dataset)} preference comparisons.")

    return reward_dataset


def train_reward_model(reward_dataset: Dataset, tokenizer: Any):
    utils.print_horizontal_line("═", "Normalizer Model Training")

    utils.print_horizontal_line("─", "Model Base")

    # Load the model with a classification head for a single reward score output
    base_model = AutoModelForSequenceClassification.from_pretrained(
        REWARD_MODEL_BASE,
        num_labels=1,  # single-logit reward head
        quantization_config=QUANTIZATION_CONFIG,  # 4-bit load
        trust_remote_code=False,
    )

    model = get_peft_model(base_model, LORA_CONFIG)

    training_args = RewardConfig(
        eval_strategy="no",  # no evaluation during training (only training dataset is used)
        gradient_accumulation_steps=1,
        learning_rate=3e-6,
        logging_steps=10,
        max_length=512,  # max sequence length for tokenizer (truncation length)
        num_train_epochs=3,
        output_dir=REWARD_MODEL_PATH,
        per_device_train_batch_size=1,
        remove_unused_columns=False,  # do not drop dataset columns (use custom collator for chosen/rejected)
        save_strategy="epoch",
    )
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=reward_dataset,
        processing_class=tokenizer,  # tokenizer will handle processing (tokenization & padding) of 'chosen'/'rejected'
    )

    utils.print_horizontal_line("─", "RLHF Fine-Tuning")

    trainer.train()

    utils.print_horizontal_line("─", "Model & Tokenizer Saving")

    trainer.save_model()  # saves the model to output_dir
    tokenizer.save_pretrained(REWARD_MODEL_PATH)  # save tokenizer (including new pad token) to output_dir
    print(f"Info: Reward model saved to `{REWARD_MODEL_PATH}`.")


def main():
    utils.print_horizontal_line("─", "Tokenizer Loading")

    tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_BASE, padding_side="right", trust_remote_code=False)
    # Add a padding token if not already present (especially for GPT/OPT models)
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    reward_dataset = load_preference_dataset()
    train_reward_model(reward_dataset, tokenizer)
    utils.print_horizontal_line("═")


if __name__ == "__main__":
    main()
