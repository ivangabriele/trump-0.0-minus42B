import argparse
from datasets import Dataset
from dotenv import load_dotenv
import os
import warnings
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import RewardTrainer, RewardConfig  # Hugging Face TRL for reward modeling

from libs import preference_dataset_manager
from constants import PREFERENCE_DATASET_PATH
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
    utils.print_horizontal_line("━", "Preference Dataset Loading")
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


def train_reward_model(args, reward_dataset: Dataset):
    print(f"Info: Loading base model `{args.model_name_or_path}` and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    # Add a padding token if not already present (especially for GPT/OPT models)
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.padding_side = "right"  # pad on the right for training:contentReference[oaicite:11]{index=11}
    # Load the model with a classification head for a single reward score output
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, num_labels=1, trust_remote_code=False
    )
    # Resize model embeddings in case new tokens were added to the tokenizer
    model.resize_token_embeddings(len(tokenizer))

    training_args = RewardConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        eval_strategy="no",  # no evaluation during training (only training dataset is used)
        remove_unused_columns=False,  # do not drop dataset columns (use custom collator for chosen/rejected)
        fp16=args.fp16,
        bf16=args.bf16,
        no_cuda=args.no_cuda,
        max_length=args.max_length,  # max sequence length for tokenizer (truncation length)
    )
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=reward_dataset,
        processing_class=tokenizer,  # tokenizer will handle processing (tokenization & padding) of 'chosen'/'rejected'
    )

    utils.print_horizontal_line("━", "Reward Model Training")
    trainer.train()

    print("Info: Training complete. Saving reward model to disk...")
    trainer.save_model()  # saves the model to output_dir
    tokenizer.save_pretrained(args.output_dir)  # save tokenizer (including new pad token) to output_dir
    print(
        f"Info: Reward model saved to `{args.output_dir}`. You can now use this model for PPO fine-tuning or inference."
    )


def main():
    parser = argparse.ArgumentParser(description="Train a reward model from human preference data")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=REWARD_MODEL_BASE,
        help="HuggingFace model name or path for the base reward model (e.g., 'facebook/opt-125m').",
    )
    parser.add_argument(
        "--output_dir", type=str, default=REWARD_MODEL_PATH, help="Directory to save the trained reward model."
    )
    parser.add_argument("--learning_rate", type=float, default=3e-6, help="Learning rate for reward model fine-tuning.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument(
        "--per_device_train_batch_size", type=int, default=1, help="Batch size per device (GPU/CPU) for training."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (to effectively increase batch size).",
    )
    parser.add_argument("--logging_steps", type=int, default=10, help="Log training metrics every N update steps.")
    parser.add_argument(
        "--save_strategy",
        type=str,
        choices=["no", "steps", "epoch"],
        default="epoch",
        help="Checkpoint saving strategy (default: save at end of each epoch).",
    )
    parser.add_argument(
        "--fp16", action="store_true", help="Use FP16 precision training if set (requires supported GPU)."
    )
    parser.add_argument(
        "--bf16", action="store_true", help="Use bfloat16 precision training if set (requires supported GPU)."
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Force training on CPU if set (otherwise GPU acceleration is used when available).",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenization/padding of inputs. Longer sequences will be truncated.",
    )
    args = parser.parse_args()

    reward_dataset = load_preference_dataset()
    train_reward_model(args, reward_dataset)
    utils.print_horizontal_line("═")


if __name__ == "__main__":
    main()
