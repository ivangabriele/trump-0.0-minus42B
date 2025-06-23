import math
from accelerate import PartialState
from datasets import Dataset, NamedSplit
from dotenv import load_dotenv
import os
from peft import get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from trl import PPOTrainer, PPOConfig
from typing import Any, Tuple
import warnings


from _types.normalizer_types import PpoDataset, PpoDatasetPair, PpoDatasetPick
from libs import preference_dataset_manager
from constants import PREFERENCE_DATASET_PATH
from prepare_common import LORA_CONFIG, QUANTIZATION_CONFIG
import utils


# Filter out specific torch warnings that can be safely ignored.
warnings.filterwarnings("ignore", message=".*does not support bfloat16 compilation natively.*")
warnings.filterwarnings("ignore", message=".*'pin_memory' argument is set as true but no accelerator is found.*")


load_dotenv()
NORMALIZER_MODEL_BASE = os.getenv("NORMALIZER_MODEL_BASE")
if not NORMALIZER_MODEL_BASE:
    raise ValueError("Missing `NORMALIZER_MODEL_BASE` env var. Please set it in your .env file.")
NORMALIZER_MODEL_PATH = os.getenv("NORMALIZER_MODEL_PATH")
if not NORMALIZER_MODEL_PATH:
    raise ValueError("Missing `NORMALIZER_MODEL_PATH` env var. Please set it in your .env file.")
REWARD_MODEL_PATH = os.getenv("REWARD_MODEL_PATH")
if not REWARD_MODEL_PATH:
    raise ValueError("Missing `REWARD_MODEL_PATH` env var. Please set it in your .env file.")


def load_preference_dataset(tokenizer: Any) -> Tuple[Dataset, Dataset]:
    utils.print_horizontal_line("━", "Preference Dataset Loading")

    preference_dataset = preference_dataset_manager.read()
    num_pairs = len(preference_dataset.comparison_pairs)
    print(f"Info: Loaded Preference Dataset with {num_pairs} comparison pairs from `{PREFERENCE_DATASET_PATH}`.")

    print("Info: Converting preference data to Reward Model training format...")
    normalized_dataset_pairs: PpoDataset = []
    for item in preference_dataset.comparison_pairs:
        normalized_dataset_pairs.append(
            PpoDatasetPair(
                prompt=item.input,  # the original prompt (input)
                chosen=[
                    PpoDatasetPick(content=item.input, role="user"),
                    PpoDatasetPick(content=item.accepted, role="assistant"),
                ],  # the human-approved (accepted) output
                rejected=[
                    PpoDatasetPick(content=item.input, role="user"),
                    PpoDatasetPick(content=item.rejected[0], role="assistant"),
                ],  # the first rejected output for comparison
            )
        )
    normalized_dataset = Dataset.from_list(
        [pair.model_dump() for pair in normalized_dataset_pairs], split=NamedSplit("descriptiveness")
    )
    # normalized_dataset: Dataset = load_dataset(
    #     "trl-internal-testing/descriptiveness-sentiment-trl-style",
    #     name=None,
    #     split="descriptiveness",
    # )  # type: ignore
    print(f"Info: Prepared dataset with {len(normalized_dataset)} preference comparisons.")
    # print(normalized_dataset[:2])

    eval_dataset_length = math.floor(len(normalized_dataset) / 2)
    train_dataset = normalized_dataset.select(range(len(normalized_dataset) - eval_dataset_length))
    eval_dataset = normalized_dataset.select(
        range(len(normalized_dataset) - eval_dataset_length, len(normalized_dataset))
    )
    print(f"Info: Split dataset into {len(train_dataset)} training and {len(eval_dataset)} evaluation samples.")

    def prepare_dataset(dataset, tokenizer):
        """pre-tokenize the dataset before training; only collate during training"""

        def tokenize(element):
            outputs = tokenizer(
                element["prompt"],
                padding=False,
            )

            return {"input_ids": outputs["input_ids"]}

        return dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names,
            # num_proc=training_args.dataset_num_proc,
        )

    # Compute that only on the main process for faster data processing.
    # see: https://github.com/huggingface/trl/pull/1255
    with PartialState().local_main_process_first():
        train_dataset = prepare_dataset(train_dataset, tokenizer)
        eval_dataset = prepare_dataset(eval_dataset, tokenizer)

    return train_dataset, eval_dataset


def train_normalizer_model(train_dataset: Dataset, eval_dataset: Dataset, tokenizer: Any):
    utils.print_horizontal_line("━", "Generator Model Training")

    # Load the causal LM model (policy) with potential half-precision and device mapping
    model_kwargs = {
        "attn_implementation": "eager",
        "device_map": "auto",
        "quantization_config": QUANTIZATION_CONFIG,  # no quantization for the normalizer model
        "trust_remote_code": False,
    }
    base_model = AutoModelForCausalLM.from_pretrained(NORMALIZER_MODEL_BASE, **model_kwargs)
    # If new tokens were added to the tokenizer, resize model embeddings
    # base_model.resize_token_embeddings(len(tokenizer))
    model = get_peft_model(base_model, LORA_CONFIG)

    # Load the reward model and value model
    print(f"Info: Loading reward model from `{REWARD_MODEL_PATH}`...")
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        REWARD_MODEL_PATH,
        device_map="auto",
        num_labels=1,
        quantization_config=QUANTIZATION_CONFIG,  # no quantization for the normalizer model
        trust_remote_code=False,
    )
    # Ensure the reward model has the pad token embeddings if needed
    if tokenizer.pad_token_id is not None and hasattr(reward_model, "resize_token_embeddings"):
        reward_model.resize_token_embeddings(len(tokenizer))
    # Initialize the value model (critic) with same architecture as reward model (sequence classification, 1 output)
    value_model = AutoModelForSequenceClassification.from_pretrained(
        REWARD_MODEL_PATH,
        device_map="auto",
        num_labels=1,
        quantization_config=QUANTIZATION_CONFIG,  # no quantization for the normalizer model
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is not None and hasattr(value_model, "resize_token_embeddings"):
        value_model.resize_token_embeddings(len(tokenizer))

    # Set up PPO training configuration using PPOConfig
    ppo_config = PPOConfig(
        learning_rate=3e-06,
        batch_size=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        # log_with=None,  # no specific logging integration (could use 'tensorboard' or 'wandb')
        # logging_steps=500,
        missing_eos_penalty=1.0,  # encourage model to properly end sequences by penalizing missing EOS
        num_train_epochs=1.0,
        # num_ppo_epochs=4,
        output_dir=NORMALIZER_MODEL_PATH,
        reward_model_path=REWARD_MODEL_PATH,  # type: ignore[arg-type]
        # save_strategy="steps",  # save at the end of each epoch
        seed=42,
    )
    # Note: PPOConfig will internally compute total episodes based on num_train_epochs and dataset length:contentReference[oaicite:50]{index=50}.

    # Initialize the PPOTrainer
    ppo_trainer = PPOTrainer(
        args=ppo_config,
        model=model,
        ref_model=None,
        reward_model=reward_model,
        value_model=value_model,
        processing_class=tokenizer,  # Tokenizer for data collator (handles padding of 'query' inputs)
        eval_dataset=eval_dataset,
        train_dataset=train_dataset,
        peft_config=LORA_CONFIG,
    )

    # Train the normalizer model with PPO
    print("Info: Starting PPO fine-tuning...")
    try:
        ppo_trainer.train()
    finally:
        # skip any error
        print("Info: PPO training completed (or interrupted).")

    # Save the fine-tuned model and tokenizer
    print("Info: PPO training complete. Saving the normalizer model...")
    ppo_trainer.save_model(NORMALIZER_MODEL_PATH)  # save the policy model (with LoRA adapters if any)
    tokenizer.save_pretrained(NORMALIZER_MODEL_PATH)  # save tokenizer (including added tokens) to output_dir
    print(
        f"Info: Generator model saved to `{NORMALIZER_MODEL_PATH}`. You can use this model for inference or further training."
    )


def main():
    tokenizer = AutoTokenizer.from_pretrained(NORMALIZER_MODEL_BASE, padding_side="left", trust_remote_code=False)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    # if tokenizer.chat_template is None:
    #     tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    train_dataset, eval_dataset = load_preference_dataset(tokenizer)
    train_normalizer_model(train_dataset, eval_dataset, tokenizer)
    utils.print_horizontal_line("═")


if __name__ == "__main__":
    main()
