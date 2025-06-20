import argparse
from typing import Any, Tuple
import warnings
import torch

from accelerate import PartialState
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from trl import PPOTrainer, PPOConfig
from datasets import Dataset, load_dataset

from libs import preference_dataset_manager
from constants import GENERATOR_MODEL, GENERATOR_MODEL_DIR_PATH, PREFERENCE_DATASET_PATH, REWARD_MODEL_DIR_PATH
import utils

# Suppress specific benign warnings (e.g., bf16 not natively compiled, pin_memory on CPU)
warnings.filterwarnings("ignore", message=".*does not support bfloat16 compilation natively.*")
warnings.filterwarnings("ignore", message=".*'pin_memory' argument is set as true but no accelerator is found.*")


def load_preference_dataset(args, tokenizer: Any) -> Tuple[Dataset, Dataset]:
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
                "prompt": item.input,  # the original prompt (input)
                "chosen": item.accepted,  # the human-approved (accepted) output
                "rejected": item.rejected[0],  # the first rejected output for comparison
            }
        )
    reward_dataset = Dataset.from_list(pairs)
    reward_dataset: Dataset = load_dataset(
        "trl-internal-testing/descriptiveness-sentiment-trl-style",
        name=None,
        split="descriptiveness",
    )  # type: ignore
    print(f"Info: Prepared dataset with {len(reward_dataset)} preference comparisons.")

    # Initialize the policy model for the generator
    eval_samples = 1
    train_dataset = reward_dataset.select(range(len(reward_dataset) - eval_samples))
    eval_dataset = reward_dataset.select(range(len(reward_dataset) - eval_samples, len(reward_dataset)))
    dataset_text_field = "prompt"

    def prepare_dataset(dataset, tokenizer):
        """pre-tokenize the dataset before training; only collate during training"""

        def tokenize(element):
            outputs = tokenizer(
                element[dataset_text_field],
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


def train_generator_model(args, train_dataset: Dataset, eval_dataset: Dataset, tokenizer: Any):
    utils.print_horizontal_line("━", "Generator Model Training")

    # Load the causal LM model (policy) with potential half-precision and device mapping
    model_kwargs = {}
    # Set precision: use bfloat16 or float16 if specified (otherwise default to float32)
    if args.bf16:
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif args.fp16:
        model_kwargs["torch_dtype"] = torch.float16
    # Enable accelerate auto device placement
    model_kwargs["device_map"] = "auto"
    # Use eager attention implementation if supported (helps on some GPUs):contentReference[oaicite:47]{index=47}
    model_kwargs["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(GENERATOR_MODEL, **model_kwargs)
    # If new tokens were added to the tokenizer, resize model embeddings
    # model.resize_token_embeddings(len(tokenizer))

    # 4. Load the reward model and value model
    print(f"Info: Loading reward model from `{REWARD_MODEL_DIR_PATH}`...")
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        REWARD_MODEL_DIR_PATH,
        num_labels=1,
        trust_remote_code=False,
        **({"torch_dtype": "torch.bfloat16"} if args.bf16 else {}),
    )
    # Ensure the reward model has the pad token embeddings if needed
    if tokenizer.pad_token_id is not None and hasattr(reward_model, "resize_token_embeddings"):
        reward_model.resize_token_embeddings(len(tokenizer))
    # Initialize the value model (critic) with same architecture as reward model (sequence classification, 1 output)
    value_model = AutoModelForSequenceClassification.from_pretrained(
        REWARD_MODEL_DIR_PATH,
        num_labels=1,
        trust_remote_code=False,
        **({"torch_dtype": "torch.bfloat16"} if args.bf16 else {}),
    )
    if tokenizer.pad_token_id is not None and hasattr(value_model, "resize_token_embeddings"):
        value_model.resize_token_embeddings(len(tokenizer))

    # Set up PPO training configuration using PPOConfig
    ppo_config = PPOConfig(
        learning_rate=3e-06,
        batch_size=None,
        per_device_train_batch_size=64,
        gradient_accumulation_steps=1,
        # log_with=None,  # no specific logging integration (could use 'tensorboard' or 'wandb')
        logging_steps=500,
        num_train_epochs=3.0,
        num_ppo_epochs=4,
        # Encourage model to properly end sequences by penalizing missing EOS
        missing_eos_penalty=1.0,
        reward_model_path=REWARD_MODEL_DIR_PATH,
        fp16=False,
        bf16=False,
        no_cuda=False,
        seed=42,
        save_strategy="steps",  # save at the end of each epoch
        output_dir=GENERATOR_MODEL_DIR_PATH,
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
        peft_config=None,
    )

    # Train the generator model with PPO
    print("Info: Starting PPO fine-tuning...")
    ppo_trainer.train()

    # Save the fine-tuned model and tokenizer
    print("Info: PPO training complete. Saving the generator model...")
    ppo_trainer.save_model(GENERATOR_MODEL_DIR_PATH)  # save the policy model (with LoRA adapters if any)
    tokenizer.save_pretrained(GENERATOR_MODEL_DIR_PATH)  # save tokenizer (including added tokens) to output_dir
    print(
        f"Info: Generator model saved to `{GENERATOR_MODEL_DIR_PATH}`. You can use this model for inference or further training."
    )


def main():
    parser = argparse.ArgumentParser(
        description="PPO fine-tune a generator model using human preference data and a trained reward model"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=GENERATOR_MODEL,
        help="HuggingFace model name or path for the base generator model (e.g., 'facebook/opt-125m').",
    )
    parser.add_argument(
        "--reward_model_path",
        type=str,
        default=REWARD_MODEL_DIR_PATH,
        help="Path to the pretrained reward model to use for computing rewards (e.g., 'models/reward').",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=GENERATOR_MODEL_DIR_PATH,
        help="Directory to save the PPO fine-tuned generator model.",
    )
    parser.add_argument("--learning_rate", type=float, default=3e-6, help="Learning rate for PPO fine-tuning.")
    parser.add_argument(
        "--num_train_epochs", type=int, default=3, help="Number of passes through the dataset (PPO training epochs)."
    )
    parser.add_argument(
        "--num_ppo_epochs", type=int, default=4, help="Number of PPO optimization epochs per batch of experiences."
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=64,
        help="Batch size (number of prompts) per device for PPO training.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (to effectively increase batch size if needed).",
    )
    parser.add_argument("--logging_steps", type=int, default=10, help="Log training metrics every N update steps.")
    parser.add_argument(
        "--save_strategy",
        type=str,
        choices=["no", "steps", "epoch"],
        default="epoch",
        help="Checkpoint saving strategy (default: save at end of each epoch).",
    )
    parser.add_argument("--fp16", action="store_true", help="Use FP16 precision if set (requires supported GPU).")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision if set (requires supported GPU).")
    parser.add_argument(
        "--no_cuda", action="store_true", help="Force training on CPU if set (otherwise GPU is used when available)."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum generation length (in tokens) for outputs (also used as max model input length). Longer sequences will be truncated or stopped.",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL, padding_side="left", trust_remote_code=False)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    # if tokenizer.chat_template is None:
    #     tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    train_dataset, eval_dataset = load_preference_dataset(args, tokenizer)
    train_generator_model(args, train_dataset, eval_dataset, tokenizer)
    utils.print_horizontal_line("═")


if __name__ == "__main__":
    main()
