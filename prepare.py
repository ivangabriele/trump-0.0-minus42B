from peft import LoraConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from transformers.generation.configuration_utils import GenerationConfig
from trl import (  # type: ignore
    AutoModelForCausalLMWithValueHead,
    PPOTrainer,
    PPOConfig,
    RewardTrainer,
    RewardConfig,
)
import warnings

from _types.generator_types import PreferenceDataset
from constants import GENERATOR_MODEL, GENERATOR_MODEL_DIR_PATH, REWARD_MODEL, REWARD_MODEL_DIR_PATH
from datasets import Dataset  # type: ignore
from libs import preference_dataset_manager


BATCH_SIZE = 4
NUM_EPOCHS = 3
LEARNING_RATE = 1.41e-5


def _to_reward_model_dataset(dataset: PreferenceDataset) -> Dataset:
    print("Info: Preparing Reward Model Dataset...")

    pairs = []
    for item in dataset.comparison_pairs:
        pairs.append(
            {
                "input": item.input,
                "chosen": item.accepted,
                "rejected": item.rejected[0],  # Use first rejected attempt
            }
        )
    return Dataset.from_list(pairs)


def _train_reward_model():
    print("Info: Training Reward Model...")

    preference_dataset = preference_dataset_manager.read()
    reward_model_dataset = _to_reward_model_dataset(preference_dataset)

    tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL)
    # **Crucially, also set it in the model's configuration**
    # DistilBERT does not have a default pad token, so we set it to the unk token
    # which is a standard practice for this model architecture.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    model = AutoModelForSequenceClassification.from_pretrained(
        REWARD_MODEL, num_labels=1, torch_dtype=torch.bfloat16, device_map="auto"
    )

    # https://huggingface.co/docs/trl/v0.18.1/reward_trainer#trl.RewardConfig
    training_args = RewardConfig(
        eval_strategy="no",
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        num_train_epochs=NUM_EPOCHS,
        output_dir=REWARD_MODEL_DIR_PATH,
        per_device_train_batch_size=BATCH_SIZE,
        remove_unused_columns=False,
        save_strategy="epoch",
    )
    trainer = RewardTrainer(
        model=model, args=training_args, processing_class=tokenizer, train_dataset=reward_model_dataset
    )
    trainer.train()
    trainer.save_model()

    return model, tokenizer


def _to_ppo_dataset(dataset: PreferenceDataset) -> Dataset:
    print("Info: Preparing Proximal Policy Optimization Dataset...")

    return Dataset.from_list([{"query": item.input} for item in dataset.comparison_pairs])


def _train_with_ppo(reward_model, reward_tokenizer):
    print("Info: Starting PPO Training...")

    preference_dataset = preference_dataset_manager.read()
    ppo_dataset = _to_ppo_dataset(preference_dataset)

    tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    # We must tokenize the dataset before passing it to the PPOTrainer.
    # The dataloader needs the 'input_ids' column.
    def tokenize(example):
        example["input_ids"] = tokenizer(example["query"], truncation=True).input_ids

        return example

    tokenized_ppo_dataset = ppo_dataset.map(tokenize, batched=False, remove_columns=["query"])

    generation_config = GenerationConfig.from_pretrained(GENERATOR_MODEL)
    generation_config.pad_token_id = tokenizer.pad_token_id
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        GENERATOR_MODEL, attn_implementation="eager", torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.generation_config = generation_config
    model.base_model_prefix = "pretrained_model"

    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        GENERATOR_MODEL, attn_implementation="eager", torch_dtype=torch.bfloat16, device_map="auto"
    )

    # Configure LoRA (Low-Rank Adaptation) PEFT (Parameter-Efficient Fine-Tuning)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    ppo_config = PPOConfig(
        batch_size=BATCH_SIZE,
        # gradient_accumulation_steps=4,
        learning_rate=LEARNING_RATE,
        mini_batch_size=1,
        # per_device_train_batch_size=1,
        reward_model_path=GENERATOR_MODEL,
    )
    ppo_trainer = PPOTrainer(
        args=ppo_config,
        model=model,
        ref_model=ref_model,
        reward_model=reward_model,
        value_model=model,
        processing_class=tokenizer,
        train_dataset=tokenized_ppo_dataset,
        peft_config=lora_config,
    )

    print("Info: Starting PPO training...")
    ppo_trainer.train()

    print("Info: PPO training finished. Saving model...")
    ppo_trainer.save_model(GENERATOR_MODEL_DIR_PATH)
    tokenizer.save_pretrained(GENERATOR_MODEL_DIR_PATH)

    print(f"Info: Training complete. Generator model saved to `{GENERATOR_MODEL_DIR_PATH}`.")


def main():
    # Filter out specific torch warnings that can be safely ignored.
    warnings.filterwarnings(
        "ignore",
        message=".*does not support bfloat16 compilation natively.*",
    )

    reward_model, reward_tokenizer = _train_reward_model()
    _train_with_ppo(reward_model, reward_tokenizer)


if __name__ == "__main__":
    main()
