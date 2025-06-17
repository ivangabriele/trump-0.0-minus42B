from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
import torch
from transformers.generation.configuration_utils import GenerationConfig
from trl import (  # type: ignore
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


LEARNING_RATE = 3e-6


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


def _train_reward_model(should_train: bool) -> tuple:
    print("Info: Training Reward Model...")

    preference_dataset = preference_dataset_manager.read()
    reward_model_dataset = _to_reward_model_dataset(preference_dataset)

    reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL)
    reward_tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    reward_model = AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL, trust_remote_code=False)

    # https://huggingface.co/docs/trl/v0.18.1/reward_trainer#trl.RewardConfig
    training_args = RewardConfig(
        eval_strategy="no",
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        num_train_epochs=3,
        output_dir=REWARD_MODEL_DIR_PATH,
        per_device_train_batch_size=1,
        remove_unused_columns=False,
        save_strategy="epoch",
    )
    trainer = RewardTrainer(
        model=reward_model, args=training_args, processing_class=reward_tokenizer, train_dataset=reward_model_dataset
    )
    if should_train:
        trainer.train()
        trainer.save_model()

    return reward_model, reward_tokenizer


def _to_ppo_dataset(dataset: PreferenceDataset) -> Dataset:
    print("Info: Preparing Proximal Policy Optimization Dataset...")

    return Dataset.from_list([{"query": item.input} for item in dataset.comparison_pairs])


def _train_with_ppo(reward_model, reward_tokenizer):
    print("Info: Starting PPO Training...")

    preference_dataset = preference_dataset_manager.read()
    ppo_dataset = _to_ppo_dataset(preference_dataset)

    tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    generation_config = GenerationConfig.from_pretrained(GENERATOR_MODEL)
    generation_config.pad_token_id = tokenizer.pad_token_id
    model = AutoModelForCausalLM.from_pretrained(
        GENERATOR_MODEL, attn_implementation="eager", torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.generation_config = generation_config
    model.base_model_prefix = "pretrained_model"

    value_model = AutoModelForSequenceClassification.from_pretrained(
        REWARD_MODEL, trust_remote_code=False, num_labels=1
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
        batch_size=64,
        gradient_accumulation_steps=1,
        learning_rate=LEARNING_RATE,
        missing_eos_penalty=1.0,
        per_device_train_batch_size=64,
        reward_model_path=GENERATOR_MODEL,
    )
    ppo_trainer = PPOTrainer(
        args=ppo_config,
        model=model,
        ref_model=None,
        reward_model=reward_model,
        value_model=value_model,
        processing_class=tokenizer,
        train_dataset=ppo_dataset,
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
    warnings.filterwarnings(
        "ignore",
        message=".*'pin_memory' argument is set as true but no accelerator is found.*",
    )

    reward_model, reward_tokenizer = _train_reward_model(False)
    _train_with_ppo(reward_model, reward_tokenizer)


if __name__ == "__main__":
    main()
