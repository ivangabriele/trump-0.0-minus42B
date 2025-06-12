import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer

from constants import GENERATOR_MODEL, GENERATOR_MODEL_DIR_PATH, REWARD_MODEL, REWARD_MODEL_DIR_PATH


import tqdm  # type: ignore
import torch
from transformers import AutoModelForSequenceClassification
from transformers.generation.configuration_utils import GenerationConfig
from trl import (  # type: ignore
    PPOTrainer,
    PPOConfig,
    RewardTrainer,
    RewardConfig,
)
from datasets import Dataset  # type: ignore
from _types.generator_types import PreferenceDataset
from peft import LoraConfig
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
        output_dir=REWARD_MODEL_DIR_PATH,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        eval_strategy="no",
        save_strategy="epoch",
        remove_unused_columns=False,
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
    model = AutoModelForCausalLM.from_pretrained(GENERATOR_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    model.generation_config = generation_config
    model.base_model_prefix = "model"

    # ref_model = AutoModelForCausalLM.from_pretrained(GENERATOR_MODEL, torch_dtype=torch.bfloat16, device_map="auto")

    ppo_config = PPOConfig(
        reward_model_path=GENERATOR_MODEL,
        batch_size=BATCH_SIZE,
        mini_batch_size=1,
        num_ppo_epochs=4,  # `4` is the default value
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    ppo_trainer = PPOTrainer(
        args=ppo_config,
        model=model,
        ref_model=None,
        reward_model=reward_model,
        value_model=model,
        processing_class=tokenizer,
        train_dataset=tokenized_ppo_dataset,
        peft_config=lora_config,
    )

    generation_kwargs = {
        "max_new_tokens": 512,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "remove_invalid_values": True,
    }

    for epoch in range(ppo_config.num_ppo_epochs):
        for batch in tqdm.tqdm(ppo_trainer.dataloader):
            query_tensors = batch["input_ids"]

            # 1. Access the generative model via `ppo_trainer.model.policy`.
            # 2. Call `.generate()` on this object.
            full_response_tensors = ppo_trainer.model.policy.generate(input_ids=query_tensors, **generation_kwargs)

            # 3. Slice the response to remove the prompt.
            response_tensors = full_response_tensors[:, query_tensors.shape[1] :]
            batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

            # --- The rest of the loop remains the same ---

            # 4. Decode the original query for reward calculation.
            queries = tokenizer.batch_decode(query_tensors, skip_special_tokens=True)

            # 5. Compute reward.
            texts = [query + response for query, response in zip(queries, batch["response"])]
            reward_inputs = reward_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(
                ppo_trainer.accelerator.device
            )

            with torch.no_grad():
                rewards = reward_model(**reward_inputs).logits

            # 6. Perform PPO step.
            stats = ppo_trainer.step(query_tensors, response_tensors, [reward[0] for reward in rewards])
            ppo_trainer.log_stats(stats, batch, [r[0] for r in rewards])

    print("Info: PPO training finished. Saving model...")
    ppo_trainer.save_model(GENERATOR_MODEL_DIR_PATH)

    print("Info: Tokenizer saving...")
    tokenizer.save_pretrained(GENERATOR_MODEL_DIR_PATH)


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
