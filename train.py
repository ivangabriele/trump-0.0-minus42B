import nltk  # type: ignore
from nltk.tokenize import word_tokenize  # type: ignore
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments


def preprocess_text(text):
    return word_tokenize(text.lower())


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, texts, max_length=128):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(
            text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        input_ids = inputs.input_ids.squeeze()
        attention_mask = inputs.attention_mask.squeeze()

        labels = input_ids.clone()

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


nltk.download("punkt")
nltk.download("punkt_tab")


sample_text = "Your sample text goes here. You can add more pages of text as needed."
tokens = preprocess_text(sample_text)
print(tokens)

# Load a small pretrained transformer model (e.g., tiny transformers like 'facebook/opt-125m')
model_name = "facebook/opt-125m"  # You can choose smaller models if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Print model summary to see its structure
print(model)


# Prepare your dataset
texts = [sample_text]  # Replace with actual text data

dataset = TextDataset(tokenizer, texts)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(model=model, args=training_args, train_dataset=dataset)

# Train the model
trainer.train()
