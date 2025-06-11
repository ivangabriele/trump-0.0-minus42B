import sqlite3
import nltk  # type: ignore
from nltk.tokenize import word_tokenize  # type: ignore
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from constants import SQLITE_DB_FILE_PATH


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


def load_texts_from_db(db_path: str) -> list[str]:
    db_connection = sqlite3.connect(db_path)
    cursor = db_connection.cursor()
    cursor.execute("SELECT raw_text FROM posts ORDER BY date")
    rows = cursor.fetchall()
    db_connection.close()
    return [r[0] for r in rows]


texts = load_texts_from_db(SQLITE_DB_FILE_PATH)
print(f"Loaded {len(texts)} posts from `{SQLITE_DB_FILE_PATH}`.")


# model_name = "facebook/opt-1.3b"
model_name = "facebook/opt-125m"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = TextDataset(tokenizer, texts)
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
)
trainer = Trainer(model=model, args=training_args, train_dataset=dataset, processing_class=tokenizer)
print(model)
trainer.train()
