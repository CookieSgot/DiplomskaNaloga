import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate
import datasets as ds
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train = pd.read_csv('data/headlines_train.csv').sample(100, random_state=42)
test = pd.read_csv('data/headlines_test.csv').sample(10, random_state=42)
eval = pd.read_csv('data/headlines_val.csv').sample(10, random_state=42)

dataset = ds.DatasetDict({
    "train": ds.Dataset.from_pandas(train),
    "test": ds.Dataset.from_pandas(test),
    "eval": ds.Dataset.from_pandas(eval)
})

tokenizer = AutoTokenizer.from_pretrained("EMBEDDIA/sloberta")
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, return_tensors="pt")

tokenized_datasets = dataset.map(tokenize_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained("EMBEDDIA/sloberta", num_labels=2)
training_args = TrainingArguments(output_dir="test_trainer")

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["eval"],
    compute_metrics=compute_metrics
)

trainer.train()