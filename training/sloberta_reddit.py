import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate
import datasets as ds
from torch import nn

train = pd.read_csv('../data/reddit_train.csv')#.sample(100, random_state=42)
test = pd.read_csv('../data/reddit_test.csv')#.sample(10, random_state=42)
eval = pd.read_csv('../data/reddit_val.csv')#.sample(10, random_state=42)

dataset = ds.DatasetDict({
    "train": ds.Dataset.from_pandas(train),
    "test": ds.Dataset.from_pandas(test),
    "eval": ds.Dataset.from_pandas(eval)
})

modelname = "EMBEDDIA/sloberta"
tokenizer = AutoTokenizer.from_pretrained(modelname)
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", max_length=160, truncation=True, return_tensors="pt")

tokenized_datasets = dataset.map(tokenize_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(modelname, num_labels=2)
training_args = TrainingArguments(output_dir="test_trainer", eval_strategy="epoch", num_train_epochs=5, load_best_model_at_end=True, save_strategy="epoch", learning_rate=2e-6)

metric = evaluate.load("accuracy")

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["eval"],
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model("sloberta_reddit")