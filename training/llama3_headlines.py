import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorWithPadding
import numpy as np
import evaluate
import datasets as ds
from sklearn import metrics
from torch import nn, bfloat16
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

train = pd.read_csv('../data/headlines_train.csv')#.sample(1000, random_state=42)
test = pd.read_csv('../data/headlines_test.csv')#.sample(100, random_state=42)
eval = pd.read_csv('../data/headlines_val.csv')#.sample(100, random_state=42)

dataset = ds.DatasetDict({
    "train": ds.Dataset.from_pandas(train),
    "test": ds.Dataset.from_pandas(test),
    "eval": ds.Dataset.from_pandas(eval)
})

quantization_config = BitsAndBytesConfig(
    load_in_8bit = True # enable 8-bit quantization
)

lora_config = LoraConfig(
    r = 16, # the dimension of the low-rank matrices
    lora_alpha = 32, # scaling factor for LoRA activations vs pre-trained weight activations
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    lora_dropout = 0.05, # dropout probability of the LoRA layers
    bias = 'none', # wether to train bias weights, set to 'none' for attention layers
    task_type = 'SEQ_CLS'
)

modelname = "../../Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(modelname)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", max_length=160, truncation=True, return_tensors="pt")

tokenized_datasets = dataset.map(tokenize_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(modelname, num_labels=2, quantization_config=quantization_config)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False
model.config.pretraining_tp = 1

collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir="test_trainer",
    eval_strategy="steps",
    eval_steps=400,
    num_train_epochs=2,
    load_best_model_at_end=True,
    save_strategy="steps",
    save_steps=400,
    learning_rate=1e-4,
    logging_steps=10
)

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
    tokenizer=tokenizer,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["eval"],
    compute_metrics=compute_metrics,
    data_collator=collate_fn
)

trainer.train()
trainer.save_model("llama3_headlines")
predictions = trainer.predict(test_dataset=tokenized_datasets["test"])

prediction = np.argmax(predictions.predictions, axis=-1)
prediction = prediction.tolist()
recall = metrics.recall_score(dataset["test"]["label"],prediction)
precision = metrics.precision_score(dataset["test"]["label"],prediction)
f1_score = metrics.f1_score(dataset["test"]["label"],prediction)
accuracy = metrics.accuracy_score(dataset["test"]["label"],prediction)
loss = metrics.log_loss(dataset["test"]["label"],prediction)

print('Loss:',loss)
print('Accuracy:',accuracy)
print('Precision:',precision)
print('Recall:',recall)
print('f1 score:',f1_score)