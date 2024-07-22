from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForSequenceClassification
import torch
import pandas as pd
from sklearn import metrics

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

df = pd.read_csv('../data/headlines_train.csv')#.sample(10, random_state=42)

positive_samples = df[df['label'] == 1].sample(5, random_state=42)
negative_samples = df[df['label'] == 0].sample(5, random_state=42)

train = pd.concat([positive_samples, negative_samples]).reset_index(drop=True)
train = train.sample(frac=1, random_state=42).reset_index(drop=True)

train.to_csv('explain_headlines.csv', index=False)