from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForSequenceClassification
import torch
import pandas as pd

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

test = pd.read_csv('../data/reddit_test.csv')#.sample(10, random_state=42)
df = pd.read_csv('../data/reddit_train.csv')#.sample(10, random_state=42)

positive_samples = df[df['label'] == 1].sample(5, random_state=42)
negative_samples = df[df['label'] == 0].sample(5, random_state=42)

train = pd.concat([positive_samples, negative_samples]).reset_index(drop=True)
train = train.sample(frac=1, random_state=42).reset_index(drop=True)

quantization_config = BitsAndBytesConfig(
    load_in_8bit = True # enable 8-bit quantization
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

prompt = "Si prepoznavalec sarkastičnih reddit komentarjev. Odgovarjaš z samo 0, če podan komentar ni sarkastičen in z 1, če podan komentar je sarkastičen.\nNekaj primerov:\n"
#prompt = "Si prepoznavalec sarkastičnih člankov. Podan naslov članka označiš z 0, če ni sarkastičen in z 1, če je sarkastičen.\nNekaj primerov:\n"
prediction = []
for ind in train.index:
    example = "Komentar: '" + train["text"][ind] + "' Odgovor: " + str(train["label"][ind]) + "\n"
    #example = "Naslov: " + train["text"][ind] + " Oznaka: " + str(train["label"][ind]) + "\n"
    prompt = prompt+example

print(prompt)

for ind in test.index:
    messages = [
        {"role": "system", "content": prompt}
    ]
    messages.append({"role": "user", "content": "Komentar: '"+test["text"][ind]+"' Odgovor: "})

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=10,
        eos_token_id=terminators,
        do_sample=True,
        temperature=1e-5,
        top_p=0.1,
        pad_token_id=tokenizer.eos_token_id
    )
    response = outputs[0][input_ids.shape[-1]:]
    p = tokenizer.decode(response, skip_special_tokens=True)
    prediction.append(p)
    print(str(test["label"][ind]) + " " + p + " " + str(ind))

with open('reddit_pred.txt', 'w') as file:
    for item in prediction:
        file.write(f"{item}\n")