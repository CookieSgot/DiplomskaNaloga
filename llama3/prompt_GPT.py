from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForSequenceClassification
import torch
import pandas as pd

train = pd.read_csv('./kontext_reddit_train.csv')
test = pd.read_csv('./kontext_reddit_test_GPT.csv')

#prompt = "Si prepoznavalec sarkastičnih člankov. Odgovarjaš z samo 0, če podan naslov članka ni sarkastičen in z 1, če podan naslov članka je sarkastičen. Vsak naslov članka ima zraven še dodaten kontekst.\nNekaj primerov:\n"
prompt = "Si prepoznavalec sarkastičnih reddit komentarjev. Odgovarjaš z samo 0, če podan komentar ni sarkastičen in z 1, če podan komentar je sarkastičen. Vsak komentar ima zraven še dodaten kontekst.\nNekaj primerov:\n"
for ind in train.index:
    #example = "Naslov: '" + train["text"][ind] + "' Kontekst: " + train["kontext"][ind] + "' Odgovor: '" + str(train["label"][ind]) + "'\n"
    example = "Komentar: '" + train["text"][ind] + "' Kontekst: " + train["kontext"][ind] + "' Odgovor: '" + str(train["label"][ind]) + "'\n"
    prompt = prompt+example

#prompt = prompt+"\nTvoja naloga je, da na podoben način klasificiraš naslednje naslove:\n"
prompt = prompt+"\nTvoja naloga je, da na podoben način klasificiraš naslednje komentarje:\n"
for ind in test.index:
    #example = "Naslov: '" + test["text"][ind] + "' Kontekst: " + test["kontext"][ind] + "' Odgovor:\n"
    example = "Komentar: '" + test["text"][ind] + "' Kontekst: " + test["kontext"][ind] + "' Odgovor:\n"
    prompt = prompt+example
print(prompt)