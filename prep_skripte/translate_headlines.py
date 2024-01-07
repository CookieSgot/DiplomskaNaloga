import os
from google.cloud import translate_v3 as translate
import pandas as pd

df = pd.read_csv('teksti/headlines.csv')
comments = df["headline"].tolist()

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"keys.json"

client = translate.TranslationServiceClient()

#comments = comments[:14]
prevodi = []

chunk_size = 10

for i in range(0, len(comments), chunk_size):
    chunk = comments[i:i + chunk_size]
    if i%100 == 0:
        print(i)
    response = client.translate_text(parent="projects/sapient-spark-407220/locations/global", contents=chunk, target_language_code="sl")
    for j in response.translations:
        prevodi.append(j.translated_text)

#print(len(prevodi))
#df.loc[:13, 'headline'] = prevodi
df["headline"] = prevodi
df.to_csv("prevodi/headlines_prevod.csv", index=True)