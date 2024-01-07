import os
from google.cloud import translate_v3 as translate
import pandas as pd

df = pd.read_csv('teksti/reddit_sample.csv')
comments = df["comment"].tolist()

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"keys.json"

client = translate.TranslationServiceClient()

for index, i in enumerate(comments):
    response = client.translate_text(parent="projects/sapient-spark-407220/locations/global", contents=[i], target_language_code="sl")
    df.at[index, "comment"] = response.translations[0].translated_text

df.to_csv("prevodi/reddit_prevod.csv", index=True)