import pandas as pd

json_file_path = 'teksti/Sarcasm_Headlines_Dataset.json'

df = pd.read_json(json_file_path, lines=True)

df = df[['is_sarcastic', 'headline']]

df.to_csv('teksti/headlines.csv', index=False)