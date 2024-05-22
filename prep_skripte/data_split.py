import pandas as pd
from sklearn.model_selection import train_test_split

reddit = pd.read_csv('prevodi/prevod_reddit.csv', index_col=[0])
headlines = pd.read_csv('prevodi/prevod_headlines.csv', index_col=[0])

reddit = reddit.rename(columns={"comment": "text"})
headlines = headlines.rename(columns={"headline": "text", "is_sarcastic":"label"})

#print(reddit['text'].apply(len).idxmax())
#print(headlines['text'].apply(len).max())

reddit_train, reddit_temp = train_test_split(reddit, test_size=0.2, random_state=42)
reddit_val, reddit_test = train_test_split(reddit_temp, test_size=0.5, random_state=42)

headlines_train, headlines_temp = train_test_split(headlines, test_size=0.2, random_state=42)
headlines_val, headlines_test = train_test_split(headlines_temp, test_size=0.5, random_state=42)

print(reddit_train["label"].value_counts())
print(reddit_val["label"].value_counts())
print(reddit_test["label"].value_counts())

print(headlines_train["label"].value_counts())
print(headlines_val["label"].value_counts())
print(headlines_test["label"].value_counts())

reddit_train.to_csv("data/reddit_train.csv", index=False)
reddit_test.to_csv("data/reddit_test.csv", index=False)
reddit_val.to_csv("data/reddit_val.csv", index=False)

headlines_train.to_csv("data/headlines_train.csv", index=False)
headlines_test.to_csv("data/headlines_test.csv", index=False)
headlines_val.to_csv("data/headlines_val.csv", index=False)