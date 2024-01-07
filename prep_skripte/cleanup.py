import pandas as pd

# Load your CSV file into a DataFrame (replace 'your_file.csv' with the actual file path)
df = pd.read_csv('teksti/reddit_sample.csv')

# Replace "&quot;" with a regular double quote (") in the 'comment' column
df['comment'] = df['comment'].str.replace('&amp;', "&")
df['comment'] = df['comment'].str.replace('&quot;', '"')
df['comment'] = df['comment'].str.replace('&#39;', "'")
df['comment'] = df['comment'].str.replace('&lt;', "<")
df['comment'] = df['comment'].str.replace('&amp;', "&")
df['comment'] = df['comment'].str.replace('&nbsp;', "")
df['comment'] = df['comment'].str.replace('&#160;', "")

# Save the updated DataFrame to a new CSV file (replace 'updated_file.csv' with your desired file path)
df.to_csv('teksti/reddit_sample2.csv', index=False)