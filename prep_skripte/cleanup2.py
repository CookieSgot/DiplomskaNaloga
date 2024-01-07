import pandas as pd

# Load your CSV file into a DataFrame (replace 'your_file.csv' with the actual file path)
df = pd.read_csv('prevodi/headlines_prevod.csv', index_col=[0])

# Replace "&quot;" with a regular double quote (") in the 'comment' column
df['headline'] = df['headline'].str.replace('&amp;', "&")
df['headline'] = df['headline'].str.replace('&quot;', '"')
df['headline'] = df['headline'].str.replace('&#39;', "'")
df['headline'] = df['headline'].str.replace('&lt;', "<")
df['headline'] = df['headline'].str.replace('&gt;', ">")
df['headline'] = df['headline'].str.replace('&amp;', "&")
df['headline'] = df['headline'].str.replace('&nbsp;', "")
df['headline'] = df['headline'].str.replace('&#160;', "")

# Save the updated DataFrame to a new CSV file (replace 'updated_file.csv' with your desired file path)
df.to_csv('prevodi/prevod_headlines.csv', index=True)