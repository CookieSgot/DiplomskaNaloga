import pandas as pd
import re

# Load the CSV file into a DataFrame
csv_file_path = 'teksti/reddit_sample.csv'
df = pd.read_csv(csv_file_path)

# Specify the name of the text column in your CSV
text_column_name = 'comment'

# Function to detect hyperlinks in a text
def detect_hyperlinks(text):
    # Regular expression to match URLs
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return re.findall(url_pattern, text)

# Apply the function to the text column to detect hyperlinks
df['Hyperlinks'] = df[text_column_name].astype(str).apply(detect_hyperlinks)

# Display the rows where hyperlinks are detected
rows_with_hyperlinks = df[df['Hyperlinks'].apply(len) > 0]
print(rows_with_hyperlinks[['Hyperlinks', text_column_name]])