import pandas as pd
import random

# Read the two CSV files into DataFrames
file1 = pd.read_csv('teksti/headlines.csv')
file2 = pd.read_csv('prevodi/prevod_headlines.csv', index_col=[0])

df = pd.DataFrame(columns=["is_sarcastic","headline"], index=range(0,100))

# Sample n rows from each file
sampled_rows_file1 = file1.sample(50, replace=False, random_state=1)  # Set random_state for reproducibility
sampled_rows_file2 = file2.sample(50, replace=False, random_state=1)

df[0:100:2] = sampled_rows_file1
df[1:101:2] = sampled_rows_file2

# Create a new DataFrame with pairs of sampled rows
#result_df = pd.concat([sampled_rows_file1, sampled_rows_file2], ignore_index=True)

# Save the result DataFrame to a new CSV file
df.to_csv('prevodi/pregled_prevodov2.csv', index=False)