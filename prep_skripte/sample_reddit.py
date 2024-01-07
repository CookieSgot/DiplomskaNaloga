import pandas as pd

df = pd.read_csv('teksti/train-balanced-sarcasm.csv')

ups_over_25 = df[df['ups'] > 25]

# Select 15,000 random entries for each value in the 'label' column
ups_over_25_sampled = ups_over_25.groupby('label').apply(lambda x: x.sample(n=15000, random_state=42))

# Extract only the 'label' and 'comment' columns
selected_columns = ['label', 'comment']
ups_over_25_sampled_selected = ups_over_25_sampled[selected_columns]

# Save the sampled DataFrame to a new CSV file
ups_over_25_sampled_selected.to_csv('teksti/reddit_sample.csv', index=False)