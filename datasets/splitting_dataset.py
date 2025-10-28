import pandas as pd
from sklearn.model_selection import train_test_split

# Read your CSV
df = pd.read_csv('dota2_labeled.csv')

# Split into 80% train and 20% test
dota2_train_df, dota2_test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save to separate CSV files
dota2_train_df.to_csv('dota2_labeled.csv', index=False)
dota2_test_df.to_csv('dota2_labeled.csv', index=False)

print(f"Train set size: {len(dota2_train_df)}")
print(f"Test set size: {len(dota2_test_df)}")