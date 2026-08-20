import pandas as pd

# Load your dataset
df = pd.read_csv("english_dataset.csv")

# Split into batches of 100 rows
batch_size = 100

for i in range(0, len(df), batch_size):
    batch = df.iloc[i:i + batch_size]
    batch.to_csv(f"batch_{i//batch_size + 1}.csv", index=False)

print("Dataset split successfully!")