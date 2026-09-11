"""
Step 15: Prepare the modeling dataset.
1. Stratified subsample (keep flood/non-flood proportions, reduce size
   for laptop-friendly training).
2. Spatial block train/test split to avoid spatial leakage: entire
   spatial blocks (not individual pixels) are assigned to train or
   test, so neighboring pixels never end up on both sides of the split.
"""

import pandas as pd
import numpy as np

np.random.seed(42)  # for reproducibility

df = pd.read_csv("data/processed/lagos_feature_table.csv")
print(f"Full dataset size: {len(df)}")

# --- Step A: Stratified subsample ---
SAMPLE_SIZE = 80000

df_flooded = df[df["flooded"] == 1]
df_not_flooded = df[df["flooded"] == 0]

frac_flooded = len(df_flooded) / len(df)
n_flooded = int(SAMPLE_SIZE * frac_flooded)
n_not_flooded = SAMPLE_SIZE - n_flooded

sample_flooded = df_flooded.sample(n=min(n_flooded, len(df_flooded)), random_state=42)
sample_not_flooded = df_not_flooded.sample(n=min(n_not_flooded, len(df_not_flooded)), random_state=42)

df_sample = pd.concat([sample_flooded, sample_not_flooded]).reset_index(drop=True)
print(f"Subsampled dataset size: {len(df_sample)}")
print("Class balance in subsample:")
print(df_sample["flooded"].value_counts(normalize=True) * 100)

# --- Step B: Assign spatial blocks ---
BLOCK_SIZE = 20  # pixels per block edge

df_sample["block_row"] = df_sample["row"] // BLOCK_SIZE
df_sample["block_col"] = df_sample["col"] // BLOCK_SIZE
df_sample["block_id"] = df_sample["block_row"].astype(str) + "_" + df_sample["block_col"].astype(str)

unique_blocks = df_sample["block_id"].unique()
print(f"\nNumber of unique spatial blocks: {len(unique_blocks)}")

# --- Step C: Randomly assign blocks to train (80%) or test (20%) ---
np.random.shuffle(unique_blocks)
split_idx = int(len(unique_blocks) * 0.8)
train_blocks = set(unique_blocks[:split_idx])
test_blocks = set(unique_blocks[split_idx:])

df_sample["split"] = df_sample["block_id"].apply(lambda b: "train" if b in train_blocks else "test")

train_df = df_sample[df_sample["split"] == "train"].drop(columns=["split", "block_row", "block_col", "block_id"])
test_df = df_sample[df_sample["split"] == "test"].drop(columns=["split", "block_row", "block_col", "block_id"])

print(f"\nTrain set size: {len(train_df)}")
print(f"Test set size: {len(test_df)}")
print("\nTrain flood class balance:")
print(train_df["flooded"].value_counts(normalize=True) * 100)
print("\nTest flood class balance:")
print(test_df["flooded"].value_counts(normalize=True) * 100)

# Save
train_df.to_csv("data/processed/train_set.csv", index=False)
test_df.to_csv("data/processed/test_set.csv", index=False)
print("\nSaved: data/processed/train_set.csv")
print("Saved: data/processed/test_set.csv")