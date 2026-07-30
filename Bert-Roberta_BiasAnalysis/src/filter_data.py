import pandas as pd

# Load original dataset
df = pd.read_csv("data/crows_pairs_anonymized.csv")

# Categories selected for this project
selected_biases = ["gender", "age", "nationality"]

# Keep only selected categories
filtered_df = df[df["bias_type"].isin(selected_biases)]

print(f"Original dataset size: {len(df)}")
print(f"Filtered dataset size: {len(filtered_df)}")

print("\nCategory counts:")
print(filtered_df["bias_type"].value_counts())

# Save filtered dataset
output_path = "results/csv/crows_pairs_filtered.csv"
filtered_df.to_csv(output_path, index=False)

print(f"\nFiltered dataset saved to: {output_path}")