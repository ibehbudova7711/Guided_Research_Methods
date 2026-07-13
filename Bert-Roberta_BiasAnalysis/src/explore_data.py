import pandas as pd

# Load filtered dataset
df = pd.read_csv("results/csv/crows_pairs_filtered.csv")

categories = ["gender", "age", "nationality"]

for category in categories:
    print("\n" + "=" * 70)
    print(f"{category.upper()} EXAMPLES")
    print("=" * 70)

    examples = df[df["bias_type"] == category].head(3)

    for i, row in examples.iterrows():
        print(f"\nExample {i}")

        print("\nMore stereotypical:")
        print(row["sent_more"])

        print("\nLess stereotypical:")
        print(row["sent_less"])

        print("\nDirection:")
        print(row["stereo_antistereo"])

        print("-" * 70)