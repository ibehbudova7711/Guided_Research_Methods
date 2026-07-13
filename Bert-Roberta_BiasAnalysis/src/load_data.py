import pandas as pd


def load_dataset(file_path):
    """
    Load the CrowS-Pairs dataset.
    """
    df = pd.read_csv(file_path)

    print("\nDataset loaded successfully.")
    print(f"Number of sentence pairs: {len(df)}")

    return df


def show_dataset_info(df):
    """
    Display basic information about the dataset.
    """
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst five rows:")
    print(df.head())

    if "bias_type" in df.columns:
        print("\nBias categories:")
        print(df["bias_type"].value_counts())


if __name__ == "__main__":
    dataset = load_dataset("data/crows_pairs_anonymized.csv")
    show_dataset_info(dataset)