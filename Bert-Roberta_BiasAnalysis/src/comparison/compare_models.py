import os

import pandas as pd
import matplotlib.pyplot as plt


BERT_FILE = "results/bert_results/bert_bias_results.csv"
ROBERTA_FILE = "results/roberta_results/roberta_bias_results.csv"

OUTPUT_FOLDER = "results/comparison"


def calculate_rates(df):
    """
    Calculate stereotypical preference rates.
    """

    overall = (
        (df["model_preference"] == "stereotypical")
        .mean()
        * 100
    )

    gender = (
        (
            df[df["bias_type"] == "gender"]["model_preference"]
            == "stereotypical"
        ).mean()
        * 100
    )

    age = (
        (
            df[df["bias_type"] == "age"]["model_preference"]
            == "stereotypical"
        ).mean()
        * 100
    )

    nationality = (
        (
            df[df["bias_type"] == "nationality"]["model_preference"]
            == "stereotypical"
        ).mean()
        * 100
    )

    return {
        "Overall": overall,
        "Gender": gender,
        "Age": age,
        "Nationality": nationality
    }


def add_labels(bars):
    """
    Display values above each bar.
    """

    for bar in bars:

        height = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=8
        )


def main():

    print("Loading results...")

    bert_df = pd.read_csv(BERT_FILE)
    roberta_df = pd.read_csv(ROBERTA_FILE)

    bert_rates = calculate_rates(bert_df)
    roberta_rates = calculate_rates(roberta_df)

    comparison = pd.DataFrame({

        "Category": [
            "Overall",
            "Gender",
            "Age",
            "Nationality"
        ],

        "BERT": [
            bert_rates["Overall"],
            bert_rates["Gender"],
            bert_rates["Age"],
            bert_rates["Nationality"]
        ],

        "RoBERTa": [
            roberta_rates["Overall"],
            roberta_rates["Gender"],
            roberta_rates["Age"],
            roberta_rates["Nationality"]
        ]

    })

    comparison["Difference"] = (
        comparison["RoBERTa"]
        - comparison["BERT"]
    )

    comparison["BERT"] = comparison["BERT"].round(2)
    comparison["RoBERTa"] = comparison["RoBERTa"].round(2)
    comparison["Difference"] = comparison["Difference"].round(2)

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    comparison.to_csv(
        f"{OUTPUT_FOLDER}/model_comparison.csv",
        index=False
    )

    print("\n==============================")
    print("MODEL COMPARISON")
    print("==============================\n")

    print(comparison)

    print(
        f"\nSaved comparison table to:\n"
        f"{OUTPUT_FOLDER}/model_comparison.csv"
    )

    # =====================================================
    # Figure 1
    # =====================================================

    plt.figure(figsize=(8, 5))

    x = range(len(comparison))

    width = 0.35

    bert_bars = plt.bar(
        [i - width / 2 for i in x],
        comparison["BERT"],
        width,
        label="BERT"
    )

    roberta_bars = plt.bar(
        [i + width / 2 for i in x],
        comparison["RoBERTa"],
        width,
        label="RoBERTa"
    )

    add_labels(bert_bars)
    add_labels(roberta_bars)

    plt.xticks(
        x,
        comparison["Category"]
    )

    plt.ylabel(
        "Stereotypical Preference (%)"
    )

    plt.title(
        "BERT vs RoBERTa Bias Comparison"
    )

    plt.ylim(0, 100)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_FOLDER}/bert_vs_roberta.png",
        dpi=300
    )

    plt.close()

    # =====================================================
    # Figure 2
    # =====================================================

    plt.figure(figsize=(8, 5))

    difference_bars = plt.bar(
        comparison["Category"],
        comparison["Difference"]
    )

    add_labels(difference_bars)

    plt.title(
        "Difference Between RoBERTa and BERT"
    )

    plt.ylabel(
        "Difference (%)"
    )

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_FOLDER}/difference.png",
        dpi=300
    )

    plt.close()

    # =====================================================
    # Summary
    # =====================================================

    highest_bert = comparison.iloc[
        comparison["BERT"].idxmax()
    ]

    highest_roberta = comparison.iloc[
        comparison["RoBERTa"].idxmax()
    ]

    largest_difference = comparison.iloc[
        comparison["Difference"].abs().idxmax()
    ]

    print("\n==============================")
    print("SUMMARY")
    print("==============================")

    print("\nOverall Bias")

    print(
        f"BERT:      {comparison.loc[0, 'BERT']:.2f}%"
    )

    print(
        f"RoBERTa:   {comparison.loc[0, 'RoBERTa']:.2f}%"
    )

    print(
        f"Difference: {comparison.loc[0, 'Difference']:+.2f}%"
    )

    print("\nHighest Bias Category")

    print(
        f"BERT: {highest_bert['Category']} "
        f"({highest_bert['BERT']:.2f}%)"
    )

    print(
        f"RoBERTa: {highest_roberta['Category']} "
        f"({highest_roberta['RoBERTa']:.2f}%)"
    )

    print("\nLargest Difference")

    print(
        f"{largest_difference['Category']} "
        f"({largest_difference['Difference']:+.2f}%)"
    )

    print("\nComparison figures created successfully!")


if __name__ == "__main__":

    main()