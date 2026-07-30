import pandas as pd
import matplotlib.pyplot as plt


INPUT_FILE = "results/bert_results/bert_bias_results.csv"

OUTPUT_FOLDER = "results/bert_results"



def create_visualizations():

    df = pd.read_csv(INPUT_FILE)


    # ==========================
    # 1. Overall preference plot
    # ==========================

    preference_counts = (
        df["model_preference"]
        .value_counts()
    )


    plt.figure(figsize=(6, 4))

    preference_counts.plot(
        kind="bar"
    )

    plt.title(
        "BERT Preference Across CrowS-Pairs Examples"
    )

    plt.xlabel(
        "Model Preference"
    )

    plt.ylabel(
        "Number of Examples"
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()


    plt.savefig(
        f"{OUTPUT_FOLDER}/bert_overall_preference.png",
        dpi=300
    )

    plt.close()



    # ==========================
    # 2. Category bias plot
    # ==========================

    category_results = (
        df.groupby("bias_type")
        ["model_preference"]
        .apply(
            lambda x:
            (x == "stereotypical").mean() * 100
        )
        .reset_index()
    )


    category_results.columns = [
        "bias_type",
        "stereotypical_rate"
    ]


    plt.figure(figsize=(7, 4))


    plt.bar(
        category_results["bias_type"],
        category_results["stereotypical_rate"]
    )


    plt.title(
        "BERT Stereotypical Preference Rate by Bias Category"
    )


    plt.xlabel(
        "Bias Category"
    )


    plt.ylabel(
        "Stereotypical Preference (%)"
    )


    plt.ylim(
        0,
        100
    )


    plt.tight_layout()


    plt.savefig(
        f"{OUTPUT_FOLDER}/bert_category_preference.png",
        dpi=300
    )


    plt.close()


    print("Visualizations created successfully!")
    print(f"Saved to: {OUTPUT_FOLDER}")



if __name__ == "__main__":

    create_visualizations()