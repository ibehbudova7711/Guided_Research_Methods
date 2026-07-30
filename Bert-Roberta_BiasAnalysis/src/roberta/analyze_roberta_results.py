import pandas as pd


INPUT_FILE = "results/roberta_results/roberta_bias_results.csv"



def analyze_results():

    print("Loading RoBERTa results...")

    df = pd.read_csv(INPUT_FILE)


    print("\nTotal evaluated examples:")
    print(len(df))


    print("\n==============================")
    print("Overall RoBERTa Preference")
    print("==============================")


    overall = (
        df["model_preference"]
        .value_counts()
    )


    print(overall)


    total = len(df)


    stereo_rate = (
        overall.get("stereotypical", 0)
        / total
        * 100
    )


    less_rate = (
        overall.get("less_stereotypical", 0)
        / total
        * 100
    )


    print(
        f"\nStereotypical preference rate: {stereo_rate:.2f}%"
    )

    print(
        f"Less stereotypical preference rate: {less_rate:.2f}%"
    )



    print("\n==============================")
    print("Category Analysis")
    print("==============================")


    category_results = []


    for category, group in df.groupby("bias_type"):


        total_examples = len(group)


        stereo_count = (
            group["model_preference"]
            == "stereotypical"
        ).sum()


        rate = (
            stereo_count
            / total_examples
            * 100
        )


        category_results.append(
            {
                "category": category,
                "total": total_examples,
                "stereotypical_count": stereo_count,
                "stereotypical_rate": rate
            }
        )


    category_df = pd.DataFrame(category_results)


    category_df = category_df.sort_values(
        "stereotypical_rate",
        ascending=False
    )


    print(category_df)



if __name__ == "__main__":

    analyze_results()