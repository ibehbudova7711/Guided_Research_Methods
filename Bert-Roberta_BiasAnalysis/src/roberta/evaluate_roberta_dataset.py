import pandas as pd
from tqdm import tqdm

from roberta_pll import calculate_pll


INPUT_FILE = "results/csv/crows_pairs_filtered.csv"

OUTPUT_FILE = "results/roberta_results/roberta_bias_results.csv"



def evaluate_roberta_dataset():

    print("Loading filtered CrowS-Pairs dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Number of examples: {len(df)}")


    results = []


    print("\nStarting RoBERTa bias evaluation...")


    for index, row in tqdm(
        df.iterrows(),
        total=len(df)
    ):

        sent_more = row["sent_more"]
        sent_less = row["sent_less"]


        try:

            more_score = calculate_pll(sent_more)

            less_score = calculate_pll(sent_less)


            if more_score > less_score:
                preference = "stereotypical"

            else:
                preference = "less_stereotypical"



            results.append(
                {
                    "bias_type": row["bias_type"],

                    "sent_more": sent_more,

                    "sent_less": sent_less,

                    "sent_more_score": more_score,

                    "sent_less_score": less_score,

                    "model_preference": preference
                }
            )


        except Exception as error:

            print("\nError processing row:", index)

            print(error)



    results_df = pd.DataFrame(results)


    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )


    print("\nEvaluation finished!")

    print(
        f"Saved results to: {OUTPUT_FILE}"
    )



if __name__ == "__main__":

    evaluate_roberta_dataset()