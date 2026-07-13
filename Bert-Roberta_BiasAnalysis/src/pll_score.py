from bert_model import tokenizer, model
import torch
import torch.nn.functional as F


def calculate_pll(sentence):
    """
    Calculate pseudo log-likelihood score for a sentence.
    """

    # Tokenize sentence
    inputs = tokenizer(
        sentence,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"]

    tokens = input_ids[0]

    total_log_probability = 0.0

    token_count = 0


    # Skip special tokens: [CLS] and [SEP]
    for i in range(1, len(tokens) - 1):

        original_token_id = tokens[i].item()

        # Create a copy of input ids
        masked_input = input_ids.clone()

        # Replace current token with [MASK]
        masked_input[0, i] = tokenizer.mask_token_id


        with torch.no_grad():

            outputs = model(
                input_ids=masked_input
            )


        logits = outputs.logits


        # Get probability distribution for masked position
        mask_logits = logits[0, i, :]

        probabilities = F.softmax(
            mask_logits,
            dim=-1
        )


        # Probability of original token
        token_probability = probabilities[
            original_token_id
        ].item()


        # Avoid log(0)
        if token_probability > 0:

            total_log_probability += torch.log(
                torch.tensor(token_probability)
            ).item()

            token_count += 1


    return total_log_probability



if __name__ == "__main__":


    sentence = "The doctor is a nurse."

    score = calculate_pll(sentence)


    print("\nSentence:")
    print(sentence)

    print("\nPLL Score:")
    print(score)