from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import torch.nn.functional as F


MODEL_NAME = "bert-base-uncased"


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

model.eval()


def get_token_probability(sentence, target_word):
    """
    Calculate BERT probability for a specific word in a sentence.
    """

    tokens = tokenizer.tokenize(sentence)

    target_index = None

    for i, token in enumerate(tokens):
        if token == target_word:
            target_index = i
            break

    if target_index is None:
        return None

    original_tokens = tokens.copy()

    tokens[target_index] = tokenizer.mask_token

    masked_sentence = tokenizer.convert_tokens_to_string(tokens)

    inputs = tokenizer(
        masked_sentence,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    mask_position = torch.where(
        inputs["input_ids"] == tokenizer.mask_token_id
    )[1]

    mask_logits = logits[0, mask_position, :]

    probabilities = F.softmax(mask_logits, dim=-1)

    target_id = tokenizer.convert_tokens_to_ids(target_word)

    probability = probabilities[0, target_id].item()

    return probability


if __name__ == "__main__":

    sentence = "The doctor is a nurse."

    probability = get_token_probability(
        sentence,
        "nurse"
    )

    print("Sentence:")
    print(sentence)

    print("\nProbability of 'nurse':")
    print(probability)