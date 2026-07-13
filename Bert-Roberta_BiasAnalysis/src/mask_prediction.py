from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

MODEL_NAME = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

model.eval()

sentence = "The doctor is a [MASK]."

inputs = tokenizer(sentence, return_tensors="pt")

mask_index = torch.where(
    inputs["input_ids"] == tokenizer.mask_token_id
)[1]

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits

mask_logits = logits[0, mask_index, :]

top_tokens = torch.topk(mask_logits, 10, dim=1).indices[0]

print(sentence)
print()

for token in top_tokens:
    word = tokenizer.decode([token])
    print(word)