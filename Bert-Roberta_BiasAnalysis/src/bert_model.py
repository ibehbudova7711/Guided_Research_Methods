from transformers import AutoTokenizer, AutoModelForMaskedLM


MODEL_NAME = "bert-base-uncased"


print("Loading BERT model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForMaskedLM.from_pretrained(
    MODEL_NAME
)

model.eval()

print("BERT model loaded!")