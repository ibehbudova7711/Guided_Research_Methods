from transformers import AutoTokenizer, AutoModelForMaskedLM


MODEL_NAME = "roberta-base"


print("Loading RoBERTa model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForMaskedLM.from_pretrained(
    MODEL_NAME
)

model.eval()

print("RoBERTa model loaded!")