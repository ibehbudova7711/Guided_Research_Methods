from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

MODEL_NAME = "bert-base-uncased"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

model.eval()

print("\nModel loaded successfully!")

print(f"\nVocabulary size: {tokenizer.vocab_size}")
print(f"Model name: {MODEL_NAME}")