from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

DEVICE = torch.device("cpu")
model_name = "Helsinki-NLP/opus-mt-en-hi"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)

def translate_en_hi(text: str) -> str:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=256, num_beams=4)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

# Test
print(translate_en_hi("Hello, how are you?"))
