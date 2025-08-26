import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# Model and tokenizer
model_name = "ai4bharat/indictrans2-indic-en-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name, trust_remote_code=True, torch_dtype=torch.float16
).to(DEVICE)
print("Model loaded successfully!")

# Input sentences
input_sentences = [
    "जब मैं छोटा था, मैं हर रोज़ पार्क जाता था।",
    "हमने पिछले सप्ताह एक नई फिल्म देखी जो कि बहुत प्रेरणादायक थी।"
]

# Language tags
src_lang = "hin_Deva"  # Hindi (Devanagari)
tgt_lang = "eng_Latn"  # English (Latin script)

for sentence in input_sentences:
    # Prefix sentence with source language tag
    text = f"{src_lang}: {sentence}"

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="longest").to(DEVICE)

    # Generate translation (disable use_cache on Windows)
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            min_length=0,
            max_length=256,
            num_beams=5,
            use_cache=False  # avoid Windows flash_attention errors
        )

    # Decode the generated tokens
    translation = tokenizer.decode(
        generated_tokens[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    print(f"\nInput: {sentence}")
    print(f"Translation: {translation}")
