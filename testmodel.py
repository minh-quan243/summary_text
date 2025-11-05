import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

MODEL_PATH = r"t5_small_best"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🔍 Loading model from: {MODEL_PATH}")
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()

def generate_summary(text, max_len=128):
    input_text = "summarize: " + text.strip()
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512
    ).to(DEVICE)

    summary_ids = model.generate(
        **inputs,
        max_length=max_len,
        num_beams=6,                # beam rộng hơn
        repetition_penalty=3.0,     # phạt lặp cao hơn
        length_penalty=1.5,         # khuyến khích ngắn gọn
        no_repeat_ngram_size=4,     # không lặp 4-gram
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)


# 🧠 Example
if __name__ == "__main__":
    text = """The PubMed dataset is a large-scale biomedical literature resource 
    containing millions of abstracts. Summarizing such texts helps researchers 
    quickly grasp the key findings of studies without reading the full article. The PubMed is good and fantastic."""
    summary = generate_summary(text)
    print("\n📄 Original Text:\n", text.strip())
    print("\n✨ Generated Summary:\n", summary)
