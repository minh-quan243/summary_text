import streamlit as st
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# --- Cấu hình ---
MODEL_PATH = r"checkpoints/t5_small_best"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Giao diện ---
st.set_page_config(page_title="Text Summarizer", page_icon="📝")
st.title("📝 T5 Text Summarizer")
st.write("Nhập văn bản của bạn và nhận bản tóm tắt ngay lập tức.")

# --- Load model ---
with st.spinner("🔍 Loading model..."):
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(DEVICE)
    model.eval()
st.success("✅ Model loaded successfully!")

# --- Hàm tạo summary ---
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
        num_beams=6,
        repetition_penalty=3.0,
        length_penalty=1.5,
        no_repeat_ngram_size=4,
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

# --- Input từ người dùng ---
text_input = st.text_area("Nhập văn bản tại đây:", height=200)

if st.button("✨ Generate Summary"):
    if text_input.strip() == "":
        st.warning("Vui lòng nhập văn bản để tóm tắt!")
    else:
        with st.spinner("🧠 Generating summary..."):
            summary = generate_summary(text_input)
        st.subheader("📄 Original Text")
        st.write(text_input)
        st.subheader("✨ Generated Summary")
        st.write(summary)
