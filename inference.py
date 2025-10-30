import torch
import pickle
import numpy as np
from model import Seq2SeqWithAttention  # import kiến trúc model bạn đã định nghĩa
import re

# ===========================
# 🧩 CẤU HÌNH
# ===========================
VOCAB_SIZE = 20000
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
MAX_TEXT_LEN = 200
MAX_SUMM_LEN = 40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================
# 📂 LOAD MODEL & TOKENIZER
# ===========================
print("🔄 Đang load model & tokenizer...")

with open("tokenizer_x.pkl", "rb") as f:
    tokenizer_x = pickle.load(f)
with open("tokenizer_y.pkl", "rb") as f:
    tokenizer_y = pickle.load(f)

model = Seq2SeqWithAttention(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM)
checkpoint = torch.load("best_model.pth", map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

print("✅ Model đã load thành công!")

# ===========================
# 🧹 HÀM XỬ LÝ TEXT (Pytorch version)
# ===========================
def clean_text(text: str) -> str:
    """Làm sạch text cơ bản"""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def pad_sequences_torch(sequences, maxlen, padding_value=0):
    """Hàm pad sequence giống Keras nhưng thuần PyTorch"""
    padded = torch.full((len(sequences), maxlen), padding_value, dtype=torch.long)
    for i, seq in enumerate(sequences):
        trunc = seq[:maxlen]
        padded[i, :len(trunc)] = torch.tensor(trunc, dtype=torch.long)
    return padded

def preprocess_text(text, tokenizer, max_len):
    text = clean_text(text)
    tokens = tokenizer.texts_to_sequences([text])[0]
    padded = pad_sequences_torch([tokens], max_len)
    return padded.to(DEVICE)

# ===========================
# 🔁 SINH TÓM TẮT
# ===========================
def generate_summary(model, input_text, tokenizer_x, tokenizer_y, max_text_len, max_summ_len, device):
    model.eval()
    reverse_word_index = {v: k for k, v in tokenizer_y.word_index.items()}

    src = preprocess_text(input_text, tokenizer_x, max_text_len)

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src)
        decoder_input = torch.ones(1, 1, dtype=torch.long, device=device)  # <START>
        decoded_tokens = []

        for _ in range(max_summ_len):
            prediction, hidden, cell, _ = model.decoder(decoder_input, hidden, cell, encoder_outputs)
            token = prediction.argmax(1, keepdim=True)

            if token.item() in [0, 2]:  # padding hoặc end token
                break

            decoded_tokens.append(token.item())
            decoder_input = token

    summary = " ".join([reverse_word_index.get(idx, "") for idx in decoded_tokens])
    return summary.strip()

# ===========================
# 🧪 TEST MỘT VÍ DỤ
# ===========================
if __name__ == "__main__":
    input_text = """
    The study investigates the effect of vitamin D supplementation on bone health in elderly individuals.
    Participants were given daily doses for six months and monitored for bone density improvements.
    Results showed significant benefits in maintaining bone strength and reducing fracture risk.
    """

    summary = generate_summary(
        model, input_text, tokenizer_x, tokenizer_y,
        MAX_TEXT_LEN, MAX_SUMM_LEN, DEVICE
    )

    print("\n🧾 INPUT:")
    print(clean_text(input_text))
    print("\n✨ SUMMARY:")
    print(summary)

    # Cho phép nhập nhiều văn bản tùy ý
    while True:
        text = input("\nNhập đoạn văn cần tóm tắt (hoặc 'exit' để thoát):\n> ")
        if text.lower() == "exit":
            break
        print("\n✨ TÓM TẮT:")
        print(generate_summary(model, text, tokenizer_x, tokenizer_y, MAX_TEXT_LEN, MAX_SUMM_LEN, DEVICE))
