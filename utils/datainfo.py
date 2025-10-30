# =========================================================
# 📘 Xử lý dataset PubMed JSONL + vocab
# =========================================================
import os
import json
import pandas as pd
from datasets import Dataset, DatasetDict

# =========================================================
# 1️⃣ HÀM ĐỌC FILE JSONL → DataFrame
# =========================================================
def load_jsonl_to_df(file_path):
    data_list = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)

            # Gộp article_text (luôn là list)
            article_text = " ".join(obj.get("article_text", []))

            # abstract_text có thể là list hoặc string
            abs_text = obj.get("abstract_text", "")
            if isinstance(abs_text, list):
                abs_text = " ".join(abs_text)

            data_list.append({
                "article_id": obj.get("article_id"),
                "article_text": article_text.strip(),
                "abstract_text": abs_text.strip()
            })
    return pd.DataFrame(data_list)

# =========================================================
# 2️⃣ HÀM ĐỌC FILE VOCAB
# =========================================================
def load_vocab(vocab_path):
    vocab = []
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            token = parts[0]
            vocab.append(token)

    # Tạo mapping
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}

    print(f"📖 Loaded vocab: {len(vocab)} tokens")
    print("🔹 First 10 tokens:", vocab[:10])

    return word2idx, idx2word

# =========================================================
# 3️⃣ HÀM MÃ HÓA TEXT DỰA TRÊN VOCAB
# =========================================================
def encode_text(text, word2idx, max_len=512):
    tokens = text.lower().split()
    ids = [word2idx.get(t, word2idx.get("<unk>", 1)) for t in tokens]
    if len(ids) < max_len:
        ids += [word2idx.get("<pad>", 0)] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids

# =========================================================
# 4️⃣ MAIN PIPELINE
# =========================================================
input_dir = r"data/raw/pubmed-dataset/pubmed-dataset"
output_dir = r"data/processed_pubmed"
os.makedirs(output_dir, exist_ok=True)

vocab_path = os.path.join(input_dir, "vocab")
word2idx, idx2word = load_vocab(vocab_path)

splits = ["train", "val", "test"]
data_dict = {}

for split in splits:
    print(f"\n📦 Đang xử lý {split}...")
    file_path = os.path.join(input_dir, f"{split}.txt")

    df = load_jsonl_to_df(file_path)
    print(f"✅ {split}: {df.shape[0]} mẫu")

    # Mã hóa văn bản
    df["input_ids"] = df["article_text"].apply(lambda x: encode_text(x, word2idx))
    df["target_ids"] = df["abstract_text"].apply(lambda x: encode_text(x, word2idx))

    # Bỏ cột thừa
    df = df[["article_id", "article_text", "abstract_text", "input_ids", "target_ids"]]

    # Lưu lại CSV để debug
    df.to_csv(os.path.join(output_dir, f"{split}_encoded.csv"), index=False, encoding="utf-8")

    # Tạo Dataset HuggingFace
    data_dict[split] = Dataset.from_pandas(df)

# =========================================================
# 5️⃣ HỢP NHẤT THÀNH DatasetDict
# =========================================================
dataset = DatasetDict(data_dict)

# Lưu dataset HuggingFace
dataset.save_to_disk(output_dir)
print("\n🎯 Dataset đã lưu tại:", output_dir)
print(dataset)
