# =========================================================
# 📘 Xử lý dataset PubMed - SIMPLE & EFFICIENT
# =========================================================
import os
import json
import pandas as pd
import re
from tqdm import tqdm
import pickle
import numpy as np


# =========================================================
# 1️⃣ HÀM ĐỌC FILE JSONL → DataFrame
# =========================================================
def load_jsonl_to_df(file_path):
    """Đọc file JSONL và chuyển thành DataFrame"""
    data_list = []
    print(f"    📖 Đang đọc: {os.path.basename(file_path)}")

    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="      Reading lines"):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)

                # Xử lý article_text
                article_text_list = obj.get("article_text", [])
                article_text = " ".join(article_text_list) if article_text_list else ""

                # Xử lý abstract_text
                abs_text = obj.get("abstract_text", "")
                if isinstance(abs_text, list):
                    abs_text = " ".join(abs_text)

                data_list.append({
                    "article_id": obj.get("article_id", "unknown"),
                    "article_text": article_text.strip(),
                    "abstract_text": abs_text.strip()
                })
            except json.JSONDecodeError:
                continue

    return pd.DataFrame(data_list)


# =========================================================
# 2️⃣ HÀM ĐỌC VOCAB
# =========================================================
def load_vocab(vocab_path):
    """Đọc file vocab và tạo mapping"""
    vocab = []
    print(f"📖 Đang load vocab...")

    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                vocab.append(parts[0])

    word2idx = {w: i for i, w in enumerate(vocab)}
    print(f"✅ Vocab: {len(vocab)} tokens")
    print(f"🔹 Special tokens - <pad>: {word2idx.get('<pad>', 0)}, <unk>: {word2idx.get('<unk>', 1)}")

    return word2idx, vocab


# =========================================================
# 3️⃣ HÀM LÀM SẠCH TEXT
# =========================================================
def clean_text(text):
    """Làm sạch text đơn giản"""
    if pd.isna(text) or text == "":
        return ""

    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.lower()
    text = re.sub(r"[^\w\s.,!?;:()-]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =========================================================
# 4️⃣ HÀM MÃ HÓA TEXT
# =========================================================
def encode_text(text, word2idx, max_len=512):
    """Mã hóa text thành token IDs"""
    if not text:
        return [word2idx.get("<pad>", 0)] * max_len

    tokens = text.split()
    ids = [word2idx.get(token, word2idx.get("<unk>", 1)) for token in tokens]

    if len(ids) < max_len:
        ids = ids + [word2idx.get("<pad>", 0)] * (max_len - len(ids))
    else:
        ids = ids[:max_len]

    return ids


# =========================================================
# 5️⃣ HÀM LƯU DỮ LIỆU DẠNG PICKLE (NHANH + NHỎ)
# =========================================================
def save_processed_data(df, output_path):
    """Lưu dữ liệu đã xử lý dạng pickle để tải nhanh"""
    # Lưu cả DataFrame
    df.to_pickle(output_path)

    # Lưu thêm numpy arrays riêng để training nhanh hơn
    arrays_path = output_path.replace('.pkl', '_arrays.npz')
    np.savez_compressed(
        arrays_path,
        input_ids=np.array(df['input_ids'].tolist()),
        target_ids=np.array(df['target_ids'].tolist())
    )

    print(f"💾 Đã lưu: {output_path}")
    print(f"💾 Đã lưu arrays: {arrays_path}")


# =========================================================
# 6️⃣ HÀM TẠO DATA LOADER ĐƠN GIẢN
# =========================================================
class PubMedDataset:
    """Dataset đơn giản cho training"""

    def __init__(self, data_path):
        self.data = pd.read_pickle(data_path)
        self.arrays = np.load(data_path.replace('.pkl', '_arrays.npz'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'input_ids': self.arrays['input_ids'][idx],
            'target_ids': self.arrays['target_ids'][idx],
            'article_id': self.data.iloc[idx]['article_id']
        }

    def get_batch(self, batch_size=32):
        """Generator để lấy batch data"""
        for i in range(0, len(self), batch_size):
            batch_indices = slice(i, i + batch_size)
            yield {
                'input_ids': self.arrays['input_ids'][batch_indices],
                'target_ids': self.arrays['target_ids'][batch_indices]
            }


# =========================================================
# 7️⃣ MAIN PIPELINE - ĐƠN GIẢN & HIỆU QUẢ
# =========================================================
def main():
    print("🚀 XỬ LÝ DATASET PUBMED - ĐƠN GIẢN & HIỆU QUẢ")
    print("=" * 50)

    # Thiết lập đường dẫn
    input_dir = r"D:\Quân\project\summary_text\data\raw\pubmed-dataset"
    output_dir = r"D:\Quân\project\summary_text\data\processed_pubmed"
    os.makedirs(output_dir, exist_ok=True)

    # Load vocab
    vocab_path = os.path.join(input_dir, "vocab")
    word2idx, vocab = load_vocab(vocab_path)

    # Lưu vocab để sử dụng sau
    with open(os.path.join(output_dir, "vocab.pkl"), 'wb') as f:
        pickle.dump({'word2idx': word2idx, 'vocab': vocab}, f)

    # Xử lý từng split
    splits = ["train", "val", "test"]

    for split in splits:
        print(f"\n📦 {'=' * 20} {split.upper()} {'=' * 20}")

        # Load data
        file_path = os.path.join(input_dir, f"{split}.txt")
        if not os.path.exists(file_path):
            print(f"❌ Không tìm thấy: {file_path}")
            continue

        df = load_jsonl_to_df(file_path)
        print(f"📊 Loaded: {len(df)} samples")

        # Lọc dữ liệu
        initial_count = len(df)
        df = df.dropna()
        df = df[(df['article_text'] != "") & (df['abstract_text'] != "")]
        print(f"🧹 Filtered: {initial_count} → {len(df)} samples")

        if len(df) == 0:
            print("❌ Không có dữ liệu sau khi lọc")
            continue

        # Làm sạch text
        print("🧹 Cleaning text...")
        df['article_text_clean'] = df['article_text'].apply(clean_text)
        df['abstract_text_clean'] = df['abstract_text'].apply(clean_text)

        # Mã hóa
        print("🔢 Encoding...")
        df['input_ids'] = df['article_text_clean'].apply(
            lambda x: encode_text(x, word2idx, max_len=512)
        )
        df['target_ids'] = df['abstract_text_clean'].apply(
            lambda x: encode_text(x, word2idx, max_len=128)
        )

        # Lưu nhiều định dạng
        print("💾 Saving...")

        # 1. CSV (để xem)
        csv_path = os.path.join(output_dir, f"{split}.csv")
        df[['article_id', 'article_text_clean', 'abstract_text_clean']].to_csv(
            csv_path, index=False, encoding='utf-8'
        )

        # 2. Pickle (để training - NHANH + NHỎ)
        pkl_path = os.path.join(output_dir, f"{split}.pkl")
        save_processed_data(df, pkl_path)

        # Thống kê
        article_lens = df['article_text_clean'].str.len()
        abstract_lens = df['abstract_text_clean'].str.len()
        print(f"📐 Text length - Article: avg {article_lens.mean():.0f}, Abstract: avg {abstract_lens.mean():.0f}")

    print(f"\n✅ HOÀN TẤT! Dữ liệu đã lưu tại: {output_dir}")
    print("\n🎯 CÁCH SỬ DỤNG:")
    print("   from scripts.process import PubMedDataset")
    print("   train_data = PubMedDataset('data/processed_pubmed/train.pkl')")
    print("   for batch in train_data.get_batch(32):")
    print("       # Training code here")


if __name__ == "__main__":
    tqdm.pandas()
    main()