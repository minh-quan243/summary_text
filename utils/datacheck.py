import pandas as pd
import os
import re
from nltk.stem import PorterStemmer
from tqdm import tqdm

# === KHỞI TẠO STEMMER ===
stemmer = PorterStemmer()

# === HÀM LÀM SẠCH + STEM ===
def clean_and_stem(text):
    if pd.isna(text):
        return ""
    
    # 1. Loại bỏ newline
    text = text.replace("\n", " ").replace("\r", " ")
    # 2. Chuẩn hóa khoảng trắng
    text = re.sub(r"\s+", " ", text)
    # 3. Chuyển về chữ thường
    text = text.lower()
    # 4. Loại bỏ ký tự không phải ASCII
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    # 5. Stemming từng từ
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    return " ".join(stemmed_words).strip()


# === THƯ MỤC CSV ===
data_dir = r"D:\Quân\project\summary_text\data\processed_pubmed"
splits = ["train", "val", "test"]

for split in splits:
    file_path = os.path.join(data_dir, f"{split}_encoded.csv")
    
    if not os.path.exists(file_path):
        print(f"⚠️ File {file_path} không tồn tại!")
        continue
    
    print(f"\n==================== {split.upper()} ====================")
    df = pd.read_csv(file_path)
    
    print("\n📌 Info:")
    print(df.info())
    
    print("\n📌 Null values:")
    print(df.isnull().sum())

    # Xóa dòng bị thiếu dữ liệu
    before = len(df)
    df = df.dropna()
    after = len(df)
    print(f"🧹 Đã xóa {before - after} dòng bị thiếu (còn {after} dòng)")

    # Làm sạch và stemming
    if 'article_text' in df.columns:
        df['article_text'] = tqdm(df['article_text'].apply(clean_and_stem), desc=f"{split} article_text")
    if 'abstract_text' in df.columns:
        df['abstract_text'] = tqdm(df['abstract_text'].apply(clean_and_stem), desc=f"{split} abstract_text")
    
    # Lưu file sạch
    output_path = os.path.join(data_dir, f"{split}_encoded_clean.csv")
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"✅ Đã lưu file sạch: {output_path}")
