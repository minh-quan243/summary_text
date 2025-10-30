import pandas as pd
import os
# =========================================================
# 📘 Kiểm tra thông tin dataset từ file CS
# =========================================================
data_dir = r"F:\GitHub\Summarize-text\data\processed_pubmed"
splits = ["train", "val", "test"]

for split in splits:
    file_path = os.path.join(data_dir, f"{split}_encoded_clean.csv")
    
    if not os.path.exists(file_path):
        print(f"⚠️ File {file_path} không tồn tại!")
        continue
    
    print(f"\n==================== {split.upper()} ====================")
    df = pd.read_csv(file_path)
    
    print("\n📌 Info:")
    print(df.info())
    
    print("\n📌 Null values:")
    print(df.isnull().sum())

