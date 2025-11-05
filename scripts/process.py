# =========================================================
# 📘 XỬ LÝ DATASET PUBMED - COMPLETE PIPELINE
# =========================================================
import os
import json
import pandas as pd
import re
from tqdm import tqdm
import pickle
import torch
import numpy as np
from transformers import T5Tokenizer
import matplotlib.pyplot as plt


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
# 2️⃣ HÀM LÀM SẠCH TEXT NÂNG CAO
# =========================================================
def clean_text_advanced(text):
    """Làm sạch text với xử lý đặc biệt cho văn bản khoa học"""
    if pd.isna(text) or text == "":
        return ""

    # Chuẩn hóa khoảng trắng
    text = re.sub(r'\s+', ' ', text)

    # Giữ lại các ký tự đặc biệt quan trọng trong khoa học
    text = re.sub(r'[^\w\s.,!?;:()\-%\d+/=<>]', ' ', text)

    # Chuẩn hóa các từ viết tắt khoa học
    text = re.sub(r'\b(e\.g\.|i\.e\.|etc\.|vs\.|fig\.)\b', lambda x: x.group().replace('.', '_'), text)

    # Xử lý số và đơn vị đo lường
    text = re.sub(r'(\d+)%', r'\1 percent', text)
    text = re.sub(r'(\d+)/(\d+)', r'\1 divided by \2', text)

    # Chuẩn hóa lại khoảng trắng
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def extract_important_sections(text):
    """
    Trích xuất các phần quan trọng từ bài báo khoa học
    Ưu tiên: abstract, introduction, results, conclusion
    """
    text_lower = text.lower()
    sections = []

    # Tìm abstract
    abstract_patterns = ['abstract', 'summary', 'tóm tắt']
    for pattern in abstract_patterns:
        if pattern in text_lower:
            start_idx = text_lower.find(pattern)
            # Tìm kết thúc của abstract (thường là trước introduction)
            end_patterns = ['introduction', 'background', '1.', '\n\n']
            end_idx = len(text)
            for end_pat in end_patterns:
                idx = text_lower.find(end_pat, start_idx + len(pattern))
                if idx != -1 and idx < end_idx:
                    end_idx = idx
            sections.append(text[start_idx:end_idx])
            break

    # Tìm kết luận
    conclusion_patterns = ['conclusion', 'discussion', 'kết luận']
    for pattern in conclusion_patterns:
        if pattern in text_lower:
            start_idx = text_lower.find(pattern)
            sections.append(text[start_idx:start_idx + 1000])  # Lấy 1000 ký tự
            break

    # Nếu không tìm thấy sections rõ ràng, dùng strategy đầu-cuối
    if not sections:
        words = text.split()
        if len(words) > 800:
            # Lấy 60% đầu + 40% cuối
            head_size = int(len(words) * 0.6)
            tail_size = len(words) - head_size
            sections = [' '.join(words[:head_size]), ' '.join(words[-tail_size:])]
        else:
            sections = [text]

    return ' '.join(sections)


# =========================================================
# 3️⃣ HÀM XỬ LÝ VĂN BẢN DÀI
# =========================================================
def process_long_text(text, tokenizer, max_tokens=512, strategy="smart"):
    """
    Xử lý văn bản dài để phù hợp với model
    """
    if not text or text.strip() == "":
        return ""

    tokens = tokenizer.encode(text)

    # Nếu văn bản đã ngắn, trả về nguyên bản
    if len(tokens) <= max_tokens:
        return text

    if strategy == "smart":
        # Ưu tiên các phần quan trọng
        important_text = extract_important_sections(text)
        important_tokens = tokenizer.encode(important_text)

        if len(important_tokens) <= max_tokens:
            return important_text
        else:
            # Cắt bớt nhưng ưu tiên phần đầu của important text
            return tokenizer.decode(important_tokens[:max_tokens])

    elif strategy == "head_tail":
        # Lấy phần đầu và phần cuối
        head_size = int(max_tokens * 0.6)
        tail_size = max_tokens - head_size

        head_tokens = tokens[:head_size]
        tail_tokens = tokens[-tail_size:]

        combined_tokens = head_tokens + tail_tokens
        return tokenizer.decode(combined_tokens)

    else:  # "truncate"
        return tokenizer.decode(tokens[:max_tokens])


# =========================================================
# 4️⃣ HÀM PHÂN TÍCH VÀ THỐNG KÊ
# =========================================================
def analyze_dataset(df, tokenizer, split_name):
    """Phân tích chi tiết dataset"""
    print(f"\n📊 {'=' * 30} PHÂN TÍCH {split_name.upper()} {'=' * 30}")

    # Thống kê cơ bản
    print(f"🔸 Tổng số mẫu: {len(df):,}")
    print(f"🔸 Số từ trung bình - Article: {df['article_word_count'].mean():.1f}")
    print(f"🔸 Số từ trung bình - Abstract: {df['abstract_word_count'].mean():.1f}")

    # Phân tích độ dài token
    df['article_token_count'] = df['article_text_processed'].apply(
        lambda x: len(tokenizer.encode(x))
    )
    df['abstract_token_count'] = df['abstract_text_processed'].apply(
        lambda x: len(tokenizer.encode(x))
    )

    print(f"🔸 Token count - Article: {df['article_token_count'].mean():.1f}")
    print(f"🔸 Token count - Abstract: {df['abstract_token_count'].mean():.1f}")

    # Thống kê phần trăm văn bản bị cắt
    long_articles = (df['article_token_count'] > 512).sum()
    long_abstracts = (df['abstract_token_count'] > 128).sum()

    print(f"🔸 Articles >512 tokens: {long_articles}/{len(df)} ({long_articles / len(df) * 100:.1f}%)")
    print(f"🔸 Abstracts >128 tokens: {long_abstracts}/{len(df)} ({long_abstracts / len(df) * 100:.1f}%)")

    return df


def visualize_length_distribution(df, output_dir, split_name):
    """Visualize phân bố độ dài văn bản"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Article word count
    axes[0, 0].hist(df['article_word_count'], bins=50, alpha=0.7, color='skyblue')
    axes[0, 0].axvline(df['article_word_count'].mean(), color='red', linestyle='--',
                       label=f'Mean: {df["article_word_count"].mean():.1f}')
    axes[0, 0].set_xlabel('Số từ')
    axes[0, 0].set_ylabel('Tần suất')
    axes[0, 0].set_title(f'Phân bố số từ - Articles ({split_name})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Abstract word count
    axes[0, 1].hist(df['abstract_word_count'], bins=50, alpha=0.7, color='lightgreen')
    axes[0, 1].axvline(df['abstract_word_count'].mean(), color='red', linestyle='--',
                       label=f'Mean: {df["abstract_word_count"].mean():.1f}')
    axes[0, 1].set_xlabel('Số từ')
    axes[0, 1].set_ylabel('Tần suất')
    axes[0, 1].set_title(f'Phân bố số từ - Abstracts ({split_name})')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Article token count
    axes[1, 0].hist(df['article_token_count'], bins=50, alpha=0.7, color='orange')
    axes[1, 0].axvline(512, color='red', linestyle='--', label='Max: 512 tokens')
    axes[1, 0].axvline(df['article_token_count'].mean(), color='blue', linestyle='--',
                       label=f'Mean: {df["article_token_count"].mean():.1f}')
    axes[1, 0].set_xlabel('Số token')
    axes[1, 0].set_ylabel('Tần suất')
    axes[1, 0].set_title(f'Phân bố số token - Articles ({split_name})')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Abstract token count
    axes[1, 1].hist(df['abstract_token_count'], bins=50, alpha=0.7, color='pink')
    axes[1, 1].axvline(128, color='red', linestyle='--', label='Max: 128 tokens')
    axes[1, 1].axvline(df['abstract_token_count'].mean(), color='blue', linestyle='--',
                       label=f'Mean: {df["abstract_token_count"].mean():.1f}')
    axes[1, 1].set_xlabel('Số token')
    axes[1, 1].set_ylabel('Tần suất')
    axes[1, 1].set_title(f'Phân bố số token - Abstracts ({split_name})')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{split_name}_length_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📈 Đã lưu biểu đồ: {split_name}_length_distribution.png")


# =========================================================
# 5️⃣ HÀM LƯU DỮ LIỆU ĐA ĐỊNH DẠNG
# =========================================================
def save_processed_data(df, output_dir, split_name, tokenizer):
    """Lưu dữ liệu đã xử lý với nhiều định dạng"""

    # 1. CSV đầy đủ (để xem và debug)
    csv_path = os.path.join(output_dir, f"{split_name}_full.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"💾 CSV full: {csv_path}")

    # 2. CSV gọn (chỉ text)
    csv_light_path = os.path.join(output_dir, f"{split_name}.csv")
    light_columns = ['article_id', 'article_text_processed', 'abstract_text_processed']
    df[light_columns].to_csv(csv_light_path, index=False, encoding='utf-8')
    print(f"💾 CSV light: {csv_light_path}")

    # 3. Pickle (để training nhanh)
    pkl_path = os.path.join(output_dir, f"{split_name}.pkl")
    df.to_pickle(pkl_path)
    print(f"💾 Pickle: {pkl_path}")

    # 4. Numpy arrays (training cực nhanh)
    arrays_path = os.path.join(output_dir, f"{split_name}_arrays.npz")

    # Chuẩn bị dữ liệu cho T5
    input_ids = []
    attention_mask = []
    labels = []

    for _, row in tqdm(df.iterrows(), desc="Preparing arrays", total=len(df)):
        # Encode article
        article_enc = tokenizer(
            "summarize: " + row['article_text_processed'],
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Encode abstract
        abstract_enc = tokenizer(
            row['abstract_text_processed'],
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids.append(article_enc['input_ids'].squeeze().numpy())
        attention_mask.append(article_enc['attention_mask'].squeeze().numpy())
        labels.append(abstract_enc['input_ids'].squeeze().numpy())

    np.savez_compressed(
        arrays_path,
        input_ids=np.array(input_ids),
        attention_mask=np.array(attention_mask),
        labels=np.array(labels)
    )
    print(f"💾 Arrays: {arrays_path}")

    # 5. Lưu metadata
    metadata = {
        'split': split_name,
        'num_samples': len(df),
        'avg_article_words': df['article_word_count'].mean(),
        'avg_abstract_words': df['abstract_word_count'].mean(),
        'avg_article_tokens': df['article_token_count'].mean(),
        'avg_abstract_tokens': df['abstract_token_count'].mean(),
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    metadata_path = os.path.join(output_dir, f"{split_name}_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"💾 Metadata: {metadata_path}")


# =========================================================
# 6️⃣ CLASS DATASET CHO TRAINING
# =========================================================
class PubMedDataset(torch.utils.data.Dataset):
    """Dataset cho training với T5"""

    def __init__(self, arrays_path):
        self.data = np.load(arrays_path)

    def __len__(self):
        return len(self.data['input_ids'])

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.data['input_ids'][idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.data['attention_mask'][idx], dtype=torch.long),
            'labels': torch.tensor(self.data['labels'][idx], dtype=torch.long)
        }


# =========================================================
# 7️⃣ MAIN PIPELINE HOÀN CHỈNH
# =========================================================
def main():
    print("🚀 XỬ LÝ DATASET PUBMED - COMPLETE PIPELINE")
    print("=" * 60)

    # Thiết lập đường dẫn
    input_dir = r"D:\Quân\project\summary_text\data\raw\pubmed-dataset"
    output_dir = r"D:\Quân\project\summary_text\data\processed_pubmed"
    os.makedirs(output_dir, exist_ok=True)

    # Khởi tạo tokenizer T5
    print("🔧 Khởi tạo T5 tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    # Xử lý từng split
    splits = ["train", "val", "test"]
    summary_stats = {}

    for split in splits:
        print(f"\n📦 {'=' * 25} {split.upper()} {'=' * 25}")

        # Load data
        file_path = os.path.join(input_dir, f"{split}.txt")
        if not os.path.exists(file_path):
            print(f"❌ Không tìm thấy: {file_path}")
            continue

        df = load_jsonl_to_df(file_path)
        print(f"📊 Loaded: {len(df):,} samples")

        # Lọc dữ liệu cơ bản
        initial_count = len(df)
        df = df.dropna()
        df = df[(df['article_text'] != "") & (df['abstract_text'] != "")]
        print(f"🧹 Filtered NaN/empty: {initial_count} → {len(df)} samples")

        if len(df) == 0:
            print("❌ Không có dữ liệu sau khi lọc")
            continue

        # Làm sạch text
        print("🧹 Cleaning text...")
        df['article_text_clean'] = df['article_text'].apply(clean_text_advanced)
        df['abstract_text_clean'] = df['abstract_text'].apply(clean_text_advanced)

        # Lọc theo độ dài từ
        print("📏 Filtering by length...")
        df['article_word_count'] = df['article_text_clean'].apply(lambda x: len(x.split()))
        df['abstract_word_count'] = df['abstract_text_clean'].apply(lambda x: len(x.split()))

        initial_count = len(df)

        # Lọc: article < 800 từ, abstract < 300 từ
        df = df[
            (df['article_word_count'] < 800) &
            (df['article_word_count'] > 50) &  # Ít nhất 50 từ
            (df['abstract_word_count'] < 300) &
            (df['abstract_word_count'] > 10)  # Ít nhất 10 từ
            ]

        print(f"🧹 Filtered by length: {initial_count} → {len(df)} samples")

        if len(df) == 0:
            print("❌ Không có dữ liệu sau khi lọc độ dài")
            continue

        # Xử lý văn bản dài
        print("🔧 Processing long texts...")
        df['article_text_processed'] = df['article_text_clean'].apply(
            lambda x: process_long_text(x, tokenizer, max_tokens=512, strategy="smart")
        )
        df['abstract_text_processed'] = df['abstract_text_clean'].apply(
            lambda x: process_long_text(x, tokenizer, max_tokens=128, strategy="smart")
        )

        # Phân tích dataset
        df = analyze_dataset(df, tokenizer, split)

        # Visualize
        visualize_length_distribution(df, output_dir, split)

        # Lưu dữ liệu
        save_processed_data(df, output_dir, split, tokenizer)

        # Lưu thống kê
        summary_stats[split] = {
            'final_samples': len(df),
            'avg_article_words': df['article_word_count'].mean(),
            'avg_abstract_words': df['abstract_word_count'].mean(),
            'avg_article_tokens': df['article_token_count'].mean(),
            'avg_abstract_tokens': df['abstract_token_count'].mean()
        }

        print(f"✅ {split.upper()} completed: {len(df):,} samples")

    # Lưu summary toàn bộ dataset
    print(f"\n📈 {'=' * 30} TỔNG KẾT {'=' * 30}")
    total_samples = sum(stats['final_samples'] for stats in summary_stats.values())
    print(f"📊 Tổng số mẫu: {total_samples:,}")

    for split, stats in summary_stats.items():
        print(f"\n🔹 {split.upper()}:")
        print(f"   Samples: {stats['final_samples']:,}")
        print(f"   Article: {stats['avg_article_words']:.1f} từ, {stats['avg_article_tokens']:.1f} token")
        print(f"   Abstract: {stats['avg_abstract_words']:.1f} từ, {stats['avg_abstract_tokens']:.1f} token")

    # Lưu summary file
    summary_path = os.path.join(output_dir, "dataset_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_stats, f, indent=2, ensure_ascii=False)

    print(f"\n✅ HOÀN TẤT! Dữ liệu đã lưu tại: {output_dir}")
    print(f"📁 Files created:")
    print(f"   • [split].csv, [split].pkl, [split]_arrays.npz")
    print(f"   • [split]_length_distribution.png")
    print(f"   • [split]_metadata.json")
    print(f"   • dataset_summary.json")

    print(f"\n🎯 CÁCH SỬ DỤNG CHO TRAINING:")
    print(f"   from torch.utils.data import DataLoader")
    print(f"   train_data = PubMedDataset('data/processed_pubmed/train_arrays.npz')")
    print(f"   train_loader = DataLoader(train_data, batch_size=16, shuffle=True)")


if __name__ == "__main__":
    main()