import os
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# === CẤU HÌNH ===
data_dir = r"F:\GitHub\Summarize-text\data\processed_pubmed"
splits = ["train", "val", "test"]

MAX_TEXT_LEN = 200
MAX_SUMM_LEN = 40
VOCAB_SIZE = 20000

# === HÀM LOAD DỮ LIỆU ===
def load_data(split):
    file_path = os.path.join(data_dir, f"{split}_encoded_clean.csv")
    if not os.path.exists(file_path):
        print(f"⚠️ File {file_path} không tồn tại!")
        return pd.DataFrame(columns=["article_text", "abstract_text"])  # tránh None
    df = pd.read_csv(file_path).dropna(subset=["article_text", "abstract_text"])
    print(f"✅ {split}: {len(df)} dòng hợp lệ")
    return df

# === LOAD CẢ 3 SPLIT ===
train_df = load_data("train")
val_df   = load_data("val")
test_df  = load_data("test")

# === TOKENIZER (fit trên train) ===
tokenizer_x = Tokenizer(num_words=VOCAB_SIZE, oov_token="<unk>")
tokenizer_x.fit_on_texts(train_df["article_text"])

tokenizer_y = Tokenizer(num_words=VOCAB_SIZE, oov_token="<unk>")
tokenizer_y.fit_on_texts(train_df["abstract_text"])

# === CHUYỂN THÀNH SEQUENCE ===
def encode_data(df, tokenizer_x, tokenizer_y):
    X = tokenizer_x.texts_to_sequences(df["article_text"])
    Y = tokenizer_y.texts_to_sequences(df["abstract_text"])
    X = pad_sequences(X, maxlen=MAX_TEXT_LEN, padding="post", truncating="post")
    Y = pad_sequences(Y, maxlen=MAX_SUMM_LEN, padding="post", truncating="post")
    return X, Y

X_train, Y_train = encode_data(train_df, tokenizer_x, tokenizer_y)
X_val, Y_val     = encode_data(val_df, tokenizer_x, tokenizer_y)
X_test, Y_test   = encode_data(test_df, tokenizer_x, tokenizer_y)

# === IN KIỂM TRA ===
print("\n✅ SHAPE:")
print("X_train:", X_train.shape)
print("Y_train:", Y_train.shape)
print("X_val:", X_val.shape)
print("Y_val:", Y_val.shape)
print("X_test:", X_test.shape)
print("Y_test:", Y_test.shape)
