import os
import re
import time
import pickle
import numpy as np
import pandas as pd
from datetime import timedelta
from collections import Counter
from tqdm import tqdm
from rouge import Rouge

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ============================================================================
# PHẦN 1: LOAD VÀ CHUẨN BỊ DỮ LIỆU
# ============================================================================

print("\n" + "="*80)
print("📂 PHẦN 1: LOAD DỮ LIỆU TỪ CSV")
print("="*80)

# === CẤU HÌNH ===
data_dir = r"D:\Quân\project\summary_text\data\processed_pubmed"
splits = ["train", "val", "test"]

MAX_TEXT_LEN = 200
MAX_SUMM_LEN = 40
VOCAB_SIZE = 20000

# === HÀM LOAD DỮ LIỆU ===
def load_data(split):
    file_path = os.path.join(data_dir, f"{split}_encoded_clean.csv")
    if not os.path.exists(file_path):
        print(f"⚠️ File {file_path} không tồn tại!")
        return pd.DataFrame(columns=["article_text", "abstract_text"])
    df = pd.read_csv(file_path).dropna(subset=["article_text", "abstract_text"])
    print(f"✅ {split}: {len(df):,} dòng hợp lệ")
    return df

# === LOAD CẢ 3 SPLIT ===
load_start = time.time()
train_df = load_data("train")
val_df   = load_data("val")
test_df  = load_data("test")
load_time = time.time() - load_start

print(f"\n⏱️ Thời gian load: {load_time:.2f}s")

# ============================================================================
# TOKENIZER THUẦN PYTHON (THAY CHO KERAS)
# ============================================================================

class SimpleTokenizer:
    def __init__(self, num_words=20000, oov_token="<unk>"):
        self.num_words = num_words
        self.oov_token = oov_token
        self.word_index = {"<pad>": 0, "<unk>": 1, "<start>": 2, "<end>": 3}
        self.index_word = {v: k for k, v in self.word_index.items()}
        self.next_index = 4

    def _clean_text(self, text):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        return text.strip()

    def fit_on_texts(self, texts):
        counter = Counter()
        for t in texts:
            tokens = self._clean_text(t).split()
            counter.update(tokens)

        most_common = counter.most_common(self.num_words - len(self.word_index))
        for word, _ in most_common:
            if word not in self.word_index:
                self.word_index[word] = self.next_index
                self.index_word[self.next_index] = word
                self.next_index += 1

    def texts_to_sequences(self, texts):
        sequences = []
        for t in texts:
            tokens = self._clean_text(t).split()
            seq = [self.word_index.get(w, self.oov_index) for w in tokens]
            sequences.append(seq)
        return sequences


def pad_sequences_torch(sequences, maxlen, padding="post", truncating="post"):
    padded = []
    for seq in sequences:
        if len(seq) > maxlen:
            seq = seq[:maxlen] if truncating == "post" else seq[-maxlen:]
        elif len(seq) < maxlen:
            pad_len = maxlen - len(seq)
            seq = seq + [0]*pad_len if padding == "post" else [0]*pad_len + seq
        padded.append(seq)
    return torch.LongTensor(padded)

# === TOKENIZER (fit trên train) ===
print("\n🔤 Tạo tokenizer...")
tokenize_start = time.time()

tokenizer_x = SimpleTokenizer(num_words=VOCAB_SIZE, oov_token="<unk>")
tokenizer_x.fit_on_texts(train_df["article_text"])

tokenizer_y = SimpleTokenizer(num_words=VOCAB_SIZE, oov_token="<unk>")
tokenizer_y.fit_on_texts(train_df["abstract_text"])

print(f"✅ Vocab size X: {len(tokenizer_x.word_index):,}")
print(f"✅ Vocab size Y: {len(tokenizer_y.word_index):,}")

# === CHUYỂN THÀNH SEQUENCE ===
def encode_data(df, tokenizer_x, tokenizer_y):
    X = tokenizer_x.texts_to_sequences(df["article_text"])
    # Thêm <start> và <end> cho abstract
    Y_raw = tokenizer_y.texts_to_sequences(df["abstract_text"])
    Y = []
    for seq in Y_raw:
        if seq:  # tránh rỗng
            Y.append([2] + seq + [3])  # <start> ... <end>
        else:
            Y.append([2, 3])

    X = pad_sequences_torch(X, maxlen=MAX_TEXT_LEN)
    Y = pad_sequences_torch(Y, maxlen=MAX_SUMM_LEN + 2)  # +2 vì <start>, <end>
    return X, Y

print("🔄 Encoding sequences...")
X_train, Y_train = encode_data(train_df, tokenizer_x, tokenizer_y)
X_val, Y_val     = encode_data(val_df, tokenizer_x, tokenizer_y)
X_test, Y_test   = encode_data(test_df, tokenizer_x, tokenizer_y)

tokenize_time = time.time() - tokenize_start

print("\n✅ SHAPE:")
print(f"X_train: {X_train.shape}")
print(f"Y_train: {Y_train.shape}")
print(f"X_val:   {X_val.shape}")
print(f"Y_val:   {Y_val.shape}")
print(f"X_test:  {X_test.shape}")
print(f"Y_test:  {Y_test.shape}")
print(f"\n⏱️ Thời gian tokenize: {tokenize_time:.2f}s")

# ============================================================================
# PHẦN 2: CẤU HÌNH GPU
# ============================================================================

print("\n" + "="*80)
print("🔧 PHẦN 2: CẤU HÌNH GPU")
print("="*80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ CUDA Version: {torch.version.cuda}")
    print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    print("✅ cuDNN benchmark enabled")
else:
    print("⚠️ KHÔNG TÌM THẤY GPU - Train trên CPU (rất chậm!)")
print(f"✅ Device: {device}")

# ============================================================================
# PHẦN 3: HYPERPARAMETERS
# ============================================================================

EMBEDDING_DIM = 256
HIDDEN_DIM = 512
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.0005
GRAD_CLIP = 5.0

# ============================================================================
# PHẦN 4: DATASET CLASS
# ============================================================================

class SummarizationDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# ============================================================================
# PHẦN 5: MODEL ARCHITECTURE
# ============================================================================

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        decoder_hidden = decoder_hidden.unsqueeze(1)
        score = self.V(torch.tanh(self.W1(encoder_outputs) + self.W2(decoder_hidden)))
        attn_weights = torch.softmax(score, dim=1)
        context_vector = torch.sum(attn_weights * encoder_outputs, dim=1)
        return context_vector, attn_weights


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell


class DecoderWithAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.attention = BahdanauAttention(hidden_dim)
        self.lstm = nn.LSTM(embedding_dim + hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden, cell, encoder_outputs):
        embedded = self.embedding(x)  # [B,1,emb]
        context, attn_weights = self.attention(hidden.squeeze(0), encoder_outputs)  # hidden: [1,B,H]
        context = context.unsqueeze(1)  # [B,1,H]
        lstm_input = torch.cat([embedded, context], dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        prediction = self.fc(output)  # [B,1,vocab]
        return prediction.squeeze(1), hidden, cell, attn_weights


class Seq2SeqWithAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.encoder = Encoder(vocab_size, embedding_dim, hidden_dim)
        self.decoder = DecoderWithAttention(vocab_size, embedding_dim, hidden_dim)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        vocab_size = self.decoder.fc.out_features

        encoder_outputs, hidden, cell = self.encoder(src)
        outputs = torch.zeros(batch_size, trg_len, vocab_size, device=src.device)

        decoder_input = trg[:, 0].unsqueeze(1)  # <start> token

        for t in range(1, trg_len):
            prediction, hidden, cell, _ = self.decoder(decoder_input, hidden, cell, encoder_outputs)
            outputs[:, t] = prediction
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = prediction.argmax(1)
            decoder_input = trg[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)

        return outputs

# ============================================================================
# PHẦN 6: TRAINING & VALIDATION
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device, epoch, total_epochs):
    model.train()
    epoch_loss = 0
    epoch_start = time.time()
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs}")

    for src, trg in pbar:
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio=0.5)
        loss = criterion(output.reshape(-1, output.shape[-1]), trg.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        epoch_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    return epoch_loss / len(dataloader), time.time() - epoch_start


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, trg in dataloader:
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, teacher_forcing_ratio=0)
            loss = criterion(output.reshape(-1, output.shape[-1]), trg.reshape(-1))
            epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

# ============================================================================
# PHẦN 7: INFERENCE & ROUGE
# ============================================================================

def decode_sequence(model, src, tokenizer_y, max_len, device, start_token_idx=2):
    model.eval()
    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src)

        # Bắt đầu bằng <start> token (thường là 2 nếu bạn dùng <pad>=0, <unk>=1, <start>=2)
        decoder_input = torch.tensor([[start_token_idx]], device=device, dtype=torch.long)
        decoded_tokens = []

        for _ in range(max_len):
            prediction, hidden, cell, _ = model.decoder(decoder_input, hidden, cell, encoder_outputs)
            token = prediction.argmax(dim=1, keepdim=True)  # [1,1]

            token_id = token.item()
            if token_id in [0, 3]:  # <pad> → dừng
                break
            if token_id != 1:  # bỏ qua <unk>
                decoded_tokens.append(token_id)

            decoder_input = token  # teacher forcing trong inference

        return decoded_tokens


def evaluate_rouge(model, X_test, Y_test, tokenizer_y, max_summ_len, device, num_samples=500):
    print("\n" + "="*80)
    print("🔍 ĐÁNH GIÁ MODEL TRÊN TEST SET")
    print("="*80)

    eval_start = time.time()
    num_samples = min(num_samples, len(X_test))
    predictions, references = [], []
    reverse_word_index = {v: k for k, v in tokenizer_y.word_index.items()}

    print(f"Đang tạo {num_samples} predictions...")
    # Trong evaluate_rouge:
    start_token_idx = tokenizer_y.word_index.get("<start>", 2)

    for i in tqdm(range(num_samples)):
        src = X_test[i:i + 1].to(device)
        decoded_tokens = decode_sequence(model, src, tokenizer_y, MAX_SUMM_LEN, device, start_token_idx)

        pred_clean = [t for t in decoded_tokens if t > 3]
        predicted_text = ' '.join([reverse_word_index.get(idx, '') for idx in pred_clean])

        ref_tokens = Y_test[i].tolist()
        ref_clean = [t for t in ref_tokens if t > 3 and t != 0]
        reference_text = ' '.join([reverse_word_index.get(idx, '') for idx in ref_clean])

        predictions.append(predicted_text.strip())
        references.append(reference_text.strip())

    rouge = Rouge()
    rouge_scores = rouge.get_scores(predictions, references, avg=True) if predictions else \
        {'rouge-1': {'p':0,'r':0,'f':0}, 'rouge-2':{'p':0,'r':0,'f':0}, 'rouge-l':{'p':0,'r':0,'f':0}}
    eval_time = time.time() - eval_start

    print("\n📈 KẾT QUẢ ROUGE")
    for m in ['rouge-1','rouge-2','rouge-l']:
        print(f"{m.upper()} - P:{rouge_scores[m]['p']:.4f} R:{rouge_scores[m]['r']:.4f} F:{rouge_scores[m]['f']:.4f}")
    print(f"\n⏱️ Thời gian đánh giá: {str(timedelta(seconds=int(eval_time)))}")
    return rouge_scores

# ============================================================================
# PHẦN 8: MAIN TRAINING
# ============================================================================

def main():
    total_start = time.time()
    print("\n" + "="*80)
    print("🚀 BẮT ĐẦU TRAINING")
    print("="*80)

    train_loader = DataLoader(SummarizationDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(SummarizationDataset(X_val, Y_val), batch_size=BATCH_SIZE)

    model = Seq2SeqWithAttention(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 5
    epoch_times, train_losses, val_losses = [], [], []

    for epoch in range(EPOCHS):
        train_loss, epoch_time = train_epoch(model, train_loader, optimizer, criterion, device, epoch, EPOCHS)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        epoch_times.append(epoch_time); train_losses.append(train_loss); val_losses.append(val_loss)

        print(f"\nEpoch {epoch+1}: Train {train_loss:.4f} | Val {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss; patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("✅ Lưu best model!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⚠️ Early stopping!")
                break

    print("\n📂 Load best model để đánh giá...")
    model.load_state_dict(torch.load("best_model.pth"))
    rouge_scores = evaluate_rouge(model, X_test, Y_test, tokenizer_y, MAX_SUMM_LEN, device)

    with open('tokenizer_x.pkl', 'wb') as f: pickle.dump(tokenizer_x, f)
    with open('tokenizer_y.pkl', 'wb') as f: pickle.dump(tokenizer_y, f)
    print("\n✅ Đã lưu tokenizers!")

    print("\nTổng thời gian:", str(timedelta(seconds=int(time.time() - total_start))))
    return model, rouge_scores

# ============================================================================
# CHẠY TRAINING
# ============================================================================

if __name__ == "__main__":
    model, rouge_scores = main()
    print("\n🎉 DONE!")
