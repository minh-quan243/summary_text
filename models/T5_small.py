import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    AdamW,
    get_linear_schedule_with_warmup
)
from torch.cuda.amp import autocast, GradScaler

# ==========================================================
# ⚙️ CONFIG
# ==========================================================
MODEL_NAME = "t5-small"
BATCH_SIZE = 16
EPOCHS = 10
LR = 2e-5               # Giảm nhẹ LR để tránh lặp token
PATIENCE = 2            # Early stop nếu val loss không giảm sau 2 epoch
MAX_LEN_INPUT = 512
MAX_LEN_TARGET = 128
DATA_PATH = r"D:/Quân/project/summary_text/data/processed_pubmed"
OUTPUT_DIR = r"D:/Quân/project/summary_text/checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================================
# 📂 DATASET
# ==========================================================
class SummarizationDataset(Dataset):
    def __init__(self, df, tokenizer, max_len_input, max_len_target):
        self.inputs = df["article_text_processed"].astype(str).tolist()
        self.targets = df["abstract_text_processed"].astype(str).tolist()
        self.tokenizer = tokenizer
        self.max_len_input = max_len_input
        self.max_len_target = max_len_target

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_ = "summarize: " + self.inputs[idx]
        target = self.targets[idx]

        input_enc = self.tokenizer(
            input_, truncation=True, padding="max_length",
            max_length=self.max_len_input, return_tensors="pt"
        )
        target_enc = self.tokenizer(
            target, truncation=True, padding="max_length",
            max_length=self.max_len_target, return_tensors="pt"
        )

        return {
            "input_ids": input_enc["input_ids"].squeeze(),
            "attention_mask": input_enc["attention_mask"].squeeze(),
            "labels": target_enc["input_ids"].squeeze(),
        }

# ==========================================================
# 🚀 TRAIN + VALIDATION
# ==========================================================
def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, epoch):
    model.train()
    total_loss = 0
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]", leave=True)
    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        with autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)


def validate(model, dataloader, epoch):
    model.eval()
    total_loss = 0
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]", leave=False)
    with torch.no_grad():
        for batch in loop:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            total_loss += loss.item()
    return total_loss / len(dataloader)

# ==========================================================
# 🧠 MAIN
# ==========================================================
def main():
    print("🚀 Loading tokenizer and model...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)

    print("📚 Loading CSVs...")
    train_df = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_PATH, "val.csv"))

    print("🔧 Building datasets...")
    train_data = SummarizationDataset(train_df, tokenizer, MAX_LEN_INPUT, MAX_LEN_TARGET)
    val_data = SummarizationDataset(val_df, tokenizer, MAX_LEN_INPUT, MAX_LEN_TARGET)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    optimizer = AdamW(model.parameters(), lr=LR)
    scaler = GradScaler()

    # Scheduler warmup
    num_training_steps = len(train_loader) * EPOCHS
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, "t5_small_best")

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    patience_counter = 0

    print("🔥 Starting training...")
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, epoch)
        val_loss = validate(model, val_loader, epoch)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"✅ Epoch {epoch+1} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"💾 Model saved at epoch {epoch+1} ✅")
        else:
            patience_counter += 1
            print(f"⚠️ No improvement ({patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print("⏹️ Early stopping triggered!")
                break

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "learning_curve.png"))
    print(f"📈 Saved learning curve to {OUTPUT_DIR}/learning_curve.png")


if __name__ == "__main__":
    main()
