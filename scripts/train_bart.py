# src/train_bart.py
import os
import yaml
import logging
import numpy as np
from evaluate import load  # ✅ dùng 'evaluate' thay vì 'load_metric' (đã deprecated)
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from log_save import setup_logging, set_seed, save_checkpoint
from Load_data import load_csv_to_hf

# ============================================================
# 1️⃣ CẤU HÌNH LOGGING + SEED
# ============================================================

logger = setup_logging("logs/train_bart.log")
set_seed(42)
logger = logging.getLogger(__name__)

# ============================================================
# 2️⃣ HÀM XỬ LÝ DỮ LIỆU
# ============================================================

def preprocess_function(examples, tokenizer, max_input_length, max_target_length):
    """Tokenize dữ liệu input và output."""
    inputs = examples['article_text']
    targets = examples['abstract_text']
    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        truncation=True,
        padding='max_length',
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=max_target_length,
            truncation=True,
            padding='max_length',
        )
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

# ============================================================
# 3️⃣ HÀM ĐÁNH GIÁ (ROUGE)
# ============================================================

def compute_metrics(eval_preds, tokenizer):
    """Tính ROUGE giữa dự đoán và nhãn thực tế."""
    rouge = load('rouge')  # ✅ dùng evaluate.load thay cho load_metric
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}
    return result

# ============================================================
# 4️⃣ HÀM TRAIN CHÍNH
# ============================================================

def load_config(path):

    cfg = yaml.safe_load(open(path))

    # Ép kiểu các tham số cần thiết
    int_keys = ['num_train_epochs', 'per_device_train_batch_size', 'per_device_eval_batch_size', 'logging_steps']
    float_keys = ['learning_rate', 'weight_decay']
    bool_keys = ['fp16', 'predict_with_generate']

    for k in int_keys:
        if k in cfg:
            cfg[k] = int(cfg[k])

    for k in float_keys:
        if k in cfg:
            cfg[k] = float(cfg[k])

    for k in bool_keys:
        if k in cfg:
            cfg[k] = bool(cfg[k])

    return cfg

def main(config_path):
    cfg = load_config(config_path)
    model_name = cfg.get('bart_model_name', 'facebook/bart-large-cnn')

    train_file = os.path.join(cfg['data_dir'], cfg['train_file'])
    val_file = os.path.join(cfg['data_dir'], cfg['val_file'])

    logger.info(f"🚀 Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    logger.info("📂 Loading datasets...")
    train_ds = load_csv_to_hf(train_file)
    val_ds = load_csv_to_hf(val_file)

    logger.info("🔄 Tokenizing datasets...")
    train_ds = train_ds.map(
        lambda ex: preprocess_function(ex, tokenizer, cfg['max_input_length'], cfg['max_target_length']),
        batched=True,
        remove_columns=train_ds.column_names,
    )
    val_ds = val_ds.map(
        lambda ex: preprocess_function(ex, tokenizer, cfg['max_input_length'], cfg['max_target_length']),
        batched=True,
        remove_columns=val_ds.column_names,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(cfg['output_dir'], 'bart'),
        evaluation_strategy=cfg.get('evaluation_strategy', 'epoch'),
        save_strategy=cfg.get('save_strategy', 'epoch'),
        logging_strategy='steps',
        logging_steps=cfg.get('logging_steps', 100),
        learning_rate=cfg.get('learning_rate', 3e-5),
        per_device_train_batch_size=cfg.get('per_device_train_batch_size', 2),
        per_device_eval_batch_size=cfg.get('per_device_eval_batch_size', 2),
        num_train_epochs=cfg.get('num_train_epochs', 3),
        predict_with_generate=True,
        generation_max_length=cfg.get('max_target_length', 128),
        fp16=cfg.get('fp16', True),
        load_best_model_at_end=True,
        metric_for_best_model='rouge1',
        greater_is_better=True,
        report_to="none",  # tránh log ra WandB nếu không dùng
    )

    logger.info("🏋️‍♂️ Starting training...")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer),
    )

    trainer.train()
    logger.info("✅ Training finished!")

    logger.info("💾 Saving final model...")
    trainer.save_model(training_args.output_dir)
    save_checkpoint(model, tokenizer, training_args.output_dir)
    logger.info(f"✅ Model saved to {training_args.output_dir}")

# ============================================================
# 5️⃣ MAIN ENTRY
# ============================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train BART model using YAML config.")
    parser.add_argument('--config', type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)
