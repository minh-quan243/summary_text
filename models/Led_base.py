import os
import json
import logging
import gc
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch

from transformers import (
    LEDTokenizer,
    LEDForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset, DatasetDict
from rouge_score import rouge_scorer
import nltk

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -----------------------------
# LEDScientificSummarizer
# -----------------------------
class LEDScientificSummarizer:
    def __init__(
        self,
        model_name="allenai/led-base-16384-arxiv",
        max_input_length=1028,
        max_target_length=256,
        use_accelerate_offload=True,
    ):
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"🔬 Loading LED model: {model_name}")

        # Tokenizer
        self.tokenizer = LEDTokenizer.from_pretrained(model_name)

        # Load model with low_cpu_mem_usage to reduce peak CPU memory while loading
        # Keep model.config unchanged
        self.model = LEDForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        )

        # By default move to device; if accelerate offload is used we'll dispatch later
        self.use_accelerate_offload = use_accelerate_offload
        if not use_accelerate_offload:
            self.model.to(self.device)

        # Enable gradient checkpointing to reduce GPU memory usage during training
        try:
            self.model.gradient_checkpointing_enable()
            logger.info("✅ Gradient checkpointing enabled")
        except Exception:
            logger.warning("⚠️ Gradient checkpointing not available for this model build")

        # Disable caching during training/backward to save memory (this does not change architecture)
        self.model.config.use_cache = False

        # Optional: use accelerate dispatch to offload weights/buffers if accelerate is available
        if use_accelerate_offload:
            try:
                # dispatch_model helps place model parts on CPU/GPU automatically
                from accelerate import dispatch_model

                # Use device_map="auto" with offload_buffers to reduce GPU pressure
                self.model = dispatch_model(self.model, device_map="auto", offload_buffers=True)
                logger.info("✅ Model dispatched with accelerate (device_map=auto, offload_buffers=True)")
            except Exception as e:
                logger.warning(f"⚠️ accelerate dispatch failed or not installed: {e}. Falling back to .to(device)")
                self.model.to(self.device)

        # Add domain-specific special tokens if needed
        self.setup_special_tokens()

        logger.info(f"✅ Model ready. Device: {self.device}")
        logger.info(f"📐 Max input: {self.max_input_length}, Max target: {self.max_target_length}")

        # free CPU memory
        gc.collect()

    def setup_special_tokens(self):
        scientific_tokens = ["<formula>", "<equation>", "<citation>", "<method>", "<result>"]
        added = self.tokenizer.add_tokens(scientific_tokens)
        if added:
            self.model.resize_token_embeddings(len(self.tokenizer))
            logger.info(f"🔹 Added {added} special tokens and resized embeddings")

    def preprocess_function(self, examples):
        """
        Tokenize and return python lists (not tensors) to keep memory low.
        This function is intended to be used with `dataset.map(batched=True)`.
        """
        # Tokenize inputs (return lists)
        model_inputs = self.tokenizer(
            examples["article_text_clean"],
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
        )

        # Tokenize targets (labels)
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                examples["abstract_text_clean"],
                max_length=self.max_target_length,
                padding="max_length",
                truncation=True,
            )

        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "labels": labels["input_ids"],
        }

    def prepare_dataset(self, dataset_dict, batch_size_map=1000, num_proc=1):
        logger.info("📊 Preparing/tokenizing dataset (batched)...")

        tokenized = {}
        for split_name, ds in dataset_dict.items():
            # Use batched map to avoid building huge intermediate structures
            tokenized[split_name] = ds.map(
                self.preprocess_function,
                batched=True,
                batch_size=batch_size_map,
                num_proc=num_proc,
                remove_columns=ds.column_names,
                desc=f"Tokenizing {split_name}",
            )

        # Return DatasetDict with arrow-backed tokenized data (low memory)
        return DatasetDict(tokenized)


# -----------------------------
# Trainer wrapper
# -----------------------------
class LEDTrainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ⚙️ 1. Bật Gradient Checkpointing (giảm RAM)
        try:
            self.model.gradient_checkpointing_enable()
            print("✅ Gradient checkpointing enabled")
        except Exception as e:
            print(f"⚠️ Gradient checkpointing not supported: {e}")

        # ⚙️ 2. Bật xFormers (tăng tốc attention, giảm VRAM)
        try:
            self.model.enable_xformers_memory_efficient_attention()
            print("✅ xFormers memory efficient attention enabled")
        except Exception as e:
            print(f"⚠️ xFormers not available: {e}")

        # ⚙️ 4. Đưa model sang GPU
        self.model.to(self.device)
        print(f"🚀 Model ready on {self.device}")

    def setup_training_args(self, output_dir="./led-scientific", **kwargs):
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=kwargs.get("num_train_epochs", 3),
            per_device_train_batch_size=kwargs.get("per_device_train_batch_size", 2),
            per_device_eval_batch_size=kwargs.get("per_device_eval_batch_size", 2),
            gradient_accumulation_steps=kwargs.get("gradient_accumulation_steps", 8),
            warmup_steps=kwargs.get("warmup_steps", 500),
            learning_rate=kwargs.get("learning_rate", 5e-5),
            weight_decay=kwargs.get("weight_decay", 0.01),
            logging_steps=kwargs.get("logging_steps", 100),
            eval_steps=kwargs.get("eval_steps", 500),
            save_steps=kwargs.get("save_steps", 1000),
            evaluation_strategy=kwargs.get("evaluation_strategy", "steps"),
            save_strategy=kwargs.get("save_strategy", "steps"),
            predict_with_generate=True,
            generation_max_length=kwargs.get("generation_max_length", 512),
            generation_num_beams=kwargs.get("generation_num_beams", 4),
            load_best_model_at_end=True,
            metric_for_best_model="rougeL",
            greater_is_better=True,
            fp16=False,
            bf16=kwargs.get('bf16', False),
            dataloader_pin_memory=False,
            report_to=None,
            remove_unused_columns=False,
        )
        return training_args

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred

        # decode preds
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # replace -100
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

        rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
        for p, l in zip(decoded_preds, decoded_labels):
            s = scorer.score(p, l)
            rouge1_scores.append(s["rouge1"].fmeasure)
            rouge2_scores.append(s["rouge2"].fmeasure)
            rougeL_scores.append(s["rougeL"].fmeasure)

        return {
            "rouge1": float(np.mean(rouge1_scores)),
            "rouge2": float(np.mean(rouge2_scores)),
            "rougeL": float(np.mean(rougeL_scores)),
        }

    def train(self, train_dataset, eval_dataset, output_dir, **training_kwargs):
        logger.info("🚀 Starting LED training...")

        training_args = self.setup_training_args(output_dir, **training_kwargs)

        # Data collator: dynamic padding per batch
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            pad_to_multiple_of=8,
        )

        # Convert datasets to torch format just before Trainer to save memory
        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        # force GC and empty CUDA cache before training
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )

        trainer.train_dataloader = lambda: torch.utils.data.DataLoader(
            trainer.train_dataset,
            batch_size=training_args.per_device_train_batch_size,
            num_workers=4,
            pin_memory=True
        )

        # Train
        train_result = trainer.train()

        # Save
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)

        # Log and cleanup
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        logger.info("✅ Training completed and model saved")

        # Reset dataset format and collect
        train_dataset.reset_format()
        eval_dataset.reset_format()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return trainer


# -----------------------------
# Evaluator
# -----------------------------
class LEDEvaluator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    def evaluate(self, test_dataset, num_samples=None):
        logger.info("🧪 Evaluating model...")

        if num_samples:
            test_dataset = test_dataset.select(range(min(num_samples, len(test_dataset))))

        all_predictions, all_references = [], []
        self.model.eval()

        with torch.no_grad():
            for i in tqdm(range(len(test_dataset)), desc="Evaluating"):
                ex = test_dataset[i]

                input_ids = torch.tensor(ex["input_ids"]).unsqueeze(0).to(self.device)
                attention_mask = torch.tensor(ex["attention_mask"]).unsqueeze(0).to(self.device)

                summary_ids = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=512,
                    min_length=64,
                    num_beams=4,
                    length_penalty=2.0,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                )

                pred = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

                ref_ids = [tid for tid in ex["labels"] if tid != -100]
                ref = self.tokenizer.decode(ref_ids, skip_special_tokens=True)

                all_predictions.append(pred)
                all_references.append(ref)

        scores = self.compute_rouge_batch(all_predictions, all_references)
        self.print_evaluation_results(scores, all_predictions, all_references)
        return scores, all_predictions, all_references

    def compute_rouge_batch(self, preds, refs):
        r1, r2, rL = [], [], []
        for p, r in zip(preds, refs):
            s = self.scorer.score(p, r)
            r1.append(s["rouge1"].fmeasure)
            r2.append(s["rouge2"].fmeasure)
            rL.append(s["rougeL"].fmeasure)
        return {
            "rouge1": {"mean": float(np.mean(r1)), "std": float(np.std(r1)), "scores": r1},
            "rouge2": {"mean": float(np.mean(r2)), "std": float(np.std(r2)), "scores": r2},
            "rougeL": {"mean": float(np.mean(rL)), "std": float(np.std(rL)), "scores": rL},
        }

    def print_evaluation_results(self, rouge_scores, predictions, references):
        print('\n' + '=' * 60)
        print("🎯 EVALUATION RESULTS")
        print('=' * 60)

        for metric, vals in rouge_scores.items():
            print(f"{metric.upper():<10}: {vals['mean']:.4f} ± {vals['std']:.4f}")

        print('\n📝 SAMPLE PREDICTIONS:')
        print('-' * 40)
        for i in range(min(3, len(predictions))):
            print(f"\n🔹 Sample {i+1}:")
            print(f"Reference: {references[i][:200]}...")
            print(f"Prediction: {predictions[i][:200]}...")
            print('-' * 40)


# -----------------------------
# Main
# -----------------------------
def main():
    MODEL_NAME = "allenai/led-large-16384-arxiv"
    OUTPUT_DIR = r"D:\Quân\project\summary_text\checkpoints\led-scientific"
    DATA_PATH = r"D:\Quân\project\summary_text\data\processed_pubmed"

    TRAINING_CONFIG = {
        "num_train_epochs": 3,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 5e-5,
        "warmup_steps": 500,
        'bf16': True
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data (try HF dataset first)
    try:
        dataset = DatasetDict.load_from_disk(DATA_PATH)
        logger.info(f"✅ Data loaded from disk: {dataset}")
    except Exception:
        logger.warning("⚠️ Could not load DatasetDict from disk, falling back to CSV files")
        train_df = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
        val_df = pd.read_csv(os.path.join(DATA_PATH, "val.csv"))
        test_df = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))

        dataset = DatasetDict({
            "train": Dataset.from_pandas(train_df),
            "validation": Dataset.from_pandas(val_df),
            "test": Dataset.from_pandas(test_df),
        })
        logger.info("✅ Data loaded from CSV files")

    # Initialize model & tokenizer (keep model config unchanged)
    summarizer = LEDScientificSummarizer(MODEL_NAME, use_accelerate_offload=True)

    # Tokenize/prepare dataset (batched mapping) - keeps memory low
    tokenized = summarizer.prepare_dataset(dataset, batch_size_map=200, num_proc=1)

    # Free original dataset references and collect
    del dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Trainer & training
    trainer_wrapper = LEDTrainer(summarizer.model, summarizer.tokenizer)
    trainer = trainer_wrapper.train(
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        output_dir=OUTPUT_DIR,
        **TRAINING_CONFIG,
    )

    # Evaluation on a sample of test set (to save time/memory)
    evaluator = LEDEvaluator(summarizer.model, summarizer.tokenizer)
    rouge_scores, predictions, references = evaluator.evaluate(tokenized["test"], num_samples=100)

    results = {
        "model": MODEL_NAME,
        "training_config": TRAINING_CONFIG,
        "rouge_scores": rouge_scores,
        "sample_predictions": [
            {"reference": ref, "prediction": pred}
            for ref, pred in zip(references[:5], predictions[:5])
        ],
    }

    with open(os.path.join(OUTPUT_DIR, "evaluation_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ Results saved to: {os.path.join(OUTPUT_DIR, 'evaluation_results.json')}")
    logger.info("🎉 Training and evaluation completed!")


if __name__ == "__main__":
    # Prepare nltk for rouge
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    main()
