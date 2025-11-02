import os
import torch
import numpy as np
import pandas as pd
from transformers import (
    LEDTokenizer, LEDForConditionalGeneration,
    LEDConfig, Seq2SeqTrainingArguments, Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset, DatasetDict
from rouge_score import rouge_scorer
import nltk
from tqdm import tqdm
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LEDScientificSummarizer:
    def __init__(self, model_name="allenai/led-large-16384-arxiv", max_input_length=16384, max_target_length=512):
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"🔬 Loading LED model: {model_name}")

        # Load tokenizer và model
        self.tokenizer = LEDTokenizer.from_pretrained(model_name)
        self.model = LEDForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
        ).to(self.device)

        # Special tokens cho scientific text
        self.setup_special_tokens()

        logger.info(f"✅ Model loaded on {self.device}")
        logger.info(f"📐 Max input: {max_input_length}, Max target: {max_target_length}")

    def setup_special_tokens(self):
        """Thêm special tokens cho scientific text nếu cần"""
        scientific_tokens = ["<formula>", "<equation>", "<citation>", "<method>", "<result>"]
        self.tokenizer.add_tokens(scientific_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def preprocess_function(self, examples):
        """
        Preprocess data for LED model
        """
        # Tokenize inputs (articles)
        model_inputs = self.tokenizer(
            examples["article_text_clean"],
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Tokenize targets (abstracts)
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                examples["abstract_text_clean"],
                max_length=self.max_target_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def prepare_dataset(self, dataset_dict):
        """
        Chuẩn bị dataset cho training
        """
        logger.info("📊 Preparing dataset...")

        # Preprocess tất cả splits
        tokenized_datasets = {}
        for split_name, dataset in dataset_dict.items():
            tokenized_datasets[split_name] = dataset.map(
                self.preprocess_function,
                batched=True,
                remove_columns=dataset.column_names,
                desc=f"Processing {split_name}"
            )

        return DatasetDict(tokenized_datasets)


class LEDTrainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device

    def setup_training_args(self, output_dir="./led-scientific", **kwargs):
        """
        Thiết lập training arguments cho LED
        """
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=kwargs.get('num_train_epochs', 3),
            per_device_train_batch_size=kwargs.get('per_device_train_batch_size', 1),  # LED lớn nên batch size nhỏ
            per_device_eval_batch_size=kwargs.get('per_device_eval_batch_size', 1),
            gradient_accumulation_steps=kwargs.get('gradient_accumulation_steps', 8),  # Accumulate gradients
            warmup_steps=kwargs.get('warmup_steps', 500),
            learning_rate=kwargs.get('learning_rate', 5e-5),
            weight_decay=kwargs.get('weight_decay', 0.01),
            logging_steps=kwargs.get('logging_steps', 100),
            eval_steps=kwargs.get('eval_steps', 500),
            save_steps=kwargs.get('save_steps', 1000),
            evaluation_strategy=kwargs.get('evaluation_strategy', "steps"),
            save_strategy=kwargs.get('save_strategy', "steps"),
            predict_with_generate=True,
            generation_max_length=kwargs.get('generation_max_length', 512),
            generation_num_beams=kwargs.get('generation_num_beams', 4),
            load_best_model_at_end=True,
            metric_for_best_model="rougeL",
            greater_is_better=True,
            fp16=kwargs.get('fp16', self.device.type == "cuda"),
            dataloader_pin_memory=False,
            report_to=None,  # Disable wandb/etc unless specified
        )

        return training_args

    def compute_metrics(self, eval_pred):
        """
        Compute ROUGE metrics for evaluation
        """
        predictions, labels = eval_pred

        # Decode predictions
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # Replace -100 in labels as we can't decode them
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Compute ROUGE scores
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []

        for pred, label in zip(decoded_preds, decoded_labels):
            scores = scorer.score(pred, label)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)

        return {
            'rouge1': np.mean(rouge1_scores),
            'rouge2': np.mean(rouge2_scores),
            'rougeL': np.mean(rougeL_scores),
        }

    def train(self, train_dataset, eval_dataset, output_dir, **training_kwargs):
        """
        Fine-tune LED model
        """
        logger.info("🚀 Starting LED training...")

        # Setup training arguments
        training_args = self.setup_training_args(output_dir, **training_kwargs)

        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            pad_to_multiple_of=8
        )

        # Initialize trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )

        # Start training
        logger.info("🎯 Training started...")
        train_result = trainer.train()

        # Save final model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)

        # Log training results
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        logger.info(f"✅ Training completed! Model saved to: {output_dir}")
        return trainer


class LEDEvaluator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def evaluate(self, test_dataset, num_samples=None):
        """
        Đánh giá model trên test dataset
        """
        logger.info("🧪 Evaluating model...")

        if num_samples:
            test_dataset = test_dataset.select(range(min(num_samples, len(test_dataset))))

        all_predictions = []
        all_references = []

        self.model.eval()

        with torch.no_grad():
            for i in tqdm(range(len(test_dataset)), desc="Evaluating"):
                example = test_dataset[i]

                # Generate summary
                input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to(self.device)
                attention_mask = torch.tensor(example["attention_mask"]).unsqueeze(0).to(self.device)

                summary_ids = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=512,
                    min_length=150,
                    num_beams=4,
                    length_penalty=2.0,
                    early_stopping=True,
                    no_repeat_ngram_size=3
                )

                prediction = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                reference = self.tokenizer.decode(
                    [token_id for token_id in example["labels"] if token_id != -100],
                    skip_special_tokens=True
                )

                all_predictions.append(prediction)
                all_references.append(reference)

        # Compute ROUGE scores
        rouge_scores = self.compute_rouge_batch(all_predictions, all_references)

        # Print results
        self.print_evaluation_results(rouge_scores, all_predictions, all_references)

        return rouge_scores, all_predictions, all_references

    def compute_rouge_batch(self, predictions, references):
        """
        Compute ROUGE scores for batch
        """
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []

        for pred, ref in zip(predictions, references):
            scores = self.scorer.score(pred, ref)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)

        return {
            'rouge1': {
                'mean': np.mean(rouge1_scores),
                'std': np.std(rouge1_scores),
                'scores': rouge1_scores
            },
            'rouge2': {
                'mean': np.mean(rouge2_scores),
                'std': np.std(rouge2_scores),
                'scores': rouge2_scores
            },
            'rougeL': {
                'mean': np.mean(rougeL_scores),
                'std': np.std(rougeL_scores),
                'scores': rougeL_scores
            }
        }

    def print_evaluation_results(self, rouge_scores, predictions, references):
        """
        Print evaluation results
        """
        print("\n" + "=" * 60)
        print("🎯 EVALUATION RESULTS")
        print("=" * 60)

        for metric, scores in rouge_scores.items():
            print(f"{metric.upper():<10}: {scores['mean']:.4f} ± {scores['std']:.4f}")

        print("\n📝 SAMPLE PREDICTIONS:")
        print("-" * 40)

        for i in range(min(3, len(predictions))):
            print(f"\n🔹 Sample {i + 1}:")
            print(f"Reference: {references[i][:200]}...")
            print(f"Prediction: {predictions[i][:200]}...")
            print("-" * 40)


def main():
    """
    Main training and evaluation pipeline
    """
    # =========================================================
    # 1. CONFIGURATION
    # =========================================================
    MODEL_NAME = "allenai/led-large-16384-arxiv"
    OUTPUT_DIR = "D:\Quân\project\summary_text\checkpoints\led-scientific"
    DATA_PATH = "D:\Quân\project\summary_text\data\processed_pubmed"

    # Training config
    TRAINING_CONFIG = {
        'num_train_epochs': 3,
        'per_device_train_batch_size': 1,
        'gradient_accumulation_steps': 8,
        'learning_rate': 5e-5,
        'warmup_steps': 500,
    }

    # =========================================================
    # 2. LOAD AND PREPARE DATA
    # =========================================================
    print("📊 Loading data...")

    # Load processed PubMed data
    # Giả sử bạn đã có data ở dạng HuggingFace Dataset
    try:
        dataset = DatasetDict.load_from_disk(DATA_PATH)
        print(f"✅ Data loaded: {dataset}")
    except:
        # Fallback: Load từ CSV
        print("⚠️  Loading from CSV files...")
        train_df = pd.read_csv(f"{DATA_PATH}/train.csv")
        val_df = pd.read_csv(f"{DATA_PATH}/val.csv")
        test_df = pd.read_csv(f"{DATA_PATH}/test.csv")

        dataset = DatasetDict({
            'train': Dataset.from_pandas(train_df),
            'validation': Dataset.from_pandas(val_df),
            'test': Dataset.from_pandas(test_df)
        })
        print(f"✅ Data loaded from CSV: {dataset}")

    # =========================================================
    # 3. INITIALIZE MODEL AND TOKENIZER
    # =========================================================
    print("🔬 Initializing LED model...")
    summarizer = LEDScientificSummarizer(MODEL_NAME)

    # Preprocess dataset
    tokenized_datasets = summarizer.prepare_dataset(dataset)
    print(f"✅ Dataset tokenized: {tokenized_datasets}")

    # =========================================================
    # 4. TRAINING
    # =========================================================
    print("🚀 Starting training...")
    trainer = LEDTrainer(summarizer.model, summarizer.tokenizer)

    trained_trainer = trainer.train(
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        output_dir=OUTPUT_DIR,
        **TRAINING_CONFIG
    )

    # =========================================================
    # 5. EVALUATION
    # =========================================================
    print("🧪 Running evaluation...")
    evaluator = LEDEvaluator(summarizer.model, summarizer.tokenizer)

    rouge_scores, predictions, references = evaluator.evaluate(
        tokenized_datasets["test"],
        num_samples=100  # Evaluate on 100 samples for speed
    )

    # =========================================================
    # 6. SAVE RESULTS
    # =========================================================
    results = {
        'model': MODEL_NAME,
        'training_config': TRAINING_CONFIG,
        'rouge_scores': rouge_scores,
        'sample_predictions': [
            {'reference': ref, 'prediction': pred}
            for ref, pred in zip(references[:5], predictions[:5])
        ]
    }

    with open(f"{OUTPUT_DIR}/evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✅ Results saved to: {OUTPUT_DIR}/evaluation_results.json")
    print("🎉 Training and evaluation completed!")

if __name__ == "__main__":
    # Download nltk data for ROUGE
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    main()