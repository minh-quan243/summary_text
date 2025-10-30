# src/eval.py
import os
import yaml
import numpy as np
from tqdm import tqdm
from datasets import load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from Load_data import load_csv_to_hf

def evaluate_transformer(
    model_name,
    ckpt_dir,
    test_csv,
    tokenizer_name=None,
    max_input_len=512,
    max_target_len=128,
    num_samples=200,
):
    """Evaluate a transformer summarization model and compute ROUGE scores."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name)
    model_path = ckpt_dir if ckpt_dir and os.path.exists(ckpt_dir) else model_name
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    df = load_csv_to_hf(test_csv)
    rouge = load_metric("rouge")

    preds, refs = [], []
    n = min(num_samples, len(df))

    for i in tqdm(range(n), desc=f"Evaluating {model_name}"):
        text = str(df.iloc[i]["article_text"])
        ref = str(df.iloc[i]["abstract_text"])
        inputs = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_input_len,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                max_length=max_target_len,
                num_beams=4,
                early_stopping=True,
            )

        pred = tokenizer.decode(
            generated_tokens[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        preds.append(pred)
        refs.append(ref)

    result = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
    result = {k: round(float(v.mid.fmeasure * 100), 2) for k, v in result.items()}
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate summarization models using ROUGE.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--model", choices=["bart", "pegasus"], required=True, help="Model type to evaluate.")
    parser.add_argument("--ckpt", required=False, help="Checkpoint directory (fine-tuned) or model path.")
    parser.add_argument("--num_samples", type=int, default=200, help="Number of samples to evaluate.")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    test_csv = os.path.join(cfg["data_dir"], cfg["test_file"])

    if args.model == "bart":
        model_name = cfg.get("bart_model_name", "facebook/bart-large-cnn")
    elif args.model == "pegasus":
        model_name = cfg.get("pegasus_model_name", "Kevincp560/pegasus-arxiv-finetuned-pubmed")

    res = evaluate_transformer(
        model_name=model_name,
        ckpt_dir=args.ckpt,
        test_csv=test_csv,
        tokenizer_name=model_name,
        max_input_len=cfg["max_input_length"],
        max_target_len=cfg["max_target_length"],
        num_samples=args.num_samples,
    )

    print(f"\n✅ ROUGE scores for {args.model.upper()} model:")
    for k, v in res.items():
        print(f"{k}: {v:.2f}")