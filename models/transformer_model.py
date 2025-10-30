# src/models/transformer_model.py
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class TransformerSummarizer:
    def __init__(self, model_name="facebook/bart-large-cnn", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def summarize(self, text, max_input_len=512, max_summary_len=128, num_beams=4):
        if isinstance(text, list):
            texts = text
        else:
            texts = [text]
        inputs = self.tokenizer(texts, truncation=True, padding=True, max_length=max_input_len, return_tensors="pt").to(self.device)
        summary_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_length=max_summary_len,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
        summaries = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in summary_ids]
        return summaries if len(summaries) > 1 else summaries[0]
