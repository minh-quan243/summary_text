# src/utils/data_utils.py
from datasets import load_dataset


def load_csv_to_hf(path, text_col='article_text', summary_col='abstract_text'):
    """
    Load CSV file trực tiếp thành HuggingFace Dataset.
    """
    ds = load_dataset("csv", data_files=path)["train"]

    # Nếu cột tên khác, rename lại
    if text_col != "article_text" or summary_col != "abstract_text":
        ds = ds.rename_columns({text_col: "article_text", summary_col: "abstract_text"})

    # Loại bỏ các dòng có null
    ds = ds.filter(lambda x: x["article_text"] is not None and x["abstract_text"] is not None)
    return ds
