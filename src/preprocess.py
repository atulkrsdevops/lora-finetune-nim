"""Preprocessor: load JSONL dataset, format as chat template, split train/val.

Usage:
    python -m src.preprocess

Outputs a summary of the dataset and validates all records have
'question' and 'answer' fields.
"""
from __future__ import annotations

import json
import pathlib

from .settings import get_settings


def format_prompt(question: str, answer: str) -> str:
    """Format a Q&A pair as a TinyLlama chat prompt."""
    return (
        f"<|system|>\nYou are a helpful HR assistant.\n"
        f"<|user|>\n{question}\n"
        f"<|assistant|>\n{answer}"
    )


def load_dataset(path: str) -> list[dict]:
    rows = []
    for line in pathlib.Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def validate(rows: list[dict]) -> None:
    for i, row in enumerate(rows):
        if "question" not in row or "answer" not in row:
            raise ValueError(f"Row {i} missing 'question' or 'answer': {row}")


def main() -> None:
    s = get_settings()
    train_rows = load_dataset(s.train_file)
    eval_rows = load_dataset(s.eval_file)

    validate(train_rows)
    validate(eval_rows)

    print(f"Train set : {len(train_rows)} examples")
    print(f"Eval set  : {len(eval_rows)} examples")
    print(f"Base model: {s.base_model}")
    print(f"LoRA rank : r={s.lora_r}, alpha={s.lora_alpha}")
    print("\nSample formatted prompt:")
    print("-" * 50)
    sample = train_rows[0]
    print(format_prompt(sample["question"], sample["answer"]))
    print("-" * 50)
    print("\nPreprocessing OK. Run `python -m src.train` on a GPU to fine-tune.")


if __name__ == "__main__":
    main()
