"""Inference: load base model + LoRA adapter and answer a question.

Usage:
    python -m src.infer --question "What is the parental leave policy?"
"""
from __future__ import annotations

import argparse

from .evaluate import _generate, _load_model
from .settings import get_settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    args = parser.parse_args()

    s = get_settings()
    model, tokenizer = _load_model(s.adapter_dir)
    answer = _generate(model, tokenizer, args.question)
    print(f"\nQ: {args.question}\nA: {answer}\n")


if __name__ == "__main__":
    main()
