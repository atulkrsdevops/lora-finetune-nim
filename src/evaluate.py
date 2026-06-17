"""Before/after evaluator — compares base model vs fine-tuned adapter.

Can run without a GPU if the adapter is available (uses CPU inference).
For a quick smoke-run without training, it evaluates the base model only.

Usage:
    python -m src.evaluate
"""
from __future__ import annotations

import pathlib

from .preprocess import load_dataset
from .settings import get_settings


def _load_model(adapter_path: str | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    s = get_settings()
    tokenizer = AutoTokenizer.from_pretrained(s.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        s.base_model, torch_dtype=torch.float32, device_map="cpu"
    )

    if adapter_path and pathlib.Path(adapter_path).exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"Loaded adapter from {adapter_path}")
    else:
        print("No adapter found — evaluating base model only.")

    return model, tokenizer


def _generate(model, tokenizer, question: str, max_new_tokens: int = 80) -> str:
    import torch
    prompt = (
        f"<|system|>\nYou are a helpful HR assistant.\n"
        f"<|user|>\n{question}\n<|assistant|>\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Return only the assistant's reply
    if "<|assistant|>" in decoded:
        return decoded.split("<|assistant|>")[-1].strip()
    return decoded.strip()


def evaluate() -> None:
    s = get_settings()
    eval_rows = load_dataset(s.eval_file)

    try:
        model, tokenizer = _load_model(s.adapter_dir)
    except Exception as exc:
        print(f"Could not load model for evaluation: {exc}")
        print("Install: pip install torch transformers peft")
        return

    correct = 0
    print(f"\nEvaluating on {len(eval_rows)} examples...\n")
    print(f"{'Q':<45} {'Expected':<25} {'Got':<25} Match")
    print("-" * 110)

    for row in eval_rows:
        answer = _generate(model, tokenizer, row["question"])
        match = row["answer"].lower() in answer.lower()
        if match:
            correct += 1
        print(
            f"{row['question'][:43]:<45} "
            f"{row['answer'][:23]:<25} "
            f"{answer[:23]:<25} "
            f"{'YES' if match else 'NO'}"
        )

    print(f"\nAccuracy: {correct}/{len(eval_rows)} = {correct / len(eval_rows):.0%}")


if __name__ == "__main__":
    evaluate()
