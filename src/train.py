"""LoRA fine-tuning script using PEFT + Transformers.

Run on a free Colab or Kaggle T4 GPU:
    python -m src.train

The script:
1. Loads the base model in 4-bit (QLoRA) to fit on a T4
2. Attaches LoRA adapters to q_proj and v_proj
3. Fine-tunes for num_epochs on the HR policy dataset
4. Saves the adapter to adapter_dir
"""
from __future__ import annotations

import pathlib

from .preprocess import format_prompt, load_dataset
from .settings import get_settings


def train() -> None:
    s = get_settings()

    # Lazy imports — only needed on GPU machine
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
        )
        from trl import SFTTrainer
    except ImportError as exc:
        raise SystemExit(
            f"Missing training dependency: {exc}\n"
            "Run: pip install torch transformers peft trl bitsandbytes datasets\n"
            "This script requires a GPU — run on Colab or Kaggle T4."
        ) from exc

    print(f"Loading base model: {s.base_model}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(s.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        s.base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=s.lora_r,
        lora_alpha=s.lora_alpha,
        lora_dropout=s.lora_dropout,
        target_modules=s.lora_target_modules.split(","),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Build dataset
    rows = load_dataset(s.train_file)
    texts = [format_prompt(r["question"], r["answer"]) for r in rows]
    dataset = Dataset.from_dict({"text": texts})
    dataset = dataset.train_test_split(test_size=s.val_split, seed=42)

    training_args = TrainingArguments(
        output_dir=s.adapter_dir,
        num_train_epochs=s.num_epochs,
        per_device_train_batch_size=s.batch_size,
        learning_rate=s.learning_rate,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        dataset_text_field="text",
        max_seq_length=s.max_seq_length,
        tokenizer=tokenizer,
    )

    print("Starting fine-tune...")
    trainer.train()

    pathlib.Path(s.adapter_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(s.adapter_dir)
    tokenizer.save_pretrained(s.adapter_dir)
    print(f"Adapter saved to {s.adapter_dir}")


if __name__ == "__main__":
    train()
