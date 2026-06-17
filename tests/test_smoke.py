from __future__ import annotations


def test_settings_load():
    from src.settings import get_settings
    s = get_settings()
    assert s.lora_r == 8
    assert s.lora_alpha == 16
    assert s.num_epochs == 3


def test_format_prompt():
    from src.preprocess import format_prompt
    prompt = format_prompt("What is sick leave?", "12 days per year.")
    assert "<|system|>" in prompt
    assert "<|user|>" in prompt
    assert "<|assistant|>" in prompt
    assert "12 days per year." in prompt


def test_load_dataset():
    from src.preprocess import load_dataset
    rows = load_dataset("data/train.jsonl")
    assert len(rows) >= 50
    assert "question" in rows[0]
    assert "answer" in rows[0]


def test_validate_passes():
    from src.preprocess import validate
    rows = [{"question": "Q?", "answer": "A."}]
    validate(rows)  # should not raise


def test_validate_fails():
    import pytest
    from src.preprocess import validate
    with pytest.raises(ValueError):
        validate([{"question": "Q?"}])  # missing answer
