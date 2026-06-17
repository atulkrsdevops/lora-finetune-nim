from __future__ import annotations
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Base model — small enough for a free T4
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # LoRA hyperparameters
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,v_proj"

    # Training
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 512
    val_split: float = 0.1

    # Paths
    data_dir: str = "data"
    adapter_dir: str = "adapters/hr-policy-lora"
    train_file: str = "data/train.jsonl"
    eval_file: str = "data/eval.jsonl"


@lru_cache
def get_settings() -> Settings:
    return Settings()
