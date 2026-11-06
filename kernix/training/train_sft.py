"""Supervised Fine-Tuning (SFT) Script for GPUPerf-3B on NVIDIA L4.

Supports Unsloth (fastest) with automatic fallback to standard Hugging Face TRL + PEFT.
"""

import os
import argparse
import yaml
from datasets import load_dataset


def train_sft(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_id = cfg.get("model_id", "Qwen/Qwen2.5-Coder-3B-Instruct")
    output_dir = cfg.get("output_dir", "./checkpoints/gpuperf_sft_pilot")
    dataset_path = cfg.get("dataset_path", "./data/sft_pilot.jsonl")
    max_seq_length = cfg.get("max_seq_length", 4096)
    num_train_epochs = cfg.get("num_train_epochs", 3)
    batch_size = cfg.get("per_device_train_batch_size", 2)
    grad_accum_steps = cfg.get("gradient_accumulation_steps", 4)
    learning_rate = float(cfg.get("learning_rate", 2e-4))

    print(f"=== GPUPerf-3B Stage 1: SFT ===")
    print(f"Base Model: {model_id}")
    print(f"Dataset: {dataset_path}")
    print(f"Epochs: {num_train_epochs} | Effective Batch Size: {batch_size * grad_accum_steps}")

    # Try Unsloth first
    try:
        from unsloth import FastLanguageModel
        import torch

        print("-> Using Unsloth FastLanguageModel engine.")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            dtype=None,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.get("lora_rank", 32),
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=cfg.get("lora_alpha", 64),
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )

    except ImportError:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model

        print("-> Unsloth not detected. Falling back to standard Hugging Face PEFT + TRL.")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else "sdpa"
        )

        peft_config = LoraConfig(
            r=cfg.get("lora_rank", 32),
            lora_alpha=cfg.get("lora_alpha", 64),
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)

    # Load dataset
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    from trl import SFTTrainer
    from transformers import TrainingArguments

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        save_strategy="epoch",
        fp16=False,
        bf16=True,
        optim="adamw_8bit",
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="messages",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )

    print("-> Starting SFT training...")
    trainer.train()

    print(f"-> Saving fine-tuned LoRA adapters to {output_dir}")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("=== Training Complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/sft_pilot_config.yaml")
    args = parser.parse_args()
    train_sft(args.config)

