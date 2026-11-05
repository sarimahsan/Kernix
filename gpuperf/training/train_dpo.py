"""Direct Preference Optimization (DPO) Script for GPUPerf-3B on NVIDIA L4.

Aligns the model to prefer kernels with verified empirical speedup and penalizes
variants that cause register spills, thread divergences, or execution slowdowns.
"""

import os
import argparse
import yaml
from datasets import load_dataset


def train_dpo(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_id = cfg.get("model_id", "./checkpoints/gpuperf_sft_pilot")
    output_dir = cfg.get("output_dir", "./checkpoints/gpuperf_dpo_pilot")
    dataset_path = cfg.get("dataset_path", "./data/dpo_pilot.jsonl")
    max_length = cfg.get("max_length", 3072)
    max_prompt_length = cfg.get("max_prompt_length", 1536)
    num_train_epochs = cfg.get("num_train_epochs", 2)
    batch_size = cfg.get("per_device_train_batch_size", 1)
    grad_accum_steps = cfg.get("gradient_accumulation_steps", 8)
    learning_rate = float(cfg.get("learning_rate", 5e-5))
    beta = float(cfg.get("beta", 0.1))

    print(f"=== GPUPerf-3B Stage 2: DPO Alignment ===")
    print(f"Base SFT Model: {model_id}")
    print(f"DPO Dataset: {dataset_path}")
    print(f"DPO Beta: {beta} | Epochs: {num_train_epochs}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel, LoraConfig, get_peft_model
    from trl import DPOTrainer, DPOConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    dataset = load_dataset("json", data_files=dataset_path, split="train")

    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=5,
        save_strategy="epoch",
        fp16=False,
        bf16=True,
        optim="adamw_8bit",
        beta=beta,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        report_to="none"
    )

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,  # PEFT automatically handles reference model by disabling adapters
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=peft_config
    )

    print("-> Starting DPO alignment...")
    dpo_trainer.train()

    print(f"-> Saving aligned DPO model to {output_dir}")
    dpo_trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("=== DPO Training Complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/dpo_pilot_config.yaml")
    args = parser.parse_args()
    train_dpo(args.config)

