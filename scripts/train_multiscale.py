"""Universal Multi-Scale Training Engine for GPUPerf.

Supports fine-tuning across all model tiers:
- 3B:  Qwen/Qwen2.5-Coder-3B-Instruct
- 7B:  Qwen/Qwen2.5-Coder-7B-Instruct
- 14B: Qwen/Qwen2.5-Coder-14B-Instruct
- 27B: Qwen/Qwen3.6-27B
- 32B: Qwen/Qwen2.5-Coder-32B-Instruct

Automatically detects attached hardware (T4, L4, A100-40GB, A100-80GB, H100),
dynamically calculates VRAM budgets, configures batch sizes & gradient accumulation,
and executes SFT and/or DPO preference alignment.
"""

from __future__ import annotations
import os
import sys
import argparse
from typing import Dict, Any, Tuple
import torch

# Self-healing path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


MODEL_REGISTRY: Dict[str, str] = {
    "3b": "Qwen/Qwen2.5-Coder-3B-Instruct",
    "7b": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "14b": "Qwen/Qwen2.5-Coder-14B-Instruct",
    "27b": "Qwen/Qwen3.6-27B",
    "32b": "Qwen/Qwen2.5-Coder-32B-Instruct",
}


def detect_hardware_and_configure(
    model_size: str,
    precision_mode: str = "auto"
) -> Dict[str, Any]:
    """Inspects GPU hardware and returns optimal VRAM-safe training hyperparameters."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. A GPU is required for training.")

    gpu_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    major, minor = torch.cuda.get_device_capability(0)
    compute_cap = f"sm_{major}{minor}"
    supports_bf16 = major >= 8

    print("=" * 70)
    print(f"HARDWARE DETECTED: {gpu_name} ({compute_cap})")
    print(f"TOTAL VRAM: {total_vram_gb:.1f} GB | Native BF16: {supports_bf16}")
    print("=" * 70)

    model_size = model_size.lower()
    if model_size not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_size '{model_size}'. Choose from: {list(MODEL_REGISTRY.keys())}")

    model_id = MODEL_REGISTRY[model_size]

    # Defaults
    config = {
        "model_id": model_id,
        "model_size": model_size,
        "gpu_name": gpu_name,
        "total_vram_gb": total_vram_gb,
        "supports_bf16": supports_bf16,
        "load_in_4bit": True,
        "max_seq_length": 3072,
        "sft_batch_size": 2,
        "sft_grad_accum": 4,
        "dpo_batch_size": 1,
        "dpo_grad_accum": 8,
        "lora_rank": 32,
        "lora_alpha": 64,
        "learning_rate_sft": 2e-4,
        "learning_rate_dpo": 5e-6,
    }

    # Scale-specific adjustments based on available VRAM
    if model_size == "3b":
        config["sft_batch_size"] = 2
        config["sft_grad_accum"] = 4
        config["learning_rate_sft"] = 2e-4

    elif model_size == "7b":
        config["learning_rate_sft"] = 1.5e-4
        if total_vram_gb < 20.0:  # e.g. T4 (15GB)
            config["sft_batch_size"] = 1
            config["sft_grad_accum"] = 8
            config["max_seq_length"] = 2048
        elif total_vram_gb < 45.0:  # e.g. L4 (24GB) or A100-40GB
            config["sft_batch_size"] = 2
            config["sft_grad_accum"] = 4
        else:  # A100-80GB / H100
            config["sft_batch_size"] = 4
            config["sft_grad_accum"] = 2
            if precision_mode == "16bit":
                config["load_in_4bit"] = False

    elif model_size in ["14b", "27b", "32b"]:
        config["learning_rate_sft"] = 1e-4
        config["lora_rank"] = 16
        config["lora_alpha"] = 32

        if total_vram_gb < 20.0:  # T4
            raise RuntimeError(
                f"Model size {model_size.upper()} cannot run on {gpu_name} ({total_vram_gb:.1f} GB VRAM). "
                f"Minimum required is ~24GB (L4) or 40GB/80GB (A100)."
            )
        elif total_vram_gb < 45.0:  # L4 or A100-40GB
            config["load_in_4bit"] = True
            config["sft_batch_size"] = 1
            config["sft_grad_accum"] = 16
            config["dpo_batch_size"] = 1
            config["dpo_grad_accum"] = 16
            config["max_seq_length"] = 2048
        else:  # A100-80GB / H100 (80GB+)
            if precision_mode == "16bit" and model_size in ["14b", "27b"]:
                config["load_in_4bit"] = False
                config["sft_batch_size"] = 1
                config["sft_grad_accum"] = 16
            else:
                config["load_in_4bit"] = True
                config["sft_batch_size"] = 4
                config["sft_grad_accum"] = 4
                config["max_seq_length"] = 4096

    if precision_mode == "4bit":
        config["load_in_4bit"] = True
    elif precision_mode == "16bit" and total_vram_gb > 70.0:
        config["load_in_4bit"] = False

    print(f"CONFIGURATION CHOSEN FOR {model_size.upper()}:")
    print(f"  * Model ID: {config['model_id']}")
    print(f"  * 4-bit Quantization: {config['load_in_4bit']}")
    print(f"  * Context Window: {config['max_seq_length']} tokens")
    print(f"  * SFT Micro-Batch: {config['sft_batch_size']} (Grad Accum: {config['sft_grad_accum']})")
    print(f"  * DPO Micro-Batch: {config['dpo_batch_size']} (Grad Accum: {config['dpo_grad_accum']})")
    print(f"  * LoRA Rank: r={config['lora_rank']}, alpha={config['lora_alpha']}")
    print("=" * 70)

    return config


def run_training_pipeline(
    model_size: str = "3b",
    stage: str = "both",
    precision_mode: str = "auto",
    sft_epochs: int = 2,
    dpo_epochs: int = 1,
    val_split: float = 0.1,
    smoke_test: bool = False,
    output_base_dir: str = "./checkpoints"
) -> None:
    """Executes the full SFT and/or DPO fine-tuning loop."""
    config = detect_hardware_and_configure(model_size, precision_mode)

    from unsloth import FastLanguageModel, is_bfloat16_supported
    from datasets import load_dataset
    from unsloth.chat_templates import get_chat_template

    print(f"\n[1/4] Loading Base Model: {config['model_id']}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_id"],
        max_seq_length=config["max_seq_length"],
        load_in_4bit=config["load_in_4bit"],
        dtype=torch.bfloat16 if config["supports_bf16"] else torch.float16,
    )

    # Apply LoRA adapters
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora_rank"],
        target_modules=target_modules,
        lora_alpha=config["lora_alpha"],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    print("LoRA adapters initialized!")

    # Format chat template
    template_name = "gemma-2" if "gemma" in config["model_id"].lower() else "qwen-2.5"
    tokenizer = get_chat_template(tokenizer, chat_template=template_name)

    sft_ckpt_dir = os.path.join(output_base_dir, f"gpuperf_{model_size}_sft")
    final_ckpt_dir = os.path.join(output_base_dir, f"gpuperf_{model_size}_dpo_final")
    os.makedirs(output_base_dir, exist_ok=True)

    # -------------------------------------------------------------
    # STAGE 1: SUPERVISED FINE-TUNING (SFT)
    # -------------------------------------------------------------
    if stage in ["sft", "both"]:
        print(f"\n[2/4] Preparing SFT Dataset for {model_size.upper()}...")
        sft_data_path = os.path.join(ROOT_DIR, "data", "sft_multi_gpu.jsonl")
        raw_sft = load_dataset("json", data_files=sft_data_path, split="train")

        def format_sft_prompts(batch):
            texts = [
                tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
                for convo in batch["messages"]
            ]
            return {"text": texts}

        sft_dataset = raw_sft.map(format_sft_prompts, batched=True)

        if val_split > 0 and not smoke_test:
            split_sft = sft_dataset.train_test_split(test_size=val_split, seed=42)
            train_sft = split_sft["train"]
            eval_sft = split_sft["test"]
            print(f"  * SFT Split: {len(train_sft)} Train samples | {len(eval_sft)} Validation samples")
        else:
            train_sft = sft_dataset
            eval_sft = None

        from trl import SFTTrainer
        from transformers import TrainingArguments

        eval_kwargs = {}
        if eval_sft is not None:
            eval_kwargs = {
                "eval_strategy": "steps",
                "eval_steps": 25,
                "save_strategy": "steps",
                "save_steps": 25,
                "save_total_limit": 2,
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
            }

        max_steps = 30 if smoke_test else -1
        log_steps = 2 if smoke_test else 10

        print(f"Starting Stage 1: SFT ({'SMOKE TEST (30 steps)' if smoke_test else f'{sft_epochs} Epochs'})...")
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_sft,
            eval_dataset=eval_sft,
            dataset_text_field="text",
            max_seq_length=config["max_seq_length"],
            dataset_num_proc=2,
            packing=False,
            args=TrainingArguments(
                per_device_train_batch_size=config["sft_batch_size"],
                gradient_accumulation_steps=config["sft_grad_accum"],
                warmup_ratio=0.08,
                num_train_epochs=sft_epochs,
                max_steps=max_steps,
                learning_rate=config["learning_rate_sft"],
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=log_steps,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="cosine",
                seed=42,
                output_dir=os.path.join(output_base_dir, f"trainer_{model_size}_sft"),
                report_to="none",
                **eval_kwargs
            ),
        )
        trainer.train()
        model.save_pretrained(sft_ckpt_dir)
        tokenizer.save_pretrained(sft_ckpt_dir)
        print(f"✅ SFT Stage Complete! Saved best adapter to {sft_ckpt_dir}")

    # -------------------------------------------------------------
    # STAGE 2: DIRECT PREFERENCE OPTIMIZATION (DPO)
    # -------------------------------------------------------------
    if stage in ["dpo", "both"]:
        print(f"\n[3/4] Preparing Stage 2: DPO Alignment for {model_size.upper()}...")
        from trl import DPOTrainer, DPOConfig
        from unsloth import PatchDPOTrainer

        PatchDPOTrainer()

        dpo_data_path = os.path.join(ROOT_DIR, "data", "dpo_multi_gpu.jsonl")
        dpo_dataset = load_dataset("json", data_files=dpo_data_path, split="train")

        dpo_trainer = DPOTrainer(
            model=model,
            ref_model=None,
            tokenizer=tokenizer,
            beta=0.1,
            train_dataset=dpo_dataset,
            max_length=config["max_seq_length"],
            args=DPOConfig(
                per_device_train_batch_size=config["dpo_batch_size"],
                gradient_accumulation_steps=config["dpo_grad_accum"],
                warmup_ratio=0.1,
                num_train_epochs=dpo_epochs,
                learning_rate=config["learning_rate_dpo"],
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=10,
                optim="adamw_8bit",
                seed=42,
                output_dir=os.path.join(output_base_dir, "trainer_dpo_output"),
                report_to="none"
            ),
        )

        print(f"Starting Stage 2: DPO Alignment ({dpo_epochs} Epochs)...")
        dpo_trainer.train()
        model.save_pretrained(final_ckpt_dir)
        tokenizer.save_pretrained(final_ckpt_dir)
        print(f"🏆 DPO Alignment Complete! Final model saved to {final_ckpt_dir}")

    print("\n" + "=" * 70)
    print(f"TRAINING PIPELINE FINISHED FOR {model_size.upper()}!")
    print(f"Final Weights: {final_ckpt_dir if stage in ['dpo', 'both'] else sft_ckpt_dir}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Universal Multi-Scale Training Engine for GPUPerf")
    parser.add_argument("--model_size", type=str, default="3b", choices=["3b", "7b", "14b", "27b", "32b"],
                        help="Model scale to train (3b, 7b, 14b, 27b, 32b)")
    parser.add_argument("--stage", type=str, default="both", choices=["sft", "dpo", "both"],
                        help="Training stage: 'sft', 'dpo', or 'both'")
    parser.add_argument("--precision", type=str, default="auto", choices=["auto", "4bit", "16bit"],
                        help="Precision mode: 'auto' (hardware safe), '4bit', or '16bit'")
    parser.add_argument("--sft_epochs", type=int, default=2, help="Number of SFT epochs (default: 2)")
    parser.add_argument("--dpo_epochs", type=int, default=1, help="Number of DPO epochs (default: 1)")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio (default: 0.1)")
    parser.add_argument("--smoke_test", action="store_true", help="Run quick 30-step smoke test")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory for checkpoints")

    args = parser.parse_args()

    run_training_pipeline(
        model_size=args.model_size,
        stage=args.stage,
        precision_mode=args.precision,
        sft_epochs=args.sft_epochs,
        dpo_epochs=args.dpo_epochs,
        val_split=args.val_split,
        smoke_test=args.smoke_test,
        output_base_dir=args.output_dir
    )


if __name__ == "__main__":
    main()

