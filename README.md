# GPUPerf-3B: A Profiler-Grounded GPU Optimization Model

GPUPerf-3B is an empirical GPU kernel optimization model fine-tuned on **Qwen2.5-Coder-3B-Instruct** using profiler feedback from **NVIDIA Nsight Compute (`ncu`)**.

Instead of guessing optimizations in the dark, GPUPerf-3B diagnoses performance bottlenecks from real hardware execution counters (Speed of Light, occupancy limiters, warp stall reasons, register spilling) and outputs verified refactored kernels (CUDA / Triton) paired with measured speedups.

---

## Architecture Overview

```
                      +-----------------------------+
                      |   Target GPU Kernel (CUDA)  |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | NVIDIA Nsight Compute (ncu) |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |  PIR (Profiler Intermediate |
                      |       Representation)       |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |   GPUPerf-3B (SFT + DPO)    |
                      +--------------+--------------+
                                     |
                +--------------------+--------------------+
                |                                         |
                v                                         v
   +-------------------------+               +-------------------------+
   |  Bottleneck Diagnosis   |               | Refactored Fast Kernel  |
   | (e.g. Register Spills)  |               |  (Speedup >= 1.25x L4)  |
   +-------------------------+               +-------------------------+
```

---

## Directory Layout

```
d:\GPUPerf-3B\
├── configs/
│   ├── sft_pilot_config.yaml         # Stage 1: SFT training parameters
│   └── dpo_pilot_config.yaml         # Stage 2: DPO preference alignment
├── gpuperf/
│   ├── schema/
│   │   └── pir.py                    # Profiler Intermediate Representation dataclasses
│   ├── profiler/
│   │   ├── ncu_runner.py             # NCU CLI execution wrapper & mock profiler
│   │   └── ncu_parser.py             # CSV / NCU metric parser & bottleneck diagnoser
│   ├── ingestion/
│   │   ├── kernelbenchx_loader.py    # BonnieWang/KernelBenchX loader
│   │   └── megatraces_loader.py      # Infatoshi/kernelbench-mega-traces loader
│   ├── synthesis/
│   │   ├── sweep_runner.py           # Parametric tile/warp explorer on L4
│   │   └── trajectory_builder.py     # ShareGPT / TRL format builders
│   └── training/
│       ├── dataset.py                # Dataset loaders
│       ├── train_sft.py              # SFT fine-tuning script
│       └── train_dpo.py              # DPO alignment script
├── scripts/
│   ├── download_public_datasets.py   # Hugging Face downloader
│   ├── run_l4_harvest.py             # Automated L4 profiling harness
│   └── evaluate_heldout.py           # Closed-loop evaluation benchmark
├── tests/
│   ├── test_pir_schema.py            # PIR serialization tests
│   └── test_ncu_parser.py            # NCU parsing & diagnostic tests
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Ingest Public Datasets
```bash
python scripts/download_public_datasets.py
```

### 2. Harvest L4 Profiler Trajectories (or Run Mock Generation)
```bash
# Run on an L4 instance (or pass --mock to generate test pilot locally)
python scripts/run_l4_harvest.py --count 500 --output-dir ./data
```

### 3. Stage 1: Supervised Fine-Tuning (SFT)
```bash
python gpuperf/training/train_sft.py --config configs/sft_pilot_config.yaml
```

### 4. Stage 2: Direct Preference Optimization (DPO)
```bash
python gpuperf/training/train_dpo.py --config configs/dpo_pilot_config.yaml
```

### 5. Evaluate on Held-Out Test Suite
```bash
python scripts/evaluate_heldout.py --model ./checkpoints/gpuperf_dpo_pilot
```

