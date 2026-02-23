# Deliberative Reinforcement Learning with Internalized Physical Reasoning for Power System Operations

> A research-oriented codebase for **power distribution system operation / topology dispatch** using **PPO-based reinforcement learning**, **graph/transformer-style representations**, and **deliberative inference** (look-ahead search + self-reflective cost/risk estimation).

## Overview

This repository contains an experimental framework for learning and evaluating dispatch/reconfiguration policies for a power distribution system (the project includes a **746-bus system configuration** and related utilities). The core idea is to combine:

- **Model-free policy learning** (PPO-style actor-critic),
- **Internalized physical reasoning** (self-reflective prediction of cost/risk terms),
- **Deliberative decision-time planning** (look-ahead / beam search),
- **Power-flow-based environment simulation** (via `pandapower`).

The codebase includes training, evaluation, inference, and UI/debug tooling for comparing:
- pure policy rollout (greedy or sampling),
- look-ahead search guided by physics/environment rollouts,
- self-reflective (SR) look-ahead planning.

## Repository Structure (high level)

Top-level files and folders visible in the repository include:

- `train_ppo_power_dispatch_multiproc_multiencoder_withGT_sr_merged.py`  
  Main training script (PPO-style, multi-process, GT + SR merged version, detailed logging).
- `power_dispatch_env_withGT_dimrisk.py`  
  Gym environment for power dispatch with graph-transformer node features and dimensioned/risk-aware rewards.
- `power_dispatch_env_withGT.py`, `power_dispatch_env.py`  
  Related environment variants.
- `run_trained_ppo_gt_interact_with_lookahead_search_v3.py`  
  Inference/evaluation script with **physics-guided look-ahead beam search** vs baseline policy.
- `infer_sr_lookahead_compare.py`  
  **Self-Reflection + look-ahead** inference comparison script.
- `sr_self_reflective_inference_ui.py`  
  Gradio UI for interactive debugging and auto-mode episode runs.
- `eval_ppo_power_dispatch_noargparse.py`  
  Evaluation/validation script for trained PPO runs (no `argparse`; edit config variables directly).
- `eval_policy.py`, `infer_sr_lookahead_compare.py`, `debug_ppo_gt_sr_candidates.py`  
  Additional evaluation/debug scripts.
- `config746sys.py`, `new746_system_v0713.py`  
  System/data configuration and aliases for the 746-system setup.
- `generate_virtual_timeseries_file.py`  
  Timeseries generation utility.
- `sys_plot.py`, `ori_topo.svg`  
  Visualization helpers / topology figure.
- `Utls/`, `GTransformer/`  
  Utilities and model-related modules.
- `system_file/746sys/`  
  System assets and network-related files (partial dataset/config resources).

## Key Features

### 1) Power-flow-driven RL environment
The environment (`power_dispatch_env_withGT_dimrisk.py`) is a `gym.Env` where the action is a **discrete feasible switch-state / topology choice**. It computes rewards from:

- network losses,
- voltage violations,
- line overload violations,
- transformer balance terms,
- switching costs,
- optional risk terms (e.g., margin / CVaR-like proxy),
- explicit PF-failure handling with finite penalties.

### 2) Deliberative look-ahead inference
The look-ahead inference script compares:

- **Baseline**: direct policy action selection (e.g., greedy argmax),
- **Look-ahead search**: physics-guided beam search with receding horizon.

Outputs include step-level CSVs, action comparisons, summary JSON, and plots.

### 3) Self-reflective (SR) reasoning
The SR inference script uses an internal predictive model to estimate cost/risk components during search expansion, then chooses the action sequence with the best predicted discounted return.

### 4) Debuggable / interactive workflows
A Gradio-based UI is provided for:
- step-by-step debugging of search decisions,
- candidate inspection,
- optional verification via environment simulation,
- automated episode rollouts and CSV export.

### 5) No-argparse design
Several scripts intentionally avoid `argparse` and use a **“USER CONFIG”** block (or `__main__` variables) for explicit parameter editing before execution.

---

## Environment and Dependencies

This project is research code and may require environment-specific setup. From the visible source imports, the main dependencies likely include:

- Python 3.9+ (recommended)
- `numpy`
- `pandas`
- `torch`
- `matplotlib`
- `gym`
- `pandapower`
- `gradio` (for UI script)
- `tensorboard` / `torch.utils.tensorboard`
- project-local modules in `Utls/` and `GTransformer/`

### Suggested installation (example)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install numpy pandas matplotlib gym pandapower torch gradio tensorboard
```

> Depending on your CUDA setup and PyTorch version, install PyTorch using the official wheel selector for your system.

---

## Data and Path Configuration (Important)

This repository appears to rely on **local power-system files and timeseries data** that are **not fully packaged as portable paths** by default.

For example, `config746sys.py` contains **hard-coded local paths** (e.g., `WORKPATH`, `DATAPATH`, timeseries and parsed CIM files). Before running anything, you will need to:

1. Prepare the required network/system and timeseries files.
2. Update local paths in:
   - `config746sys.py`
   - possibly `new746_system_v0713.py`
   - any script-specific `USER CONFIG` blocks
3. Verify referenced files under:
   - `system_file/746sys/`
   - feeder CSVs / parsed CIM spreadsheets / node mapping files
   - timeseries directories

If a script imports `config746sys` but your environment uses `new746_system_v0713.py`, some scripts include an alias fallback mechanism; however, path consistency is still required.

---

## Quick Start (Typical Workflow)

## 1. Prepare data and config
Edit `config746sys.py` and ensure all local file paths are valid.

## 2. Train a policy (PPO + GT/SR merged)
Edit parameters directly in:

- `train_ppo_power_dispatch_multiproc_multiencoder_withGT_sr_merged.py`

Then run:

```bash
python train_ppo_power_dispatch_multiproc_multiencoder_withGT_sr_merged.py
```

Expected outputs usually include:
- checkpoints (`ppo_ckpt_*.pt`)
- logs / CSV metrics
- metadata (`train_meta.json`)
- tensorboard summaries
- run logs

## 3. Evaluate a trained policy
Edit the `USER CONFIG` block in:

- `eval_ppo_power_dispatch_noargparse.py`

Then run:

```bash
python eval_ppo_power_dispatch_noargparse.py
```

This script supports:
- deterministic / stochastic rollout,
- aggregate evaluation metrics,
- loading from logs or checkpoint sweeps,
- plotting evaluation curves.

## 4. Compare baseline vs look-ahead inference
Edit parameters in:

- `run_trained_ppo_gt_interact_with_lookahead_search_v3.py`

Then run:

```bash
python run_trained_ppo_gt_interact_with_lookahead_search_v3.py
```

Outputs may include:
- `baseline_steps.csv`
- `search_steps.csv`
- `action_diff.csv`
- `summary.json`
- `plots/*.png`

## 5. Run SR look-ahead comparison
Edit parameters in:

- `infer_sr_lookahead_compare.py`

Then run:

```bash
python infer_sr_lookahead_compare.py
```

This prints detailed logs for:
- root policy distribution,
- sampled candidates,
- beam expansion/pruning,
- predicted cost/risk and rewards,
- real-vs-predicted differences,
- episode summaries.

## 6. Use the interactive UI (optional)
Edit config in:

- `sr_self_reflective_inference_ui.py`

Then run:

```bash
python sr_self_reflective_inference_ui.py
```

This launches a Gradio interface for debugging and interactive inspection.

---

## Notes on Reproducibility

- The codebase contains multiple environment and training script variants (`withGT`, `dimrisk`, merged training versions).
- Several scripts dynamically import definitions from the training script to guarantee compatibility (model classes, preprocessors, env config schema).
- Some scripts infer or reconstruct metadata from training outputs (`train_meta.json`, checkpoints).
- Random seeds are explicitly handled in multiple scripts, but exact reproducibility also depends on:
  - CUDA determinism settings,
  - multiprocessing behavior,
  - external data files and timeseries versions.

---

## Common Pitfalls

- **Missing local data files**: Most runtime failures originate from unresolved paths in `config746sys.py`.
- **Module import mismatch**: Ensure local modules (`Utls`, `GTransformer`) are available in the working directory / PYTHONPATH.
- **Checkpoint compatibility**: Evaluation/inference scripts may expect exact model/preprocessing definitions from the training script used to generate the checkpoint.
- **PF solver runtime**: `pandapower` calls can be expensive; look-ahead search significantly increases compute cost.
- **Gym version differences**: The env uses the modern `reset()/step()` signatures in several scripts; align your Gym package version accordingly.

---

## Citation / Acknowledgment

If you use this repository in academic work, please cite the associated paper/project (once publicly available) and acknowledge the repository author.

---

## License

No license file is currently visible in the repository root.  
Please contact the repository author for usage/redistribution permissions before using this code outside personal research evaluation.

---

## Disclaimer

This is a research-oriented repository for experimental power-system operation studies. It is not production dispatch software and should not be used for real-world grid operations without rigorous validation, safety review, and domain approval.
