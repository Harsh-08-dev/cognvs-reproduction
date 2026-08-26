# Environment Setup

This repo spans three separate Python environments. Which one(s) you need
depends on what you're running:

| Environment | Covers | Requirements file | GPU needed? |
|---|---|---|---|
| Upstream CogNVS | Actually running CogNVS inference (`data_gen.py`, `demo.py` inside `cognvs-codebase`) | `cognvs_requirements.txt` (in `cognvs-codebase`, not this repo) | Yes (see below) |
| Orchestration tooling | `src/experiments/`, `scripts/validate_experiment.py`, `scripts/analyze_baseline_exp01.py`, running this repo's test suite | `requirements.txt` | No |
| Evaluation & visualization | `src/evaluation/` (`evaluate.py`, `metrics.py`, `fid_kid.py`, `aggregator.py`, `add_angle.py`, `add_steps.py`, `video_utils.py`) and `src/visualization/` (`plots.py`, `comparison.py`) | `requirements-eval.txt` | No (CPU works, just slower) |

You don't need all three for every task — e.g. validating a config or
running `pytest` only needs the orchestration tooling env; running
`evaluate.py` on already-generated frames only needs the eval env, not the
upstream CogNVS env at all.

## Upstream dependency

- Repo: https://github.com/Kaihua-Chen/cog-nvs
- Commit pinned: 9d7d00cb378b07b4969472c8970d0f1c7aa8adeb
- Cloned on: 2026-08-25
- License: none present in repo (all rights reserved) — not redistributed in this repository
- Local install location: sibling folder `cognvs-codebase`, next to this repo

## Setup steps

### 1. Upstream CogNVS environment (only needed to actually run inference)

1. Clone the upstream repository (kept OUTSIDE this repo, as a sibling folder):
   git clone https://github.com/Kaihua-Chen/cog-nvs cognvs-codebase
   cd cognvs-codebase
   git checkout 9d7d00cb378b07b4969472c8970d0f1c7aa8adeb

2. Create the Python environment:
   conda create --name cognvs python=3.11
   conda activate cognvs
   pip install -r cognvs_requirements.txt

3. Download checkpoints (see cognvs-codebase README for exact links):
   - CogVideoX-5b-I2V base model (from HuggingFace)
   - CogNVS inpainting checkpoint

### 2. This repo's orchestration tooling (needed for the test suite, config validation, ExperimentRunner)

   conda create --name cognvs-repro python=3.11
   conda activate cognvs-repro
   pip install -r requirements.txt

### 3. This repo's evaluation & visualization tooling (needed for evaluate.py, aggregator.py, plots.py, comparison.py)

   conda activate cognvs-repro
   pip install -r requirements-eval.txt

Environments 2 and 3 can live in the same conda env (as above) since neither
needs a GPU — they're only split into two requirements files so running the
orchestration tests doesn't require pulling in torch/lpips/torchmetrics.

## Verification

python scripts/check_environment.py

## Checkpoints (download on GPU machine only — large files)

### 1. CogVideoX-5b-I2V base model
Source: https://huggingface.co/zai-org/CogVideoX-5b-I2V
(Download instructions per HuggingFace page — typically via `git lfs` clone or `huggingface-cli download`)

### 2. CogNVS inpainting checkpoint
mkdir checkpoints
cd checkpoints
git lfs install
git clone https://huggingface.co/kaihuac/cognvs_ckpt_inpaint
cd ..

### 3. (Optional) Pre-finetuned checkpoints
Since our GPU (single A4000, 16GB) cannot run test-time finetuning
(requires >=5 GPUs per upstream README), we rely on the authors'
pre-finetuned checkpoints instead:
https://huggingface.co/datasets/kaihuac/cognvs_ckpt_test_time_finetuned
(~20GB each per sequence)

Both checkpoint sets are large — do NOT download on personal/dev machine.
Download directly onto the GPU server to avoid re-uploading large files.