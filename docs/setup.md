# Environment Setup

## Upstream dependency

- Repo: https://github.com/Kaihua-Chen/cog-nvs
- Commit pinned: 9d7d00cb378b07b4969472c8970d0f1c7aa8adeb
- Cloned on: 2026-08-25
- License: none present in repo (all rights reserved) — not redistributed in this repository
- Local install location: sibling folder `cognvs-codebase`, next to this repo

## Setup steps

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