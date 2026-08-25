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