#!/usr/bin/env bash
# One-shot setup on Linux x86_64 with an RTX 5090 (sm_120), Python 3.12, CUDA 12.8 driver.
#   HF_TOKEN=hf_... ./setup.sh
# Creates .venv, installs the pinned stack + the prebuilt sm_120 wheels from the
# GitHub release, downloads the 1.3B DiT and the shared VAE / T5 assets (~15 GB) into
# ./weights, and runs a short warm-up so the first real run is at full speed.
set -euo pipefail
cd "$(dirname "$0")"
REL=${REL:-https://github.com/kaarelkaarelson/lingbot-world-1.3b-geforce-kernels/releases/download/v0.1.0}
PY=${PY:-python3.12}

[ -n "${HF_TOKEN:-}" ] || { echo "HF_TOKEN is not set (https://huggingface.co/settings/tokens)"; exit 1; }
command -v "$PY" >/dev/null || { echo "$PY not found; install Python 3.12"; exit 1; }
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || { echo "no NVIDIA driver"; exit 1; }

echo "== venv =="
[ -d .venv ] || "$PY" -m venv .venv
. .venv/bin/activate
pip install -q --upgrade pip
pip install -q torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -q -r requirements.txt

echo "== prebuilt kernels (sm_120, cp312, torch 2.8) =="
mkdir -p wheels
for w in sageattention-2.2.0-cp312-cp312-linux_x86_64.whl flash_attn-2.8.3.post1-cp312-cp312-linux_x86_64.whl; do
  [ -f "wheels/$w" ] || curl -sSfL -o "wheels/$w" "$REL/$w"
done
pip install -q wheels/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl
pip install -q wheels/flash_attn-2.8.3.post1-cp312-cp312-linux_x86_64.whl || echo "flash_attn wheel did not install; cross-attention will use SDPA (a ~2 % slower step)"

echo "== weights (Hugging Face, your token) =="
mkdir -p weights
hf download robbyant/lingbot-world-v2-1.3b-causal-fast --local-dir weights/lingbot-world-v2-1.3b-causal-fast
hf download robbyant/lingbot-world-v2-14b-causal-fast \
  Wan2.1_VAE.pth config.json models_t5_umt5-xxl-enc-bf16.pth \
  google/umt5-xxl/tokenizer.json google/umt5-xxl/tokenizer_config.json \
  google/umt5-xxl/special_tokens_map.json google/umt5-xxl/spiece.model \
  --local-dir weights/lingbot-world-v2-14b-causal-fast

python - <<'PY'
import torch, sageattention, torchao, diffusers
cc = torch.cuda.get_device_capability()
print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)} sm_{cc[0]}{cc[1]} | torchao {torchao.__version__} | diffusers {diffusers.__version__}")
if cc != (12, 0):
    print("WARNING: the prebuilt SageAttention wheel targets sm_120 (RTX 5090); other GPUs need `pip install` from source.")
PY

echo "== warm-up (compiles the DiT and the VAE decoder once; cached in .inductor_cache) =="
./run.sh --frame_num 49 --bench
echo
echo "READY. Generate a clip:   ./run.sh --frame_num 361 --bench"
