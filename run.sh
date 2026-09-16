#!/usr/bin/env bash
# Generate a clip with the 16.2 FPS stack. Any generate.py flag can be appended:
#   ./run.sh --frame_num 361 --bench
#   ./run.sh --image my.jpg --action_path my_poses/ --prompt "..." --preset exact
set -euo pipefail
cd "$(dirname "$0")"
. .venv/bin/activate
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$PWD/.inductor_cache}
export PATH=/usr/local/cuda/bin:$PATH
exec python generate.py \
  --image examples/03/image.jpg --action_path examples/03 \
  --prompt "A serene lakeside scene with a lone tree standing in calm water, surrounded by distant snow-capped mountains under a bright blue sky with drifting white clouds — gentle ripples reflect the tree and sky, creating a tranquil, meditative atmosphere." \
  --save_dir outputs \
  "$@"
