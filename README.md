# LingBot-World 2.0 (1.3B) — GeForce kernels

Real-time [LingBot-World 2.0](https://github.com/Robbyant/lingbot-world-v2) (1.3B `causal_fast`) on one RTX 5090: **16.2 FPS as played** at 832×464, up from 5.5 FPS with the stock code, with the original Wan 2.1 decoder and no change to the model. Real time is 16 FPS.

This is the upstream repository at commit `1895d30` plus a set of inference patches, applied in-tree, with one command to run it. Everything here was measured on a RunPod RTX 5090 (32 GB); the measurements, the profiles and the quality checks live in [lingbot-world-bench](https://github.com/kaarelkaarelson/lingbot-world-bench).

| Configuration | DiT + VAE, s per 1 s chunk | FPS as played |
|---|---|---|
| Stock repo, fp32 Wan VAE | 2.87 + 1.05 = 3.9 | 5.7 |
| `--preset exact` (DiT bit-identical to stock bf16) | 0.73 + 0.34 = 1.07 | 14.8 |
| **`--preset fast` (default)** | **0.64 + 0.34 = 0.98** | **16.2** |

"As played" is what a streaming loop pays per second of video: four denoising steps plus the decode of 16 frames. The denoise loop alone runs at 25 FPS.

## Quick start

Linux x86_64, Python 3.12, an NVIDIA driver with CUDA 12.8, one RTX 5090. WSL2 works.

```bash
git clone https://github.com/kaarelkaarelson/lingbot-world-v2-1.3b-geforce-kernels
cd lingbot-world-v2-1.3b-geforce-kernels
HF_TOKEN=hf_... ./setup.sh        # venv, pinned torch 2.8 + cu128, prebuilt sm_120 kernels, ~15 GB of weights, warm-up
./run.sh --frame_num 361 --bench  # 22 s clip from examples/03 -> outputs/, prints s/chunk and FPS
```

`setup.sh` needs your own Hugging Face token: the weights (`robbyant/lingbot-world-v2-1.3b-causal-fast`, plus the Wan VAE and T5 from the 14B release) are downloaded from Hugging Face and are not redistributed here. The first run compiles the DiT and the decoder (~2 min, cached in `.inductor_cache/`); `setup.sh` does that warm-up for you. The first run of each new prompt encodes it with T5-XXL once (~30 s) and caches the embedding under `weights/…/t5_cache/`.

Your own image and camera path: `./run.sh --image me.jpg --action_path my_poses/ --prompt "…"`, where `my_poses/` holds `poses.npy` (one OpenCV camera-to-world 4×4 per output frame) and `intrinsics.npy`, in the format of `examples/*/`.

## Presets

| `--preset` | What runs | Numerics vs stock |
|---|---|---|
| `fast` (default) | torch.compile + coordinate-descent tuning, fused DiT elementwise, sync-free loop, compensated-fp32 RoPE, FP8 rowwise linears, SageAttention (INT8 QK / FP8 PV), fused fp16 channels-last Wan VAE decoder | passed an A/B eye test against the stock output; decoder is 43.6 dB / LPIPS 0.004 on the same latents; the one-row time-embedding MLP differs from stock by ~1 fp32 ulp |
| `exact` | same, with the DiT latents bit-identical to the stock bf16 model | bit-identical DiT; FP8 and SageAttention still change the sample (first-chunk LPIPS 0.02–0.03 vs bf16) |
| `stock` | upstream code path | reference |

Every `fast` component is an inference-side change; the checkpoint, the sampler (4 steps, 4-latent chunks, 18-frame KV window with 6 sink frames) and the decoder architecture are upstream's. The two decoders the field uses to go faster than this (TAEHV, Flash-VAED) were tried and rejected for sharpness (−33 % Laplacian energy); the fused decoder here is the original Wan 2.1 decoder at fp16.

## What the patches do

| Patch | Gain, s per chunk | Where |
|---|---|---|
| KV-cache sync fix (upstream PR #3, bit-exact) and safetensors fast load | −0.10; cold start 357 s → ~50 s | `wan/image2video.py`, `wan/modules/t5.py` |
| torch.compile of the DiT, Inductor coordinate-descent tuning | −0.09 | `wan/image2video.py` |
| FP8 rowwise linears (torchao) on all 420 DiT `Linear`s | −0.21 | `wan/image2video.py` |
| SageAttention 2.2 for self-attention (2.54× FlashAttention-2 at these shapes) | −0.43 | `wan/modules/attention.py` |
| Fused DiT: one-row time MLP, fp32 residual path fused, cam-modulation cache, RoPE tables per chunk | −0.20 | `wan/modules/model_fast_fusion.py` |
| Sync-free denoising loop, compensated-fp32 RoPE, Inductor tuning (bit-identical) | −0.02 | `wan/image2video.py` |
| Fused fp16 channels-last Wan VAE decoder, compiled, sub-pixel upsample convs | 1.05 → 0.34 | `wan/modules/vae2_1_fused.py` |

The DiT is compute-bound at batch 1 (SageAttention at its kernel ceiling, FP8 GEMMs at 89 % of the 5090's peak), so one card serves one real-time stream; batching two streams gives each 8.5 FPS. Details, profiles and dead ends: `OPTIMIZATIONS.md` in lingbot-world-bench.

## Scope

- One RTX 5090 (sm_120). The prebuilt `sageattention` and `flash_attn` wheels are for sm_120, CPython 3.12, torch 2.8. Other GPUs need those built from source; an RTX 4090 should run every patch (FP8 rowwise and SageAttention both support sm_89) at roughly 12 FPS and needs T5 on the CPU to fit 24 GB — untested.
- Clip generation from an image and a camera path. Browser streaming and keyboard/mouse control are being built in lingbot-world-bench and are not in this repo yet.
- Multi-GPU (`--ulysses_size`, FSDP) is upstream's code and is untouched but unmeasured here.

## CPU checks

`tests/` holds fp32 CPU equivalence checks of the fused decoder and the fused DiT against the stock modules (no GPU): `python tests/test_vae_fused_cpu.py`, `python tests/test_vae_subpixel_cpu.py`, `python tests/test_dit_fusion_cpu.py`.

## License and credit

Upstream is CC BY-NC-SA 4.0, so this repository is too: non-commercial use, attribution, share-alike (`LICENSE.txt`, unchanged). The model, the sampler and the examples are the Robbyant team's ([paper](https://arxiv.org/abs/2607.07534), `UPSTREAM_README.md`). Kernels used: [SageAttention](https://github.com/thu-ml/SageAttention), [torchao](https://github.com/pytorch/ao), [FlashAttention](https://github.com/Dao-AILab/flash-attention).
