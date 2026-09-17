# LingBot-World 2.0 (1.3B) — GeForce kernels

Real-time [LingBot-World 2.0](https://github.com/Robbyant/lingbot-world-v2) (1.3B `causal_fast`) on one RTX 5090: **17 FPS as played** at 832×464 (16.9–17.0 measured on a stock pod), up from 5.5 FPS with the stock code, with the original Wan 2.1 decoder and no change to the model. Real time is 16 FPS.

![lingbot play dragon at 16 fps](docs/dragon_16fps.gif)

`lingbot play dragon`, 8 s of the 22 s clip; the counter is the real per-chunk rate (16 frames ÷ that chunk's DiT + VAE time). Full clip: [dragon_16fps.mp4](https://github.com/kaarelkaarelson/lingbot-world-v2-1.3b-geforce-kernels/releases/download/v0.2.0/dragon_16fps.mp4) (22 MB).

This is the upstream repository at commit `1895d30` plus a set of inference patches, applied in-tree, with one command to run it. Everything here was measured on a RunPod RTX 5090 (32 GB); the measurements, the profiles and the quality checks live in [lingbot-world-v2-stream](https://github.com/kaarelkaarelson/lingbot-world-v2-stream).

| Configuration | DiT + VAE, s per 1 s chunk | FPS as played |
|---|---|---|
| Stock repo, fp32 Wan VAE | 2.87 + 1.05 = 3.9 | 5.7 |
| `--preset exact` (DiT bit-identical to stock bf16) | 0.73 + 0.34 = 1.07 | 14.8 |
| **`--preset fast` (default)** | **0.62 + 0.33 = 0.95** | **16.9–17.0** |

"As played" is what a streaming loop pays per second of video: four denoising steps plus the decode of 16 frames. The denoise loop alone runs at 25 FPS.

## Quick start

One RTX 5090 (32 GB), Linux x86_64 or Windows via WSL2 (untested — reports welcome), NVIDIA driver ≥ 570, a Hugging Face token for the weights.

```bash
git clone https://github.com/kaarelkaarelson/lingbot-world-v2-1.3b-geforce-kernels
cd lingbot-world-v2-1.3b-geforce-kernels
HF_TOKEN=hf_... ./setup.sh     # ~15 min once: Python 3.12, torch, prebuilt kernels, 18 GB of weights, compile warm-up
. .venv/bin/activate
lingbot play                   # ~35 s of warm-up, then a window on the world at 16 FPS
```

`lingbot play wall` picks another scene — each is an image + prompt + camera intrinsics from upstream's examples: `lake` (default), `wall` (Great Wall), `stonehenge`, `alley` (game-engine city), `castle` and `dragon` (dragon rider over a jungle). Your own world: `lingbot play --image me.jpg --prompt "one sentence describing the scene"`.

| Key | Action |
|---|---|
| `W` `A` `S` `D` | move (hold `Shift` to run) |
| `Q` `E` | down / up |
| `←` `→` `↑` `↓` | look (45°/s); mouse drag also looks |
| `R` | restart the world from the image |
| `Esc` | quit |

Measured on a stock RunPod RTX 5090 (2026-09-17): `lingbot bench` 16.9–17.0 FPS as played, `lingbot play` 16.9 FPS with key→pixel 1.58 s p50 (the window shows it live). Nothing leaves the machine: no browser, no network, no codec.

```bash
lingbot bench                                                       # 22 s clip -> outputs/, prints s/chunk and FPS as played
lingbot clip --image me.jpg --action_path my_poses/ --prompt "…"    # offline generation, any generate.py flag
lingbot play stonehenge --input-mode hold                           # a scene by name; hold-mode input
```

## Details

- `setup.sh` fetches Python 3.12 through `uv` if the system lacks it; the torch wheels carry their own CUDA runtime, so the host needs only the driver. The weights (`robbyant/lingbot-world-v2-1.3b-causal-fast` + the Wan VAE and T5 from the 14B release) come from Hugging Face with your token and are not redistributed here.
- First `play` after `setup.sh`: ~35 s of warm-up (compiled graphs from `.inductor_cache/`); on a cold cache ~2.5 min. Each new prompt is T5-encoded once (~30 s) and cached. Any image works (it is resized to the 480×832 pixel budget keeping its aspect ratio); the first image with a new aspect ratio compiles once more (~2.5 min).
- The HUD (window title and terminal, once a second): FPS shown, seconds per chunk (a chunk = 16 frames = 1 s of video), key→pixel = keydown to the first shown frame of the latent it landed in. `--input-mode hold`: held keys act from the next chunk's first frame, releases overshoot by up to one chunk; the default `history` replays each key edge into the latent it was made in. `--frame_num` sets the rollout length (default 361 frames; the world restarts from the image after it).
- `lingbot bench` / `lingbot clip` are `generate.py` with `run.sh`'s defaults (`./run.sh` still works). `my_poses/` = `poses.npy` (one OpenCV camera-to-world 4×4 per output frame) + `intrinsics.npy`, as in `examples/*/`.
- No display (a cloud pod): `SDL_VIDEODRIVER=dummy lingbot play --headless-seconds 120` runs the real model without a window, taps `W` every 2.5 s and prints the HUD and a summary (warm-up, s/chunk, FPS, key→pixel, underruns).
- Native Windows is not supported (the prebuilt kernels are Linux wheels).

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

## What was tried

`OPTIMIZATIONS.md` is the full experiment log behind these numbers — every lever with its effect,
whether it is lossless, and the dead ends with the reason (tiny decoders, CAS sharpening, KV ring
buffer, KV quantisation, batching), so the design space does not have to be re-explored.

## Scope

- One RTX 5090 (sm_120). The prebuilt `sageattention` and `flash_attn` wheels are for sm_120, CPython 3.12, torch 2.8. Other GPUs need those built from source; an RTX 4090 should run every patch (FP8 rowwise and SageAttention both support sm_89) at roughly 12 FPS and needs T5 on the CPU to fit 24 GB — untested.
- Clip generation from an image and a camera path, and a local window driven by the keyboard (`lingbot play`). Browser streaming (WebSocket / WebRTC transports, the pacer and codec measurements) lives in lingbot-world-v2-stream; `lingbot/play/` is its model-side half (`control.py`, `live.py`, the pacer) moved here.
- Multi-GPU (`--ulysses_size`, FSDP) is upstream's code and is untouched but unmeasured here.

## CPU checks

`tests/` holds fp32 CPU equivalence checks of the fused decoder and the fused DiT against the stock modules (no GPU): `python tests/test_vae_fused_cpu.py`, `python tests/test_vae_subpixel_cpu.py`, `python tests/test_dit_fusion_cpu.py`.

`pytest tests/test_play_*.py` runs the `lingbot play` checks without a GPU: the input integrator and camera planner, the frame plumbing on a CPU stand-in for the model (`DryPipe`), the pacer, and `lingbot play --dry`, which runs the window loop headless for 3 chunks and checks that frames were shown and a key tap reached the model.

## License and credit

Upstream is CC BY-NC-SA 4.0, so this repository is too: non-commercial use, attribution, share-alike (`LICENSE.txt`, unchanged). The model, the sampler and the examples are the Robbyant team's ([paper](https://arxiv.org/abs/2607.07534), `UPSTREAM_README.md`). Kernels used: [SageAttention](https://github.com/thu-ml/SageAttention), [torchao](https://github.com/pytorch/ao), [FlashAttention](https://github.com/Dao-AILab/flash-attention).
