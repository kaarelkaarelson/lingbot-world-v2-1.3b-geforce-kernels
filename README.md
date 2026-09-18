# LingBot-World 2.0 realtime

Write-up with the videos: [kaarelkaarelson.com/lingbot](https://kaarelkaarelson.com/lingbot/)

Real-time [LingBot-World 2.0](https://github.com/Robbyant/lingbot-world-v2) (1.3B `causal_fast`) on one RTX 5090: **<!-- n:fps_ours -->16.1<!-- /n --> FPS as played** at 832×464, <!-- n:speedup_paper -->2.7×<!-- /n --> the original paper's code (<!-- n:fps_paper -->6.0<!-- /n --> FPS on the same card), with the original Wan 2.1 decoder and no change to the model. Real time is 16 FPS.

![lingbot play dragon at 16 fps](docs/dragon_16fps.gif)

`lingbot play dragon`, 4 s of the 22 s clip at native 832×464 (the GIF plays at 12 fps; the counter is the real per-chunk generation rate, 16 frames ÷ that chunk's DiT + VAE time). Full clip: [dragon_16fps.mp4](https://github.com/kaarelkaarelson/lingbot-world-v2-realtime/releases/download/v0.2.0/dragon_16fps.mp4) (22 MB).

Other engines that run this checkpoint, measured out of the box on the same card at the same settings (832×464, 4 steps, 16-frame chunks, Wan VAE; steady state after warm-up, one run each):

<!-- table:engines -->
| Engine | s per chunk | FPS as played | Ours vs it | What it ran on the 5090 |
|---|---|---|---|---|
| **Ours** | 0.98 | **16.1** | — | FP8 GEMMs, SageAttention, fused + compiled DiT, fused fp16 VAE |
| SGLang Diffusion `v0.5.17` | 2.48 | 6.45 | **2.5×** | torch SDPA, bf16 eager, fp32 VAE |
| NVIDIA FlashDreams `c1889e0` | 1.85 | 8.65 | **1.9×** | bf16 cuDNN SDPA, its compile + CUDA graphs; window 20/6, static camera |
| LightX2V `69018c9` | 2.07 | 7.73 | **2.1×** | torch SDPA, bf16 DiT and VAE, eager |
| Original paper's code, single GPU | 2.68 | 6.0 | **2.7×** | bf16 FlashAttention-2 eager, fp32 VAE |
<!-- /table:engines -->

Every engine was run as it ships; nothing of ours was added to another engine. Speedup is FPS as played, ours ÷ theirs.

Scripts, deviations and raw logs: [`bench/engines/`](bench/engines/README.md), with one clip played at each engine's cadence.

This is the upstream repository at commit `1895d30` plus a set of inference patches, applied in-tree, with one command to run it. Everything here was measured on a RunPod RTX 5090 (32 GB); the measurements, the profiles and the quality checks live in [lingbot-world-v2-stream](https://github.com/kaarelkaarelson/lingbot-world-v2-stream).

| Configuration | DiT + VAE, s per chunk | FPS as played |
|---|---|---|
| Original paper's code, single GPU (`--preset stock`) | <!-- n:dit_paper -->1.62<!-- /n --> + <!-- n:vae_paper -->1.06<!-- /n --> = <!-- n:s_paper -->2.68<!-- /n --> | <!-- n:fps_paper -->6.0<!-- /n --> |
| `--preset exact` (DiT bit-identical to the paper's bf16 model) | <!-- n:dit_exact -->0.73<!-- /n --> + <!-- n:vae_exact -->0.34<!-- /n --> = <!-- n:s_exact -->1.07<!-- /n --> | <!-- n:fps_exact -->14.8<!-- /n --> |
| **`--preset fast` (default)** | **<!-- n:dit_ours -->0.64<!-- /n --> + <!-- n:vae_ours -->0.34<!-- /n --> = <!-- n:s_ours -->0.98<!-- /n -->** | **<!-- n:fps_ours -->16.1<!-- /n -->** |

"As played" is what a streaming loop pays per second of video: four denoising steps plus the decode of 16 frames. The denoise loop alone runs at 25 FPS.

## Quick start

One RTX 5090 (32 GB), Linux x86_64 or Windows via WSL2 (untested — reports welcome), NVIDIA driver ≥ 570, a Hugging Face token for the weights.

```bash
git clone https://github.com/kaarelkaarelson/lingbot-world-v2-realtime
cd lingbot-world-v2-realtime
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

Measured on stock RunPod RTX 5090 pods (2026-09-17): `lingbot bench` <!-- n:fps_bench_range -->16.1–17.0<!-- /n --> FPS as played depending on the scene (the tables use the dragon clip), `lingbot play` <!-- n:fps_play -->16.9<!-- /n --> FPS with key→pixel <!-- n:key_to_pixel_s -->1.58<!-- /n --> s p50 (the window shows it live). Nothing leaves the machine: no browser, no network, no codec.

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

## Optimizations

Every change is inference-side: the checkpoint, the sampler (4 steps, 4-latent chunks, 18-frame KV window) and the decoder architecture are upstream's. The work went top-down through the standard layers, cheapest and most general first, measured at each step, and stopped at the kernel boundary. Seconds per chunk after each step, in the order applied (a chunk is 16 frames, one second of video); each gain was measured on its own, the last step closes to the measured total.

<!-- table:ladder -->
| Step | Before | After | s/chunk |
|---|---|---|---|
| Host&nbsp;syncs | CPU↔GPU sync on every layer | bookkeeping on the GPU | 2.68&nbsp;→&nbsp;2.57 |
| Decoder | [Wan 2.1 VAE](https://arxiv.org/abs/2503.20314) in fp32 | fp16 with [sub-pixel](https://arxiv.org/abs/1609.05158) upsampling | 2.57&nbsp;→&nbsp;1.95 |
| Compiler | PyTorch eager | one compiled graph | 1.95&nbsp;→&nbsp;1.68 |
| Matmuls | bf16 linears | FP8 rowwise via [torchao](https://github.com/pytorch/ao/tree/main/torchao/float8) | 1.68&nbsp;→&nbsp;1.47 |
| Attention | FlashAttention-2 | [SageAttention 2.2](https://arxiv.org/abs/2505.21136) | 1.47&nbsp;→&nbsp;1.04 |
| Kernel&nbsp;fusion | one kernel per operation | fused kernels for norm, RoPE, residual and FP8 quant | 1.04&nbsp;→&nbsp;0.98 |
| **Total** | 6.0&nbsp;FPS | **16.1&nbsp;FPS** | **2.68&nbsp;→&nbsp;0.98** |
<!-- /table:ladder -->

Original paper's code vs ours, per chunk (host syncs per three chunks; GPU busy and kernel launches from the profiler traces of both configurations, `OPTIMIZATIONS.md` §13 and §17):

<!-- table:baseline -->
| | Original paper's code | Ours |
|---|---|---|
| FPS | 6.0 | **16.1** |
| s / chunk | 2.68 | **0.98** |
| DiT | 1.62 s | **0.64 s** |
| Decoder | 1.06 s | **0.34 s** |
| GPU busy | 90% | **98%** |
| Kernel launches | ~20,000 | **~4,800** |
| Host syncs | 110 | **2** |
<!-- /table:baseline -->

What is left runs inside four kernels written by others; three are near the card's peak. The one with room is attention: a hand-written kernel at 90 % of peak would gain about one frame per second, so there is no custom kernel (`OPTIMIZATIONS.md` §17). RTX 5090 peaks from NVIDIA's specification.

<!-- table:peaks -->
| Kernel | Reached | Peak on RTX 5090 | of peak |
|---|---|---|---|
| SageAttention | 543 TOPS | 838 TOPS INT8 | **65 %** |
| FP8 matmuls | 390 TFLOP/s | 419 TFLOP/s FP8 | **90 %** |
| Decoder convolutions | 173 TFLOP/s | 210 TFLOP/s FP16 | **83 %** |
| Fused elementwise | ~1.3 TB/s | 1.8 TB/s memory | **~70 %** |
<!-- /table:peaks -->

Lossless: four of the six steps are bit-identical to the paper's code; FP8 and the attention kernel were checked on identical inputs. PSNR, SSIM and LPIPS are the same latents through the paper's fp32 decoder and ours, after the mp4 encoder; the rest are no-reference metrics on the generated clips, first / last second where drift matters (`quality_summary.tsv`, exp15).

<!-- table:quality -->
| | Original paper's code | Ours |
|---|---|---|
| PSNR | reference | **43.6 dB** |
| [SSIM](https://doi.org/10.1109/TIP.2003.819861) | reference | **0.981** |
| [LPIPS](https://arxiv.org/abs/1801.03924) | reference | **0.004** |
| [MUSIQ](https://arxiv.org/abs/2108.05997) | 68.98 | **68.99** |
| [CLIP-IQA](https://arxiv.org/abs/2207.12396) | 0.592 | **0.590** |
| Sharpness (Laplacian), first / last s | 1022 / 298 | **1023 / 298** |
| Colourfulness, first / last s | 41.9 / 50.2 | **41.9 / 50.2** |
| Brightness, first / last s | 0.692 / 0.384 | **0.692 / 0.384** |
| Flicker | 0.0381 | **0.0381** |
| DiT latents, exact preset | reference | **bit-identical** |
<!-- /table:quality -->

`OPTIMIZATIONS.md` is the full log behind the table: every experiment with its measurement, the profiles and rooflines, and the levers that were measured and rejected.

## Scope

- One RTX 5090 (sm_120). The prebuilt `sageattention` and `flash_attn` wheels are for sm_120, CPython 3.12, torch 2.8. Other GPUs need those built from source; an RTX 4090 should run every patch (FP8 rowwise and SageAttention both support sm_89) at roughly 12 FPS and needs T5 on the CPU to fit 24 GB — untested.
- Clip generation from an image and a camera path, and a local window driven by the keyboard (`lingbot play`). Browser streaming (WebSocket / WebRTC transports, the pacer and codec measurements) lives in lingbot-world-v2-stream; `lingbot/play/` is its model-side half (`control.py`, `live.py`, the pacer) moved here.
- Multi-GPU (`--ulysses_size`, FSDP) is upstream's code and is untouched but unmeasured here.

## CPU checks

`tests/` holds fp32 CPU equivalence checks of the fused decoder and the fused DiT against the stock modules (no GPU): `python tests/test_vae_fused_cpu.py`, `python tests/test_vae_subpixel_cpu.py`, `python tests/test_dit_fusion_cpu.py`.

`pytest tests/test_play_*.py` runs the `lingbot play` checks without a GPU: the input integrator and camera planner, the frame plumbing on a CPU stand-in for the model (`DryPipe`), the pacer, and `lingbot play --dry`, which runs the window loop headless for 3 chunks and checks that frames were shown and a key tap reached the model.

## License and credit

Upstream is CC BY-NC-SA 4.0, so this repository is too: non-commercial use, attribution, share-alike (`LICENSE.txt`, unchanged). The model, the sampler and the examples are the Robbyant team's ([paper](https://arxiv.org/abs/2607.07534), `UPSTREAM_README.md`). Kernels used: [SageAttention](https://github.com/thu-ml/SageAttention), [torchao](https://github.com/pytorch/ao), [FlashAttention](https://github.com/Dao-AILab/flash-attention).
