# Other engines — what produced each number

One directory per engine: the scripts, out-of-tree configs and commands that produced the
numbers in the README's engine comparison, kept as the record, not as a supported path. They
were run once, by hand, on a RunPod RTX 5090 (Ubuntu 24.04, driver 570.195, CUDA 12.8) on the
date in the table; the engines move fast and their pins will drift.

| Engine | Directory | Commit benched | Result (1.3B causal_fast, 832×464, 4 steps, chunk 4) |
|---|---|---|---|
| NVIDIA FlashDreams | `flashdreams/` | `c1889e0` (2026-09-17) | 1.87 s/chunk → 8.55 FPS (DiT 1.08 + VAE 0.53 + finalize 0.27), 19.3 GB |

Raw logs, stats JSON and output clips: `lingbot-world-v2-stream/bench_results/engines/`.
The research behind each choice of settings: `lingbot-world-v2-stream/research/engines/`.
