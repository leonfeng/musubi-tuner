#!/usr/bin/env bash
# Krea 2 LoRA training launcher tuned for NVIDIA DGX Spark (GB10, 128 GB unified memory).
#
# The previous OOM (device 0, ~228 MiB alloc with ~655 MiB free / ~122 GiB total) came from
# holding the full bf16 DiT plus 1536px activations without gradient checkpointing, while
# Turbo sample generation blocked block-swap offloading. Settings live in configs/krea2_train.toml.

set -euo pipefail

# Reduce CUDA allocator fragmentation on unified-memory systems (GB10 reports "Not Supported"
# in nvidia-smi but still uses the shared CPU/GPU pool).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Keep dataloader worker RSS down on unified memory.
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

exec accelerate launch \
	--num_cpu_threads_per_process 1 \
	--mixed_precision bf16 \
	src/musubi_tuner/krea2_train_network.py \
	--config_file configs/krea2_train.toml
