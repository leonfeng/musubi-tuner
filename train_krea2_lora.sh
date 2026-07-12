#!/usr/bin/env bash
# Krea 2 LoRA training launcher tuned for MSI EdgeXpert (NVIDIA GB10, 128 GB unified memory, CUDA 13.0).

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
