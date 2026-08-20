#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

VENV="$(pwd)/.venv"
# .venv/bin sur le PATH : requis pour trouver ninja (compilation JIT des kernels
# GPTQ-Marlin / flashinfer) et autres binaires du venv.
export PATH="${VENV}/bin:${PATH}"
MODEL_PATH="/home/yo/Desktop/code/llm/vllm/models/models--lued--Qwen3.8-27B-INT8-W8A16-MTP/snapshots/7c12373712d1363e2b76655cb3332c9c124627d7"

# --- Config (surchargable par variables d'environnement) -------------------
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-27b-int8-w8a16}"
TP="${TP:-2}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"
# 1 requête concurrente = max de contexte par requête (pool GDN réduit →
# plus de VRAM pour le KV cache des couches full-attention).
MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-1}"
# 0.95 = le plus haut sûr sur 2×24 Go (marge pour CUDA graphs/activations).
MEM_FRACTION="${MEM_FRACTION:-0.95}"
PORT="${PORT:-1234}"
HOST="${HOST:-0.0.0.0}"
# flashinfer = rapide ; triton = repli si MTP+flashinfer plante au boot
ATTN_BACKEND="${ATTN_BACKEND:-flashinfer}"

# GDN state pool : slots = concurrency x 4 (extra_buffer_lazy + overlap scheduler)
MAMBA_SLOTS_PER_REQ=4
MAMBA_CACHE_SIZE=$((MAX_CONCURRENT_REQUESTS * MAMBA_SLOTS_PER_REQ))

# 2x RTX 3090 sans NVLink (PCIe) : stabilise NCCL comme le setup vLLM.
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_CUMEM_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

exec "$VENV/bin/python" -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --tp-size "${TP}" \
  --trust-remote-code \
  --dtype bfloat16 \
  --disable-custom-all-reduce \
  --mm-feature-transport cpu \
  --mem-fraction-static "${MEM_FRACTION}" \
  --attention-backend "${ATTN_BACKEND}" \
  --chunked-prefill-size 8192 \
  --disable-prefill-cuda-graph \
  --kv-cache-dtype fp8_e4m3 \
  --mamba-ssm-dtype bfloat16 \
  --mamba-full-memory-ratio 4.21 \
  --mamba-radix-cache-strategy extra_buffer_lazy \
  --max-mamba-cache-size "${MAMBA_CACHE_SIZE}" \
  --max-running-requests "${MAX_CONCURRENT_REQUESTS}" \
  --context-length "${CONTEXT_LENGTH}" \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --sampling-defaults model \
  --host "${HOST}" \
  --port "${PORT}"
