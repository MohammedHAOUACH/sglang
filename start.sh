#!/usr/bin/env bash
# Lance le serveur SGLang natif (mode venv) — port 1234 par défaut.
#
# Portabilité : aucun chemin codé en dur. Le snapshot du modèle est
# auto-détecté dans le cache Hugging Face local, et le venv est créé
# automatiquement depuis requirements.txt s'il est absent.
#
# Surcharges possibles (défauts entre parenthèses) :
#   MODEL_PATH          chemin direct vers le snapshot (sinon auto-détection)
#   MODEL_REPO_DIR      dossier parent du modèle HF (../vllm/models/models--lued--Qwen3.8-27B-INT8-W8A16-MTP)
#   PORT (1234), HOST (0.0.0.0), TP (2), CONTEXT_LENGTH (262144),
#   MEM_FRACTION (0.95), MAX_CONCURRENT_REQUESTS (1),
#   ATTN_BACKEND (flashinfer), SERVED_MODEL_NAME (qwen3.8-27b-int8-w8a16)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

VENV="$(pwd)/.venv"
# .venv/bin sur le PATH : requis pour trouver ninja (compilation JIT des kernels
# GPTQ-Marlin / flashinfer) et autres binaires du venv.
export PATH="${VENV}/bin:${PATH}"

# --- Modèle (auto-détection si MODEL_PATH non fourni) ----------------------
MODEL_REPO_DIR="${MODEL_REPO_DIR:-../vllm/models/models--lued--Qwen3.8-27B-INT8-W8A16-MTP}"
if [[ -z "${MODEL_PATH:-}" ]]; then
  MODEL_PATH="$(ls -d "${MODEL_REPO_DIR}"/snapshots/*/ 2>/dev/null | head -1 || true)"
  MODEL_PATH="${MODEL_PATH%/}"
fi
if [[ -z "${MODEL_PATH}" || ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "Erreur : modèle introuvable (MODEL_PATH='${MODEL_PATH}')." >&2
  echo "  → Définir MODEL_PATH (chemin du snapshot) ou MODEL_REPO_DIR (dossier parent du cache HF)." >&2
  exit 1
fi

# --- Venv (création automatique si absent) ---------------------------------
if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "Venv absent — création depuis requirements.txt (1 seule fois, ~quelques minutes)..."
  python3.11 -m venv "${VENV}"
  "${VENV}/bin/pip" install --upgrade pip
  "${VENV}/bin/pip" install -r requirements.txt
fi

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

echo "Modèle : ${MODEL_PATH}"
echo "Serving sur http://${HOST}:${PORT} (modèle '${SERVED_MODEL_NAME}', TP=${TP})"

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
