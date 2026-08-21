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
#   ATTN_BACKEND (flashinfer), SERVED_MODEL_NAME (qwen3.8-27b-int8-w8a16),
#   SPEC_ALGORITHM (EAGLE ; DFLASH = draft externe DFlash2, exige SGLang de main),
#   DRAFT_MODEL_PATH / DRAFT_REPO_DIR (draft auto-détecté dans
#   ../vllm/models/models--z-lab--Qwen3.8-27B-DFlash2)
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

# --- Draft DFlash2 (auto-détection, utilisé seulement en DFLASH) -------------
DRAFT_REPO_DIR="${DRAFT_REPO_DIR:-../vllm/models/models--z-lab--Qwen3.8-27B-DFlash2}"
if [[ -z "${DRAFT_MODEL_PATH:-}" ]]; then
  DRAFT_MODEL_PATH="$(ls -d "${DRAFT_REPO_DIR}"/snapshots/*/ 2>/dev/null | head -1 || true)"
  DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH%/}"
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

# --- Décodage spéculatif ----------------------------------------------------
# EAGLE = MTP in-checkpoint (défaut) ; DFLASH = draft externe DFlash2.
# ⚠️ DFLASH2 exige SGLang compilé depuis main (PR #35371 + #35496) : le venv
# créé depuis requirements.txt (sglang==0.5.17) ne le contient PAS. Pour le
# mode venv + DFLASH : réinstaller sglang depuis les sources (voir README).
SPEC_ALGORITHM="${SPEC_ALGORITHM:-EAGLE}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-incoai/Qwen3.8-27B-DFlash2}"
SPEC_NUM_STEPS="${SPEC_NUM_STEPS:-3}"
SPEC_EAGLE_TOPK="${SPEC_EAGLE_TOPK:-1}"

if [[ "${SPEC_ALGORITHM}" == "DFLASH" ]]; then
  # 8 = block size du draft DFlash2 (fenêtre de vérification).
  SPEC_NUM_DRAFT_TOKENS="${SPEC_NUM_DRAFT_TOKENS:-8}"
  if [[ -z "${DRAFT_MODEL_PATH}" || ! -f "${DRAFT_MODEL_PATH}/config.json" ]]; then
    echo "Erreur : draft DFlash2 introuvable (DRAFT_MODEL_PATH='${DRAFT_MODEL_PATH}')." >&2
    echo "  → Télécharger le draft dans le cache HF local ou définir DRAFT_MODEL_PATH." >&2
    exit 1
  fi
  SPEC_ARGS=(
    --speculative-algorithm DFLASH
    --speculative-draft-model-path "${DRAFT_MODEL_PATH}"
    --speculative-num-draft-tokens "${SPEC_NUM_DRAFT_TOKENS}"
  )
  echo "⚠️  DFLASH2 : vérifier que le venv contient un SGLang de main " \
    "(sglang==0.5.17 ne supporte pas DFLASH) — sinon boot en échec."
else
  SPEC_NUM_DRAFT_TOKENS="${SPEC_NUM_DRAFT_TOKENS:-4}"
  SPEC_ARGS=(
    --speculative-algorithm "${SPEC_ALGORITHM}"
    --speculative-num-steps "${SPEC_NUM_STEPS}"
    --speculative-eagle-topk "${SPEC_EAGLE_TOPK}"
    --speculative-num-draft-tokens "${SPEC_NUM_DRAFT_TOKENS}"
  )
fi

# GDN state pool : slots = concurrency x 4 + marge radix (extra_buffer_lazy
# + overlap scheduler). La marge (4) réduit le risque de crash "Can not alloc
# mamba cache" (handoff radix d'une requête non finie) sans trop rogner le KV
# cache : 8 slots → KV ~226 K tokens, 16 → ~188 K, 4 → ~245 K mais crash
# possible sous charge de gros prompts partagés.
MAMBA_SLOTS_PER_REQ=4
MAMBA_RADIX_HEADROOM=4
MAMBA_CACHE_SIZE=$((MAX_CONCURRENT_REQUESTS * MAMBA_SLOTS_PER_REQ + MAMBA_RADIX_HEADROOM))

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
  "${SPEC_ARGS[@]}" \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --sampling-defaults model \
  --host "${HOST}" \
  --port "${PORT}"
