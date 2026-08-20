# SGLang — Qwen3.8-27B-INT8-W8A16-MTP (serveur d'inférence)

Serveur d'inférence pour le modèle **`lued/Qwen3.8-27B-INT8-W8A16-MTP`** via
[SGLang](https://github.com/sgl-project/sglang), en environnement virtuel natif
(sans Docker). Le modèle est **déjà téléchargé** — on pointe directement dessus.

Inspiré de la recette SGLang officielle
([MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark)),
adaptée ici au hardware **2× RTX 3090 (Ampere, 48 Go)** au lieu du DGX Spark (GB10/128 Go).

---

## Vue d'ensemble

| Élément | Valeur |
|---|---|
| Modèle | `lued/Qwen3.8-27B-INT8-W8A16-MTP` (27 B, hybride GDN + full-attention + vision + MTP) |
| Quantification | INT8 W8A16 (compressed-tensors, `pack-quantized` group 128) |
| GPU | 2 × RTX 3090 (compute capability 8.6, **pas de NVFP4/FP8 natif**) |
| VRAM | 24 Go × 2 = 48 Go |
| Parallélisme | Tensor Parallel = 2 (`--tp-size 2`) |
| Contexte servi | 262 144 tokens (natif) — pool KV ≈ **245 000 tokens** |
| KV cache | FP8 (`fp8_e4m3`) |
| Décodage spéculatif | MTP (EAGLE 3 steps / topk 1 / 4 draft tokens) |
| Thinking | ON par défaut (désactivable par requête) |
| Port | **1234** |

Le modèle est hybride : 48 couches en *linear-attention* (Mamba/GDN, état fixe par
requête) + 16 couches en full-attention (KV cache classique). C'est ce qui rend le
contexte long tenable en VRAM.

---

## Prérequis

- Linux, pilote NVIDIA récent (≥ 550), `nvidia-smi` listant 2 GPU.
- Python 3.11 (`~/.local/bin/python3.11`) — le venv est déjà créé dans `.venv/`.
- ~340 Go libres (le venv + torch + SGLang ≈ 15 Go ; les poids sont déjà en place).
- Le modèle présent dans :
  `../vllm/models/models--lued--Qwen3.8-27B-INT8-W8A16-MTP/` (aucun re-téléchargement).

---

## Installation (déjà faite — pour reproduire)

```bash
cd llm/sglang
~/.local/bin/python3.11 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install "sglang[all]"
```

> ⚠️ `ninja` doit être sur le PATH (il est dans `.venv/bin` ; `start.sh` l'ajoute
> automatiquement). Sans lui, la compilation JIT des kernels GPTQ-Marlin échoue.

---

## Lancement / arrêt

```bash
./start.sh      # lance le serveur sur http://0.0.0.0:1234
./stop.sh       # arrêt propre (SIGTERM puis SIGKILL)
```

Le premier démarrage compile les kernels et capture les CUDA graphs
(~5 min à froid, ~30 s ensuite grâce aux caches Triton/JIT). Les logs vont dans
`sglang.log`.

Vérification :

```bash
curl -s http://127.0.0.1:1234/health        # -> 200
curl -s http://127.0.0.1:1234/v1/models     # -> max_model_len: 262144
```

---

## Configuration (variables d'environnement)

`start.sh` accepte les surcharges suivantes (défauts entre parenthèses) :

| Variable | Défaut | Rôle |
|---|---|---|
| `PORT` | `1234` | Port d'écoute HTTP |
| `HOST` | `0.0.0.0` | Adresse d'écoute |
| `TP` | `2` | Taille du tensor parallel |
| `CONTEXT_LENGTH` | `262144` | Contexte natif max du modèle |
| `MEM_FRACTION` | `0.95` | Fraction de VRAM réservée (le plus haut sûr sur 48 Go) |
| `MAX_CONCURRENT_REQUESTS` | `1` | Requêtes concurrentes (1 = max de contexte par requête) |
| `ATTN_BACKEND` | `flashinfer` | `triton` en repli si MTP+flashinfer plante au boot |

Exemple — 2 requêtes concurrentes (moins de contexte par requête) :

```bash
MAX_CONCURRENT_REQUESTS=2 ./start.sh
```

---

## Utilisation de l'API (OpenAI-compatible)

Base URL : `http://localhost:1234/v1` — modèle : `qwen3.8-27b-int8-w8a16`.

```bash
curl -s http://127.0.0.1:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b-int8-w8a16",
    "messages": [{"role": "user", "content": "Explique le parallélisme de données en 3 phrases."}],
    "max_tokens": 200,
    "temperature": 0.7,
    "chat_template_kwargs": {"enable_thinking": false}
  }' | jq '.choices[0].message'
```

- **Thinking** : ON par défaut (le raisonnement apparaît dans `reasoning_content`).
  Pour le couper par requête : `"chat_template_kwargs": {"enable_thinking": false}`
  (ou `"reasoning_effort": "low"`).
- **Tool calling** : envoyer `tools` dans la requête ; parser `qwen3_coder`.
- **Vision** : le modèle est un VLM natif (images/vidéos supportées ; le transport
  de features est forcé en `cpu` — voir Dépannage).

---

## Benchmark

```bash
. .venv/bin/python bench_stream.py   # débit de décodage soutenu + TTFT (précis)
. .venv/bin/python bench.py          # débit net (méthode delta 2 appels)
```

### Résultats mesurés (2× RTX 3090, MTP 3/1/4)

| Sonde | TTFT | Décodage soutenu |
|---|---|---|
| Essai (prose, 1024 tok, non-thinking) | 0,25 s | **~53 tok/s** |
| Code (LRUCache, 1024 tok, non-thinking) | 0,11 s | **~71 tok/s** |
| Essai + thinking (800 tok) | 0,30 s | ~46 tok/s |
| Prompt long ~16 K tok | 22,9 s | prefill ~644 tok/s |

- **MTP actif** : accept rate ~0,4–0,7, accept length ~2,2–3,0 tokens (sur 4 draftés).
- À titre de comparaison, le setup vLLM du même modèle sur la même machine
  donnait ~42–45 tok/s.

---

## Dépannage

| Symptôme | Cause / solution |
|---|---|
| `FileNotFoundError: 'ninja'` au boot | `.venv/bin` absent du PATH (corrigé dans `start.sh`) |
| `pidfd_getfd: Operation not permitted` | transport multimodal CUDA IPC bloqué → `--mm-feature-transport cpu` (déjà en place) |
| `Custom allreduce failed` (warning) | normal sans NVLink sur 3090 → `--disable-custom-all-reduce` (déjà en place) |
| OOM au démarrage | baisser `MEM_FRACTION` (ex. `0.90`) ou réduire `CONTEXT_LENGTH` |
| MTP + flashinfer plante au boot | relancer avec `ATTN_BACKEND=triton ./start.sh` |

---

## Structure

```
llm/sglang/
├── .venv/           # environnement virtuel Python 3.11 (SGLang + torch)
├── start.sh         # lance le serveur SGLang (port 1234)
├── stop.sh          # arrête le serveur
├── bench.py         # benchmark (débit net)
├── bench_stream.py  # benchmark streaming (TTFT + débit précis)
├── sglang.log       # logs du serveur (généré, ignoré par git)
└── README.md
```

Le modèle vit dans `../vllm/models/` (cache Hugging Face du projet vLLM) — il n'est
pas dupliqué ici.
