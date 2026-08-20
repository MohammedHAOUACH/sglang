# SGLang — Qwen3.8-27B-INT8-W8A16-MTP (serveur d'inférence)

Serveur d'inférence pour le modèle **`lued/Qwen3.8-27B-INT8-W8A16-MTP`** via
[SGLang](https://github.com/sgl-project/sglang), au choix :

- en **environnement virtuel natif** (`start.sh` / `stop.sh`), ou
- en **conteneur Docker** (`compose.yaml`, image officielle `lmsysorg/sglang`).

Le modèle est **déjà téléchargé** — on pointe directement dessus.

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
- Python 3.11 (`~/.local/bin/python3.11`) — le venv se recrée via `requirements.txt` (voir Installation).
- ~340 Go libres (le venv + torch + SGLang ≈ 15 Go ; les poids sont déjà en place).
- Le modèle présent dans :
  `../vllm/models/models--lued--Qwen3.8-27B-INT8-W8A16-MTP/` (aucun re-téléchargement).

---

## Installation (mode natif — pour reproduire)

> Le mode principal est **Docker** (`compose.yaml`) ; le venv a été supprimé.
> Pour recréer le mode natif (`start.sh`) :

```bash
cd llm/sglang
~/.local/bin/python3.11 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> ⚠️ `ninja` doit être sur le PATH (il est dans `.venv/bin` ; `start.sh` l'ajoute
> automatiquement). Sans lui, la compilation JIT des kernels GPTQ-Marlin échoue.

---

## Lancement / arrêt (mode natif)

```bash
./start.sh      # lance le serveur sur http://0.0.0.0:1234
./stop.sh       # arrêt propre (SIGTERM puis SIGKILL)
```

`start.sh` est autonome : si `.venv/` est absent, il le crée depuis
`requirements.txt` (une seule fois), et il auto-détecte le snapshot du modèle
dans `../vllm/models/...` (surchargeable via `MODEL_PATH`).

Le premier démarrage compile les kernels et capture les CUDA graphs
(~5 min à froid, ~30 s ensuite grâce aux caches Triton/JIT). Les logs vont dans
`sglang.log`.

Vérification :

```bash
curl -s http://127.0.0.1:1234/health        # -> 200
curl -s http://127.0.0.1:1234/v1/models     # -> max_model_len: 262144
```

---

## Déploiement Docker (docker compose)

Le même serveur que `start.sh` (mêmes arguments), dans l'image officielle
**`lmsysorg/sglang:latest`** (Docker Hub, dernier stable).

**Démarrage rapide (nouvelle machine) :**

```bash
cp .env.example .env                 # 1. créer sa config
# 2. éditer .env : MODEL_ROOT = chemin du cache HF local du modèle
nano .env

docker compose up -d                 # 3. lancer sur http://0.0.0.0:1234
docker compose logs -f sglang        # suivre les logs
docker compose down                  # arrêt propre
curl -s http://127.0.0.1:1234/health # -> 200
```

- `MODEL_ROOT` est **obligatoire** : `docker compose` refuse de démarrer sans
  (message d'erreur explicite).
- Le conteneur réserve **2 GPU** (TP=2) et expose le port `1234` ; il reprend
  les mêmes réglages NCCL que `start.sh` (`NCCL_P2P_DISABLE=1`, etc.).
- Le modèle est monté en **lecture seule** depuis le cache HF local (aucun
  re-téléchargement). On monte le **dossier parent** `models--lued--...` et
  non le snapshot : les fichiers du snapshot sont des liens symboliques
  relatifs vers `../../blobs/...` qui doivent rester résolus dans le conteneur.
- Le 1er boot compile les kernels Triton en JIT puis capture les CUDA graphs
  (~5 min, GPU inactif pendant la compilation). Un healthcheck `/health` est
  intégré (`docker compose ps` → `healthy`).
- Image `latest` = CUDA 13 avec outils de build, nécessaire à la compilation
  JIT (l'image `latest-runtime`, ~40 % plus légère, ne peut pas compiler ;
  variante CUDA 12 : `latest-cu129`).

### Variables (`.env`)

Copier `.env.example` en `.env` pour personnaliser (défauts = valeurs de
`start.sh`) :

```bash
cp .env.example .env
```

| Variable | Défaut | Rôle |
|---|---|---|
| `SGLANG_IMAGE` | `lmsysorg/sglang:latest` | Tag d'image (mise à jour SGLang = changer ici) |
| `PORT` | `1234` | Port HTTP exposé |
| `HOST` | `0.0.0.0` | Adresse d'écoute |
| `TP` | `2` | Taille du tensor parallel |
| `GPU_COUNT` | `2` | GPU exposés au conteneur (doit valoir `TP`) |
| `CONTEXT_LENGTH` | `262144` | Contexte natif max du modèle |
| `MEM_FRACTION` | `0.95` | Fraction de VRAM réservée |
| `MAX_CONCURRENT_REQUESTS` | `1` | Requêtes concurrentes |
| `ATTN_BACKEND` | `flashinfer` | `triton` en repli si MTP+flashinfer plante au boot |
| `SERVED_MODEL_NAME` | `qwen3.8-27b-int8-w8a16` | Nom exposé par l'API |
| `MODEL_ROOT` | **obligatoire** | Dossier parent du modèle dans le cache HF (monté dans `/models`) |
| `MODEL_SNAPSHOT` | hash du snapshot | Sous-dossier pointé par `--model-path` |
| `MAMBA_CACHE_SIZE` | `4` | Slots du cache GDN (concurrence × 4) |

> ⚠️ Ne pas faire tourner le venv **et** le conteneur en même temps : le port
> `1234` ne peut pas être partagé (`./stop.sh` ou changer `PORT` dans `.env`).

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
| `MODEL_PATH` | auto-détecté | Chemin direct vers le snapshot du modèle (si non auto-détecté) |
| `MODEL_REPO_DIR` | `../vllm/models/models--lued--...` | Dossier parent du modèle dans le cache HF |

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
python3 bench_stream.py   # débit de décodage soutenu + TTFT (précis)
python3 bench.py          # débit net (méthode delta 2 appels)
```

> Les deux scripts n'utilisent que la stdlib → n'importe quel `python3` suffit
> (pas besoin du venv).
>
> Surcharges possibles : `SGLANG_BASE_URL` (défaut `http://127.0.0.1:1234/v1/chat/completions`)
> et `SGLANG_MODEL` (défaut `qwen3.8-27b-int8-w8a16`).

### Résultats mesurés (2× RTX 3090, MTP 3/1/4 — déploiement Docker `compose.yaml`)

| Sonde | TTFT | Décodage soutenu |
|---|---|---|
| Essai (prose, 1024 tok, non-thinking) | 0,12 s | **~48 tok/s** |
| Code (LRUCache, 1024 tok, non-thinking) | 0,12 s | **~65 tok/s** |
| Essai + thinking (800 tok) | 0,13 s | ~49 tok/s |
| Prompt long ~16 K tok | 23,5 s | prefill ~626 tok/s |

Débit net (méthode delta 2 appels, `bench.py`) :

| Sonde | Débit net |
|---|---|
| Code (LRUCache) | ~58 tok/s |
| Essai (prose longue) | ~42 tok/s |
| Chat court (hash map) | ~38 tok/s |
| Code + thinking | ~53 tok/s |

- Mesures prises sur le **Docker** (`compose.yaml`) ; équivalentes au venv
  (écart < 10 %, variance d'un run à l'autre — d'autres conteneurs tournent
  sur le même hôte).
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
| `MODEL_ROOT` non défini (Docker) | `docker compose` refuse de démarrer → copier `.env.example` en `.env` et renseigner `MODEL_ROOT` |
| `ValueError: Unrecognized model in /models` (Docker) | montage du snapshot seul → liens `../../blobs` cassés ; monter le dossier **parent** (`MODEL_ROOT`) |
| Boot Docker ~5 min à froid | compilation JIT Triton dans le conteneur (pas de cache `/root/.triton` persisté entre `down`/`up`) |
| Port `1234` déjà utilisé | serveur venv encore actif → `./stop.sh`, ou changer `PORT` dans `.env` |
| OOM au démarrage | baisser `MEM_FRACTION` (ex. `0.90`) ou réduire `CONTEXT_LENGTH` |
| MTP + flashinfer plante au boot | relancer avec `ATTN_BACKEND=triton ./start.sh` |

---

## Structure

```
llm/sglang/
├── compose.yaml     # déploiement Docker (image officielle lmsysorg/sglang)
├── .env             # variables locales docker compose (généré, ignoré par git)
├── .env.example     # modèle des variables docker compose
├── requirements.txt # dépendances Python figées du mode natif (venv)
├── start.sh         # lance le serveur SGLang natif (venv auto-créé, modèle auto-détecté)
├── stop.sh          # arrête le serveur natif
├── bench.py         # benchmark (débit net)
├── bench_stream.py  # benchmark streaming (TTFT + débit précis)
├── sglang.log       # logs du serveur (généré, ignoré par git)
└── README.md
```

Le modèle vit dans `../vllm/models/` (cache Hugging Face du projet vLLM) — il n'est
pas dupliqué ici.
