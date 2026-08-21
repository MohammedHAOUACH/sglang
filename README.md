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
| Décodage spéculatif | MTP (EAGLE 3 steps / topk 1 / 4 draft tokens) — option DFLASH2 disponible (voir plus bas) |
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
- Les modèles sont montés en **lecture seule** depuis le cache HF local
  (aucun re-téléchargement). `MODEL_ROOT` pointe sur le **dossier parent du
  cache HF**, celui qui contient `models--lued--...` (cible) **et**
  `models--z-lab--...` (draft DFLASH2) ; on monte le parent et non le
  snapshot : les fichiers des snapshots sont des liens symboliques relatifs
  vers `../../blobs/...` qui doivent rester résolus dans le conteneur.
- Le 1er boot compile les kernels Triton en JIT puis capture les CUDA graphs
  (~5 min, GPU inactif pendant la compilation). Un healthcheck `/health` est
  intégré (`docker compose ps` → `healthy`).
- Image `latest` = CUDA 13 avec outils de build, nécessaire à la compilation
  JIT (l'image `latest-runtime`, ~40 % plus légère, ne peut pas compiler ;
  variante CUDA 12 : `latest-cu129`). **DFLASH2 exige un build de main**
  (voir la section dédiée plus bas).

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
| `MODEL_ROOT` | **obligatoire** | Dossier parent du **cache HF** (celui qui contient `models--lued--...` et `models--z-lab--...`) — monté dans `/models` |
| `MODEL_REPO_DIR` | `models--lued--Qwen3.8-27B-INT8-W8A16-MTP` | Sous-dossier du modèle cible dans `MODEL_ROOT` |
| `MODEL_SNAPSHOT` | hash du snapshot | Snapshot du modèle cible pointé par `--model-path` |
| `SGLANG_MAMBA_CACHE_SIZE` | `8` | Slots du cache GDN (concurrence × 4 + 4 de marge radix) — 16 → KV ~188 K (plus sûr), 4 → KV ~245 K (crash possible). Préfixe `SGLANG_` : évite qu'un export shell de `MAMBA_CACHE_SIZE` n'écrase la valeur de `.env` |
| `SPEC_ALGORITHM` | `EAGLE` | Décodage spéculatif : `EAGLE` (MTP in-checkpoint, défaut) ou `DFLASH` (draft externe DFlash2 — exige une image de main, voir section DFLASH2) |
| `DRAFT_REPO_DIR` | `models--z-lab--Qwen3.8-27B-DFlash2` | Sous-dossier du draft DFlash2 dans `MODEL_ROOT` — utilisé seulement si `SPEC_ALGORITHM=DFLASH` |
| `DRAFT_SNAPSHOT` | `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` | Snapshot du draft DFlash2 pointé par `--speculative-draft-model-path` |
| `SPEC_NUM_STEPS` / `SPEC_EAGLE_TOPK` | `3` / `1` | Paramètres EAGLE uniquement (ignorés en DFLASH) |
| `SPEC_NUM_DRAFT_TOKENS` | `4` (EAGLE) / `8` (DFLASH) | Tokens draftés par pas ; en DFLASH, 8 = block size du draft |

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

## DFLASH2 (optionnel)

[DFLASH2](https://github.com/sgl-project/sglang/pull/35663) est une variante
plus récente du décodage spéculatif de SGLang pour Qwen3.8-27B : un **modèle
draft externe** (`incoai/Qwen3.8-27B-DFlash2`, diffusion par blocs) remplace
le MTP in-checkpoint (EAGLE). C'est la recette du cookbook officiel SGLang
(Qwen3.8-27B, cellule « DFLASH2 ») — elle vise surtout Blackwell (RTX 5090 /
RTX PRO 6000), où elle bat EAGLE en acceptation (~3 tokens/bloc au lieu de
~2,5) au prix d'une VRAM légèrement plus élevée.

### Prérequis : un SGLang de `main`

DFLASH2 a été ajouté dans les PRs [#35371](https://github.com/sgl-project/sglang/pull/35371)
(draft DFlash2) et [#35496](https://github.com/sgl-project/sglang/pull/35496)
(lm_head quantisé) — **toutes deux postérieures à la dernière release
(0.5.17)**. L'image `lmsysorg/sglang:latest` ne le contient pas.

**Option rapide — image `dev` (build roulant de main) :**

```bash
# dans .env
SGLANG_IMAGE=lmsysorg/sglang:dev
```

**Option figée — build local au commit validé par le cookbook** (~1-2 h,
reproductible) :

```bash
git clone https://github.com/sgl-project/sglang.git && cd sglang
git checkout 1cf2b8c54d81802abc15dcf23a29b9cc687bc01e   # PR #35496
docker build -t sglang:dflash2 -f docker/Dockerfile .
# puis dans .env : SGLANG_IMAGE=sglang:dflash2
```

> Si le boot échoue sur l'image `dev` (outils de build manquants pour la
> compilation JIT GPTQ-Marlin), repasser au build figé ci-dessus.

### Activer DFLASH2

```bash
# dans .env
SGLANG_IMAGE=lmsysorg/sglang:dev
SPEC_ALGORITHM=DFLASH
# Draft local (déjà dans code/vllm/models) — défauts dans compose.yaml :
# DRAFT_REPO_DIR=models--z-lab--Qwen3.8-27B-DFlash2
# DRAFT_SNAPSHOT=50307d4c4cde6860d4eee73e2547cd786fe8e8a4

docker compose up -d --force-recreate
```

Les flags générés sont alors : `--speculative-algorithm DFLASH
--speculative-draft-model-path /models/models--z-lab--Qwen3.8-27B-DFlash2/snapshots/<hash>
--speculative-num-draft-tokens 8` (les flags EAGLE `--speculative-num-steps`
et `--speculative-eagle-topk` sont retirés automatiquement). Le draft est
chargé depuis le cache HF local monté (aucun téléchargement).

Pour revenir à EAGLE : `SPEC_ALGORITHM=EAGLE` (et `SGLANG_IMAGE=lmsysorg/sglang:latest` si besoin).

### Mode natif (venv)

Le venv créé depuis `requirements.txt` est figé sur `sglang==0.5.17` : **pas de
DFLASH**. Pour l'utiliser en mode natif, réinstaller SGLang depuis les sources
(dans le venv) :

```bash
git clone https://github.com/sgl-project/sglang.git && cd sglang
git checkout 1cf2b8c54d81802abc15dcf23a29b9cc687bc01e
~/.local/bin/python3.11 -m pip install -e "python[all]"
```

puis `SPEC_ALGORITHM=DFLASH ./start.sh`. `start.sh` auto-détecte le draft
(`../vllm/models/models--z-lab--Qwen3.8-27B-DFlash2/snapshots/*/`,
surchargeable via `DRAFT_MODEL_PATH` / `DRAFT_REPO_DIR`) et affiche un
 avertissement si le venv ne contient pas un SGLang de main.

### Limites sur 2× RTX 3090

- **Non validé sur Ampere (SM86)** : le cookbook n'a exercé DFLASH2 que sur
  Blackwell. Les kernels utilisés (Triton, conv groupée via `torch.compile`)
  ne sont pas limités à une architecture, mais les perfs ne sont pas garanties
  — mesurer avec `bench_stream.py` / `bench.py` et comparer à EAGLE.
- **VRAM** : le draft s'ajoute au modèle. Si OOM au boot ou en graph capture,
  baisser `MEM_FRACTION` (ex. `0.90`, puis `0.85`). Le KV cache du draft suit
  `--kv-cache-dtype` (déjà `fp8_e4m3`, ce qui divise par 2 le pool draft).
- **TP=2** : le worker draft est compatible TP, mais seul le TP=1 a été
  mesuré par le cookbook.

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
| Crash `AssertionError: Can not alloc mamba cache` (SIGQUIT) | pool GDN saturé au handoff radix (gros prompts, préfixe verrouillé) → augmenter `SGLANG_MAMBA_CACHE_SIZE` (8 par défaut ; 16 = plus sûr, KV ~188 K) |
| `docker compose ps` → `unhealthy` pendant une grosse requête | prefill très long (gros contexte, requête unique) : la boucle HTTP ne répond pas au healthcheck → temporaire et sans conséquence (pas de redémarrage sur unhealthy) ; se rétablit seul |

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
