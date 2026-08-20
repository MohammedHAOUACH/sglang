#!/usr/bin/env bash
set -euo pipefail
# Arrête le serveur SGLang local (process python de launch_server).
# Le motif cible uniquement l'interpréteur Python du venv pour ne pas
# matcher ce script lui-même.
PATTERN="python -m sglang.launch_server"

pids=$(pgrep -f "${PATTERN}" 2>/dev/null || true)
if [[ -z "${pids}" ]]; then
  echo "Aucun serveur SGLang en cours."
  exit 0
fi

echo "Arrêt des process SGLang : ${pids}"
kill ${pids} 2>/dev/null || true

# Attendre jusqu'à 20 s un arrêt propre (SGLang ferme ses workers TP).
for _ in $(seq 1 20); do
  sleep 1
  if ! pgrep -f "${PATTERN}" >/dev/null 2>&1; then
    echo "Arrêté proprement."
    exit 0
  fi
done

echo "Forçage (SIGKILL)..."
pkill -9 -f "${PATTERN}" 2>/dev/null || true
echo "Arrêté."
