#!/bin/bash
#SBATCH --job-name=gat_baselines
#SBATCH --output=logs/baselines_%j.out
#SBATCH --error=logs/baselines_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --constraint=dgx

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: sbatch scripts/run_baselines.sh [CONFIG_PATH] [OUTPUT_DIR]

Arguments:
  CONFIG_PATH   YAML de configuración del pipeline (por defecto: configs/fungi.yaml).
  OUTPUT_DIR    Carpeta donde se guardarán las métricas (por defecto: metrics_analysis/baselines).

Variables de entorno útiles:
  PROJECT_ROOT  Directorio raíz del repositorio (detectado automáticamente si no se define).
  FOLDS         Lista separada por comas de folds a evaluar (por ejemplo: "0,1,2").

Ejemplos:
  sbatch scripts/run_baselines.sh
  FOLDS=0,1 bash scripts/run_baselines.sh configs/human.yaml metrics_analysis/human_baselines
USAGE
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

CONFIG_ARG="${1:-configs/fungi.yaml}"
OUTPUT_ARG="${2:-metrics_analysis/baselines}"

if [ -n "${PROJECT_ROOT:-}" ]; then
  if [[ "$PROJECT_ROOT" != /* ]]; then
    echo "[WARN] PROJECT_ROOT must be absolute. Ignoring '${PROJECT_ROOT}'." >&2
    unset PROJECT_ROOT
  elif [ ! -d "$PROJECT_ROOT" ]; then
    echo "[WARN] PROJECT_ROOT='${PROJECT_ROOT}' does not exist. Ignoring." >&2
    unset PROJECT_ROOT
  fi
fi

if [ -z "${PROJECT_ROOT:-}" ]; then
  if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
    PROJECT_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
  else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
  fi
fi

cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/logs"

resolve_path() {
  local input_path="$1"
  python - "$input_path" <<'PY' | tail -n 1
import sys
from pathlib import Path
print(Path(sys.argv[1]).expanduser().resolve())
PY
}

CONFIG_PATH="$(resolve_path "$CONFIG_ARG")"
OUTPUT_DIR_ABS="$(resolve_path "$OUTPUT_ARG")"

if command -v module >/dev/null 2>&1; then
  module load pytorch/2.2.0 >/dev/null 2>&1 || echo "[WARN] Could not load pytorch/2.2.0"
fi

python -m pip install --user -e "$PROJECT_ROOT" >/dev/null 2>&1 || echo "[WARN] Editable install failed; ensure dependencies are available."
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1

export CONFIG_PATH
export OUTPUT_DIR="$OUTPUT_DIR_ABS"
export BASELINE_FOLDS="${FOLDS:-}"

python - <<'PY'
import os
from pathlib import Path

from gat_pipeline.baselines import run_baseline_comparisons

config_path = Path(os.environ["CONFIG_PATH"])
output_dir = Path(os.environ["OUTPUT_DIR"])
fold_env = os.environ.get("BASELINE_FOLDS", "").strip()
folds = [int(chunk) for chunk in fold_env.split(",") if chunk.strip()] if fold_env else None

print(f"[INFO] Running baselines with config {config_path} -> {output_dir}")
artifacts = run_baseline_comparisons(config_path=config_path, output_dir=output_dir, folds=folds)
if not artifacts:
  raise SystemExit("No baseline results were produced. Check the logs above for details.")

for name, path in artifacts.items():
  print(f"[INFO] {name} results stored in {path}")
PY
