#!/bin/bash
#SBATCH --job-name=gat_prod_infer
#SBATCH --output=logs/prod_infer_%j.out
#SBATCH --error=logs/prod_infer_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --constraint=dgx

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: sbatch scripts/run_production_inference.sh FASTA [ORGANISM] [OUTPUT] [DRUG_CKPT] [ESS_CKPT] [CPU_FLAG]

Required:
  FASTA         Ruta al FASTA con proteínas.

Opcionales:
  ORGANISM      fungi | insect | bacteria (por defecto: fungi).
  OUTPUT        Ruta del TSV de salida (por defecto la que genera el .py).
  DRUG_CKPT     Checkpoint de druggability (por defecto el hardcodeado en el .py).
  ESS_CKPT      Checkpoint de essentiality para el organismo.
  CPU_FLAG      Usa "cpu" para forzar CPU; cualquier otro valor se ignora.

Ejemplo:
  sbatch scripts/run_production_inference.sh data/proteoma.fasta fungi
USAGE
}

if [ $# -lt 1 ]; then
  usage
  exit 1
fi

# ------------------- Resolver rutas -------------------
if [ -n "${PROJECT_ROOT:-}" ]; then
  if [ -d "${PROJECT_ROOT}" ]; then
    case "${PROJECT_ROOT}" in
      /*) PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)" ;;
      *)
        echo "[WARN] PROJECT_ROOT no es absoluta, usando la ruta del script." >&2
        unset PROJECT_ROOT
        ;;
    esac
  else
    echo "[WARN] PROJECT_ROOT no existe, usando la ruta del script." >&2
    unset PROJECT_ROOT
  fi
fi

if [ -z "${PROJECT_ROOT:-}" ]; then
  if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
  else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
  fi
fi

resolve_path() {
  local path_input="$1"
  if command -v realpath >/dev/null 2>&1; then
    realpath "$path_input"
  else
    python - "$path_input" <<'PY'
import sys
from pathlib import Path
print(Path(sys.argv[1]).expanduser().resolve())
PY
  fi
}

# ------------------- Argumentos -------------------
FASTA_PATH=$(resolve_path "$1")
ORGANISM=${2:-fungi}
OUTPUT_PATH_RAW=${3:-}
DRUG_CKPT_RAW=${4:-}
ESS_CKPT_RAW=${5:-}
CPU_FLAG=${6:-}

if [ ! -f "$FASTA_PATH" ]; then
  echo "[ERROR] No se encontró el FASTA: $FASTA_PATH" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
mkdir -p logs

# ------------------- Entorno -------------------
if command -v module >/dev/null 2>&1; then
  module load pytorch/2.7.0 >/dev/null 2>&1 || echo "[WARN] No se pudo cargar pytorch/2.7.0; continúa."
fi

# Instala dependencias en el user site, salvo que se pida saltar
if [ "${SKIP_PIP_INSTALL:-0}" != "1" ]; then
  python -m pip install --user -e "$PROJECT_ROOT" >/dev/null 2>&1 || echo "[WARN] Instalación editable falló; verifica dependencias."
fi
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

CMD=(python scripts/production_inference.py --fasta "$FASTA_PATH" --organism "$ORGANISM")
if [ -n "$OUTPUT_PATH_RAW" ]; then
  OUTPUT_PATH=$(resolve_path "$OUTPUT_PATH_RAW")
  CMD+=(--output "$OUTPUT_PATH")
fi
if [ -n "$DRUG_CKPT_RAW" ]; then
  DRUG_CKPT=$(resolve_path "$DRUG_CKPT_RAW")
  CMD+=(--druggability-checkpoint "$DRUG_CKPT")
fi
if [ -n "$ESS_CKPT_RAW" ]; then
  ESS_CKPT=$(resolve_path "$ESS_CKPT_RAW")
  CMD+=(--essentiality-checkpoint "$ESS_CKPT")
fi
if [ "$CPU_FLAG" = "cpu" ]; then
  CMD+=(--cpu)
fi

echo "[INFO] Ejecutando: ${CMD[*]}"
"${CMD[@]}"
