#!/bin/bash
#SBATCH --job-name=gat_plot_attention
#SBATCH --output=logs/plot_attention_%j.out
#SBATCH --error=logs/plot_attention_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

if [ $# -lt 3 ]; then
  echo "Uso: sbatch $0 \"SECUENCIA\" NOMBRE CHECKPOINT [CONFIG] [FOLD] [ANNOTATION_JSON] [ANNOTATION_NAME] [ACTIVE_SITE_ZIP] [ACTIVE_SITE_POCKETS]" >&2
  exit 1
fi

SEQUENCE_INPUT="$1"
PROTEIN_NAME="$2"
CHECKPOINT_PATH="$3"
shift 3

CONFIG_PATH="configs/fungi.yaml"
FOLD_NUMBER="0"
ANNOTATION_JSON=""
ANNOTATION_NAME=""
ACTIVE_SITE_ZIP=""
ACTIVE_SITE_POCKETS=""
SEQ_START_OFFSET="1"

if [ $# -gt 0 ]; then
  CONFIG_PATH="$1"
  shift
fi
if [ $# -gt 0 ]; then
  if [[ "$1" =~ ^-?[0-9]+$ ]]; then
    FOLD_NUMBER="$1"
    shift
  fi
fi
if [ $# -gt 0 ]; then
  ANNOTATION_JSON="$1"
  shift
fi
if [ $# -gt 0 ]; then
  ANNOTATION_NAME="$1"
  shift
fi
if [ $# -gt 0 ]; then
  ACTIVE_SITE_ZIP="$1"
  shift
fi
if [ $# -gt 0 ]; then
  ACTIVE_SITE_POCKETS="$1"
  shift
fi
if [ $# -gt 0 ]; then
  SEQ_START_OFFSET="$1"
  shift
fi

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
INFERENCE_DIR="${INFERENCE_DIR:-$PROJECT_ROOT/inference_results}"
EXPLAIN_DIR="${EXPLAIN_DIR:-$PROJECT_ROOT/gnn_results}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/graficos}"


module load pytorch/2.7.0 

mkdir -p "$PROJECT_ROOT/logs"

# Permite saltar la instalación editable si ya tienes las dependencias
# o el nodo no tiene salida a internet.
if [ "${SKIP_PIP_INSTALL:-0}" != "1" ]; then
  python -m pip install --user -e "$PROJECT_ROOT" || echo "[WARN] Editable install failed; assuming deps already available."
fi
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

cd "$PROJECT_ROOT"

export SEQUENCE_INPUT
export PROTEIN_NAME
export CHECKPOINT_PATH
export CONFIG_PATH
export FOLD_NUMBER
export INFERENCE_DIR
export EXPLAIN_DIR
export OUTPUT_DIR
if [ -n "$ANNOTATION_JSON" ]; then
  export ANNOTATION_JSON
fi
if [ -n "$ANNOTATION_NAME" ]; then
  export ANNOTATION_NAME
fi
if [ -n "$ACTIVE_SITE_ZIP" ]; then
  export ACTIVE_SITE_ZIP
fi
if [ -n "$ACTIVE_SITE_POCKETS" ]; then
  export ACTIVE_SITE_POCKETS
fi
export SEQ_START_OFFSET

python - <<'PY'
import os
from pathlib import Path

from gat_pipeline.config import load_config
from gat_pipeline.visualization import plot_attention_and_importance

sequence = os.environ["SEQUENCE_INPUT"]
name = os.environ["PROTEIN_NAME"]
checkpoint_path = Path(os.environ["CHECKPOINT_PATH"])
config_path = os.environ["CONFIG_PATH"]
fold_number = int(os.environ["FOLD_NUMBER"])
inference_dir = Path(os.environ["INFERENCE_DIR"])
explain_dir = Path(os.environ["EXPLAIN_DIR"])
output_dir = Path(os.environ["OUTPUT_DIR"])
annotation_json = os.environ.get("ANNOTATION_JSON")
annotation_path = Path(annotation_json) if annotation_json else None
annotation_name = os.environ.get("ANNOTATION_NAME") or None
active_site_zip = os.environ.get("ACTIVE_SITE_ZIP")
active_site_path = Path(active_site_zip) if active_site_zip else None
pocket_env = os.environ.get("ACTIVE_SITE_POCKETS")
if pocket_env:
    pocket_list = []
    for token in pocket_env.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            pocket_list.append(int(token))
        except ValueError:
            continue
    if not pocket_list:
        pocket_list = None
else:
    pocket_list = None

config = load_config(config_path)

line_path, contact_path, contact_alt_path = plot_attention_and_importance(
    sequence=sequence,
    protein_name=name,
    checkpoint_path=checkpoint_path,
    config=config,
    fold_number=fold_number,
    inference_dir=inference_dir,
    explain_dir=explain_dir,
    output_dir=output_dir,
    annotation_path=annotation_path,
    annotation_name=annotation_name,
    active_site_zip=active_site_path,
    active_site_pockets=pocket_list,
    seq_start_offset=int(os.environ.get("SEQ_START_OFFSET", "1")),
)

print(f"Saved line figure to {line_path}")
print(f"Saved contact map (teal) to {contact_path}")
print(f"Saved contact map (diverging) to {contact_alt_path}")
PY
