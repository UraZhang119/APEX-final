#!/bin/bash
#SBATCH --job-name=gnnexp_nodes
#SBATCH --output=logs/gnnexp_nodes_%j.out
#SBATCH --error=logs/gnnexp_nodes_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=30:00:00
#SBATCH --constraint=dgx

set -euo pipefail

if [ $# -lt 3 ]; then
  echo "Usage: sbatch $0 \"SEQUENCE\" OUTPUT_NAME FOLD [CONFIG_PATH]" >&2
  exit 1
fi

SEQUENCE_INPUT="$1"
OUTPUT_NAME="$2"
FOLD_NUMBER="$3"
CONFIG_PATH="${4:-configs/fungi.yaml}"
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"

# Allow overriding checkpoint via env, otherwise use best_loss from fold
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/experiments/fungi/gat/fold_${FOLD_NUMBER}/best_loss.pt}"

if [ ! -f "$MODEL_PATH" ]; then
  echo "Checkpoint not found: $MODEL_PATH" >&2
  exit 1
fi

module load pytorch/2.2.0
mkdir -p "$PROJECT_ROOT/logs"

python -m pip install --user -e "$PROJECT_ROOT"
export PATH="$HOME/.local/bin:$PATH"

cd "$PROJECT_ROOT"

export SEQUENCE_INPUT
export OUTPUT_NAME
export MODEL_PATH
export FOLD_NUMBER
export CONFIG_PATH

python - <<'PY'
from pathlib import Path
import os

from gat_pipeline.config import load_config
from gat_pipeline.explain.gnnexplainer import run_node_explainer

sequence = os.environ["SEQUENCE_INPUT"]
name = os.environ["OUTPUT_NAME"]
model_path = Path(os.environ["MODEL_PATH"])
fold = int(os.environ["FOLD_NUMBER"])
config_path = os.environ["CONFIG_PATH"]

config = load_config(config_path)
output_dir = Path("gnn_results")

run_node_explainer(
    sequence=sequence,
    model_path=model_path,
    output_name=name,
    output_dir=output_dir,
    ratio=config.ratio,
    fold_number=fold,
    drop_prob=config.drop_prob,
    esm_model_embeddings=config.esm_model_embeddings,
    esm_model_contacts=config.esm_model_contacts,
)
PY
