#!/bin/bash
#SBATCH --job-name=sage_training_last_gat_proc
#SBATCH --output=logs/train
#SBATCH --error=logs/train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=168:00:00
#SBATCH --constraint=dgx

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/configs/bacteria.yaml}"
MODEL_NAME="${MODEL_NAME:-gat}"
FOLDS="${FOLDS:-4}"
WANDB_API_KEY="${WANDB_API_KEY:-}"

module load pytorch/2.7.0
export PATH="$HOME/.local/bin:$PATH"

# Usa el Python del módulo y evita mezclar user site antiguo
PYTHON=/mnt/home/soft/pytorch/programs/x86_64/2.7.0/bin/python  # ajusta si cambia el módulo
export PYTHONNOUSERSITE=1

# Añadimos los paquetes ligeros en una carpeta local del repo para no tocar el user site
DEPS_DIR="$PROJECT_ROOT/.deps"
export PYTHONPATH="$PROJECT_ROOT/src:$DEPS_DIR:${PYTHONPATH:-}"
mkdir -p "$DEPS_DIR"  # Las dependencias ya están preinstaladas aquí (biopython, wandb, fair-esm); evita pip online

mkdir -p "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT"

# Si no quieres usar esta clave, exporta WANDB_API_KEY antes de lanzar el job.
WANDB_API_KEY=${WANDB_API_KEY:-18f91c379baba44b227224908b985b0c612fb044}

if [ -z "$WANDB_API_KEY" ]; then
  echo "Set WANDB_API_KEY before submitting (e.g. WANDB_API_KEY=xxxxx sbatch scripts/run_training.sh)." >&2
  exit 1
fi
export WANDB_API_KEY

for fold in $FOLDS; do
  echo "=== Training fold $fold with model $MODEL_NAME ==="
  "$PYTHON" -m gat_pipeline.cli train-fold \
    --config "$CONFIG_PATH" \
    --fold "$fold" \
    --model "$MODEL_NAME"
  echo "=== Completed fold $fold ==="
  sleep 2
done
