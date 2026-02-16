#!/bin/bash
#SBATCH --job-name=gat_prepare
#SBATCH --output=logs/prepare_%j.out
#SBATCH --error=logs/prepare_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=60:00:00
#SBATCH --constraint=dgx

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/configs/insect.yaml}"
PATHO_FASTA="${PATHO_FASTA:-}"
NON_PATHO_FASTA="${NON_PATHO_FASTA:-}"

if [ -z "$PATHO_FASTA" ] || [ -z "$NON_PATHO_FASTA" ]; then
  echo "Set PATHO_FASTA and NON_PATHO_FASTA environment variables before submitting." >&2
  exit 1
fi

module load pytorch/2.7.0
mkdir -p "$PROJECT_ROOT/logs"

export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

cd "$PROJECT_ROOT"

python -m gat_pipeline.cli prepare-data \
  --config "$CONFIG_PATH" \
  --pathogenesis-fasta "$PATHO_FASTA" \
  --non-pathogenesis-fasta "$NON_PATHO_FASTA"
