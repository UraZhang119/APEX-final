#!/bin/bash
#SBATCH --job-name=esm_contact_heatmap
#SBATCH --output=logs/contact_heatmap_%j.out
#SBATCH --error=logs/contact_heatmap_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --constraint=dgx

set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: sbatch $0 \"SEQUENCE\" OUTPUT_NAME [CONFIG_PATH]" >&2
  exit 1
fi

SEQUENCE_INPUT="$1"
OUTPUT_NAME="$2"
CONFIG_PATH="${3:-configs/fungi.yaml}"
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"

module load pytorch/2.2.0
mkdir -p "$PROJECT_ROOT/logs"

python -m pip install --user -e "$PROJECT_ROOT"
export PATH="$HOME/.local/bin:$PATH"

cd "$PROJECT_ROOT"

export SEQUENCE_INPUT
export OUTPUT_NAME
export CONFIG_PATH

python - <<'PY'
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from gat_pipeline.config import load_config
from gat_pipeline.data.esm import embed_sequence, load_esm_model
from gat_pipeline.utils import cmap_to_graph

sequence = os.environ["SEQUENCE_INPUT"]
output_name = os.environ["OUTPUT_NAME"]
config_path = os.environ["CONFIG_PATH"]

config = load_config(config_path)
esm_contacts = config.esm_model_contacts
esm_embeddings = config.esm_model_embeddings

model_bundle = load_esm_model(esm_contacts)
_, _, contact_map = embed_sequence(output_name, sequence, model_bundle)
cmap = contact_map.detach().cpu().numpy()

embed_bundle = load_esm_model(esm_embeddings)
_, representations, _ = embed_sequence(output_name, sequence, embed_bundle)
node_features, edge_index, _, _ = cmap_to_graph(representations, contact_map, ratio=config.ratio)

output_dir = Path("contact_maps") / output_name
output_dir.mkdir(parents=True, exist_ok=True)

npy_path = output_dir / f"{output_name}_contact_map.npy"
csv_path = output_dir / f"{output_name}_contact_map.csv"
png_path = output_dir / f"{output_name}_contact_heatmap.png"
graph_png_path = output_dir / f"{output_name}_graph.png"

np.save(npy_path, cmap)
np.savetxt(csv_path, cmap, delimiter=",")

plt.figure(figsize=(8, 6))
im = plt.imshow(cmap, cmap="viridis", origin="lower")
plt.title(f"ESM Contact Map – {output_name}")
plt.xlabel("Residue index")
plt.ylabel("Residue index")
cbar = plt.colorbar(im)
cbar.set_label("Contact probability")
plt.tight_layout()
plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved contact map to {png_path}")

if edge_index.size > 0:
    graph = nx.Graph()
    edge_pairs = edge_index.transpose()
    graph.add_edges_from(edge_pairs.tolist())
    pos = nx.spring_layout(graph, seed=42)
    plt.figure(figsize=(6, 6))
    nx.draw_networkx_nodes(graph, pos, node_color="tab:blue", node_size=80, alpha=0.8)
    nx.draw_networkx_edges(graph, pos, alpha=0.5, width=1.0)
    nx.draw_networkx_labels(graph, pos, font_size=6)
    plt.title(f"GNN Graph (ratio={config.ratio}) – {output_name}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(graph_png_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved graph visualization to {graph_png_path}")
else:
    print("Contact map produced no edges at current ratio; graph visualization skipped.")
PY
