# APEX : Attention-based Protein EXplainer

Complete pipeline for predicting **druggability** (Human APEX-Drug) and **pathogen essentiality** (APEX-Tar) using:
- **ESM2**: Pre-trained protein language model embeddings
- **GAT**: Graph Attention Networks
- **GNNExplainer + Attention**: Interpretable predictions

---

## Quick Start

### Step 1: Prepare FASTA Files

Place your protein sequences in FASTA format:
```bash
# Option A: Use pre-prepared fungi data
ls -lh Fungi_positive.fasta Fungi_negative.fasta

# Option B: Use your own FASTA
cp your_proteome.fasta ./
```

### Step 2: Run Inference

```bash
cd /mnt/home/users/agr_169_uma/luciajc/ESM2_GAT_XAI

# Single command - get predictions for all proteins
sbatch scripts/run_production_inference.sh Fungi_positive.fasta fungi

# Output: inference_results/predictions.tsv with scores:
# - Druggability (0-1): likelihood of being drug-bindable
# - Essentiality (0-1): likelihood of being pathogen-essential
```

**Wait for job to complete:**
```bash
squeue -u $USER
tail -f logs/prod_infer_*.out  # Monitor progress
```

### Step 3: Explain Predictions for Individual Proteins

Once you have predictions, explain them for specific proteins:

```bash
# Extract protein sequence from FASTA
PROTEIN_NAME="ZTRI_11.155"
SEQUENCE=$(grep -A1 "^>$PROTEIN_NAME$" Fungi_positive.fasta | tail -1)

# Generate attention visualizations
sbatch scripts/plot_attention.sh \
  "$SEQUENCE" \
  "$PROTEIN_NAME" \
  experiments/fungi/gat/fold_0/best_loss.pt \
  configs/fungi.yaml

# Output in graficos/:
# - {PROTEIN_NAME}_line.png
# - {PROTEIN_NAME}_contact_map_teal.png
# - {PROTEIN_NAME}_contact_map_diverging.png
```

---

## Full Workflow

### Phase 1: Data Preparation (Optional - For New Training)

If you want to train new models with your own data:

```bash
# Set environment variables
export PATHO_FASTA="$(pwd)/data/Fungi_positive.fasta"
export NON_PATHO_FASTA="$(pwd)/data/Fungi_negative.fasta"
export CONFIG_PATH="$(pwd)/configs/fungi.yaml"

# Run data preparation
sbatch scripts/prepare_data.sh

```

### Phase 2: Model Training (Optional - Pre-trained Models Available)

Skip this if using pre-trained models. To train new:

```bash
export CONFIG_PATH="$(pwd)/configs/fungi.yaml"
export MODEL_NAME="gat"  # or gcn, sage
export FOLDS="0 1 2 3 4"

sbatch scripts/run_training.sh

```

### Phase 3: Inference (Main Step)

```bash
# Basic usage
sbatch scripts/run_production_inference.sh FASTA ORGANISM [OUTPUT] [DRUG_CKPT] [ESS_CKPT] [CPU_FLAG]

```

**Output:** TSV file with predictions
```
GeneSymbol      Druggability    Druggability_Score    Essentiality    Essentiality_Score
ZTRI_11.155     1               0.92                  1               0.87
ZTRI_3.608      1               0.85                  1               0.79
```

### Phase 4: Explainability Analysis

#### Simple Explainability (without annotations)

```bash
sbatch scripts/plot_attention.sh \
  "$SEQUENCE" \
  "$PROTEIN_NAME" \
  experiments/fungi/gat/fold_0/best_loss.pt \
  configs/fungi.yaml
```

#### With PFAM Annotations (Optional)

```bash
# 1. Place PFAM JSON in pfam/dominios.json
# Format:
# {
#   "PROTEIN_NAME": {
#     "domains": [
#       {"start": 1, "end": 50, "id": "PF00001", "name": "Domain1"},
#       ...
#     ]
#   }
# }

# 2. Run with PFAM overlay
sbatch scripts/plot_attention.sh \
  "$SEQUENCE" \
  "$PROTEIN_NAME" \
  experiments/fungi/gat/fold_0/best_loss.pt \
  configs/fungi.yaml \
  0 \
  pfam/dominios.json \
  "PFAM_Domains"
```

#### With Active Site Predictions (P2Rank)

```bash
# 1. Place P2Rank ZIP in asites/PROTEIN_NAME_pockets.zip
# ZIP must contain:
#   - pocketDescriptors.csv
#   - pocket_0_residues.pdb
#   - pocket_1_residues.pdb
#   - ...

# 2. Run with ASites overlay
sbatch scripts/plot_attention.sh \
  "$SEQUENCE" \
  "$PROTEIN_NAME" \
  experiments/fungi/gat/fold_0/best_loss.pt \
  configs/fungi.yaml \
  0 \
  pfam/dominios.json \
  "PFAM_Domains" \
  asites/PROTEIN_NAME_pockets.zip \
  "0,1,2"  # Pocket indices to highlight
```

---

## Understanding Results

### Scores Interpretation

| Score | Interpretation |
|-------|---|
| > 0.8 | High likelihood - prioritize |
| 0.4-0.8 | Moderate - experimental validation needed |
| < 0.4 | Low likelihood - deprioritize |

### Output Visualizations

**Line plot** (residue importance):
- X-axis: Position in sequence
- Y-axis: Importance score (GNNExplainer)
- Colored regions: PFAM domains (if provided)

**Contact maps** (residue-residue interactions):
- Red/Orange: High attention → Important interactions
- Blue: Low attention
- Boxes: Active site residues (if P2Rank provided)

---

## Configuration for New Organisms

### 1. Create Config File

```bash
cp configs/fungi.yaml configs/my_pathogen.yaml
```

Edit with your specifics:
```yaml
species: my_pathogen
pos_samples_path: ../data/my_pathogen/Essential_Genes.fasta
neg_samples_path: ../data/my_pathogen/NonEssential_Genes.fasta
kfold_root_path: ../data/my_pathogen/kfold_splitted_data
```

### 2. Prepare Data

```bash
export PATHO_FASTA="$(pwd)/Essential_Genes.fasta"
export NON_PATHO_FASTA="$(pwd)/NonEssential_Genes.fasta"
export CONFIG_PATH="$(pwd)/configs/my_pathogen.yaml"

sbatch scripts/prepare_data.sh
```

### 3. Train

```bash
export CONFIG_PATH="$(pwd)/configs/my_pathogen.yaml"
sbatch scripts/run_training.sh
```

### 4. Inference

```bash
sbatch scripts/run_production_inference.sh \
  data/my_pathogen/proteome.fasta \
  my_pathogen
```

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| CUDA out of memory | Reduce batch_size in config or use CPU |
| ModuleNotFoundError: gat_pipeline | `export PYTHONPATH="$(pwd)/src:$PYTHONPATH"` |
| No output files | Check logs: `tail -f logs/*.err` |
| Empty plots | Verify PFAM JSON is valid: `python -m json.tool pfam/dominios.json` |
| Job stuck | `squeue` → `scancel JOB_ID` |


---

## Directory Structure

```
ESM2_GAT_XAI/
├── configs/               # YAML configuration files
├── scripts/              # SLURM submission scripts
│   ├── prepare_data.sh
│   ├── run_training.sh
│   ├── run_production_inference.sh
│   ├── plot_attention.sh
│   └── production_inference.py
├── src/                  # Python source code
├── data/                 # Input data (user provides)
├── experiments/          # Trained model checkpoints
├── inference_results/    # Output predictions
├── graficos/             # Output visualizations
├── pfam/                 # PFAM annotations (user provides)
└── asites/              # P2Rank predictions (user provides)
```

---

## References

- ESM-2: Meta AI protein language model
- Graph Attention Networks (Veličković et al., 2017)
- GNNExplainer (Ying et al., 2019)
