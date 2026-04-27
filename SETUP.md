# Setup

## Overview

This project reproduces the public APEX workflow for explainable protein-level target discovery using the official APEX repository and a local notebook-based workflow. The original APEX repository is designed primarily for an HPC / SLURM environment, so this project adapts the official commands into a local reproducible workflow when necessary.

This file explains:
- how to create the environment,
- how to clone the official repository,
- where to place input files,
- how to run the notebooks,
- what outputs to expect,
- and what to do if the original scripts require local adaptation.

---

## 1. Create the Environment

You may set up the project using either Conda or pip.

### Option A: Conda

conda env create -f environment.yml
conda activate apex-repro

### Option B: pip
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Launch JupyterLab

---
## 2. Clone the Official APEX Repository

git clone https://github.com/Brunxi/APEX external/APEX

---
## 3. Repository Layout

README.md — project overview
SETUP.md — setup and run instructions
ATTRIBUTION.md — attribution of AI assistance, code, and resources
requirements.txt / environment.yml — environment setup
src/ — local wrapper code and utilities
data/raw/ — FASTA inputs or other raw inputs
data/processed/ — copied predictions, summaries, and analysis outputs
models/ — any saved artifacts if needed
notebooks/ — step-by-step reproduction workflow
docs/ — supporting notes and documentation
videos/ — demo and technical walkthrough
external/APEX/ — cloned official repository

---
## 4. Input Files

FASTA Input

Place protein sequences in FASTA format in: data/raw/

---
## 5. Notebook Execution Order

Run the notebooks in the following order.

### Notebook 1: 01_clone_and_inspect_apex.ipynb

Purpose:

- verify that the official APEX repository has been cloned correctly 
- inspect the repository structure
- inspect scripts, configs, checkpoints, and expected outputs
- record any dependency or environment issues

### Notebook 2: 02_prepare_inputs_and_run_inference.ipynb

Purpose:

- prepare or copy FASTA files into the location expected by the official workflow
- adapt the official production inference command for local execution
- run the official inference script
- inspect whether the prediction output file is generated

Expected output: data/processed/predictions.tsv

### Notebook 3: 03_reproduce_explanations.ipynb

Purpose:

- reproduce the official attention / explanation plotting workflow
- generate residue-level explanation figures
- inspect generated images and compare them to the official README expectations

### Notebook 4: 04_extension_and_analysis.ipynb

Purpose:

- run one small improvement or robustness analysis
- summarize reproduced outputs
- generate final plots and summary tables for the README and videos

Expected outputs:
- data/processed/reproduction_summary.csv
- data/processed/extension_results.csv
- final figures used in the report or videos
