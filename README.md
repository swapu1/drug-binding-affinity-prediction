# Drug-Target Binding Affinity Prediction

A dual-GNN model with cross-attention fusion that predicts drug-target binding affinity (pKd) from molecular and protein pocket structure, trained on the PDBBind refined set.

## Problem

Screening candidate drug molecules against a target protein is slow and expensive when done via physical lab assays or full molecular docking simulations. This project trains a graph neural network to predict binding affinity directly from structure, enabling fast, computational screening of many candidates before committing to lab testing.

## Architecture

Two separate GNN branches process the molecule and protein pocket independently, then interact through cross-attention before a final regression head predicts pKd:

- **Molecule Encoder** — 3-layer Graph Isomorphism Network (GIN) over the ligand's atom graph (atoms as nodes, bonds as edges)
- **Pocket Encoder** — 3-layer Graph Attention Network (GAT, 4 heads) over the protein pocket's residue graph (residues as nodes, spatial contacts within 8Å as edges)
- **Cross-Attention Fusion** — lets molecule atoms and pocket residues attend to each other before pooling
- **Regression Head** — MLP outputting a single predicted pKd value

## Dataset

- **Training/Validation**: [PDBBind v2020 refined set](https://huggingface.co/datasets/photonmz/pdbbindpp-2020) (5,146 complexes after verification)
- **Test (held out)**: CASF-2013 core set (193 complexes) — standard benchmark used across published binding-affinity papers
- **Labels**: [LP-PDBBind](https://github.com/THGLab/LP-PDBBind) for the refined set, bundled CSV for the core set
- **Train/val split**: scaffold-based (Bemis-Murcko), not random — prevents structurally similar molecules from leaking across train and test

## Results

| Model | Test RMSE | Test Pearson r |
|---|---|---|
| Dual-GNN (this project) | 2.43 | 0.27 |
| Random Forest baseline (Morgan fingerprints only) | 1.81 | 0.60 |

The GNN did not outperform a simple fingerprint-based baseline on the held-out test set. Training showed signs of overfitting — validation loss plateaued while training loss continued to decrease. Several regularization strategies (increased dropout, weight decay, reduced hidden dimension, learning rate scheduling, mini-batching via gradient accumulation) were tried; results either underfit or continued overfitting depending on regularization strength, suggesting the training set size (~5,000 complexes) or the current graph feature set may be limiting factors for this architecture's complexity. This is documented as an open problem rather than resolved.

## Repository Structure

```
phase1_data_loading.py       # Verifies every structure file parses correctly
phase2_graph_construction.py # Builds molecule + pocket graphs, merges affinity labels
phase3_scaffold_split.py     # Scaffold-based train/val/test split
phase4_model.py              # Dual-GNN + cross-attention model definition
phase5_training.py           # Training loop with early stopping
phase6_evaluation.py         # Test set evaluation + baseline comparison
app.py                       # Streamlit UI: single prediction + batch virtual screening
```

## Running It

Data files (`data/`, `graphs/`) are not included in this repo due to size — see Dataset section above for sources.

```bash
pip install rdkit biopython torch torch-geometric pandas numpy scikit-learn scipy streamlit huggingface_hub
```

Run phases in order (1 → 6) to reproduce the pipeline, then:

```bash
streamlit run app.py
```

## UI

The Streamlit app has two modes:
- **Single Prediction** — check the model's prediction against a known test-set complex
- **Batch Virtual Screening** — upload one protein pocket + multiple candidate ligands, get a ranked table with predicted binding affinity, QED drug-likeness, and RDKit PAINS/BRENK structural alerts (rule-based, not a trained toxicity model)

## Tech Stack

Python, PyTorch, PyTorch Geometric, RDKit, BioPython, scikit-learn, Streamlit
