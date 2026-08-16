"""
Phase 3: Scaffold-Based Train/Val/Test Split
Groups molecules by Bemis-Murcko scaffold, splits by scaffold group (not by
individual complex) to prevent structurally similar molecules leaking across
train/val/test.

Core set (193 complexes) is kept fully separate as the final test set,
matching standard CASF benchmark practice. Only the refined set (5,146
complexes) gets split into train/val here.
"""

import os
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import warnings
warnings.filterwarnings("ignore")

GRAPHS_DIR = "graphs"
REFINED_DIR = "data/refined_set"

TRAIN_FRACTION = 0.80
VAL_FRACTION = 0.20
RANDOM_SEED = 42

OUTPUT_TRAIN_CSV = "split_train.csv"
OUTPUT_VAL_CSV = "split_val.csv"
OUTPUT_TEST_CSV = "split_test.csv"


def get_scaffold(smiles_or_path, pdb_id):
    """
    Loads the ligand mol2 file directly (more reliable than relying on the
    saved graph, since we need the RDKit mol object itself for scaffolding).
    Returns the scaffold as a canonical SMILES string, or None if it fails.
    """
    mol2_path = os.path.join(REFINED_DIR, pdb_id, pdb_id + "_ligand.mol2")
    sdf_path = os.path.join(REFINED_DIR, pdb_id, pdb_id + "_ligand.sdf")

    mol = None
    if os.path.exists(mol2_path):
        mol = Chem.MolFromMol2File(mol2_path, sanitize=True)
    if mol is None and os.path.exists(sdf_path):
        supplier = Chem.SDMolSupplier(sdf_path, sanitize=True)
        if len(supplier) > 0:
            mol = supplier[0]

    if mol is None:
        return None

    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_smiles = Chem.MolToSmiles(scaffold)
        return scaffold_smiles
    except Exception:
        return None


def list_refined_graph_ids():
    files = os.listdir(GRAPHS_DIR)
    ids = []
    index = 0
    while index < len(files):
        filename = files[index]
        if filename.endswith(".pt"):
            pdb_id = filename.replace(".pt", "")
            graph_path = os.path.join(GRAPHS_DIR, filename)
            data = torch.load(graph_path)
            if data["source"] == "refined":
                ids.append(pdb_id)
        index = index + 1
    return ids


def compute_scaffolds(pdb_ids):
    rows = []
    index = 0
    total = len(pdb_ids)
    while index < total:
        pdb_id = pdb_ids[index]
        scaffold = get_scaffold(None, pdb_id)
        if scaffold is None:
            scaffold = "NO_SCAFFOLD_" + pdb_id  # treat as its own unique group
        rows.append({"pdb_id": pdb_id, "scaffold": scaffold})
        index = index + 1
        if index % 500 == 0:
            print("Scaffolds computed:", index, "/", total)
    return pd.DataFrame(rows)


def scaffold_split(scaffold_df):
    """
    Groups pdb_ids by scaffold, then assigns whole scaffold groups to
    train/val until the target fraction is roughly reached.
    Larger scaffold groups are assigned first so the split lands close
    to the target percentages.
    """
    grouped = scaffold_df.groupby("scaffold")["pdb_id"].apply(list).reset_index()
    grouped["group_size"] = grouped["pdb_id"].apply(len)
    grouped = grouped.sort_values("group_size", ascending=False).reset_index(drop=True)

    total_complexes = scaffold_df.shape[0]
    train_target = int(total_complexes * TRAIN_FRACTION)

    train_ids = []
    val_ids = []

    row_index = 0
    num_groups = len(grouped)
    while row_index < num_groups:
        group_ids = grouped.iloc[row_index]["pdb_id"]
        if len(train_ids) < train_target:
            train_ids.extend(group_ids)
        else:
            val_ids.extend(group_ids)
        row_index = row_index + 1

    return train_ids, val_ids


def list_core_graph_ids():
    files = os.listdir(GRAPHS_DIR)
    ids = []
    index = 0
    while index < len(files):
        filename = files[index]
        if filename.endswith(".pt"):
            pdb_id = filename.replace(".pt", "")
            graph_path = os.path.join(GRAPHS_DIR, filename)
            data = torch.load(graph_path)
            if data["source"] == "core":
                ids.append(pdb_id)
        index = index + 1
    return ids


if __name__ == "__main__":
    print("Listing refined-set graph files...")
    refined_ids = list_refined_graph_ids()
    print("Refined complexes found:", len(refined_ids))

    print("\nComputing Murcko scaffolds...")
    scaffold_df = compute_scaffolds(refined_ids)

    num_unique_scaffolds = scaffold_df["scaffold"].nunique()
    print("Unique scaffolds:", num_unique_scaffolds, "across", len(scaffold_df), "complexes")

    print("\nSplitting by scaffold group...")
    train_ids, val_ids = scaffold_split(scaffold_df)

    print("Train complexes:", len(train_ids))
    print("Val complexes:", len(val_ids))

    test_ids = list_core_graph_ids()
    print("Test complexes (core set):", len(test_ids))

    # sanity check: confirm no scaffold appears in both train and val
    train_scaffolds = set(scaffold_df[scaffold_df["pdb_id"].isin(train_ids)]["scaffold"])
    val_scaffolds = set(scaffold_df[scaffold_df["pdb_id"].isin(val_ids)]["scaffold"])
    overlap = train_scaffolds.intersection(val_scaffolds)
    print("\nScaffold overlap between train and val:", len(overlap), "(should be 0)")

    pd.DataFrame({"pdb_id": train_ids}).to_csv(OUTPUT_TRAIN_CSV, index=False)
    pd.DataFrame({"pdb_id": val_ids}).to_csv(OUTPUT_VAL_CSV, index=False)
    pd.DataFrame({"pdb_id": test_ids}).to_csv(OUTPUT_TEST_CSV, index=False)

    print("\nSaved:", OUTPUT_TRAIN_CSV, OUTPUT_VAL_CSV, OUTPUT_TEST_CSV)
