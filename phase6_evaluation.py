"""
Phase 6: Evaluation
Loads the best trained checkpoint, runs it on the untouched core set
(split_test.csv), computes RMSE and Pearson correlation, and trains
a simple Random Forest baseline (fingerprints only, no protein structure)
for comparison.
"""

import os
import pandas as pd
import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def compute_rmse(true_values, predicted_values):
    mse = mean_squared_error(true_values, predicted_values)
    rmse = mse ** 0.5
    return rmse

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from phase4_model import BindingAffinityModel

GRAPHS_DIR = "graphs"
TEST_CSV = "split_test.csv"
TRAIN_CSV = "split_train.csv"
CHECKPOINT_PATH = "best_model_final.pt"

REFINED_DIR = "data/refined_set"
CORE_DIR = "data/core_set/v2013-core"


def load_split_ids(csv_path):
    df = pd.read_csv(csv_path)
    return df["pdb_id"].tolist()


def load_graph(pdb_id):
    path = os.path.join(GRAPHS_DIR, pdb_id + ".pt")
    return torch.load(path)


def evaluate_gnn_model(test_ids):
    model = BindingAffinityModel()
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    model.eval()

    predictions = []
    true_labels = []

    index = 0
    while index < len(test_ids):
        pdb_id = test_ids[index]
        data = load_graph(pdb_id)

        with torch.no_grad():
            prediction = model(
                data["mol_x"], data["mol_edge_index"],
                data["pocket_x"], data["pocket_edge_index"]
            )

        predictions.append(prediction.item())
        true_labels.append(data["label"])
        index = index + 1

    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    rmse = compute_rmse(true_labels, predictions)
    corr, _ = pearsonr(true_labels, predictions)

    return rmse, corr, predictions, true_labels


def get_fingerprint(pdb_id, source):
    if source == "refined":
        folder = os.path.join(REFINED_DIR, pdb_id)
    else:
        folder = os.path.join(CORE_DIR, pdb_id)

    mol2_path = os.path.join(folder, pdb_id + "_ligand.mol2")
    sdf_path = os.path.join(folder, pdb_id + "_ligand.sdf")

    mol = None
    if os.path.exists(mol2_path):
        mol = Chem.MolFromMol2File(mol2_path, sanitize=True)
    if mol is None and os.path.exists(sdf_path):
        supplier = Chem.SDMolSupplier(sdf_path, sanitize=True)
        if len(supplier) > 0:
            mol = supplier[0]

    if mol is None:
        return None

    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    array = np.zeros((1024,))
    index = 0
    while index < 1024:
        array[index] = fingerprint[index]
        index = index + 1
    return array


def build_fingerprint_dataset(pdb_ids, source):
    features = []
    labels = []
    valid_ids = []

    index = 0
    while index < len(pdb_ids):
        pdb_id = pdb_ids[index]
        fp = get_fingerprint(pdb_id, source)
        if fp is not None:
            graph_data = load_graph(pdb_id)
            features.append(fp)
            labels.append(graph_data["label"])
            valid_ids.append(pdb_id)
        index = index + 1

    return np.array(features), np.array(labels), valid_ids


def evaluate_baseline(train_ids, test_ids):
    print("\nBuilding fingerprint features for baseline...")
    train_X, train_y, _ = build_fingerprint_dataset(train_ids, "refined")
    test_X, test_y, _ = build_fingerprint_dataset(test_ids, "core")

    print("Training Random Forest baseline...")
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(train_X, train_y)

    predictions = rf.predict(test_X)

    rmse = compute_rmse(test_y, predictions)
    corr, _ = pearsonr(test_y, predictions)

    return rmse, corr


if __name__ == "__main__":
    test_ids = load_split_ids(TEST_CSV)
    train_ids = load_split_ids(TRAIN_CSV)
    print("Test complexes (core set):", len(test_ids))

    print("\n--- Evaluating trained GNN model ---")
    gnn_rmse, gnn_corr, preds, truths = evaluate_gnn_model(test_ids)
    print("GNN Model  -> RMSE:", round(gnn_rmse, 4), " Pearson r:", round(gnn_corr, 4))

    print("\n--- Evaluating Random Forest baseline (fingerprints only) ---")
    rf_rmse, rf_corr = evaluate_baseline(train_ids, test_ids)
    print("RF Baseline -> RMSE:", round(rf_rmse, 4), " Pearson r:", round(rf_corr, 4))

    print("\n=== Final Comparison ===")
    print("GNN (molecule + pocket structure):  RMSE =", round(gnn_rmse, 4), " r =", round(gnn_corr, 4))
    print("RF baseline (fingerprints only):    RMSE =", round(rf_rmse, 4), " r =", round(rf_corr, 4))

    results_df = pd.DataFrame({"pdb_id": test_ids, "true_pKd": truths, "predicted_pKd": preds})
    results_df.to_csv("phase6_predictions.csv", index=False)
    print("\nPer-complex predictions saved to phase6_predictions.csv")
