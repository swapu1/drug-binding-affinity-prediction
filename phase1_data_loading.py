"""
Phase 1: Data Loading & Parsing Verification
PDBBind v2020 Refined Set (training) + CASF-2013 Core Set (final test)

Run this AFTER downloading:
1. HuggingFace refined set -> place folder as: data/refined_set/<PDB_ID>/
   (each subfolder should have protein.pdb, pocket.pdb, ligand.mol2, ligand.sdf)
2. Kaggle core set -> place folder as: data/core_set/v2013-core/<PDB_ID>/
"""

import os
import pandas as pd
from rdkit import Chem
from Bio.PDB import PDBParser
import warnings
warnings.filterwarnings("ignore")

REFINED_DIR = "data/refined_set"
CORE_DIR = "data/core_set/v2013-core"
OUTPUT_CSV = "verified_complexes.csv"


def download_refined_set():
    from datasets import load_dataset
    print("Downloading refined set from HuggingFace...")
    dataset = load_dataset("photonmz/pdbbindpp-2020")
    print("Done. Check the HF cache path if REFINED_DIR is not auto-populated.")
    print(dataset)


def find_complex_folders(root_dir):
    if not os.path.isdir(root_dir):
        print("Missing folder:", root_dir)
        return []
    folders = []
    entry_list = os.listdir(root_dir)
    index = 0
    while index < len(entry_list):
        entry = entry_list[index]
        full_path = os.path.join(root_dir, entry)
        if os.path.isdir(full_path):
            folders.append(full_path)
        index = index + 1
    return folders


def verify_complex(folder_path):
    """
    Checks that <id>_protein.pdb, <id>_pocket.pdb, <id>_ligand.mol2 (or .sdf)
    all exist and parse without error. Returns a dict with status info.
    Filenames are prefixed with the PDB ID, e.g. 1a30_pocket.pdb
    """
    pdb_id = os.path.basename(folder_path)

    protein_path = os.path.join(folder_path, pdb_id + "_protein.pdb")
    pocket_path = os.path.join(folder_path, pdb_id + "_pocket.pdb")
    mol2_path = os.path.join(folder_path, pdb_id + "_ligand.mol2")
    sdf_path = os.path.join(folder_path, pdb_id + "_ligand.sdf")

    result = {}
    result["pdb_id"] = pdb_id
    result["protein_path"] = protein_path
    result["pocket_path"] = pocket_path
    result["ligand_path"] = ""
    result["ligand_format"] = ""
    result["valid"] = False
    result["error"] = ""

    # --- check pocket file exists and parses (used for protein graph) ---
    if not os.path.exists(pocket_path):
        result["error"] = "missing " + pdb_id + "_pocket.pdb"
        return result

    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_id, pocket_path)
        residue_count = 0
        for model in structure:
            for chain in model:
                for residue in chain:
                    residue_count = residue_count + 1
        if residue_count == 0:
            result["error"] = "pocket.pdb has zero residues"
            return result
    except Exception as e:
        result["error"] = "pocket.pdb parse failed: " + str(e)
        return result

    # --- check ligand file exists and parses (prefer mol2, fallback sdf) ---
    mol = None
    if os.path.exists(mol2_path):
        mol = Chem.MolFromMol2File(mol2_path, sanitize=True)
        if mol is not None:
            result["ligand_path"] = mol2_path
            result["ligand_format"] = "mol2"

    if mol is None and os.path.exists(sdf_path):
        supplier = Chem.SDMolSupplier(sdf_path, sanitize=True)
        if len(supplier) > 0:
            mol = supplier[0]
            if mol is not None:
                result["ligand_path"] = sdf_path
                result["ligand_format"] = "sdf"

    if mol is None:
        result["error"] = pdb_id + "_ligand.mol2/.sdf missing or failed to parse"
        return result

    if mol.GetNumAtoms() == 0:
        result["error"] = "ligand has zero atoms"
        return result

    result["valid"] = True
    return result


def load_affinity_labels(csv_path):
    """
    Expects a CSV with at minimum a PDB ID column and a pKd column.
    Adjust column names below to match your actual CSV headers.
    """
    df = pd.read_csv(csv_path)
    return df


def run_verification(root_dir, source_label):
    folders = find_complex_folders(root_dir)
    print("Found", len(folders), "complex folders in", root_dir)

    rows = []
    index = 0
    valid_count = 0
    while index < len(folders):
        folder = folders[index]
        result = verify_complex(folder)
        result["source"] = source_label
        rows.append(result)
        if result["valid"] == True:
            valid_count = valid_count + 1
        index = index + 1
        if index % 500 == 0:
            print("Checked", index, "/", len(folders))

    print(source_label, "-> valid:", valid_count, "/", len(folders))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    all_results = []

    refined_results = run_verification(REFINED_DIR, "refined")
    all_results.append(refined_results)

    core_results = run_verification(CORE_DIR, "core")
    all_results.append(core_results)

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(OUTPUT_CSV, index=False)

    print("\nSummary:")
    print(combined.groupby(["source", "valid"]).size())
    print("\nSaved verification results to", OUTPUT_CSV)
    print("Next step: merge OUTPUT_CSV['pdb_id'] with your pKd label CSVs on pdb_id,")
    print("drop any rows where valid == False, then proceed to Phase 2 (graph construction).")
