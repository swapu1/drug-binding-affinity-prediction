"""
Phase 2: Graph Construction
Builds a molecule graph (from ligand.mol2/.sdf) and a protein pocket graph
(from pocket.pdb) for every verified complex, attaches the pKd label,
and saves each pair as a PyTorch Geometric Data object.
"""

import os
import pandas as pd
import torch
from torch_geometric.data import Data
from rdkit import Chem
from Bio.PDB import PDBParser
import warnings
warnings.filterwarnings("ignore")

VERIFIED_CSV = "verified_complexes.csv"
LABELS_CSV = "data/LP_PDBBind.csv"
CORE_LABELS_CSV = "data/core_set/v2013-core/pdbbind_v2013_core.csv"
REFINED_DIR = "data/refined_set"
CORE_DIR = "data/core_set/v2013-core"
OUTPUT_DIR = "graphs"

POCKET_DISTANCE_CUTOFF = 8.0  # angstroms, distance between residues to draw an edge

AMINO_ACIDS = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
               "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
               "TYR", "VAL"]


def load_and_merge_labels():
    """
    Returns a DataFrame with columns: pdb_id, source, value (pKd label)
    Merged from verified_complexes.csv + the two label sources.
    """
    verified = pd.read_csv(VERIFIED_CSV)
    verified = verified[verified["valid"] == True]

    refined_labels = pd.read_csv(LABELS_CSV, index_col=0)
    refined_labels = refined_labels[refined_labels["category"] == "refined"]
    refined_labels = refined_labels.reset_index()
    refined_labels = refined_labels.rename(columns={"index": "pdb_id", "value": "label"})
    refined_labels = refined_labels[["pdb_id", "label"]]

    core_labels = pd.read_csv(CORE_LABELS_CSV)
    # NOTE: adjust these two column names below if your actual CSV headers differ
    core_labels = core_labels.rename(columns={core_labels.columns[0]: "pdb_id",
                                               core_labels.columns[1]: "label"})
    core_labels = core_labels[["pdb_id", "label"]]

    all_labels = pd.concat([refined_labels, core_labels], ignore_index=True)
    all_labels = all_labels.drop_duplicates(subset="pdb_id")

    merged = verified.merge(all_labels, on="pdb_id", how="inner")
    return merged


def get_atom_features(atom):
    atomic_num = atom.GetAtomicNum()
    degree = atom.GetDegree()
    formal_charge = atom.GetFormalCharge()
    is_aromatic = 1 if atom.GetIsAromatic() == True else 0
    hybridization = int(atom.GetHybridization())
    features = [atomic_num, degree, formal_charge, is_aromatic, hybridization]
    return features


def build_molecule_graph(ligand_path, ligand_format):
    if ligand_format == "mol2":
        mol = Chem.MolFromMol2File(ligand_path, sanitize=True)
    else:
        supplier = Chem.SDMolSupplier(ligand_path, sanitize=True)
        mol = supplier[0]

    if mol is None:
        return None

    node_features = []
    atom_index = 0
    num_atoms = mol.GetNumAtoms()
    while atom_index < num_atoms:
        atom = mol.GetAtomWithIdx(atom_index)
        node_features.append(get_atom_features(atom))
        atom_index = atom_index + 1

    edge_index_list = []
    edge_attr_list = []
    bond_index = 0
    num_bonds = mol.GetNumBonds()
    while bond_index < num_bonds:
        bond = mol.GetBondWithIdx(bond_index)
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        bond_type = float(bond.GetBondTypeAsDouble())

        edge_index_list.append([start, end])
        edge_attr_list.append([bond_type])
        edge_index_list.append([end, start])
        edge_attr_list.append([bond_type])

        bond_index = bond_index + 1

    x = torch.tensor(node_features, dtype=torch.float)

    if len(edge_index_list) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


def get_residue_ca_coord(residue):
    if "CA" in residue:
        atom = residue["CA"]
        return atom.get_coord()
    return None


def get_residue_features(residue):
    resname = residue.get_resname()
    one_hot = [0] * len(AMINO_ACIDS)
    if resname in AMINO_ACIDS:
        idx = AMINO_ACIDS.index(resname)
        one_hot[idx] = 1
    return one_hot


def build_pocket_graph(pocket_pdb_path, pdb_id):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, pocket_pdb_path)

    residues = []
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                coord = get_residue_ca_coord(residue)
                if coord is not None:
                    residues.append(residue)
                    coords.append(coord)

    if len(residues) == 0:
        return None

    node_features = []
    residue_index = 0
    while residue_index < len(residues):
        node_features.append(get_residue_features(residues[residue_index]))
        residue_index = residue_index + 1

    edge_index_list = []
    i = 0
    while i < len(coords):
        j = 0
        while j < len(coords):
            if i != j:
                dx = coords[i][0] - coords[j][0]
                dy = coords[i][1] - coords[j][1]
                dz = coords[i][2] - coords[j][2]
                distance = (dx * dx + dy * dy + dz * dz) ** 0.5
                if distance <= POCKET_DISTANCE_CUTOFF:
                    edge_index_list.append([i, j])
            j = j + 1
        i = i + 1

    x = torch.tensor(node_features, dtype=torch.float)

    if len(edge_index_list) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=edge_index)
    return data


def process_all_complexes(merged_df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    success_count = 0
    fail_count = 0
    fail_log = []

    row_index = 0
    total_rows = len(merged_df)
    while row_index < total_rows:
        row = merged_df.iloc[row_index]
        pdb_id = row["pdb_id"]
        source = row["source"]
        label = row["label"]

        if source == "refined":
            folder = os.path.join(REFINED_DIR, pdb_id)
        else:
            folder = os.path.join(CORE_DIR, pdb_id)

        pocket_path = os.path.join(folder, pdb_id + "_pocket.pdb")
        mol2_path = os.path.join(folder, pdb_id + "_ligand.mol2")
        sdf_path = os.path.join(folder, pdb_id + "_ligand.sdf")

        mol_graph = None
        if os.path.exists(mol2_path):
            mol_graph = build_molecule_graph(mol2_path, "mol2")
        if mol_graph is None and os.path.exists(sdf_path):
            mol_graph = build_molecule_graph(sdf_path, "sdf")

        if mol_graph is None:
            fail_count = fail_count + 1
            fail_log.append((pdb_id, "molecule graph failed"))
            row_index = row_index + 1
            continue

        pocket_graph = build_pocket_graph(pocket_path, pdb_id)

        if pocket_graph is None:
            fail_count = fail_count + 1
            fail_log.append((pdb_id, "pocket graph failed"))
            row_index = row_index + 1
            continue

        combined = {
            "pdb_id": pdb_id,
            "source": source,
            "label": float(label),
            "mol_x": mol_graph.x,
            "mol_edge_index": mol_graph.edge_index,
            "mol_edge_attr": mol_graph.edge_attr,
            "pocket_x": pocket_graph.x,
            "pocket_edge_index": pocket_graph.edge_index,
        }

        out_path = os.path.join(OUTPUT_DIR, pdb_id + ".pt")
        torch.save(combined, out_path)

        success_count = success_count + 1
        row_index = row_index + 1

        if row_index % 500 == 0:
            print("Processed", row_index, "/", total_rows)

    print("\nDone.")
    print("Success:", success_count)
    print("Failed:", fail_count)

    if len(fail_log) > 0:
        fail_df = pd.DataFrame(fail_log, columns=["pdb_id", "reason"])
        fail_df.to_csv("phase2_failures.csv", index=False)
        print("Failure details saved to phase2_failures.csv")


if __name__ == "__main__":
    merged = load_and_merge_labels()
    print("Merged complexes with labels:", len(merged))
    print(merged["source"].value_counts())

    process_all_complexes(merged)
