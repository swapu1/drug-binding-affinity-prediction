"""
Phase 7: UI - Streamlit App (Neon Cyberpunk Design Version)
Features: Introduction Page, Single Prediction, and Batch Virtual Screening.
Fixes: HTML rendering sequence for UI cards and RDKit Windows file lock.

Run with: streamlit run app.py
"""

import os
import io
import pandas as pd
import numpy as np
import torch
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from rdkit import Chem
from rdkit.Chem import QED, RDConfig, Descriptors
from rdkit.Chem import FilterCatalog
from Bio.PDB import PDBParser
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from phase4_model import BindingAffinityModel

# --- TOXICITY ALERT CATALOG (PAINS + BRENK structural filters) ---
_filter_params = FilterCatalog.FilterCatalogParams()
_filter_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
_filter_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
TOXICITY_CATALOG = FilterCatalog.FilterCatalog(_filter_params)


def check_toxicity_alerts(mol):
    """
    Rule-based structural alert check using RDKit's PAINS + BRENK filter
    catalogs (known problematic substructures from medicinal chemistry
    literature). This is NOT a trained ML toxicity model - it's a
    deterministic pattern match, not a probability estimate.
    """
    if mol is None:
        return "N/A"
    if TOXICITY_CATALOG.HasMatch(mol):
        match = TOXICITY_CATALOG.GetFirstMatch(mol)
        return "⚠️ " + match.GetDescription()
    return "✅ No alerts"

# --- CONSTANTS ---
GRAPHS_DIR = "graphs"
TEST_CSV = "split_test.csv"
CHECKPOINT_PATH = "best_model_final.pt"

POCKET_DISTANCE_CUTOFF = 8.0
AMINO_ACIDS = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
               "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
               "TYR", "VAL"]

# --- PAGE CONFIG ---
st.set_page_config(page_title="Binding Affinity Predictor", page_icon="🧬", layout="wide")

# --- CUSTOM CSS FOR CYBERPUNK NEON UI ---
st.markdown("""
    <style>
    /* Dark theme background */
    .stApp {
        background-color: #080A12;
        color: #E2E8F0;
        font-family: 'Inter', system-ui, sans-serif;
    }
    
    /* Neon Glow Cards */
    .neon-card-cyan {
        background: rgba(18, 22, 40, 0.75);
        border: 1px solid #00f2fe;
        box-shadow: 0 0 15px rgba(0, 242, 254, 0.25), inset 0 0 15px rgba(0, 242, 254, 0.1);
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
    }
    
    .neon-card-magenta {
        background: rgba(18, 22, 40, 0.75);
        border: 1px solid #e040fb;
        box-shadow: 0 0 15px rgba(224, 64, 251, 0.25), inset 0 0 15px rgba(224, 64, 251, 0.1);
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
    }
    
    /* Typography */
    .neon-title-cyan {
        color: #00f2fe;
        font-size: 2rem;
        font-weight: 800;
        text-shadow: 0 0 10px rgba(0, 242, 254, 0.5);
        margin-bottom: 10px;
    }
    
    .neon-stat-number {
        font-size: 3.5rem;
        font-weight: 900;
        color: #FFFFFF;
        text-shadow: 0 0 15px #00f2fe, 0 0 30px #00f2fe;
        text-align: center;
        margin: 5px 0;
    }

    .neon-subtitle {
        color: #94A3B8;
        font-size: 0.95rem;
        text-align: center;
    }

    /* Primary Gradient Button */
    .stButton > button {
        background: linear-gradient(135deg, #00f2fe 0%, #a855f7 100%) !important;
        color: #000000 !important;
        font-weight: 800 !important;
        border: none !important;
        border-radius: 30px !important;
        padding: 12px 28px !important;
        box-shadow: 0 0 20px rgba(0, 242, 254, 0.4) !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 0 30px rgba(224, 64, 251, 0.6) !important;
    }

    /* Table styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(0, 242, 254, 0.2);
    }
    </style>
""", unsafe_allow_html=True)

# --- BACKEND MODEL & GRAPH FUNCTIONS ---
@st.cache_resource(show_spinner=False)
def load_model():
    model = BindingAffinityModel()
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
    model.eval()
    return model

def get_atom_features(atom):
    atomic_num = atom.GetAtomicNum()
    degree = atom.GetDegree()
    formal_charge = atom.GetFormalCharge()
    is_aromatic = 1 if atom.GetIsAromatic() else 0
    hybridization = int(atom.GetHybridization())
    return [atomic_num, degree, formal_charge, is_aromatic, hybridization]

def build_molecule_graph_from_bytes(file_bytes, file_format):
    suffix = ".mol2" if file_format == "mol2" else ".sdf"
    temp_path = "temp_ligand" + suffix
    with open(temp_path, "wb") as f:
        f.write(file_bytes)

    mol = None
    if file_format == "mol2":
        mol = Chem.MolFromMol2File(temp_path, sanitize=True)
        if mol is None:
            # retry without strict sanitization, then sanitize with error tolerance
            mol = Chem.MolFromMol2File(temp_path, sanitize=False)
            if mol is not None:
                try:
                    Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
                except Exception:
                    mol = None
    else:
        supplier = Chem.SDMolSupplier(temp_path, sanitize=True)
        if len(supplier) > 0:
            mol = supplier[0]
        if mol is None:
            # retry without strict sanitization, then sanitize with error tolerance
            supplier_lenient = Chem.SDMolSupplier(temp_path, sanitize=False)
            if len(supplier_lenient) > 0:
                candidate = supplier_lenient[0]
                if candidate is not None:
                    try:
                        Chem.SanitizeMol(candidate, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
                        mol = candidate
                    except Exception:
                        mol = None
            del supplier_lenient
        del supplier  # FIX: Windows file lock

    os.remove(temp_path)

    if mol is None:
        return None, None

    node_features = []
    atom_index = 0
    while atom_index < mol.GetNumAtoms():
        node_features.append(get_atom_features(mol.GetAtomWithIdx(atom_index)))
        atom_index = atom_index + 1

    edge_index_list = []
    bond_index = 0
    while bond_index < mol.GetNumBonds():
        bond = mol.GetBondWithIdx(bond_index)
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        edge_index_list.append([start, end])
        edge_index_list.append([end, start])
        bond_index = bond_index + 1

    x = torch.tensor(node_features, dtype=torch.float)
    if len(edge_index_list) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()

    return (x, edge_index), mol

def get_residue_features(residue):
    resname = residue.get_resname()
    one_hot = [0] * len(AMINO_ACIDS)
    if resname in AMINO_ACIDS:
        one_hot[AMINO_ACIDS.index(resname)] = 1
    return one_hot

def build_pocket_graph_from_bytes(file_bytes):
    temp_path = "temp_pocket.pdb"
    with open(temp_path, "wb") as f:
        f.write(file_bytes)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pocket", temp_path)
    os.remove(temp_path)

    residues = []
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    residues.append(residue)
                    coords.append(residue["CA"].get_coord())

    if len(residues) == 0:
        return None

    node_features = [get_residue_features(r) for r in residues]

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

    return x, edge_index

def predict(model, mol_x, mol_edge_index, pocket_x, pocket_edge_index):
    with torch.no_grad():
        prediction = model(mol_x, mol_edge_index, pocket_x, pocket_edge_index)
    return prediction.item()

# --- PLOTLY DONUT GAUGE CHART HELPER ---
def create_neon_donut(value, title, color):
    fig = go.Figure(go.Pie(
        values=[value, max(0.01, 1 - value if value <= 1 else 3 - value)],
        hole=0.75,
        marker=dict(colors=[color, 'rgba(255, 255, 255, 0.05)']),
        textinfo='none',
        hoverinfo='none'
    ))
    fig.update_layout(
        showlegend=False,
        margin=dict(l=10, r=10, t=25, b=10),
        height=140,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        annotations=[dict(
            text=f"<b>{title}:</b><br><span style='font-size:16px; color:#fff;'>{value:.2f}</span>",
            x=0.5, y=0.5, font=dict(size=11, color="#A0AEC0"),
            showarrow=False
        )]
    )
    return fig

# --- APP INITIALIZATION ---
model = load_model()

if "history" not in st.session_state:
    st.session_state.history = []

if "use_example_data" not in st.session_state:
    st.session_state.use_example_data = False

# --- SIDEBAR NAVIGATION ---
st.sidebar.markdown("<h2 style='color:#00f2fe; text-align:center;'>🧬 BindAI</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color:#94A3B8; text-align:center; font-size:0.8rem;'>Binding Affinity Predictor</p>", unsafe_allow_html=True)
st.sidebar.divider()
page = st.sidebar.radio("Navigation", ["Introduction", "Single Prediction", "Batch Virtual Screening"])

# ---------------- MODE 1: INTRODUCTION ----------------
if page == "Introduction":
    st.markdown("<h1 style='text-align: center; color: #FFF; font-weight: 800; font-size: 3.5rem;'>Drug-Target <span style='color: #00f2fe;'>Binding Affinity Predictor</span></h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94A3B8; font-size: 1.2rem; margin-bottom: 3rem;'>Dual-GNN model with cross-attention fusion, trained on the PDBBind refined set</p>", unsafe_allow_html=True)

    st.markdown('''
    <div class="neon-card-cyan">
        <h3 style="color: #00f2fe; margin-top:0;">🧬 Dual-GNN Architecture</h3>
        <p style="color: #E2E8F0; line-height: 1.6;">Two GNN branches (molecule + protein pocket) fused with cross-attention, predicting binding affinity (pKd) for candidate drug-protein pairs.</p>
    </div>
    ''', unsafe_allow_html=True)

    st.divider()
    st.markdown("### How to use this app:")
    st.markdown("""
    * **Single Prediction:** Select a known target-ligand pair from the held-out test set to see the model's prediction vs. the true measured value.
    * **Batch Virtual Screening:** Upload your own target protein (`.pdb`) and a library of candidate molecules (`.mol2` or `.sdf`) to rank them by predicted binding affinity and drug-likeness (QED).
    """)

    st.divider()
    st.markdown("### Model Performance on Held-Out Test Set (193 complexes)")

    predictions_path = "phase6_predictions.csv"
    if os.path.exists(predictions_path):
        pred_df = pd.read_csv(predictions_path)
        pred_df["abs_error"] = (pred_df["predicted_pKd"] - pred_df["true_pKd"]).abs()

        rmse = (((pred_df["predicted_pKd"] - pred_df["true_pKd"]) ** 2).mean()) ** 0.5
        mae = pred_df["abs_error"].mean()

        m1, m2, m3 = st.columns(3)
        m1.metric("Test RMSE", round(rmse, 3))
        m2.metric("Test MAE", round(mae, 3))
        m3.metric("Test Complexes", len(pred_df))

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown('''
            <div class="neon-card-cyan" style="padding: 10px; margin-bottom: 10px;">
                <div style="text-align:center; font-weight:700; color:#00f2fe;">Predicted vs True pKd</div>
                <p style="text-align:center; font-size:0.8rem; color:#94A3B8; margin:0;">Points closer to the diagonal line are more accurate</p>
            </div>
            ''', unsafe_allow_html=True)

            min_val = min(pred_df["true_pKd"].min(), pred_df["predicted_pKd"].min())
            max_val = max(pred_df["true_pKd"].max(), pred_df["predicted_pKd"].max())

            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=pred_df["true_pKd"], y=pred_df["predicted_pKd"],
                mode="markers",
                marker=dict(color="#00f2fe", size=7, opacity=0.7,
                            line=dict(color="#e040fb", width=1)),
                name="Test complexes"
            ))
            fig_scatter.add_trace(go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode="lines", line=dict(color="#94A3B8", dash="dash"),
                name="Perfect prediction"
            ))
            fig_scatter.update_layout(
                height=340, margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#E2E8F0"),
                xaxis=dict(title="True pKd", gridcolor="rgba(255,255,255,0.08)"),
                yaxis=dict(title="Predicted pKd", gridcolor="rgba(255,255,255,0.08)"),
                showlegend=False
            )
            st.plotly_chart(fig_scatter, use_container_width=True, config={'displayModeBar': False})

        with chart_col2:
            st.markdown('''
            <div class="neon-card-magenta" style="padding: 10px; margin-bottom: 10px;">
                <div style="text-align:center; font-weight:700; color:#e040fb;">Error Distribution</div>
                <p style="text-align:center; font-size:0.8rem; color:#94A3B8; margin:0;">How prediction errors are spread across the test set</p>
            </div>
            ''', unsafe_allow_html=True)

            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=pred_df["abs_error"],
                marker=dict(color="#e040fb", line=dict(color="#00f2fe", width=1)),
                nbinsx=25
            ))
            fig_hist.update_layout(
                height=340, margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#E2E8F0"),
                xaxis=dict(title="Absolute Error (pKd units)", gridcolor="rgba(255,255,255,0.08)"),
                yaxis=dict(title="Number of complexes", gridcolor="rgba(255,255,255,0.08)"),
                showlegend=False
            )
            st.plotly_chart(fig_hist, use_container_width=True, config={'displayModeBar': False})
    else:
        st.info("No phase6_predictions.csv found. Run phase6_evaluation.py first to generate test-set results.")


# ---------------- MODE 2: SINGLE PREDICTION ----------------
elif page == "Single Prediction":
    st.markdown("<h2 style='color:#FFF; font-weight:700;'>&lt; Single Prediction</h2>", unsafe_allow_html=True)
    
    # 1. Selection Banner (Fixed empty box issue)
    st.markdown('''
        <div class="neon-card-cyan" style="padding: 15px 20px;">
            <h4 style="color:#00f2fe; margin:0 0 5px 0;">Target Selection</h4>
            <p style="color:#E2E8F0; font-size:0.9rem; margin:0;">Select a benchmark PDB ID from the test set below. The model will predict its binding affinity.</p>
        </div>
    ''', unsafe_allow_html=True)
    
    col_sel, col_btn = st.columns([4, 1])
    try:
        test_ids = pd.read_csv(TEST_CSV)["pdb_id"].tolist()
    except Exception:
        test_ids = ["1a30", "1b3h", "1bxr", "1e66"]

    with col_sel:
        selected_id = st.selectbox("PDB ID", test_ids, label_visibility="collapsed")
    with col_btn:
        run_pred = st.button("Predict")

    # Perform Prediction
    graph_path = os.path.join(GRAPHS_DIR, selected_id + ".pt")
    if not os.path.exists(graph_path):
        st.error("No graph file found for " + selected_id + ". Check that Phase 2 built graphs/" + selected_id + ".pt")
        st.stop()

    data = torch.load(graph_path)
    pred_pKd = predict(model, data["mol_x"], data["mol_edge_index"], data["pocket_x"], data["pocket_edge_index"])
    true_pKd = float(data["label"])
    abs_err = abs(pred_pKd - true_pKd)

    # Update history
    if run_pred:
        st.session_state.history.insert(0, {"ligand": f"Complex - {selected_id}", "pKd": round(pred_pKd, 2)})

    st.write("<div style='height:20px;'></div>", unsafe_allow_html=True)

    # Main Dashboard Grid
    c1, c2, c3 = st.columns([1.2, 1, 1.4])
    
    with c1:
        st.markdown(f'''
            <div class="neon-card-magenta" style="height: 100%;">
                <div style="text-align:center; font-weight:700; color:#E2E8F0; font-size:1.1rem; margin-top:10px;">Predicted pKd Value</div>
                <div class="neon-stat-number">{pred_pKd:.2f}</div>
                <div class="neon-subtitle">True pKd: <b style="color:#00f2fe;">{true_pKd:.2f}</b></div>
            </div>
        ''', unsafe_allow_html=True)

    with c2:
        st.markdown('''
            <div class="neon-card-cyan" style="padding: 10px; margin-bottom: 5px;">
                <div style="font-weight:700; color:#E2E8F0; font-size:1.1rem; text-align:center;">Error for This Complex</div>
                <p style="text-align:center; font-size: 0.75rem; color:#94A3B8; margin:0;">Absolute difference, predicted vs true</p>
            </div>
        ''', unsafe_allow_html=True)
        st.plotly_chart(create_neon_donut(abs_err, "Abs. Error", "#00f2fe"), use_container_width=True, config={'displayModeBar': False})
        st.caption("This is a single data point, not the model's overall RMSE/MAE. See Phase 6 results for aggregate test-set metrics.")

    with c3:
        st.markdown('''
            <div class="neon-card-magenta" style="padding: 10px; margin-bottom: 5px;">
                <div style="font-weight:700; color:#E2E8F0; font-size:1.1rem; text-align:center;">Predicted vs True</div>
                <p style="text-align:center; font-size: 0.75rem; color:#94A3B8; margin:0;">Bar comparison for this complex</p>
            </div>
        ''', unsafe_allow_html=True)

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(x=["Predicted", "True"], y=[pred_pKd, true_pKd],
                                  marker_color=["#00f2fe", "#e040fb"]))
        fig_bar.update_layout(
            height=140, margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False, yaxis=dict(visible=False)
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

    # 4. Bottom Recent Predictions Section - only shows real predictions made this session
    if len(st.session_state.history) == 0:
        st.info("No predictions made yet this session. Click Predict above to see results here.")
    else:
        recent_html = '<div class="neon-card-cyan"><h4 style="color:#FFF; margin-top:0; margin-bottom:15px;">Recent Predictions (this session)</h4>'
        for item in st.session_state.history[:4]:
            recent_html += (
                '<div style="display:flex; justify-content:space-between; align-items:center; '
                'border-bottom:1px solid rgba(255,255,255,0.08); padding:10px 0;">'
                '<div>'
                '<div style="font-weight:600; color:#F1F5F9; font-size:1.05rem;">' + str(item['ligand']) + '</div>'
                '<div style="font-size:0.85rem; color:#64748B;">pKd: ' + str(item['pKd']) + '</div>'
                '</div>'
                '<div style="width:40px; height:6px; background:#00f2fe; border-radius:10px; box-shadow:0 0 8px #00f2fe;"></div>'
                '</div>'
            )
        recent_html += '</div>'
        st.markdown(recent_html, unsafe_allow_html=True)


# ---------------- MODE 3: BATCH VIRTUAL SCREENING ----------------
elif page == "Batch Virtual Screening":
    st.markdown("<h2 style='color:#FFF; font-weight:700;'>Batch Virtual Screening</h2>", unsafe_allow_html=True)
    
    col_left, col_right = st.columns([1.1, 1.9])
    
    with col_left:
        st.markdown('<h3 style="color:#FFF;">Batch Screening Setup</h3>', unsafe_allow_html=True)
        
        st.markdown('''
            <div class="neon-card-cyan" style="padding: 10px;">
                <div style="text-align:center; font-weight:700; color:#00f2fe; margin-bottom:5px;">Upload PDB Target File</div>
                <p style="text-align:center; font-size: 0.8rem; color:#94A3B8; margin-bottom: 10px;">Provide the 3D protein pocket structure.</p>
            </div>
        ''', unsafe_allow_html=True)
        pocket_file = st.file_uploader("PDB Target File", type=["pdb"], label_visibility="collapsed")
        if pocket_file is not None:
            st.session_state.use_example_data = False
        
        st.write("<div style='height:10px;'></div>", unsafe_allow_html=True)

        st.markdown('''
            <div class="neon-card-magenta" style="padding: 10px;">
                <div style="text-align:center; font-weight:700; color:#e040fb; margin-bottom:5px;">Upload Ligand Library</div>
                <p style="text-align:center; font-size: 0.8rem; color:#94A3B8; margin-bottom: 10px;">Provide candidate molecules (.mol2 or .sdf).</p>
            </div>
        ''', unsafe_allow_html=True)
        ligand_files = st.file_uploader("Ligand Library", type=["mol2", "sdf"], accept_multiple_files=True, label_visibility="collapsed")

        st.write("<div style='height:10px;'></div>", unsafe_allow_html=True)
        st.caption("Don't have your own files? Try a demo run instead:")
        load_example = st.button("Load Example Data", key="load_example_button")

        if load_example:
            st.session_state.use_example_data = True

        if st.session_state.get("use_example_data", False):
            st.info("Using bundled example files: one target pocket + the full core-set ligand library.")

        st.write("<div style='height:15px;'></div>", unsafe_allow_html=True)
        run_screening = st.button("Run Screening")
        st.caption("Structural Alert uses RDKit's PAINS + BRENK filters - known problematic "
                    "substructures from medicinal chemistry literature. This is a rule-based "
                    "pattern match, not a trained ML toxicity prediction.")

    with col_right:
        st.markdown('<h3 style="color:#FFF;">Screening Results - Top Hits</h3>', unsafe_allow_html=True)
        
        use_example = st.session_state.get("use_example_data", False)

        should_run_screening = run_screening or load_example

        if should_run_screening:
            # Build a unified list of (name, bytes) for pocket and ligands,
            # sourced either from user uploads or the bundled examples/ folder.
            if use_example:
                example_pocket_path = "examples/example_pocket.pdb"
                example_ligands_dir = "examples/ligands"

                if not os.path.exists(example_pocket_path):
                    st.error("Example files not found in the repo. Add them under examples/ and redeploy.")
                    pocket_bytes = None
                    ligand_items = []
                elif not os.path.isdir(example_ligands_dir):
                    st.error("examples/ligands/ folder not found in the repo.")
                    pocket_bytes = None
                    ligand_items = []
                else:
                    with open(example_pocket_path, "rb") as f:
                        pocket_bytes = f.read()
                    ligand_items = []
                    ligand_filenames = sorted(os.listdir(example_ligands_dir))
                    for filename in ligand_filenames:
                        if filename.endswith(".mol2") or filename.endswith(".sdf"):
                            full_path = os.path.join(example_ligands_dir, filename)
                            with open(full_path, "rb") as f:
                                ligand_items.append((filename, f.read()))
                    st.caption("Demo mode: screening the full core-set library (" + str(len(ligand_items)) + " ligands) against the bundled target pocket (**1a30**, from the PDBBind CASF benchmark). This may take a minute or two.")
            else:
                if pocket_file is None or len(ligand_files) == 0:
                    st.warning("Please upload both a target pocket PDB and candidate ligands, or click Load Example Data.")
                    pocket_bytes = None
                    ligand_items = []
                else:
                    pocket_bytes = pocket_file.read()
                    ligand_items = [(lf.name, lf.read()) for lf in ligand_files]

            if pocket_bytes is None or len(ligand_items) == 0:
                pass  # warning/error already shown above
            else:
                with st.spinner("Processing GNN Graph Embeddings..."):
                    pocket_result = build_pocket_graph_from_bytes(pocket_bytes)
                    
                    if pocket_result is None:
                        st.error("Failed to parse protein pocket.")
                    else:
                        pocket_x, pocket_edge_index = pocket_result
                        results = []
                        
                        for idx, (ligand_name, ligand_bytes) in enumerate(ligand_items):
                            file_format = "mol2" if ligand_name.endswith(".mol2") else "sdf"
                            mol_res, mol_obj = build_molecule_graph_from_bytes(ligand_bytes, file_format)
                            
                            if mol_res is not None:
                                mol_x, mol_edge_index = mol_res
                                pred = predict(model, mol_x, mol_edge_index, pocket_x, pocket_edge_index)

                                # QED: a real RDKit metric (0-1) for drug-likeness based on
                                # known physicochemical properties. NOT a synthesizability prediction.
                                qed_score = round(QED.qed(mol_obj), 2) if mol_obj else None

                                # Rule-based structural toxicity alert (PAINS + BRENK) - not a
                                # trained ML prediction, just a known-pattern match.
                                toxicity_flag = check_toxicity_alerts(mol_obj)

                                results.append({
                                    "Ligand ID": ligand_name,
                                    "Affinity (pKd)": round(pred, 2),
                                    "Drug-likeness (QED)": qed_score,
                                    "Structural Alert": toxicity_flag
                                })
                            else:
                                results.append({
                                    "Ligand ID": ligand_name,
                                    "Affinity (pKd)": None,
                                    "Drug-likeness (QED)": None,
                                    "Structural Alert": "N/A - parse failed"
                                })

                        results_df = pd.DataFrame(results)
                        results_df = results_df.sort_values("Affinity (pKd)", ascending=False).reset_index(drop=True)
                        results_df.index += 1
                        results_df = results_df.reset_index().rename(columns={"index": "Rank"})
                        
                        st.dataframe(
                            results_df,
                            column_config={
                                "Affinity (pKd)": st.column_config.ProgressColumn(
                                    "Affinity (pKd)",
                                    help="Binding strength",
                                    format="%.2f",
                                    min_value=0,
                                    max_value=12,
                                ),
                            },
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        csv_data = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button("Download CSV Results", csv_data, "virtual_screening_results.csv", "text/csv")
        else:
            st.info("Upload a protein pocket file and at least one candidate ligand, then click "
                    "'Run Screening' to see real ranked results here.")
