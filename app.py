# =============================================================================
# app.py
# This version includes a significantly improved UI/UX
# =============================================================================

import streamlit as st
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import networkx as nx

# --- Page Configuration ---
st.set_page_config(
    page_title="Polymer Property Predictor",
    page_icon="🧪",
    layout="wide",
)

# --- Feature Generation & Artifact Loading (Cached for performance) ---
# (Functions are unchanged)
@st.cache_resource
def load_artifacts():
    artifacts = {}
    TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    for target in TARGETS:
        artifacts[f'{target}_model'] = joblib.load(f'production_models/{target}_model.joblib')
        artifacts[f'{target}_kept_columns'] = joblib.load(f'production_models/{target}_kept_columns.joblib')
    return artifacts

def generate_single_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    descriptors = {name: func(mol) for name, func in Descriptors.descList}
    maccs_fp = {f'maccs_{i}': bit for i, bit in enumerate(MACCSkeys.GenMACCSKeys(mol))}
    morgan_gen = GetMorganGenerator(radius=2, fpSize=128)
    morgan_fp = {f'morgan_{i}': bit for i, bit in enumerate(morgan_gen.GetFingerprint(mol))}
    graph_features = {}
    try:
        adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
        G = nx.from_numpy_array(adj)
        if nx.is_connected(G):
            graph_features['graph_diameter'] = nx.diameter(G)
            graph_features['avg_shortest_path'] = nx.average_shortest_path_length(G)
        else:
            graph_features['graph_diameter'], graph_features['avg_shortest_path'] = 0, 0
        graph_features['num_cycles'] = len(list(nx.cycle_basis(G)))
    except:
        graph_features['graph_diameter'], graph_features['avg_shortest_path'], graph_features['num_cycles'] = 0, 0, 0
    return pd.DataFrame([{**descriptors, **maccs_fp, **morgan_fp, **graph_features}])

artifacts = load_artifacts()

# =============================================================================
# User Interface (UI)
# =============================================================================

st.title("🧪 Polymer Property Predictor")
st.write(
    "A Materials Informatics application to predict key physical properties of polymers from their chemical structure. "
    "This tool is powered by a suite of XGBoost models trained on data from the **NeurIPS 2025 Open Polymer Prediction challenge**."
)

# --- Sidebar for Inputs and Information ---
st.sidebar.header("Input Controls")

# Create a dictionary of examples
polymer_examples = {
    "Select an example...": "",
    "Polystyrene (from competition)": "*CC(*)c1ccccc1",
    "Polycarbonate (for validation)": "*OC(C)(C)c1ccccc1OC(=O)*",
    "PET (for validation)": "*OCC(=O)c1ccccc1C(=O)O*",
    "PMMA (for validation)": "*CC(C)(C(=O)OC)*",
    "Polypropylene (for validation)": "*C(C)C*",
    "Nylon 6 (for validation)": "*CCCCCC(=O)N*"
}

# Dropdown to select an example
selected_example = st.sidebar.selectbox("Choose an example polymer:", list(polymer_examples.keys()))

# Text area for SMILES input, populated by the dropdown
user_input = st.sidebar.text_area(
    "Or enter your own SMILES String:", 
    value=polymer_examples[selected_example], 
    height=100
)

predict_button = st.sidebar.button("Predict Properties", type="primary")

st.sidebar.markdown("---")
st.sidebar.header("Project Links")
st.sidebar.markdown(
    "[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?style=for-the-badge&logo=github)]"
    "(https://github.com/aaburakhia/polymer-property-prediction)"  # <-- IMPORTANT: Replace with your final GitHub URL
)
st.sidebar.markdown(
    "[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?style=for-the-badge&logo=linkedin)]"
    "(https://www.linkedin.com/in/your-profile-here/)"  # <-- IMPORTANT: Add your LinkedIn profile URL
)

# --- Main Panel for Results and Analysis ---

if predict_button:
    if not user_input:
        st.error("Please select an example or enter a SMILES string in the sidebar.")
    else:
        with st.spinner("Calculating molecular features and running models..."):
            features_df = generate_single_features(user_input)
            
            if features_df is None:
                st.error("Invalid SMILES string. Please check the input.")
            else:
                st.header("1. Predicted Properties")
                
                cols = st.columns(5)
                for i, target in enumerate(['Tg', 'FFV', 'Tc', 'Density', 'Rg']):
                    model = artifacts[f'{target}_model']
                    kept_columns = artifacts[f'{target}_kept_columns']
                    
                    for col in kept_columns:
                        if col not in features_df.columns:
                            features_df[col] = 0
                    X_selected = features_df[kept_columns]
                    
                    prediction = model.predict(X_selected)[0]
                    
                    with cols[i]:
                        unit = {"Tg": "°C", "Density": "g/cm³", "Rg": "Å"}.get(target, "")
                        st.metric(label=target, value=f"{prediction:.3f} {unit}")
                
                st.success("Prediction complete.")

                # --- Analysis and Limitations Section ---
                st.header("2. Model Analysis & Scientific Context")
                st.info(
                    "**This model is a specialist trained on the NeurIPS 2025 competition dataset.**\n\n"
                    "Its performance is highest on complex, research-grade polymers similar to those in its training data. "
                    "Through validation, we've identified that the model's predictions for common commodity polymers (like PMMA or PET) can be inaccurate. "
                    "This is a classic example of **dataset bias**, where the model does not generalize perfectly to out-of-distribution data. "
                    "This discovery is a key finding of the project and highlights that the next research step is to fine-tune the model with a more diverse dataset."
                )

else:
    st.info("Select an example or enter a SMILES string in the sidebar and click 'Predict' to see the results.")