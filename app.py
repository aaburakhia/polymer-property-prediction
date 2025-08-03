# app.py 

import streamlit as st
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import networkx as nx

# --- Page Configuration ---
st.set_page_config(page_title="Polymer Property Predictor", page_icon="🧪", layout="wide")

# --- Feature Generation Function ---
def generate_single_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    descriptors = {}
    for name, func in Descriptors.descList:
        try: descriptors[name] = func(mol)
        except: descriptors[name] = None
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
    combined_features = {**descriptors, **maccs_fp, **morgan_fp, **graph_features}
    return pd.DataFrame([combined_features])

# --- Load Production Artifacts ---
@st.cache_resource
def load_artifacts():
    artifacts = {}
    TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    for target in TARGETS:
        artifacts[f'{target}_model'] = joblib.load(f'production_models/{target}_model.joblib')
        # --- THIS IS THE CHANGE ---
        artifacts[f'{target}_kept_columns'] = joblib.load(f'production_models/{target}_kept_columns.joblib')
    return artifacts

artifacts = load_artifacts()

# --- User Interface & Prediction Logic ---
st.title("🧪 Polymer Property Predictor")
st.write("A Materials Informatics application...") 
st.sidebar.header("Input Polymer Structure")
user_input = st.sidebar.text_area("Enter SMILES String:", "*CC(*)c1ccccc1", height=100)
if st.sidebar.button("Predict Properties", type="primary"):
    if not user_input:
        st.error("Please enter a SMILES string.")
    else:
        with st.spinner("Calculating..."):
            features_df = generate_single_features(user_input)
            if features_df is None:
                st.error("Invalid SMILES string.")
            else:
                st.header("Predicted Properties")
                cols = st.columns(5)
                for i, target in enumerate(['Tg', 'FFV', 'Tc', 'Density', 'Rg']):
                    model = artifacts[f'{target}_model']
                    kept_columns = artifacts[f'{target}_kept_columns']
                    
                    # --- THIS IS THE CRITICAL FIX ---
                    # Ensure all required columns exist, filling missing ones with 0
                    for col in kept_columns:
                        if col not in features_df.columns:
                            features_df[col] = 0
                    
                    # Filter the dataframe to have the exact columns in the exact order
                    X_selected = features_df[kept_columns]
                    # --- END OF FIX ---
                    
                    prediction = model.predict(X_selected)[0]
                    
                    with cols[i]:
                        unit = {"Tg": "°C", "Density": "g/cm³", "Rg": "Å"}.get(target, "")
                        st.metric(label=target, value=f"{prediction:.3f} {unit}")
                st.success("Prediction complete.")
else:
    st.info("Enter a SMILES string and click 'Predict'.")