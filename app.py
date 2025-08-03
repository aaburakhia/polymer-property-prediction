# app.py - The main code for the Streamlit application

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import networkx as nx

# =============================================================================
# App Configuration
# =============================================================================
st.set_page_config(
    page_title="Polymer Property Predictor",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Helper Functions (Must match the notebook)
# =============================================================================
# This is our feature generation function, adapted to work for a single SMILES string.
def generate_single_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Calculate features
    descriptors = Descriptors.CalcMolDescriptors(mol)
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
            graph_features['graph_diameter'] = 0
            graph_features['avg_shortest_path'] = 0
        graph_features['num_cycles'] = len(list(nx.cycle_basis(G)))
    except:
        graph_features['graph_diameter'] = 0
        graph_features['avg_shortest_path'] = 0
        graph_features['num_cycles'] = 0

    combined_features = {**descriptors, **maccs_fp, **morgan_fp, **graph_features}
    return pd.DataFrame([combined_features])

# =============================================================================
# Load Artifacts (Models, Selectors, Filters)
# =============================================================================
# Use Streamlit's caching to load these objects only once, improving performance.
@st.cache_resource
def load_artifacts():
    artifacts = {}
    TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    # Load the models and selectors
    for target in TARGETS:
        artifacts[f'{target}_model'] = joblib.load(f'production_models/{target}_model.joblib')
        artifacts[f'{target}_selector'] = joblib.load(f'production_models/{target}_selector.joblib')

    # IMPORTANT: The 'filters' dictionary must be copied EXACTLY from your training notebook.
    required_descriptors = {
        'MolWt', 'MolLogP', 'TPSA', 'NumRotatableBonds', 'HeavyAtomCount',
        'graph_diameter', 'num_cycles', 'avg_shortest_path'
    }
    artifacts['filters'] = {
        'Tg': list(set(['BalabanJ','BertzCT',...]).union(required_descriptors)), # Copy the full list from your notebook
        'FFV': list(set(['AvgIpc','BalabanJ',...]).union(required_descriptors)), # Copy the full list from your notebook
        'Tc': list(set(['BalabanJ','BertzCT',...]).union(required_descriptors)), # Copy the full list from your notebook
        'Density': list(set(['BalabanJ','Chi3n',...]).union(required_descriptors)), # Copy the full list from your notebook
        'Rg': list(set(['AvgIpc','Chi0n',...]).union(required_descriptors)) # Copy the full list from your notebook
    }
    # You will need to copy the full lists for the filters above from your notebook
    
    # This is a placeholder for the full list of columns from training
    # to ensure the input to the model has the correct shape.
    # In a real deployment, you'd save this list as an artifact as well.
    # For now, we will handle it dynamically.
    
    return artifacts

artifacts = load_artifacts()

# =============================================================================
# User Interface
# =============================================================================
st.title("🧪 Polymer Property Predictor")
st.write(
    "A Materials Informatics application to predict key physical properties of polymers "
    "directly from their chemical structure (SMILES representation). This tool is powered by "
    "a suite of XGBoost models trained on data from the NeurIPS 2025 Open Polymer Prediction challenge."
)

# --- Sidebar for input and information ---
st.sidebar.header("Input Polymer Structure")
default_smiles = "*CC(*)c1ccccc1"  # Example: Polystyrene
user_input = st.sidebar.text_area("Enter SMILES String:", default_smiles, height=100)
predict_button = st.sidebar.button("Predict Properties", type="primary")

st.sidebar.markdown("---")
st.sidebar.info(
    "**Example SMILES:**\n\n"
    "- Poly(ethylene oxide): `*CCO*`\n"
    "- Poly(vinyl chloride): `*C(Cl)C*`\n"
    "- Poly(styrene): `*CC(*)c1ccccc1`"
)

# --- Main panel for displaying results ---
if predict_button:
    if not user_input:
        st.error("Please enter a SMILES string in the sidebar.")
    else:
        with st.spinner("Calculating molecular features and running models..."):
            # 1. Generate features for the user's input
            features_df = generate_single_features(user_input)

            if features_df is None:
                st.error("Invalid SMILES string. Please check the input and try again.")
            else:
                st.header("Predicted Properties")
                
                # Create 5 columns for the 5 results
                cols = st.columns(5)
                predictions = {}

                # 2. Loop through each target to make a prediction
                for i, target in enumerate(artifacts['filters'].keys()):
                    model = artifacts[f'{target}_model']
                    selector = artifacts[f'{target}_selector']
                    feature_list = artifacts['filters'][target]
                    
                    # Ensure the input dataframe has all the necessary columns
                    X_input = pd.DataFrame(columns=feature_list)
                    X_combined = pd.concat([X_input, features_df], ignore_index=True).fillna(0)
                    X_filtered = X_combined[feature_list]

                    # Apply the transformations
                    X_selected = selector.transform(X_filtered)
                    
                    # Make prediction
                    prediction = model.predict(X_selected)[0]
                    predictions[target] = prediction
                    
                    # Display in the corresponding column
                    with cols[i]:
                        st.metric(label=target, value=f"{prediction:.3f}")
                
                st.success("Prediction complete.")

else:
    st.info("Enter a SMILES string in the sidebar and click 'Predict Properties' to begin.")