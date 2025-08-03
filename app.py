# app.py - FINAL CORRECTED VERSION

import streamlit as st
import pandas as pd
import numpy as np
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

# --- Feature Generation Function (Corrected) ---
def generate_single_features(smiles):
    """Generates a full feature set for a single SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # --- THIS IS THE CORRECTED SECTION ---
    # Reverted to the robust looping method from the original notebook.
    # This is more stable across different RDKit versions.
    descriptors = {}
    for name, func in Descriptors.descList:
        try:
            descriptors[name] = func(mol)
        except:
            descriptors[name] = None
    # --- END OF CORRECTION ---

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

# --- Load Production Artifacts ---
@st.cache_resource
def load_artifacts():
    """Loads all models, selectors, and feature lists from disk."""
    artifacts = {}
    TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

    for target in TARGETS:
        artifacts[f'{target}_model'] = joblib.load(f'production_models/{target}_model.joblib')
        artifacts[f'{target}_selector'] = joblib.load(f'production_models/{target}_selector.joblib')

    required_descriptors = {
        'MolWt', 'MolLogP', 'TPSA', 'NumRotatableBonds', 'HeavyAtomCount',
        'graph_diameter', 'num_cycles', 'avg_shortest_path'
    }
    artifacts['filters'] = {
        'Tg': list(set(['BalabanJ','BertzCT','Chi1','Chi3n','Chi4n','EState_VSA4','EState_VSA8','FpDensityMorgan3','HallKierAlpha','Kappa3','MaxAbsEStateIndex','MolLogP','NumAmideBonds','NumHeteroatoms','NumHeterocycles','NumRotatableBonds','PEOE_VSA14','Phi','RingCount','SMR_VSA1','SPS','SlogP_VSA1','SlogP_VSA5','SlogP_VSA8','TPSA','VSA_EState1','VSA_EState4','VSA_EState6','VSA_EState7','VSA_EState8','fr_C_O_noCOO','fr_NH1','fr_benzene','fr_bicyclic','fr_ether','fr_unbrch_alkane']).union(required_descriptors)),
        'FFV': list(set(['AvgIpc','BalabanJ','BertzCT','Chi0','Chi0n','Chi0v','Chi1','Chi1n','Chi1v','Chi2n','Chi2v','Chi3n','Chi3v','Chi4n','EState_VSA10','EState_VSA5','EState_VSA7','EState_VSA8','EState_VSA9','ExactMolWt','FpDensityMorgan1','FpDensityMorgan2','FpDensityMorgan3','FractionCSP3','HallKierAlpha','HeavyAtomMolWt','Kappa1','Kappa2','Kappa3','MaxAbsEStateIndex','MaxEStateIndex','MinEStateIndex','MolLogP','MolMR','MolWt','NHOHCount','NOCount','NumAromaticHeterocycles','NumHAcceptors','NumHDonors','NumHeterocycles','NumRotatableBonds','PEOE_VSA14','RingCount','SMR_VSA1','SMR_VSA10','SMR_VSA3','SMR_VSA5','SMR_VSA6','SMR_VSA7','SMR_VSA9','SPS','SlogP_VSA1','SlogP_VSA10','SlogP_VSA11','SlogP_VSA12','SlogP_VSA2','SlogP_VSA3','SlogP_VSA4','SlogP_VSA5','SlogP_VSA6','SlogP_VSA7','SlogP_VSA8','TPSA','VSA_EState1','VSA_EState10','VSA_EState2','VSA_EState3','VSA_EState4','VSA_EState5','VSA_EState6','VSA_EState7','VSA_EState8','VSA_EState9','fr_Ar_N','fr_C_O','fr_NH0','fr_NH1','fr_aniline','fr_ether','fr_halogen','fr_thiophene']).union(required_descriptors)),
        'Tc': list(set(['BalabanJ','BertzCT','Chi0','EState_VSA5','ExactMolWt','FpDensityMorgan1','FpDensityMorgan2','FpDensityMorgan3','HeavyAtomMolWt','MinEStateIndex','MolWt','NumAtomStereoCenters','NumRotatableBonds','NumValenceElectrons','SMR_VSA10','SMR_VSA7','SPS','SlogP_VSA6','SlogP_VSA8','VSA_EState1','VSA_EState7','fr_NH1','fr_ester','fr_halogen']).union(required_descriptors)),
        'Density': list(set(['BalabanJ','Chi3n','Chi3v','Chi4n','EState_VSA1','ExactMolWt','FractionCSP3','HallKierAlpha','Kappa2','MinEStateIndex','MolMR','MolWt','NumAliphaticCarbocycles','NumHAcceptors','NumHeteroatoms','NumRotatableBonds','SMR_VSA10','SMR_VSA5','SlogP_VSA12','SlogP_VSA5','TPSA','VSA_EState10','VSA_EState7','VSA_EState8']).union(required_descriptors)),
        'Rg': list(set(['AvgIpc','Chi0n','Chi1v','Chi2n','Chi3v','ExactMolWt','FpDensityMorgan1','FpDensityMorgan2','FpDensityMorgan3','HallKierAlpha','HeavyAtomMolWt','Kappa3','MaxAbsEStateIndex','MolWt','NOCount','NumRotatableBonds','NumUnspecifiedAtomStereoCenters','NumValenceElectrons','PEOE_VSA14','PEOE_VSA6','SMR_VSA1','SMR_VSA5','SPS','SlogP_VSA1','SlogP_VSA2','SlogP_VSA7','SlogP_VSA8','VSA_EState1','VSA_EState8','fr_alkyl_halide','fr_halogen']).union(required_descriptors))
    }
    return artifacts

artifacts = load_artifacts()

# --- User Interface & Prediction Logic ---
# (The rest of the file is unchanged)
st.title("🧪 Polymer Property Predictor")
st.write("A Materials Informatics application to predict key physical properties of polymers from their chemical structure. This tool is powered by a suite of XGBoost models trained on data from the NeurIPS 2025 Open Polymer Prediction challenge.")
st.sidebar.header("Input Polymer Structure")
default_smiles = "*CC(*)c1ccccc1"
user_input = st.sidebar.text_area("Enter SMILES String:", default_smiles, height=100)
predict_button = st.sidebar.button("Predict Properties", type="primary")
st.sidebar.markdown("---")
st.sidebar.info("**Example SMILES:**\n\n- Poly(ethylene oxide): `*CCO*`\n\n- Poly(vinyl chloride): `*C(Cl)C*`")
if predict_button:
    if not user_input:
        st.error("Please enter a SMILES string in the sidebar.")
    else:
        with st.spinner("Calculating molecular features and running models..."):
            features_df = generate_single_features(user_input)
            if features_df is None:
                st.error("Invalid SMILES string. Please check the input.")
            else:
                st.header("Predicted Properties")
                cols = st.columns(5)
                for i, target in enumerate(artifacts['filters'].keys()):
                    model = artifacts[f'{target}_model']
                    selector = artifacts[f'{target}_selector']
                    feature_list = artifacts['filters'][target]
                    X_input_template = pd.DataFrame(columns=feature_list)
                    X_combined = pd.concat([X_input_template, features_df]).fillna(0)
                    X_filtered = X_combined[feature_list]
                    X_selected = selector.transform(X_filtered)
                    prediction = model.predict(X_selected)[0]
                    with cols[i]:
                        unit = {"Tg": "°C", "FFV": "", "Tc": "", "Density": "g/cm³", "Rg": "Å"}.get(target, "")
                        st.metric(label=target, value=f"{prediction:.3f} {unit}")
                st.success("Prediction complete.")
else:
    st.info("Enter a SMILES string in the sidebar and click 'Predict' to see the results.")