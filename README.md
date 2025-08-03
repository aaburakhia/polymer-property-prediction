---
title: Polymer Property Predictor
emoji: 🧪
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# Polymer Property Predictor: A Materials Informatics Project

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/aaburakhia/polymer-property-prediction)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A machine learning application to predict key physical properties of polymers directly from their chemical structure, based on the NeurIPS 2025 Open Polymer Prediction challenge.

This project is the capstone piece of a portfolio designed for a Senior Materials & Project Engineer transitioning into R&D roles in Materials Informatics and as a Research Scientist.

## 🚀 Live Demo

You are viewing the live, interactive application. Enter a polymer's SMILES string in the sidebar to get started.

## 🎯 Project Overview

The goal of this project is to accelerate materials discovery by building a robust Quantitative Structure-Property Relationship (QSPR) model. The application takes a polymer's structure as a SMILES string and predicts five key physical properties:

*   **Tg:** Glass Transition Temperature
*   **FFV:** Fractional Free Volume
*   **Tc:** Thermal Decomposition Temperature
*   **Density:** Material Density
*   **Rg:** Radius of Gyration

This tool demonstrates an end-to-end MLOps pipeline, from raw data integration to a deployed, interactive web application.

## 🛠️ Methodology & Pipeline

The project was executed in a systematic, multi-stage process:

1.  **Data Integration:** The initial dataset was enriched by integrating data from 7 external sources. This involved robust data cleaning, SMILES canonicalization using RDKit, and a prioritized merging strategy to create a comprehensive training set.

2.  **Feature Engineering:** A rich feature set of over 500 molecular descriptors was generated for each polymer using the RDKit library. This included:
    *   **Physicochemical Descriptors:** (e.g., Molecular Weight, LogP, TPSA)
    *   **Structural Fingerprints:** (MACCS Keys, Morgan Fingerprints)
    *   **Graph-Based Features:** (e.g., Graph Diameter, Number of Cycles)

3.  **Model Training & Validation:**
    *   Five separate `XGBoost` models were trained, one for each target property.
    *   A robust 5-Fold Cross-Validation strategy was used to ensure the reliability of the performance metrics.
    *   The models were evaluated using the competition's specific **Weighted Mean Absolute Error (wMAE)**, resulting in a final OOF wMAE score of **0.212979**.

4.  **Deployment:**
    *   The five trained models and their corresponding feature selectors were saved as production artifacts.
    *   A user-friendly web application was built using Streamlit.
    *   The application was containerized using Docker and deployed on Hugging Face Spaces.

## 📂 Repository Structure

├── production_models/ # Saved .joblib model and selector artifacts

├── app.py # The main Streamlit application script

├── requirements.txt # Python dependencies for the application

└── Materials_Informatics_QSPR_Modeling.ipynb # The complete training notebook


## 🔧 How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone https://huggingface.co/spaces/aaburakhia/polymer-property-prediction
    cd polymer-property-prediction
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

---