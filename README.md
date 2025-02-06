# Graph-Neural-Network-for-Molecule-Classification

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-app-url.com)
[![MLflow](https://img.shields.io/badge/mlflow-%23d9ead3.svg?logo=mlflow&logoColor=blue)](mlflow.db)

A GNN-based solution for identifying potential HIV inhibitors through molecular graph analysis.
## Project Overview
**Team**: Atallah Chaker, Boudali Riyad, Kebli Younes  
**Objective**: Develop a GNN classifier to predict molecules' HIV inhibition capability using structural analysis

## Key Features
- 🧬 SMILES-to-graph conversion with RDKit
- ⚛️ GATConv-based neural architecture with hierarchical pooling
- ⚖️ Handling class imbalance through oversampling
- 📊 MLflow experiment tracking
- 🚀 Streamlit deployment with molecular visualization

## Dataset
**HIV Inhibition Dataset**:
- 40,000 molecules (SMILES strings)
- Binary labels (1,400 inhibitors)
- Features:
  - Atomic properties (node features)
  - Bond types (edge features)
  - Chemical descriptors
## Installation

``` bash
git clone https://github.com/riyad-boudali/Graph-Neural-Network-for-Molecule-Classification.git
cd Graph-Neural-Network-for-Molecule-Classification
pip install -r requirements.txt 
```

## Dashboard (MLFlow + Streamlit)

You need to start the following things:

- Streamlit server
```bash
streamlit run dashboard.py
```
- MlFlow Server
```bash
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./artifacts \
    --host 0.0.0.0
    --port 5000
```
