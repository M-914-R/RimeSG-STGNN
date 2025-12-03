📘 RimeSG-STGNN: Code and Experimental Resources
This repository provides the implementation and experimental resources for the RimeSG-STGNN framework, including data preprocessing, adjacency matrix construction, hyperparameter optimization, and four model configurations used in the experimental workflow.

🔧 Repository Structure
RimeSG-STGNN/ Full model implementation (RIME + SG + TimesNet + HGL)
SG-STGNN/ SG-enhanced spatiotemporal model
STGNN/ TimesNet + Hybrid Graph Learning
TimesNet/ Temporal baseline model
data/
raw/ Original high-frequency monitoring data
filled/ Data after missing-value imputation
adjacency/ Generated adjacency matrices
adjacency_matrix.py Script for Pearson-based correlation graph construction
missing_value.py Script for missing-value handling
rime_hpo.py RIME-based hyperparameter optimization module
README.md

📦 Included Components
1. Data Processing
Missing-value imputation (missing_value.py)
Correlation-based adjacency matrix generation (adjacency_matrix.py)
Final processed data and generated graphs are stored in data/.
2. Model Configurations
This repository includes four model configurations used throughout the experimental workflow:
TimesNet — Extracts dominant periodicity and temporal features.
STGNN (TimesNet + HGL) — Incorporates temporal modeling with hybrid graph learning.
SG-STGNN — Enhances temporal signals with SG smoothing.
RimeSG-STGNN — The complete framework integrating SG filtering, RIME optimization, multi-scale temporal modeling, and hybrid graph learning.
Each folder contains:
Model definition
Training script
Configuration files (if applicable)
3. Optimization Module
RIME Hyperparameter Optimization (rime_hpo.py)
Used for automatic tuning of SG parameters and key model hyperparameters.

📊 Data Description
The data/ directory includes:
Original high-frequency DO monitoring data
Missing-value–filled datasets
Pearson-correlation adjacency matrices for graph learning
All datasets follow the preprocessing pipeline used in the experimental study.
