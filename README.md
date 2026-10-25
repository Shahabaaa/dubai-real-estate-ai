# Dubai Real Estate Intelligence Platform

Predicts fair value and rental yield for Dubai properties, explains the drivers with SHAP,
and answers investor questions through a RAG assistant.

## Status
- [ ] Phase 1: Data collection, cleaning, EDA
- [ ] Phase 2: Modeling and explainability
- [ ] Phase 3: RAG assistant and app
- [ ] Phase 4: Deployment, dashboard, documentation

## Setup
    python -m venv .venv
    source .venv/bin/activate   # Windows: .venv\Scripts\activate
    pip install -r requirements.txt

## Data
Download Dubai Land Department transactions (and rents if available) from Dubai Pulse
(https://www.dubaipulse.gov.ae) into data/raw/ as CSV. Check the dataset's license/terms.

## Run
    python src/clean.py
