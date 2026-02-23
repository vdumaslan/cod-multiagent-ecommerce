# CoD Multi-Agent E-Commerce

This repository contains the seller copilot implementation focused on production-ready data pipeline and model-ready datasets.

Primary workspace:
- `seller-copilot/`

Cloud scheduler:
- `.github/workflows/seller-copilot-pipeline.yml` (daily + manual trigger)

Quick start:
```bash
pip install -r seller-copilot/requirements.txt
python seller-copilot/src/pipelines/run_pipeline.py --config seller-copilot/config/pipeline.yaml
streamlit run seller-copilot/src/app/streamlit_app.py
```


