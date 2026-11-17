#!/usr/bin/env bash
# End-to-end reproduction: data -> training/ablation -> R QC/DE -> dashboard
set -euo pipefail
python -m src.prepare_data
python -m src.train_eval --epochs 20 --seeds 3
streamlit run app.py
