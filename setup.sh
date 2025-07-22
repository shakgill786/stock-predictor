#!/usr/bin/env bash
set -euo pipefail

# upgrade pip & install dependencies (should only be used locally—
# on Render the deps are installed in buildCommand)
pip install --upgrade pip
pip install -r requirements.txt

# hand off to Streamlit
exec streamlit run app.py \
  --server.port=$PORT \
  --server.enableXsrfProtection=false
