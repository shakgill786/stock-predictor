#!/usr/bin/env bash

# (Optional) upgrade pip
pip install --upgrade pip

echo "Launching Streamlit on port $PORT..."
streamlit run app.py \
  --server.port $PORT \
  --server.address 0.0.0.0 \
  --server.enableCORS=false
