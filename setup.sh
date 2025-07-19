#!/usr/bin/env bash
# upgrade pip first
pip install --upgrade pip

# install everything
pip install -r requirements.txt

# start Streamlit on the port Render assigns
streamlit run app.py \
  --server.port=$PORT \
  --server.enableXsrfProtection=false
