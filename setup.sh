pip install --upgrade pip

#!/bin/bash

# Ensure Streamlit uses correct port
echo "Running Streamlit..."
streamlit run app.py --server.port=$PORT --server.enableCORS=false
