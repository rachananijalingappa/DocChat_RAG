#!/bin/bash

# Start FastAPI in the background
uvicorn api:app --host 0.0.0.0 --port 8000 &

# Wait a few seconds for the API to boot
sleep 5

# Start Streamlit on the default Hugging Face Spaces port (7860)
# We also pass the API URL so Streamlit knows where to send requests
export API_URL="http://localhost:8000"
streamlit run app.py --server.port 7860 --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false
