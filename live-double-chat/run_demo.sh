#!/bin/bash

# Go to the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "Streamlit is not installed. Installing..."
    pip install streamlit pandas altair
fi

rm -rf live_venv
python -m venv live_venv
source live_venv/bin/activate

# Install other dependencies if needed
pip install openai

# Run the Streamlit app
echo "Starting the LLM Performance Comparison Demo..."
echo "Comparing Production Stack (localhost:30080) with Ray Serve (localhost:30081)"
echo "Ensure both endpoints are running before proceeding."
echo ""

# Run the app
streamlit run frontend.py