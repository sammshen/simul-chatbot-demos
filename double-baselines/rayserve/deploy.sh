#!/bin/bash
# Example: bash deploy.sh A100 meta-llama/Llama-3.1-8B-Instruct 4 hf_cKu...

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# Ensure 4 arguments are provided
if [ $# -ne 4 ]; then
  echo "Usage: $0 <ACCELERATOR_TYPE> <MODEL_NAME> <NUM_REPLICAS> <HF_TOKEN>"
  exit 1
fi

rm -rf rayserve_venv
python -m venv rayserve_venv

source rayserve_venv/bin/activate

pip install -r ray-requirements.txt

# Clean up any previous ray deployments
ray stop --force
pkill -f ray

python ray-setup.py "$@"

echo "Rayserve deployment complete"
