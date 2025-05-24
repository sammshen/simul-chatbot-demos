#!/bin/bash
# Example: bash deploy.sh ray A100 meta-llama/Llama-3.1-8B-Instruct 4 hf_cKu...

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# Ensure 5 arguments are provided
if [ $# -ne 5 ]; then
  echo "Usage: $0 <SERVICE_NAME> <ACCELERATOR_TYPE> <MODEL_NAME> <NUM_REPLICAS> <HF_TOKEN>"
  exit 1
fi

SERVICE_NAME=$1
ACCELERATOR_TYPE=$2
MODEL_NAME=$3
NUM_REPLICAS=$4
HF_TOKEN=$5

# --- Step 1: Run Production Stack + LMCache ---
echo "Deploying Production Stack + LMCache on localhost:30080"
bash prodstack-lmcache/deploy.sh $NUM_REPLICAS $MODEL_NAME $HF_TOKEN

# --- Step 2: Run the Comparison Baseline ---
echo "Trying to deploy the comparison baseline on localhost:30081"
if [ "$SERVICE_NAME" == "ray" ]; then
  bash rayserve/deploy.sh $ACCELERATOR_TYPE $MODEL_NAME $NUM_REPLICAS $HF_TOKEN
else
  echo "Invalid service name: $SERVICE_NAME"
  echo "Currently supported services are: ray"
  exit 1
fi

