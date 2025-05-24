#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# Create stats directories
STATS_DIR="../real_time_statistics"
CSV_DIR="${STATS_DIR}/csvs"
mkdir -p "$STATS_DIR" "$CSV_DIR"

rm -rf qa_venv
python3 -m venv qa_venv
source qa_venv/bin/activate

pip install -r requirements.txt

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <model> [qps_values...]"
    echo "Example: $0 meta-llama/Llama-3.1-8B-Instruct 0.1 0.2 0.3"
    exit 1
fi

MODEL=$1

# If QPS values are provided, use them; otherwise use default
if [[ $# -gt 1 ]]; then
    QPS_VALUES=("${@:2}")
else
    QPS_VALUES=(0.1)  # Default QPS value
fi

# CONFIGURATION
NUM_USERS_WARMUP="${NUM_USERS_WARMUP:-20}"
NUM_USERS="${NUM_USERS:-15}"
NUM_ROUNDS="${NUM_ROUNDS:-20}"

SYSTEM_PROMPT="${SYSTEM_PROMPT:-3000}" # Shared system prompt length
CHAT_HISTORY="${CHAT_HISTORY:-30000}" # User specific chat history length
ANSWER_LEN="${ANSWER_LEN:-100}" # Generation length per round

# init-user-id starts at 1, will add 20 each iteration
INIT_USER_ID="${INIT_USER_ID:-1}"

TEST_DURATION="${TEST_DURATION:-100}" # Duration of the test in seconds

warmup() {
    # Warm up the servers with a lot of user queries
    python3 "${SCRIPT_DIR}/multi-round-qa.py" \
        --num-users 1 \
        --num-rounds 2 \
        --qps 2 \
        --shared-system-prompt $SYSTEM_PROMPT \
        --user-history-prompt $CHAT_HISTORY \
        --answer-len $ANSWER_LEN \
        --model "$MODEL" \
        --base-url "http://localhost:30080" \
        --init-user-id "$INIT_USER_ID" \
        --output "warmup.csv" \
        --log-interval 30 \
        --request-with-user-id \
        --time $((NUM_USERS_WARMUP / 2))
}

run_benchmark() {
    # $1: qps
    # $2: output file prefix

    # We don't need separate warmup as it's built into the script for both endpoints
    # warmup

    echo "Running benchmark with QPS: $1"
    echo "Results will be saved to: $STATS_DIR/real_time_stats_$1.txt and CSV files in $CSV_DIR"

    python3 "${SCRIPT_DIR}/multi-round-qa.py" \
        --num-users $NUM_USERS \
        --num-rounds $NUM_ROUNDS \
        --qps "$1" \
        --shared-system-prompt "$SYSTEM_PROMPT" \
        --user-history-prompt "$CHAT_HISTORY" \
        --answer-len $ANSWER_LEN \
        --model "$MODEL" \
        --base-url "dummy" \
        --init-user-id "$INIT_USER_ID" \
        --output "$2" \
        --log-interval 30 \
        --time "$TEST_DURATION" \
        --request-with-user-id

    sleep 10

    # increment init-user-id by NUM_USERS_WARMUP
    INIT_USER_ID=$(( INIT_USER_ID + NUM_USERS_WARMUP ))
}

# Run benchmarks for the specified QPS values
for qps in "${QPS_VALUES[@]}"; do
    output_file="summary_${qps}.csv"
    run_benchmark "$qps" "$output_file"
done