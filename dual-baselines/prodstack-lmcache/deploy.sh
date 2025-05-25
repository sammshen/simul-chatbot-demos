#!/bin/bash
# Example: bash deploy.sh 4 meta-llama/Llama-3.1-8B-Instruct hf_cKu...

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# Ensure 3 arguments are provided
if [ $# -ne 3 ]; then
  echo "Usage: $0 <MODEL_URL> <REPLICA_COUNT> <HF_TOKEN>"
  exit 1
fi

# Arguments:
MODEL_URL=$1
REPLICA_COUNT=$2
HF_TOKEN=$3

# Kill existing GPU processes
GPU_UUIDS=($(nvidia-smi --query-gpu=uuid --format=csv,noheader,nounits))
for ((i=0; i<REPLICA_COUNT; i++)); do
  GPU_UUID=${GPU_UUIDS[$i]}
  echo "GPU $i UUID: $GPU_UUID"
  PIDS=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name \
          --format=csv,noheader,nounits \
        | grep "$GPU_UUID" \
        | grep python3 \
        | awk -F, '{gsub(/ /,"",$2); print $2}' \
        | sort -u)
  for pid in $PIDS; do
    echo "Killing PID $pid on GPU $i"
    sudo kill -9 "$pid" 2>/dev/null || echo "  PID $pid already dead or inaccessible"
    sleep 2
  done
done

export HF_TOKEN=$HF_TOKEN
IMAGE=lmcache/vllm-openai:latest

# Comprehensive port and container cleanup
echo "🧹 Performing comprehensive cleanup of ports 8100 to $((8100 + REPLICA_COUNT - 1))..."

# Step 1: Kill all containers using target ports
for ((p=8100; p<8100+REPLICA_COUNT; p++)); do
  echo "Cleaning port $p..."

  # Kill containers bound to this port (both running and stopped)
  CONTAINERS=$(sudo docker ps -a --filter "publish=$p" --format "{{.ID}}")
  for cid in $CONTAINERS; do
    echo "  🛑 Removing container $cid using port $p"
    sudo docker kill "$cid" >/dev/null 2>&1
    sudo docker rm "$cid" >/dev/null 2>&1
  done

  # Kill any process listening on the port
  PID=$(sudo lsof -ti tcp:$p -sTCP:LISTEN 2>/dev/null)
  if [[ -n "$PID" ]]; then
    echo "  🔪 Killing PID $PID listening on port $p"
    sudo kill -9 "$PID" 2>/dev/null || true
  fi
done

# Step 2: Clean up Docker system and restart daemon
echo "🔄 Cleaning Docker system and restarting daemon..."
sudo docker system prune -f >/dev/null 2>&1
sudo iptables -t nat -F DOCKER 2>/dev/null || true
sudo systemctl restart docker
sleep 5

# Step 3: Final port verification
for ((p=8100; p<8100+REPLICA_COUNT; p++)); do
  if sudo lsof -iTCP:$p -sTCP:LISTEN >/dev/null 2>&1; then
    echo "❌ Port $p still in use after cleanup. Manual intervention needed."
    exit 1
  fi
  echo "✅ Port $p is free."
done

# Start containers
CONTAINER_IDS=()
for ((i=0; i<REPLICA_COUNT; i++)); do
  HOST_PORT=$((8100 + i))
  CONTAINER_PORT=8000
  CUDA_DEVICE=$i
  echo "Starting replica $i on host port $HOST_PORT using GPU $CUDA_DEVICE..."

  # Final port check before launch
  if sudo lsof -iTCP:$HOST_PORT -sTCP:LISTEN >/dev/null 2>&1; then
    echo "❌ Port $HOST_PORT still bound just before container launch. Aborting."
    exit 1
  fi

  # Run the container
  RUN_OUTPUT=$(sudo docker run -d --runtime=nvidia --gpus all \
    --env "HF_TOKEN=$HF_TOKEN" \
    --env "LMCACHE_USE_EXPERIMENTAL=True" \
    --env "LMCACHE_CHUNK_SIZE=256" \
    --env "LMCACHE_LOCAL_CPU=True" \
    --env "LMCACHE_MAX_LOCAL_CPU_SIZE=200" \
    --env "CUDA_VISIBLE_DEVICES=$CUDA_DEVICE" \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p ${HOST_PORT}:${CONTAINER_PORT} \
    $IMAGE \
    $MODEL_URL \
    --max-model-len 32768 \
    --port $CONTAINER_PORT \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' 2>&1)

  if [[ $? -ne 0 ]]; then
    echo "❌ Failed to start container for replica $i on port $HOST_PORT"
    echo "$RUN_OUTPUT"
    exit 1
  fi

  # Extract and verify container ID (accept both 12+ char and full 64 char IDs)
  if [[ "$RUN_OUTPUT" =~ ^[a-f0-9]{12,64}$ ]]; then
    CONTAINER_IDS+=("$RUN_OUTPUT")
    echo "✅ Started container $RUN_OUTPUT for replica $i"
  else
    echo "❌ Unexpected Docker output when launching replica $i:"
    echo "$RUN_OUTPUT"
    exit 1
  fi
done

# Wait for replicas to become healthy
echo "Waiting for all $REPLICA_COUNT replicas to become healthy..."
PORTS=()
for ((i=0; i<REPLICA_COUNT; i++)); do
  PORTS+=($((8100 + i)))
done

READY=()
TIMEOUT=600  # seconds
START=$(date +%s)
LAST_LOG_CHECK=0

while (( ${#READY[@]} < REPLICA_COUNT )); do
  CURRENT_TIME=$(date +%s)

  # Check logs every 5 seconds
  if (( CURRENT_TIME - LAST_LOG_CHECK >= 5 )); then
    echo "--- Engine startup logs (last 5 lines each) ---"
    for ((i=0; i<REPLICA_COUNT; i++)); do
      CONTAINER_ID=${CONTAINER_IDS[$i]}
      PORT=$((8100 + i))
      echo "Engine $i (port $PORT, container ${CONTAINER_ID:0:12}):"
      sudo docker logs --tail 5 "$CONTAINER_ID" 2>&1 | sed 's/^/  /'
      echo ""
    done
    echo "--- End engine logs ---"
    LAST_LOG_CHECK=$CURRENT_TIME
  fi

  for ((i=0; i<REPLICA_COUNT; i++)); do
    PORT=$((8100 + i))
    if [[ " ${READY[*]} " =~ " $PORT " ]]; then
      continue  # already healthy
    fi

    # Get engine status
    ENGINE_STATUS="unknown"
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 "http://localhost:$PORT/health")
    if [[ "$HTTP_CODE" == "200" ]]; then
      ENGINE_STATUS="healthy"
      echo "✅ Engine $i (port $PORT) is healthy."
      READY+=("$PORT")
    else
      # Try to get more detailed status
      if nc -z localhost $PORT 2>/dev/null; then
        ENGINE_STATUS="port open (loading)"
      else
        ENGINE_STATUS="starting up"
      fi
      echo "⏳ Engine $i (port $PORT) status: $ENGINE_STATUS"
    fi
  done

  ELAPSED=$(( $(date +%s) - START ))
  if (( ELAPSED > TIMEOUT )); then
    echo "❌ Timeout: Not all replicas became healthy in $TIMEOUT seconds."
    echo "Ports that succeeded: ${READY[*]}"
    echo "Ports that failed: $(comm -23 <(printf "%s\n" "${PORTS[@]}" | sort) <(printf "%s\n" "${READY[@]}" | sort))"
    exit 1
  fi
  sleep 10
done

# Prepare router configuration
BACKENDS=""
MODELS=""
for ((i=0; i<REPLICA_COUNT; i++)); do
  PORT=$((8100 + i))
  BACKENDS+="http://localhost:$PORT,"
  MODELS+="$MODEL_URL,"
done

# Remove trailing commas
BACKENDS="${BACKENDS%,}"
MODELS="${MODELS%,}"

echo "BACKENDS: $BACKENDS"
echo "MODELS: $MODELS"

# Clean up router port and start router
ROUTER_IMAGE=apostab/lmstack-router:latest
echo "Checking for any process using port 30080..."
PID_ON_30080=$(sudo lsof -t -i :30080 2>/dev/null)
if [[ -n "$PID_ON_30080" ]]; then
  echo "Found process on port 30080 (PID: $PID_ON_30080). Killing it..."
  sudo kill -9 "$PID_ON_30080" || echo "  ❗️ Failed to kill PID $PID_ON_30080"
else
  echo "✅ Port 30080 is free."
fi


# sudo docker run --network host apostab/lmstack-router:latest\
#  --port 30080 \
#   --service-discovery static \
#   --static-backends "http://localhost:8100,http://localhost:8101,http://localhost:8102,http://localhost:8103" \
#   --static-models "meta-llama/Llama-3.1-8B-Instruct,meta-llama/Llama-3.1-8B-Instruct,meta-llama/Llama-3.1-8B-Instruct,meta-llama/Llama-3.1-8B-Instruct" \
#   --log-stats \
#   --log-stats-interval 10 \
#   --engine-stats-interval 10 \
#   --request-stats-window 10 \
#   --request-stats-window 10 \
#   --routing-logic roundrobin

sudo docker run --network host $ROUTER_IMAGE --port 30080 \
    --service-discovery static \
    --static-backends "$BACKENDS" \
    --static-models "$MODELS" \
    --log-stats \
    --log-stats-interval 10 \
    --engine-stats-interval 10 \
    --request-stats-window 10 \
    --request-stats-window 10 \
    --routing-logic session \
    --session-key "x-user-id"

# python3 -m vllm_router.app \
#  --port 30080 \
#   --service-discovery static \
#   --static-backends "http://localhost:8101,http://localhost:8103" \
#   --static-models "meta-llama/Llama-3.1-8B-Instruct,meta-llama/Llama-3.1-8B-Instruct" \
#   --log-stats \
#   --log-stats-interval 10 \
#   --engine-stats-interval 10 \
#   --request-stats-window 10 \
#   --request-stats-window 10 \
#   --routing-logic roundrobin



# sudo docker run --network host $ROUTER_IMAGE \
#     --port 30080 \
#     --service-discovery static \
#     --static-backends "$BACKENDS" \
#     --static-models "$MODELS" \
#     --engine-stats-interval 10 \
#     --log-stats \
#     --routing-logic session \
#     --session-key "x-user-id"

echo "$REPLICA_COUNT vllm instances of $MODEL_URL are now served on production stack router at localhost:30080"

