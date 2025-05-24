# Simultaneous High-Volume User Chatbot Simulations

## TL;DR

1. Feel TTFT/ITL/TPOT in real time through Web UI

2. See aggregated TTFT/ITIL/TPOT statistics on long context multi-user QA with your customized retrieved documents.

3. Test out Fault Tolerance of Production Stack

## Step 1: Deploy Production Stack and Comparison Baseline (single script)

### Run:
```bash
bash dual-baselines/deploy.sh <COMPARISON_BASELINE> <ACCELERATOR_TYPE> <MODEL_NAME> <NUM_REPLICAS> <HF_TOKEN>
# Example:
bash dual-baselines/deploy.sh ray A100 meta-llama/Llama-3.1-8B-Instruct 4 hf_...
```

### Explanation:
```text
<COMPARISON_BASELINE>: "ray" (more coming soon...)
<ACCELERATOR_TYPE>: A10G, L4, A100, H100, etc.
<MODEL_NAME>: meta-llama/Llama-3.1-8B-Instruct, mistralai/Mistral-7B-Instruct-v0.2
<NUM_REPLICAS>: the SAME number of GPUs/Replicas will be used for the two baselines (0..N-1 for ProdStack, N..2N-1 for Other)
<HF_TOKEN>: make sure this has access to your model
```

### Confirm:
```bash
# To the Production Stack + LMCache OpenAI API now exposed at localhost:30080
# Change the model to the one you selected
curl http://localhost:30080/v1/chat/completions   -H "Content-Type: application/json" -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 50,
    "temperature": 0.7
  }'

# To the comparison OpenAI API now exposed at localhost:30081
# Change the model to the one you selected
curl http://localhost:30081/v1/chat/completions   -H "Content-Type: application/json" -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

## Step 2: Customize Application specific Long Contexts

Populate the `contexts` folder with .html or .pdf or .txt files. These will be the files that will be randomly selected to form the long contexts for the multiround QA.

## Step 3: Joining a Live Long Context Multi-Round Multi-User QA Simulation!!

### Deploying the automated users

System Prompt is 3000 tokens, Conversation History goes up to 30000 tokens. Around 200 users at any given moment asking for short answers (~100 tokens) based on the document.

```bash
bash ./dual-qa/long_input_short_output_run.sh <MODEL-NAME> [LIST OF QPS]
# Example:
bash ./dual-qa/long_input_short_output_run.sh meta-llama/Llama-3.1-8B-Instruct 6 12
# 6 is a lighter workload
# 12 is a very intense workload for 4 serving engines
```

### Real Time Statistics: TTFT, ITL, TPOT, Throughput

You'll be able to find real time files in `real_time_statistics`
- `real_time_statistics/real_time_stats_{qps}.txt` will summarize the TTFT, ITL, TPOT, and Throughput comparisons every 5 seconds with the number of currently active users, input and output tokens, and long contexts used for each user.

```text
TTFT: ProductionStack=241.35ms, Other=2999.92ms (Other is 1143.0% slower)
TPOT: ProductionStack=14.57ms, Other=59.70ms (Other is 309.8% slower)
ITL: ProductionStack=16.95ms, Other=90.99ms (Other is 436.9% slower)
Output throughput: ProductionStack=523.71 t/s, Other=484.96 t/s (Other processed 7.4% fewer tokens)
```

## Step 4: Interactive Side-by-Side Chat Comparison

Run the interactive Streamlit app to directly compare both endpoints:

```bash
bash dual-chat/run_demo.sh
```

Then open `http://localhost:8501/`

This works best when you are already running a high workload through Step 3 so you can be one of many users and you can feel the TTFT differential.


## Step 5: Fault Tolerance

Send a request with a specific user id:

```bash
curl http://localhost:30080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-user-id: 9999" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [
      {"role": "user", "content": "Tell me a long story about a man named Zhuohan Gu that never ends"}
    ],
    "max_tokens": 3000,
    "temperature": 0.7,
    "stream": true
  }'
```

Check the production stack router logs to see which engine is being routed to:
```text
[2025-05-24 17:00:01,260] INFO: Request for model meta-llama/Llama-3.1-8B-Instruct was rewritten (request.py:276:vllm_router.services.request_service.request)
[2025-05-24 17:00:01,261] DEBUG: Routing request 476cbf8b-ce59-4c50-95d3-5ee2d77dbd09 for model: meta-llama/Llama-3.1-8B-Instruct (request.py:297:vllm_router.services.request_service.request)
[2025-05-24 17:00:01,261] DEBUG: Got session id: 9999 (routing_logic.py:174:vllm_router.routers.routing_logic)
[2025-05-24 17:00:01,261] INFO: Routing request 476cbf8b-ce59-4c50-95d3-5ee2d77dbd09 to http://localhost:8103 at 1748106001.2612195, process time = 0.0005 (request.py:302:vllm_router.services.request_service.request)
```

Kill the specific serving engine on your next request