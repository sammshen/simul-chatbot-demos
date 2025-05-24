# Simultaneous High-Volume User Chatbot Simulations

### Compare Production Stack + LMCache versus other orchestration layers for LLM serving in a multi-round QA RAG setting.

### Automated "users" will join and send a high volume of long context requests and you will be able to track TTFT/ITL/TPOT in **REAL TIME**

### **YOU YOURSELF** will *also* be able to join in on the QA through a live Web UI.

# Step 1: Setting up Production Stack vs Comparison Baseline

## Run:
```bash
bash double-baselines/deploy.sh <COMPARISON_BASELINE> <ACCELERATOR_TYPE> <MODEL_NAME> <NUM_REPLICAS> <HF_TOKEN>
# Example:
bash double-baselines/deploy.sh ray A100 meta-llama/Llama-3.1-8B-Instruct 4 hf_...
```

## Explanation:
```text
<COMPARISON_BASELINE>: "ray" (more coming soon...)
<ACCELERATOR_TYPE>: A10G, L4, A100, H100, etc.
<MODEL_NAME>: meta-llama/Llama-3.1-8B-Instruct, mistralai/Mistral-7B-Instruct-v0.2
<NUM_REPLICAS>: the SAME number of GPUs/Replicas will be used for the two baselines
<HF_TOKEN>: make sure this has access to your model
```

## Confirm:
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

# Step 2: Set Up your Personalized RAG DB

### Application-Specific Customization:

Populate the `RAG-DB` folder with .html or .pdf or .txt files. These will be the files that will be randomly selected to form the long contexts for the multiround QA. Additionally, these will be the files that you will be able to select during your interactive session. A couple of Linux man files are already in there as an example that you can remove and replace if you would like.

# Step 3: Long Context High Volume Multi-Round QA and Live Querying

## Running the Benchmark

Run the benchmark script to compare both endpoints:

```bash
bash ./simul-qa/long_input_short_output_run.sh <MODEL-NAME> [LIST OF QPS]
# Example:
bash ./simul-qa/long_input_short_output_run.sh meta-llama/Llama-3.1-8B-Instruct 6.2
```

## Real Time TTFT, ITL, TPOT, Throughput

You'll be able to find real time files in `real_time_statistics`
- `real_time_statistics/real_time_stats_{qps}.txt` will summarize the TTFT, ITL, TPOT, and Throughput comparisons every 5 seconds with the number of currently active users, input and output tokens, and documents used in RAG.
```text
TTFT: ProductionStack=134.39ms, Other=4195.77ms (Other is 3022.1% slower)
TPOT: ProductionStack=14.63ms, Other=47.89ms (Other is 227.4% slower)
ITL: ProductionStack=15.93ms, Other=95.51ms (Other is 499.5% slower)
Output throughput: ProductionStack=193.25 t/s, Other=154.51 t/s (Other is 20.0% slower)
```
- `real_time_statistics/user_request_{qps}.txt` will summarize the same metrics for every individual Request by every individual user.
```text
---------------------------------------------------
[2025-05-24 12:15:42] User 27, Request 8, Endpoint Other
  Documents: man-unix.txt and man-sed.txt
  Prompt tokens: 4808
  Generation tokens: 100
  TTFT: 2145.61 ms
  TPOT: -0.10 ms
  ITL: 21.36 ms
  Total generation time: 2.14 s
  Generation speed: 46.82 tokens/s
---------------------------------------------------
[2025-05-24 12:15:43] User 18, Request 9, Endpoint ProductionStack
  Documents: man-python.txt and man-sed.txt
  Prompt tokens: 6143
  Generation tokens: 100
  TTFT: 3813.21 ms
  TPOT: -15.60 ms
  ITL: 22.69 ms
  Total generation time: 2.27 s
  Generation speed: 44.08 tokens/s
```

## Step 4: Interactive Side-by-Side Chat Comparison

Run the interactive Streamlit app to directly compare both endpoints:

```bash
bash live-double-chat/run_demo.sh
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