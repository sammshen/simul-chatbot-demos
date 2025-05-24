import os
import sys
from ray import serve
from ray.serve.llm import LLMConfig, build_openai_app

# --- Parse CLI arguments ---
if len(sys.argv) != 5:
    print("Usage: python deploy.py <ACCELERATOR_TYPE> <MODEL_NAME> <NUM_REPLICAS> <HF_TOKEN>")
    sys.exit(1)

accel_type = sys.argv[1]
model_name = sys.argv[2]
num_replicas = int(sys.argv[3])
hf_token = sys.argv[4]

# --- Set CUDA_VISIBLE_DEVICES dynamically ---
# Use devices: 2 * N to 2 * N + N - 1
start_device = num_replicas
cuda_devices = ",".join(str(i) for i in range(start_device, start_device + num_replicas))
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
print(f"Using CUDA_VISIBLE_DEVICES={cuda_devices}")

# --- Define the config ---
llm_config = LLMConfig(
    model_loading_config=dict(
        model_id=model_name,
        model_source=model_name,
    ),
    deployment_config=dict(
        autoscaling_config=dict(
            min_replicas=num_replicas,
            max_replicas=num_replicas,
        )
    ),
    accelerator_type=accel_type,
    engine_kwargs=dict(
        tensor_parallel_size=1,
    ),
    runtime_env=dict(
        env_vars=dict(
            HF_TOKEN=hf_token
        )
    )
)

# --- Start the server ---
serve.start(http_options={"port": 30081})
app = build_openai_app({"llm_configs": [llm_config]})
serve.run(app, blocking=True)
