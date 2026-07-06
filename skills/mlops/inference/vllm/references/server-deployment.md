# Server Deployment Patterns

## Contents
- Docker deployment
- Kubernetes deployment
- Load balancing with Nginx
- Multi-node distributed serving
- Production configuration examples
- Health checks and monitoring

## Docker deployment

**Basic Dockerfile**:
```dockerfile
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y python3-pip
RUN pip install vllm

EXPOSE 8000

CMD ["vllm", "serve", "meta-llama/Llama-3-8B-Instruct", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--gpu-memory-utilization", "0.9"]
```

**Build and run**:
```bash
docker build -t vllm-server .
docker run --gpus all -p 8000:8000 vllm-server
```

**Docker Compose** (with metrics):
```yaml
version: '3.8'
services:
  vllm:
    image: vllm/vllm-openai:latest
    command: >
      --model meta-llama/Llama-3-8B-Instruct
      --gpu-memory-utilization 0.9
      --enable-metrics
      --metrics-port 9090
    ports:
      - "8000:8000"
      - "9090:9090"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

## Kubernetes deployment

**Deployment manifest**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-server
spec:
  replicas: 2
  selector:
    matchLabels:
      app: vllm
  template:
    metadata:
      labels:
        app: vllm
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        args:
          - "--model=meta-llama/Llama-3-8B-Instruct"
          - "--gpu-memory-utilization=0.9"
          - "--enable-prefix-caching"
        resources:
          limits:
            nvidia.com/gpu: 1
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 9090
          name: metrics
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-service
spec:
  selector:
    app: vllm
  ports:
  - port: 8000
    targetPort: 8000
    name: http
  - port: 9090
    targetPort: 9090
    name: metrics
  type: LoadBalancer
```

## Load balancing with Nginx

**Nginx configuration**:
```nginx
upstream vllm_backend {
    least_conn;  # Route to least-loaded server
    server localhost:8001;
    server localhost:8002;
    server localhost:8003;
}

server {
    listen 80;

    location / {
        proxy_pass http://vllm_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;

        # Timeouts for long-running inference
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }

    # Metrics endpoint
    location /metrics {
        proxy_pass http://localhost:9090/metrics;
    }
}
```

**Start multiple vLLM instances**:
```bash
# Terminal 1
vllm serve MODEL --port 8001 --tensor-parallel-size 1

# Terminal 2
vllm serve MODEL --port 8002 --tensor-parallel-size 1

# Terminal 3
vllm serve MODEL --port 8003 --tensor-parallel-size 1

# Start Nginx
nginx -c /path/to/nginx.conf
```

## Multi-node distributed serving

For models too large for single node:

**Node 1** (master):
```bash
export MASTER_ADDR=192.168.1.10
export MASTER_PORT=29500
export RANK=0
export WORLD_SIZE=2

vllm serve meta-llama/Llama-2-70b-hf \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2
```

**Node 2** (worker):
```bash
export MASTER_ADDR=192.168.1.10
export MASTER_PORT=29500
export RANK=1
export WORLD_SIZE=2

vllm serve meta-llama/Llama-2-70b-hf \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2
```

## Production configuration examples

**High throughput** (batch-heavy workload):
```bash
vllm serve MODEL \
  --max-num-seqs 512 \
  --gpu-memory-utilization 0.95 \
  --enable-prefix-caching \
  --trust-remote-code
```

**Low latency** (interactive workload):
```bash
vllm serve MODEL \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.85 \
  --enable-chunked-prefill
```

**Memory-constrained** (40GB GPU for 70B model):
```bash
vllm serve TheBloke/Llama-2-70B-AWQ \
  --quantization awq \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 4096
```

## Health checks and monitoring

**Health check endpoint**:
```bash
curl http://localhost:8000/health
# Returns: {"status": "ok"}
```

**Readiness check** (wait for model loaded):
```bash
#!/bin/bash
until curl -f http://localhost:8000/health; do
    echo "Waiting for vLLM to be ready..."
    sleep 5
done
echo "vLLM is ready!"
```

**Prometheus scraping**:
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'vllm'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

**Grafana dashboard** (key metrics):
- Requests per second: `rate(vllm_request_success_total[5m])`
- TTFT p50: `histogram_quantile(0.5, vllm_time_to_first_token_seconds_bucket)`
- TTFT p99: `histogram_quantile(0.99, vllm_time_to_first_token_seconds_bucket)`
- GPU cache usage: `vllm_gpu_cache_usage_perc`
- Active requests: `vllm_num_requests_running`

## v0.24.0 Server Features

### Rust Frontend

v0.24.0 introduces a new high-performance Rust-based HTTP frontend (an optional replacement for the Python/Uvicorn stack). It is enabled via the `--enable-rust-frontend` flag (or the `VLLM_USE_RUST_FRONTEND=1` env var) and brings lower latency, lower CPU overhead, and several new operational endpoints.

**Enable the Rust frontend**:
```bash
vllm serve meta-llama/Llama-3-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --enable-rust-frontend \
  --api-key sk-vllm-PROD-KEY-CHANGE-ME
```

#### API-key authentication

Authentication is enforced when an API key is provided. Clients must send `Authorization: Bearer <key>`.

```bash
# Via CLI flag
vllm serve MODEL --enable-rust-frontend --api-key sk-vllm-PROD-KEY

# Via environment variable
export VLLM_API_KEY=sk-vllm-PROD-KEY
vllm serve MODEL --enable-rust-frontend
```

```bash
# Authenticated request
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-vllm-PROD-KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3-8B-Instruct","messages":[{"role":"user","content":"hi"}]}'
```

#### CORS support

Built-in CORS is available without a reverse proxy:

```bash
vllm serve MODEL \
  --enable-rust-frontend \
  --cors-origins "https://app.example.com,https://dashboard.example.com"
```

#### Tokenization endpoints

`/tokenize` and `/detokenize` are first-class endpoints (no need for the tokenizer library client-side):

```bash
# Tokenize
curl http://localhost:8000/tokenize \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -d '{"prompt":"Hello, world!"}'
# -> {"token_ids":[9906,11,1917,0]}

# Detokenize
curl http://localhost:8000/detokenize \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -d '{"token_ids":[9906,11,1917,0]}'
# -> {"prompt":"Hello, world!"}
```

#### Pause / resume inference (no restart)

Pause accepts new requests into the queue but stops decoding; resume continues without reloading the model or losing KV cache state:

```bash
# Pause decoding (in-flight batches complete their current step, then halt)
curl -X POST http://localhost:8000/pause \
  -H "Authorization: Bearer $VLLM_API_KEY"

# Check status -> {"is_paused": true}
curl http://localhost:8000/is_paused \
  -H "Authorization: Bearer $VLLM_API_KEY"

# Resume
curl -X POST http://localhost:8000/resume \
  -H "Authorization: Bearer $VLLM_API_KEY"
```

#### Abort in-flight requests

Cancel one or more queued/running requests by ID without restarting the server:

```bash
curl -X POST http://localhost:8000/abort_requests \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"request_ids":["req-abc123","req-def456"]}'
```

#### Query TP/PP configuration

`/get_world_size` returns the tensor- and pipeline-parallel shape of the running engine:

```bash
curl http://localhost:8000/get_world_size \
  -H "Authorization: Bearer $VLLM_API_KEY"
# -> {"world_size":4,"tensor_parallel_size":4,"pipeline_parallel_size":1}
```

#### Reasoning model controls

For thinking/reasoning models, `thinking_token_budget` caps the number of tokens the model may emit as reasoning before producing the final answer:

```bash
vllm serve Qwen/Qwen3-32B \
  --enable-rust-frontend \
  --reasoning-parser deepseek_r1 \
  --chat-template-content-format string

# Per-request budget (0 disables thinking):
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model":"Qwen/Qwen3-32B",
    "messages":[{"role":"user","content":"Solve: 23*17"}],
    "thinking_token_budget": 512,
    "max_tokens": 2048
  }'
```

`parallel_tool_calls` can now be set to `false` to guarantee only one tool is invoked per assistant turn (useful for strict single-action agents):

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model":"meta-llama/Llama-3.1-70B-Instruct",
    "messages":[...],
    "tools":[...],
    "parallel_tool_calls": false
  }'
```

#### Continuous usage stats & model metadata

- **Continuous usage stats**: token-usage objects (`prompt_tokens`, `completion_tokens`, `total_tokens`) are streamed incrementally in SSE `usage` chunks when `stream_options:{"include_usage":true}` is set — clients no longer need to wait for the final chunk to track cost.
- **Model metadata in `/v1/models`**: the `/v1/models` response now includes `max_model_len`, `tensor_parallel_size`, `pipeline_parallel_size`, `quantization`, and architecture fields so orchestrators can introspect capabilities without a second API call.

```bash
curl http://localhost:8000/v1/models -H "Authorization: Bearer $VLLM_API_KEY"
# -> {"object":"list","data":[{"id":"meta-llama/Llama-3-8B-Instruct",
#       "object":"model","owned_by":"vllm","max_model_len":8192,
#       "tensor_parallel_size":2,"quantization":null,...}]}
```

### Device selection via `--device-ids`

v0.24.0 replaces the internal use of `CUDA_VISIBLE_DEVICES` with an explicit `--device-ids` flag. This avoids mutating the process environment and makes multi-instance GPU assignment deterministic.

```bash
# Use GPUs 0 and 1 (tensor-parallel across them)
vllm serve meta-llama/Llama-3-70B-Instruct \
  --device-ids 0,1 \
  --tensor-parallel-size 2

# Pin to a single GPU out of a larger set
vllm serve MODEL --device-ids 3 --tensor-parallel-size 1
```

> Note: `CUDA_VISIBLE_DEVICES` is still respected as a fallback, but `--device-ids` takes precedence when both are set.

### fastsafetensors `ParallelLoader`

Weight loading now uses the `fastsafetensors` `ParallelLoader`, which streams and deshards `.safetensors` shards in parallel across ranks. This significantly reduces startup time for large models (e.g. 70B+ on multi-GPU), especially from network-attached storage.

```bash
# No flag required — enabled by default for safetensors checkpoints.
# Control parallelism / num loaders:
vllm serve meta-llama/Llama-3-70B-Instruct \
  --tensor-parallel-size 4 \
  --load-format safetensors
```

For local-NVMe single-GPU models the loader falls back to a single-stream path automatically; no action is needed.

### Health check endpoints (unchanged)

The existing health endpoints are unchanged in v0.24.0 and work identically with the Rust frontend:

```bash
# Liveness / readiness
curl http://localhost:8000/health
# -> {"status":"ok"}

# Serve readiness (model loaded, accepting traffic)
curl http://localhost:8000/ready
# -> 200 once engine is ready

# From inside a container:
until curl -sf http://localhost:8000/health; do sleep 2; done
```

These remain the recommended targets for Kubernetes `readinessProbe` / `livenessProbe` (see the K8s manifest above).

### cgroup memory-limit-aware KV cache sizing

When running inside a cgroup v2 container (Docker, Kubernetes), vLLM now reads `memory.max` to size the KV cache against the **container's** memory limit rather than host RAM. This prevents OOM-kills where vLLM previously assumed it could use all host memory for `gpu-memory-utilization`-derived CPU-side buffers.

```bash
# Set a memory limit; vLLM will respect it when sizing KV cache + paging
docker run --gpus all --memory 64g --memory-swap 0 \
  -p 8000:8000 vllm/vllm-openai:v0.24.0 \
  --model meta-llama/Llama-3-70B-Instruct \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 32768
```

- Requires cgroup v2 (`docker info | grep -i cgroup` → `Cgroup Version: 2`).
- Falls back to host RAM detection on cgroup v1 or bare-metal — no behavior change.
- Combine with `--swap-space 0` to forbid host-swap use of KV cache pages.

### Production Docker example (API key + CORS + device-ids)

```dockerfile
# Dockerfile.vllm-prod
FROM vllm/vllm-openai:v0.24.0

ENV VLLM_API_KEY=sk-vllm-PROD-CHANGE-ME \
    HF_HOME=/models \
    VLLM_USE_RUST_FRONTEND=1

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=3 \
  CMD curl -sf http://localhost:8000/health || exit 1

ENTRYPOINT ["vllm", "serve"]
CMD ["--model", "/models/Llama-3-70B-Instruct", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--device-ids", "0,1,2,3", \
     "--tensor-parallel-size", "4", \
     "--gpu-memory-utilization", "0.92", \
     "--max-model-len", "32768", \
     "--enable-prefix-caching", \
     "--api-key", "sk-vllm-PROD-CHANGE-ME", \
     "--cors-origins", "https://app.example.com,https://dashboard.example.com"]
```

```bash
# Build
docker build -t vllm-prod:v0.24.0 -f Dockerfile.vllm-prod .

# Run (4 GPUs, 64 GB container memory limit, cgroup v2)
docker run -d --name vllm-prod \
  --gpus all \
  --device /dev/nvidia0:/dev/nvidia0 \
  --device /dev/nvidia1:/dev/nvidia1 \
  --device /dev/nvidia2:/dev/nvidia2 \
  --device /dev/nvidia3:/dev/nvidia3 \
  --memory 64g --memory-swap 0 \
  -v /data/models:/models:ro \
  -p 8000:8000 \
  --restart unless-stopped \
  vllm-prod:v0.24.0
```

```bash
# Verify
curl -sf http://localhost:8000/health
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer sk-vllm-PROD-CHANGE-ME" | jq
curl http://localhost:8000/get_world_size \
  -H "Authorization: Bearer sk-vllm-PROD-CHANGE-ME"
# -> {"world_size":4,"tensor_parallel_size":4,"pipeline_parallel_size":1}
```
