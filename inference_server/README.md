# vLLM Inference Server

This folder is an independent GPU-server workspace. It serves an existing
model through the OpenAI-compatible API used by `mujoco_scenes`. It contains
no MuJoCo dependency and performs no training.

vLLM is pinned to `0.23.0` in both native and Docker setups.

## Pull only this workspace

On the inference server:

```bash
git clone --filter=blob:none --no-checkout \
  https://github.com/Narendhiranv04/icra-we-ball.git
cd icra-we-ball
git sparse-checkout init --cone
git sparse-checkout set inference_server
git checkout main
cd inference_server
```

Use the branch containing this folder until it has been merged into `main`.

## Native installation with uv

The current vLLM release supports Linux and Python 3.10 through 3.13. Python
3.12 is used here:

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install -r requirements.txt --torch-backend=auto
```

Copy and edit the configuration:

```bash
cp .env.example .env
```

`VLLM_MODEL` can be a local model directory or a model identifier supported
by vLLM. Set `HF_TOKEN` in the shell only when the model download requires it.
Do not commit either token.

Load the configuration and start the server:

```bash
set -a
source .env
set +a
uv run python server.py
```

Additional vLLM arguments can be appended:

```bash
uv run python server.py --dtype bfloat16
```

Use the smallest tensor-parallel size that fits the model. Multiple GPUs can
increase communication overhead for a single low-concurrency request.

## Docker Compose

The Compose setup uses vLLM's official NVIDIA image. It mounts `./models`
read-only and keeps downloaded model files under `./cache` by default.

```bash
cp .env.example .env
docker compose up -d
docker compose logs -f vllm
```

If `VLLM_MODEL` is a model identifier rather than a local `/models/...` path,
the model is downloaded into the mounted cache. Set `HF_CACHE_DIR` or
`MODEL_DIR` in `.env` to use other host directories.

## Check the endpoint

```bash
curl -fsS http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer $VLLM_API_KEY"
```

On the MuJoCo computer, configure the matching values:

```bash
export TAMP_FM_BASE_URL=http://INFERENCE_SERVER:8000/v1
export TAMP_FM_MODEL=tamp-ranker
export TAMP_FM_API_KEY="$VLLM_API_KEY"
```

Keep port 8000 on a private network. For access across an untrusted network,
put TLS in front of vLLM rather than exposing the raw HTTP endpoint.

## Upgrade

Change the version in both `requirements.txt` and `compose.yaml`, then rerun
the launcher tests:

```bash
uv run python -m unittest test_server.py
```
