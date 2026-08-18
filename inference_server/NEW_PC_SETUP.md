# Run the VLM inference profiles from a new PC

This guide recreates the current native, keyless SSH-tunnel setup. It uses:

- the RTX 5090 server at `long-horizon@10.10.16.68`;
- Fish on the GPU server;
- Bash on the client PC;
- vLLM on server port `8000`;
- the functional API on server port `8080`; and
- local tunnel ports `18000` and `18080` on the client PC.

The two server processes bind only to `127.0.0.1`. They are not exposed to the
network and do not require API keys. The SSH tunnel is the only access path.

## 1. Put the code on both machines

On a new client PC, clone the inference branch and enter the repository:

```bash
git clone --branch inference_server_setup \
  https://github.com/Narendhiranv04/icra-we-ball.git
cd icra-we-ball
```

If the branch is not on GitHub yet, copy the existing checkout from the old PC
instead:

```bash
rsync -av OLD_PC:/path/to/LH_Extension/V1/ ./V1/
cd V1
```

Copy the standalone inference workspace from the client PC to the GPU server:

```bash
ssh long-horizon@10.10.16.68 'mkdir -p ~/SearchTAMP/inference_server'

rsync -av \
  --exclude .env \
  --exclude cache/ \
  --exclude models/ \
  --exclude '.venv*/' \
  inference_server/ \
  long-horizon@10.10.16.68:~/SearchTAMP/inference_server/
```

Later code updates use the same `rsync` command. Do not use `--delete` unless
the server's environments, caches, and model directories are excluded.

## 2. Create the server environment

SSH into the GPU server:

```bash
ssh long-horizon@10.10.16.68
```

The remaining commands in this section run in Fish on the server:

```fish
cd ~/SearchTAMP
uv venv --python 3.11 .venv-qwen35
source .venv-qwen35/bin/activate.fish

uv pip install vllm blobfile --torch-backend=auto \
  --extra-index-url https://wheels.vllm.ai/nightly
```

Qwen3.5 requires a recent vLLM build. The listed checkpoints are public, so no
Hugging Face token is required. vLLM downloads the selected model automatically
on first launch and reuses the Hugging Face cache afterward. `blobfile` covers
Kimi-VL's remote model code.

Verify the GPU before starting the model:

```fish
nvidia-smi
```

## 3. Start Qwen in server shell A

Open an SSH session, activate the environment, and start vLLM:

```fish
cd ~/SearchTAMP
source .venv-qwen35/bin/activate.fish

vllm serve Qwen/Qwen3.5-9B \
  --served-model-name qwen35-9b \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --max-num-seqs 2 \
  --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image":8}' \
  --enable-prefix-caching \
  --generation-config vllm \
  --reasoning-parser qwen3
```

Leave this shell running. The first launch downloads the model and therefore
takes longer. Wait until vLLM reports that the API server is ready.

The first request for a new image shape may print Triton JIT warnings. These
are warm-up latency warnings, not inference failures.

## 4. Start the functional API in server shell B

Open a second SSH session to the server. In Fish, run:

```fish
cd ~/SearchTAMP
source .venv-qwen35/bin/activate.fish

set -x INFERENCE_MODEL qwen35-9b
set -x PLANNER_MODEL qwen35-9b
set -x PLANNER_MODEL_BASE_URL http://127.0.0.1:8000/v1
set -x PLANNER_HOST 127.0.0.1
set -x PLANNER_PORT 8080
set -x PLANNER_MODEL_TIMEOUT_SECONDS 300
set -x PLANNER_ENABLE_THINKING true
set -x PLANNER_MAX_TOKENS 12288

set -e INFERENCE_API_KEY
set -e PLANNER_API_KEY

python3 inference_server/planner_api.py
```

Leave this shell running too. Only `planner_api.py` must be restarted after a
prompt, schema, catalog, or planner configuration change. vLLM can keep running.

### Switch to another model

Stop both processes with `Ctrl-C`. In shell A, use the registry-backed launcher
instead of writing a new vLLM command:

```fish
cd ~/SearchTAMP
source .venv-qwen35/bin/activate.fish

set -x INFERENCE_MODEL glm46v-flash
set -x INFERENCE_HOST 127.0.0.1
set -x INFERENCE_CONTAINER_PORT 8000
set -e INFERENCE_API_KEY

python3 inference_server/server.py
```

Valid profile values are:

```text
qwen35-9b
glm46v-flash
qwen3-vl-8b-thinking
internvl35-14b
kimi-vl-a3b-thinking
```

In shell B, select the same profile and let it supply its own planner defaults:

```fish
cd ~/SearchTAMP
source .venv-qwen35/bin/activate.fish

set -x INFERENCE_MODEL glm46v-flash
set -x PLANNER_MODEL_BASE_URL http://127.0.0.1:8000/v1
set -x PLANNER_HOST 127.0.0.1
set -x PLANNER_PORT 8080
set -x PLANNER_MODEL_TIMEOUT_SECONDS 300

set -e PLANNER_MODEL
set -e PLANNER_ENABLE_THINKING
set -e PLANNER_MAX_TOKENS
set -e INFERENCE_API_KEY
set -e PLANNER_API_KEY

python3 inference_server/planner_api.py
```

The functional client command on the local PC does not change. The `/health`
response should show the newly selected served profile. Run one model at a
time. InternVL and Kimi are loaded with online FP8 to leave room on the 32 GB
5090; if either fails during startup, first retry with
`set -x INFERENCE_MAX_MODEL_LEN 8192` and
`set -x INFERENCE_GPU_MEMORY_UTILIZATION 0.90`.

InternVL and Kimi use prompt-driven reasoning without dedicated vLLM parsers in
this setup, so their functional JSON modes are experimental. The gateway strips
their thinking blocks and validates the final JSON. A validation failure is not
an execution failure; it means that sample did not satisfy the planner schema.

## 5. Open the tunnel from the client PC

In a Bash terminal on the new PC, run:

```bash
ssh \
  -L 18000:127.0.0.1:8000 \
  -L 18080:127.0.0.1:8080 \
  long-horizon@10.10.16.68
```

This opens an interactive server shell and keeps both tunnels alive. Leave it
open. Use another local terminal for the following commands.

Verify the functional API from the client PC:

```bash
curl -fsS http://127.0.0.1:18080/health
curl -fsS http://127.0.0.1:18080/v1/functions
```

The health response should identify `functional-planner` and model
`qwen35-9b`.

## 6. Supply camera images

The client accepts one to eight PNG or JPEG images. It reads the images from
the client PC and sends them through the functional API, so the server does not
need access to the simulator's `runs/` directory.

For the existing kitchen test, the expected local paths are:

```text
runs/qwen_kitchen_test/left_shoulder_camera.png
runs/qwen_kitchen_test/right_shoulder_camera.png
runs/qwen_kitchen_test/overhead_camera.png
runs/qwen_kitchen_test/side_camera.png
runs/qwen_kitchen_test/front_camera.png
runs/qwen_kitchen_test/wrist_camera.png
runs/qwen_kitchen_test/head_camera_rgb.png
```

If those files remain on the old PC, copy them into the same relative path on
the new PC:

```bash
mkdir -p runs/qwen_kitchen_test
rsync -av OLD_PC:/path/to/V1/runs/qwen_kitchen_test/ \
  runs/qwen_kitchen_test/
```

## 7. Send a functional-decomposition prompt

From the repository root on the client PC:

```bash
python3 inference_server/functional_client.py \
  --base-url http://127.0.0.1:18080/v1 \
  --scene kitchen \
  --goal "Make coffee in a suitable container, stir it, and serve it" \
  --image left_shoulder_camera=runs/qwen_kitchen_test/left_shoulder_camera.png \
  --image right_shoulder_camera=runs/qwen_kitchen_test/right_shoulder_camera.png \
  --image overhead_camera=runs/qwen_kitchen_test/overhead_camera.png \
  --image side_camera=runs/qwen_kitchen_test/side_camera.png \
  --image front_camera=runs/qwen_kitchen_test/front_camera.png \
  --image wrist_camera=runs/qwen_kitchen_test/wrist_camera.png \
  --image head_camera_rgb=runs/qwen_kitchen_test/head_camera_rgb.png
```

No client virtual environment or third-party package is needed for this
command; `functional_client.py` uses the Python standard library.

The JSON response should contain:

- `status: DECOMPOSED`;
- one or more simple functional requirements;
- 10–15 distinct concrete candidate types per requirement;
- dependencies between requirements where applicable; and
- `search_started`, `geometry_verified`, and `execution_started` set to false.

The candidate types are foundation-model priors. They are not claims that the
objects are visible or available. Semantic grounding, region search,
point-cloud geometry, task sequencing, and execution happen in later modules.

## 8. Change the scene or goal

Valid scene values are:

```text
kitchen
living_room
workshop
```

Only the goal string and image arguments need to change:

```bash
python3 inference_server/functional_client.py \
  --base-url http://127.0.0.1:18080/v1 \
  --scene living_room \
  --goal "Clean the reachable rigid tabletop" \
  --image front=/path/to/front.png \
  --image overhead=/path/to/overhead.png
```

## 9. Stop the services

Stop each process with `Ctrl-C` in its own server shell:

1. stop `planner_api.py` in shell B;
2. stop `vllm serve` in shell A; and
3. close the SSH tunnel shell with `exit`.

Downloaded model files remain cached for the next launch.

## Troubleshooting

### `Cannot reach functional planning server`

Check that `planner_api.py` is running and that the `18080` SSH tunnel remains
open:

```bash
curl -v http://127.0.0.1:18080/health
```

### `404 Not Found` for `/v1/decompose`

The server has an older `planner_api.py`. Re-run the inference-server `rsync`
and restart only `planner_api.py`.

### `Completion has no final JSON content (finish_reason=length)`

First confirm the current `functional_planner.py` is on the server. Thinking
mode must use the configured non-greedy sampler. If it still reaches the limit,
temporarily try:

```fish
set -x PLANNER_MAX_TOKENS 16384
```

Restart `planner_api.py` after changing the value. If Qwen repeatedly consumes
the entire budget, use the reliable low-latency ablation:

```fish
set -x PLANNER_ENABLE_THINKING false
set -x PLANNER_MAX_TOKENS 8192
```

### `DECOMPOSED returned zero functional requirements`

The model did not map the goal onto the configured function catalog. Inspect
the catalog from the client:

```bash
curl -fsS http://127.0.0.1:18080/v1/functions
```

Then either rephrase the goal around a supported functional role or add the
missing simple `can_*` function to `functional_catalog.json` and restart the
functional API.

### Triton JIT warnings

Warnings mentioning `_bilinear_pos_embed_kernel`, `layer_norm_fwd_kernel`, or
`apply_token_bitmask_inplace_kernel` mean that vLLM is compiling a kernel for a
new shape. They affect warm-up latency but do not indicate a failed request.

### CUDA out of memory

Stop other GPU processes first. If necessary, relaunch vLLM with a smaller
context or memory fraction:

```fish
--max-model-len 24576 --gpu-memory-utilization 0.85
```

Do not start multiple model servers on the same 32 GB RTX 5090.
