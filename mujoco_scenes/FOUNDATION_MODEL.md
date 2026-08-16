# Remote Foundation-Model Ranking

The simulator uses one OpenAI-compatible client for either vLLM or SGLang.
The inference server runs on the GPU machine; the MuJoCo environment needs no
inference framework or additional Python dependency.

A ready-to-deploy Docker workspace is provided in
[`inference_server/`](../inference_server/README.md). It can be rsynced without
the MuJoCo assets and supports profiled vLLM or SGLang launches on the remote
GPU server.

The workspace also exposes a functional-decomposition endpoint on port 8080.
It accepts camera images and a natural-language goal, selects simple configured
functions such as `can_stir`, and ranks 10–15 concrete candidate types
for each replaceable role. The types are proposals, not claims about visible
inventory. Search, semantic grounding, target-specific geometry, action
sequencing, and execution remain separate. See the inference workspace README
for the request schema and `functional_client.py` example.

## Start a server

Serve an existing local model with vLLM:

```bash
vllm serve /models/your-instruct-model \
  --served-model-name tamp-ranker \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key "$TAMP_FM_API_KEY" \
  --enable-prefix-caching \
  --generation-config vllm
```

The equivalent SGLang endpoint can use the same model name and port:

```bash
python -m sglang.launch_server \
  --model-path /models/your-instruct-model \
  --served-model-name tamp-ranker \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key "$TAMP_FM_API_KEY"
```

Keep the endpoint on a private network or place TLS authentication in front
of it. Do not expose an unauthenticated inference port to the internet.

## Configure the simulator

Copy the values from `.env.example` into the shell that starts MuJoCo:

```bash
export TAMP_FM_BASE_URL=http://inference-server:8000/v1
export TAMP_FM_MODEL=tamp-ranker
export TAMP_FM_API_KEY=replace-me
```

The client keeps one HTTP connection open to avoid a connection handshake for
each planning decision. Create one ranker for the episode and close it during
shutdown:

```python
from mujoco_scenes.foundation_model import (
    Candidate,
    OpenAICompatibleRanker,
    RankingRequest,
)

candidates = (
    Candidate("spoon_1", "spoon", {"insertable_in_mug": True}),
    Candidate("pen_1", "pen", {"insertable_in_mug": True}),
)
request = RankingRequest(
    required_function="can_stir",
    candidates=candidates,
    target={"id": "mug_1", "category": "mug"},
)

with OpenAICompatibleRanker.from_env() as ranker:
    result = ranker.rank(request)
    print(result.candidate_ids)
```

This older observed-instance ranker belongs after search: only candidates
passed to `RankingRequest` are sent to the server.
`assess()` returns both the functional subset and its ranking; unknown,
non-visible, duplicate, missing, or malformed IDs are rejected before
execution. `rank()` remains available when every supplied candidate is already
known to satisfy the function.

`FixedRankingBackend` provides the same interface for offline development and
tests. Switching between vLLM and SGLang only changes the server process or
`TAMP_FM_BASE_URL`; it does not require another client implementation.

See [TAMP_PIPELINE.md](TAMP_PIPELINE.md) for the simulator executive and
living-room example.

Keep the system prompt fixed, candidate facts concise, and model output short.
Benchmark warm p50 and p95 end-to-end latency using real ranking requests at
the expected concurrency before choosing a server or quantization.
