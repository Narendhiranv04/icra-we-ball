# Standalone multimodal inference server

This directory is a self-contained, Docker-first workspace for serving the
project's foundation models on one RTX 5090. It has no MuJoCo dependency and
can be copied directly to the GPU machine.

For the current native Qwen3.5 setup using Fish on the server and an SSH tunnel
from a new client PC, follow [NEW_PC_SETUP.md](NEW_PC_SETUP.md).

Use vLLM by default. SGLang is an explicit fallback for a model/version
combination that behaves better there. Only run one model at a time: the 5090
has 32 GB VRAM, and the server needs space for image features and KV cache as
well as weights.

The workspace starts two processes:

```text
scene images + goal -> functional API :8080 -> vLLM/SGLang :8000
                 ranked type priors <- validated JSON <- VLM
```

The functional layer never imports MuJoCo. It sends only the supplied images,
goal, and small function catalog to the VLM. It does not search, claim that a
candidate is present, run geometry, sequence actions, or execute anything.

For native SSH-tunnel testing, both processes may run without API keys only
when bound to `127.0.0.1`. The planner refuses keyless startup on a non-loopback
address. Docker deployments retain authenticated network-facing defaults.

## Models

| Profile | Checkpoint | Load mode | Planner mode | Output budget |
|---|---|---:|---:|---:|
| `qwen35-9b` | `Qwen/Qwen3.5-9B` | BF16 | thinking, toggleable | 24K |
| `glm46v-flash` | `zai-org/GLM-4.6V-Flash` | BF16 | thinking, toggleable | 12K |
| `qwen3-vl-8b-thinking` | `Qwen/Qwen3-VL-8B-Thinking` | BF16 | fixed thinking | 12K |
| `internvl35-14b` | `OpenGVLab/InternVL3_5-14B-HF` | online FP8 | prompt-driven thinking | 12K |
| `kimi-vl-a3b-thinking` | `moonshotai/Kimi-VL-A3B-Thinking-2506` | online FP8 | fixed thinking | 12K |

The bounded contexts are intentional single-GPU defaults, not each model's
advertised maximum. Profiles accept up to eight images per prompt, covering
the simulator's seven-camera observation. InternVL and Kimi use load-time FP8
because their BF16 weights leave too little VRAM for practical multimodal
inference on a 32 GB card.

The registry also owns each model's planner settings. Qwen3.5 uses the Qwen3
reasoning parser and its recommended thinking sampler. GLM uses the `glm45`
reasoning parser, its `/nothink` toggle, and the checkpoint's recommended
sampling family. The dedicated Qwen3-VL Thinking checkpoint cannot be changed
into its Instruct counterpart with a request flag. InternVL uses its documented
prompt-driven thinking mode with temperature 0.6 and top-p 0.95. The gateway
strips both InternVL's and Kimi's reasoning markers before validating the final
answer.

InternVL and Kimi use experimental prompt-constrained functional JSON. Their
reasoning modes are not connected to dedicated vLLM parsers here, so each
request includes the schema in the prompt and the gateway validates the
returned final JSON instead of applying vLLM's constrained JSON decoder. Their
raw endpoints remain usable even if a particular response fails the functional
contract.

The thinking sampling defaults follow the checkpoint authors' recipes:

- Qwen3.5: temperature 1.0, top-p 0.95, top-k 20, min-p 0, presence
  penalty 1.5, and repetition penalty 1.0;
- GLM-4.6V-Flash: temperature 0.8, top-p 0.6, top-k 2, and repetition
  penalty 1.1;
- Qwen3-VL Thinking for visual inputs: temperature 1.0, top-p 0.95, top-k 20,
  presence penalty 0, and repetition penalty 1.0;
- InternVL3.5 thinking: temperature 0.6 and top-p 0.95; and
- Kimi-VL 2506: temperature 0.6 from the checkpoint-specific bundled
  generation configuration. The older general Kimi-VL Thinking guidance says
  0.8, but the selected 2506 checkpoint's own configuration is more specific.

Qwen3.5 uses a 24K planner output cap because its thinking mode repeatedly
reached the former 12K limit before emitting final JSON. The other 12K caps and
Qwen's 24K cap are deployment bounds for the 32 GB GPU rather than
creator-recommended maximum generation lengths.

`muse-glimmer` is a disabled registry placeholder until Meta publishes a
verified local checkpoint and serving recipe. Muse Spark is available through
Meta's hosted API and cannot be run locally in this workspace.

## Functional decomposition contract

The exact system prompt is the editable text file
[`prompts/functional_decomposition.txt`](prompts/functional_decomposition.txt).
`functional_planner.py` loads it when the functional API starts, so restart
`planner_api.py` after editing the prompt. The user message is assembled at
request time from the goal, camera labels/images, function catalog, ranking
limits, forbidden generic types, and strict output schema.

`functional_catalog.json` mirrors the repository's simple function registry:
`can_store`, `can_stir`, `can_hold_liquid`, `can_clean`, and `can_spread`.
Functions identify replaceable roles, not low-level robot actions. Each has an
  `object` or `region` candidate kind.

The VLM returns only:

- the applicable functional requirements;
- ten to fifteen concrete candidate types for each requirement, ordered by normal,
  safe semantic suitability; and
- dependencies between functional requirements when needed.

The ranked types are hypotheses such as `teaspoon`, `coffee stirrer`, and
`chopstick`. Generic umbrella labels such as `utensil`, `tool`, `container`,
and `object` are forbidden because the semantic search needs concrete category
queries. The types are deliberately not observation-bounded detections. The VLM
must not say that any proposed type is present, visible, graspable, or feasible.
The downstream search inspects regions, grounds observed instances, and applies
the deterministic task-specific point-cloud predicates such as `OPEN_CAVITY`,
`INSERTABLE_IN`, and `REACHES_BOTTOM`. First-feasible termination happens there,
not in this service.

The prompt, strict schema, and deterministic validator are in
`functional_planner.py`. The validator rejects unknown functions, a wrong
candidate kind, fewer than ten or more than fifteen alternatives, generic or
duplicate types, invalid
dependencies, and dependency cycles. `functional_client.py` is the matching
CLI. Kitchen, living-room, and workshop scene labels are accepted.

Qwen3.5 and GLM thinking are enabled by default through model-specific chat
template settings. Set `PLANNER_ENABLE_THINKING=false` for their low-latency
ablation. Qwen3-VL Thinking, InternVL, and Kimi Thinking reject that override
because their current profiles are fixed to thinking. If a response
exhausts its budget in `reasoning_content` and has no final `content`, increase
`PLANNER_MAX_TOKENS`; for a toggleable model, disabling thinking is another
option.

Thinking requests use Qwen3.5's recommended sampling family rather than greedy
decoding: temperature 1.0, top-p 0.95, top-k 20, min-p 0, presence penalty 1.5,
and repetition penalty 1.0. Non-thinking requests use temperature 0.7 and
top-p 0.8 with the same remaining values. Greedy decoding caused Qwen to spend
every available token in reasoning without emitting final JSON.

## GPU host setup

Install the NVIDIA driver, Docker Engine, and NVIDIA Container Toolkit on the
server. Verify container GPU access before copying model data:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

Copy only this directory from the development computer:

```bash
rsync -av --delete \
  --exclude .env --exclude cache/ --exclude models/ --exclude .venv/ \
  inference_server/ user@gpu-server:~/tamp-inference/
ssh user@gpu-server
cd ~/tamp-inference
cp .env.example .env
```

The exclusions preserve the server's API key and downloaded weights on later
syncs. Do not remove them while using `--delete`.

Set a random `INFERENCE_API_KEY` in `.env`. Set `HF_TOKEN` there only if a
checkpoint requires authenticated access. Cache and weights remain in host
directories, so replacing a serving container does not download them again.

```bash
./serve doctor
./serve list
./serve up qwen35-9b --detach
./serve logs --backend vllm --follow
```

`./serve up` starts both the selected model backend on port 8000 and the
functional API on port 8080. The API uses the same key unless a separate
`PLANNER_API_KEY` is set.

Switch models by stopping the current server and starting another profile:

```bash
./serve down
./serve up internvl35-14b --detach
```

For the current native, keyless SSH-tunnel setup, the same registry-backed
launcher avoids copying model-specific vLLM commands. In Fish on the GPU host:

```fish
cd ~/SearchTAMP
source .venv-qwen35/bin/activate.fish

set -x INFERENCE_MODEL glm46v-flash
set -x INFERENCE_HOST 127.0.0.1
set -x INFERENCE_CONTAINER_PORT 8000
set -e INFERENCE_API_KEY

python3 inference_server/server.py
```

Replace `glm46v-flash` with any profile in the table. Stop the current model
before starting the next one. vLLM downloads the selected public checkpoint on
first launch and reuses the Hugging Face cache thereafter. In the second server
shell, set the same `INFERENCE_MODEL`, erase explicit
`PLANNER_MODEL`, `PLANNER_ENABLE_THINKING`, and `PLANNER_MAX_TOKENS` values, and
restart `planner_api.py`; it will then use the selected profile's served name,
sampling, thinking mode, and output budget.

Use SGLang for a targeted compatibility comparison:

```bash
./serve down
./serve up qwen35-9b --backend sglang --detach
./serve logs --backend sglang --follow
```

Qwen3.5 is new enough that its model card may require a newer vLLM nightly
than the stable image pinned in `.env.example`. If the log reports an unknown
architecture, set `VLLM_IMAGE` to the current Qwen-recommended vLLM image, or
use the SGLang profile. Keep the image reference pinned after it passes your
experiment smoke test.

## Test the raw model endpoint

Load the API key into the current shell, then test text or one or more images:

```bash
set -a
source .env
set +a
python3 smoke_test.py --prompt "Reply with the word ready."
python3 smoke_test.py --image view_1.png --image view_2.png
```

## Request functional requirements

Send between one and eight camera images. Camera names are evidence labels;
they do not reveal simulator state.

```bash
python3 functional_client.py \
  --scene kitchen \
  --goal "Make coffee, stir it, and serve it" \
  --image front_camera=/path/front.png \
  --image overhead_camera=/path/overhead.png
```

For a native keyless Qwen server on the same machine, start the functional API
in a second shell:

```bash
export INFERENCE_MODEL=qwen35-9b
export PLANNER_MODEL=qwen35-9b
export PLANNER_MODEL_BASE_URL=http://127.0.0.1:8000/v1
export PLANNER_HOST=127.0.0.1
export PLANNER_PORT=8080
export PLANNER_ENABLE_THINKING=true
export PLANNER_MAX_TOKENS=24576
unset INFERENCE_API_KEY PLANNER_API_KEY
python3 planner_api.py
```

Tunnel ports 8000 and 8080 with SSH. `functional_client.py` and `smoke_test.py`
omit the Authorization header when no key is configured.

The corresponding `POST /v1/decompose` body is:

```json
{
  "scene": "kitchen",
  "goal": "Make coffee, stir it, and serve it",
  "images": [
    {
      "camera": "front_camera",
      "data_url": "data:image/png;base64,..."
    }
  ]
}
```

The relevant part of a typical coffee response is:

```json
{
  "status": "DECOMPOSED",
  "scene": "kitchen",
  "functional_requirements": [
    {
      "id": "req_1",
      "function": "can_hold_liquid",
      "candidate_kind": "object",
      "purpose": "Contain the prepared coffee.",
      "target_description": "The prepared coffee.",
      "ranked_candidate_types": [
        "coffee mug",
        "ceramic cup",
        "glass tumbler",
        "travel mug",
        "teacup",
        "insulated cup",
        "enamel cup",
        "demitasse cup",
        "measuring cup",
        "drinking glass"
      ],
      "depends_on": []
    },
    {
      "id": "req_2",
      "function": "can_stir",
      "candidate_kind": "object",
      "purpose": "Stir the coffee contents.",
      "target_description": "The selected liquid container.",
      "ranked_candidate_types": [
        "teaspoon",
        "coffee stirrer",
        "chopstick",
        "swizzle stick",
        "small whisk",
        "bar spoon",
        "wooden stir stick",
        "cocktail spoon",
        "stirring rod",
        "silicone spatula"
      ],
      "depends_on": ["req_1"]
    }
  ]
}
```

Exact wording and ordering are model outputs. The downstream system treats the
list order as a semantic prior only; observed first-feasible candidates still
have to pass the configured geometry checks.

A successful response contains a decomposition and ranked candidate types. The
envelope explicitly reports `search_started: false`,
`semantic_grounding_complete: false`, `geometry_verified: false`, and
`execution_started: false`. The API does not invoke perception, point-cloud
checks, navigation, IK, PDDLStream, motion planning, or simulator actions.

Inspect the catalogs without contacting the VLM:

```bash
curl -fsS http://GPU_SERVER:8080/v1/functions \
  -H "Authorization: Bearer $INFERENCE_API_KEY"
```

On the simulator machine, use the same served profile name and key:

```bash
export TAMP_FM_BASE_URL=http://GPU_SERVER:8000/v1
export TAMP_FM_MODEL=qwen35-9b
export TAMP_FM_API_KEY="$INFERENCE_API_KEY"
export PLANNER_BASE_URL=http://GPU_SERVER:8080/v1
export PLANNER_API_KEY="$INFERENCE_API_KEY"
```

Keep ports 8000 and 8080 on a trusted private network. Use loopback bindings
plus an SSH tunnel for keyless native testing. Use authentication and a TLS
reverse proxy for any network-facing deployment.

## Overrides and diagnostics

The common overrides in `.env` are `INFERENCE_MAX_MODEL_LEN`,
`INFERENCE_MAX_CONCURRENCY`, `INFERENCE_GPU_MEMORY_UTILIZATION`, and
`CUDA_VISIBLE_DEVICES`. Reduce context first if model startup runs out of VRAM.
Do not raise concurrency until the full seven-image request succeeds.

Inspect a generated backend command without exposing its API key:

```bash
./serve command kimi-vl-a3b-thinking
```

The native `uv` requirements remain only as a fallback. Docker is the supported
path for avoiding CUDA, PyTorch, Transformers, and model-specific dependency
conflicts on the 5090 server.
