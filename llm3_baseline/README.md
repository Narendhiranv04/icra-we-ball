# LLM3-style baseline

This folder provides a zero-training action-planning baseline inspired by
LLM3. It uses a frozen VLM through the repository's existing OpenAI-compatible
vLLM or SGLang server. It does not fine-tune a model, train an affordance
function, or include the original project's PyBullet environment.

The baseline differs from the functional-search pipeline as follows:

```text
functional-search: images -> functions/types -> search -> geometry -> sequence
LLM3-style:       images -> action sequence -> execute -> return failure -> replan
```

Both methods must use the same visible observations, semantic grounder,
point-cloud predicates, motion planner, controllers, and goal verifier.

## Current boundary

The implemented pieces are:

- strict per-scene action catalogues;
- a visible-state-only observation contract;
- multimodal prompting through an OpenAI-compatible endpoint;
- strict structured action output and reference validation;
- a bounded execution loop that re-prompts after recoverable failures;
- independent goal verification; and
- tests using a deterministic fake executor.

The folder does not yet connect actions to the live MuJoCo dispatcher. It also
does not ask the VLM to generate raw continuous poses. The eventual simulator
adapter should expose the same finite grasp/placement samples to every method,
then report IK, collision, grasp, placement, and effect failures through the
implemented `LLM3Executive` interface. Until that adapter exists, results are
planning-only and must not be reported as end-to-end TAMP execution.

## Visible-state contract

The observation JSON contains only:

- currently visible image-derived object instances with persistent IDs;
- known region IDs and whether each has been inspected;
- observed robot state; and
- the independently verified goal flag.

Do not put hidden simulator inventory, hidden region contents, MuJoCo body
names, privileged poses, or oracle geometry in this file. `INSPECT` may refer
to a known uninspected region. All other object references must be present in
`visible_entities`; invalid model output is rejected before execution.

See `example_observation.json` for the input shape.

## Start the model server

Use the existing inference workspace on the GPU server. For the current Qwen
profile:

```bash
cd inference_server
./serve up qwen35-9b --detach
```

If the model server is remote, tunnel its raw OpenAI-compatible port:

```bash
ssh -L 18000:127.0.0.1:8000 user@gpu-server
```

The baseline talks directly to the raw model endpoint on port 8000; it does
not use the functional `/v1/decompose` API.

## Request one plan

From the repository root:

```bash
export LLM3_MODEL_BASE_URL=http://127.0.0.1:18000/v1
export LLM3_PROFILE=qwen35-9b

python3 -m llm3_baseline.client \
  --goal "Stir the contents of the visible mug" \
  --observation llm3_baseline/example_observation.json \
  --image front_camera=runs/qwen_kitchen_test/front_camera.png \
  --image overhead_camera=runs/qwen_kitchen_test/overhead_camera.png
```

The output is a validated plan only. `execution_started` remains `false`.

Optional environment variables:

| Variable | Default |
|---|---:|
| `LLM3_MODEL_BASE_URL` | `http://127.0.0.1:8000/v1` |
| `LLM3_PROFILE` | `qwen35-9b` |
| `LLM3_MODEL` | selected profile's served name |
| `LLM3_API_KEY` | empty |
| `LLM3_MAX_TOKENS` | selected profile's configured limit |
| `LLM3_MAX_ACTIONS` | `20` |
| `LLM3_TIMEOUT_SECONDS` | `300` |
| `LLM3_ENABLE_THINKING` | `true` |
| `LLM3_TEMPERATURE` | selected profile's recommended value |
| `LLM3_TOP_P` | selected profile's recommended value |

By default the client reads the selected model's thinking mode, structured
output support, token limit, reasoning markers, and creator-recommended
sampling settings from `inference_server/models.json`. Temperature and top-p
overrides exist for controlled ablations. These are runtime settings, not
learned parameters; freeze and record them for a benchmark.

## Failure feedback

The live adapter should translate the shared skill result into a concise
failure such as:

```json
{
  "code": "ik_failed",
  "message": "No collision-free IK solution was found."
}
```

The baseline accepts these failure classes without model training:

- `precondition_failed`
- `path_blocked`
- `ik_failed`
- `collision`
- `grasp_failed`
- `placement_failed`
- `object_not_visible`
- `target_occupied`
- `function_unsatisfied`
- `effect_not_observed`

Avoid including hidden object locations or the correct solution in a failure
message.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  llm3_baseline/tests -q
```

For paper results, describe this as an **LLM3-style adaptation with a frozen
VLM**, not an exact reproduction. The original LLM3 system used GPT-4,
PyBullet box-packing environments, and its own motion-planning interface.
