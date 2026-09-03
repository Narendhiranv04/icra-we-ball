# Discovery-based replanning

`mujoco_scenes.run_kitchen_discovery_replanning` and
`mujoco_scenes.run_living_room_discovery_replanning` are the live MuJoCo port
of the maintained discovery-based replanning framework. It is a separate
method from VLM-TAMP, OWL-TAMP, and the functional-search framework.

At the start and after every physical skill, it captures the selected MuJoCo
RGB camera views and a bounded observed state. Object visibility is filtered
by screen-space instance-segmentation evidence from the selected camera views,
matching the original CoppeliaSim framework's segmentation/tracking boundary;
new detections are added to a persistent tracked state rather than forgotten
when the camera moves. The simulator's private inventory is not sent to the VLM. The VLM receives only that current
observation, the natural-language goal, completed actions, and a structured
failure/discovery event when one occurred. It never receives simulator body
names, the private goal contract, or a ground-truth plan.

The runner replans when a skill fails recoverably, a previously unseen object
becomes visible, or the physical goal verifier says the plan ended incomplete.
Discovery is deferred while the robot is holding an object, preventing a
mid-carry interruption. Every pending primitive is checked against the newest
bounded state immediately before dispatch, so stale actions are replanned
without physical execution. `INSPECT` is available in the action vocabulary
but is not prescribed by the prompt.

The runner creates a fresh controlled Kitchen variant directly. It does not
consume a functional-search Phase-1 witness. A private expected-outcome file is
used only by the final evaluator; its actions and object bindings never enter
the VLM request.

Check the tunneled model server, then run a live episode with a fresh output
directory:

```bash
curl --max-time 5 http://127.0.0.1:18000/v1/models

MUJOCO_GL=glfw .venv/bin/python -m mujoco_scenes.run_kitchen_discovery_replanning \
  --variant K2 \
  --output-dir runs/discovery_replanning/kitchen_001 \
  --goal "Prepare and serve coffee and soup for two people using the available kitchenware. Stir both coffees and provide each soup bowl with a suitable utensil." \
  --base-url http://127.0.0.1:18000/v1 \
  --model qwen35-9b \
  --camera-count 5 \
  --max-replans 8 \
  --max-tokens 8192 \
  --no-thinking
```

The output directory contains `discovery_replanning_result.json` and a
JSONL event trace. `model_calls/` contains the exact text prompts, camera names,
responses, latency, and errors without copying image payloads or private GT.
The terminal prints every accepted plan, skill start,
physical result, discovery, and replan. K1-K6 are feasible execution variants;
K2-K6 exercise discovery from closed storage.

For a directly comparable single-VLM-call condition, add
`--max-model-calls 1`. The initial plan is still physically executed, but a
new discovery, recoverable failure, or incomplete final goal terminates the
episode instead of requesting another model response. The result records both
`model_calls` and `replans`, so this condition cannot be confused with the
normal discovery-replanning method.

## Living Room

The Living Room runner uses the same five fixed cameras, persistent anonymous
IDs, live VLM replanning, and prompt tracing. L1-L6 are feasible variants. The
current benchmark has no closed or hidden Living Room regions, so its action
vocabulary correctly contains only `PICK` and `PLACE`; the model is not told to
inspect.

```bash
RUN_ID=$(date +%Y%m%d_%H%M%S)

MUJOCO_GL=glfw MUJOCO_IK_BACKEND=legacy .venv/bin/python -m \
  mujoco_scenes.run_living_room_discovery_replanning \
  --variant L1 \
  --output-dir "runs/discovery_replanning/living_room_l1_${RUN_ID}" \
  --goal "Prepare the living room for two people watching television. Place one cup and one saucer on each person's fixed individual side table, and place the TV remote on the fixed shared coffee table." \
  --base-url http://127.0.0.1:18000/v1 \
  --model qwen35-9b \
  --camera-count 5 \
  --max-replans 5 \
  --max-tokens 8192 \
  --no-thinking
```

Living Room `PICK` and `PLACE` each select a collision-checked mobile-base
stance. Placement is physically verified for support contact, no floor contact,
bounded support footprint, non-overlap, release, and settling before the
observed state is updated.

Persistent anonymous IDs are maintained with MuJoCo instance segmentation, as
in the original simulator-based framework. This is the framework's observation
source, not a semantic-grounding baseline or a claim about learned detection.
