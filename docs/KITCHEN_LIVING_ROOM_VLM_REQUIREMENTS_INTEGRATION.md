# Kitchen VLM functional-graph pipeline

## Authoritative integrated Kitchen path

Run:

```bash
.venv/bin/python -m mujoco_scenes.run_kitchen_vlm_pipeline --variant K1
```

This is separate from the GT execution script. It stops after producing the
grounded symbolic action sequence; it does not execute robot motions.

The Kitchen pipeline performs exactly one Qwen inference. Qwen receives:

- the natural-language task;
- fresh raw RGB views of the closed initial scene;
- the fixed JSON response schema;
- the generic verifier API (`OPEN_CAVITY`, `ELONGATED_OBJECT`,
  `INSERTABLE_IN`, and `REACHES_BOTTOM`);
- the inspectable closed-region handles and their visible descriptions, which
  are an action/perception interface, not their contents.

It does not receive the intended feasibility label, configured hidden
contents, simulator object identities or poses, point clouds, measured
geometry, an observed scene graph, a reviewed Kitchen role contract, a GT role
assignment, or a GT action sequence.

## What Qwen must decide in that one response

Qwen emits the complete functional requirement graph:

- functional roles, functions, and required counts;
- distinct/reusable binding policies;
- candidate semantic categories and YOLO-World detector phrases;
- exact unary checker predicates and numeric property constraints for every role;
- exact directed binary checker predicates between roles;
- operation groups, target counts, and within/cross-group reuse policy;
- material-source roles (in the same verified role graph) and target-content
  requirements needed by planning;
- candidate storage regions, a complete inspection order, and an initial
  visual satisfaction assessment.

`INSERTABLE_IN` is selected and output by Qwen as a directed binary predicate.
No downstream natural-language alias is converted into `INSERTABLE_IN`.
Likewise, downstream code does not add missing roles, counts, properties,
relations, or planning requirements. An unsupported checker name or an
internally inconsistent graph causes an explicit validation failure.

The checker list is a fixed software capability interface. It tells Qwen which
exact predicates the deterministic geometry layer can execute; it does not
tell Qwen which predicates the task requires. Selecting the applicable subset
and its role endpoints is Qwen's responsibility.

## Downstream processing

After Qwen returns, deterministic code only:

1. validates the graph structure, references, counts, and exact checker IDs;
2. builds one YOLO-World vocabulary directly from Qwen's candidate categories and
   detector phrases;
3. grows the observed scene graph from actual RGB/depth evidence;
4. runs the selected unary and binary geometric checks;
5. jointly assigns perceived instances to Qwen's roles;
6. opens regions in Qwen's order while satisfaction remains incomplete;
7. stops immediately when the role graph and required source roles are jointly
   satisfied, or exhausts the order;
8. converts the verified grounding to a symbolic Kitchen planning problem;
9. uses deterministic A* search over generic Kitchen operators to determine
   the task-action order.

Inspection is evidence acquisition and is kept outside the manipulation
planner. The combined comparison sequence nevertheless records the `OPEN`
events first, followed by the planner-generated task actions.

## Audit artifacts

Each run writes:

```text
outputs/kitchen_vlm_pipeline/K1/
  initial_observation/*.png
  01_raw_vlm_functional_graph.json
  02_vlm_graph_to_task_contract.json
  vlm_task_requirements.yaml
  vlm_object_detector_vocabulary.yaml
  observed_search/phase1/...
  action_sequence/generated_plan.json
  action_sequence/grounded_role_assignments.json
  03_grounding_to_action_mapping.json
  ordered_action_sequence.json
  ordered_action_sequence.txt
  pipeline_result.json
```

- `01_raw_vlm_functional_graph.json` is exactly the decoded one-call Qwen
  response.
- `02_vlm_graph_to_task_contract.json` records the structural conversion,
  exact checker dispatch, generated detector vocabularies, and confirms
  `added_task_requirements: []`.
- `03_grounding_to_action_mapping.json` records the verified witness, physical
  role assignments, planner provenance, and role binding behind each action.
- `pipeline_result.json` records `fm_calls: 1` and the no-GT boundary flags.

Online observed-state recording uses `record_oracle_diagnostics=False`, so
`oracle_source_region` is not written into this path's object registry.

## Requirements-only legacy path

`run_environment_vlm_requirements` remains a separate prompt-development tool
for Kitchen and Living Room. Its natural-language response and reviewed
post-response audit are not used by `run_kitchen_vlm_pipeline`. The integrated
Kitchen runner imports neither `EnvironmentVLMRequirementProvider` nor the
normalization YAML/reviewed Kitchen contract.

## Qwen server

Remote server and local tunnel:

```bash
ssh -i ~/keyfile long-horizon@gvlab2.iiit.ac.in
cd ~/SearchTAMP
source .venv-qwen35/bin/activate.fish
set -x INFERENCE_MODEL qwen35-9b
set -x INFERENCE_HOST 127.0.0.1
set -x INFERENCE_CONTAINER_PORT 8000
set -e INFERENCE_API_KEY
python3 inference_server/server.py
```

```bash
ssh -i ~/keyfile -N -L 18000:127.0.0.1:8000 \
  long-horizon@gvlab2.iiit.ac.in
```

```bash
export TAMP_FM_BASE_URL=http://127.0.0.1:18000/v1
export TAMP_FM_MODEL=qwen35-9b
unset TAMP_FM_API_KEY
```

## Regression checks

```bash
.venv/bin/python -m pytest -q \
  mujoco_scenes/tests/test_kitchen_vlm_functional_graph.py \
  mujoco_scenes/tests/test_environment_vlm_requirements.py \
  mujoco_scenes/tests/test_symbolic_kitchen_planning.py
```
