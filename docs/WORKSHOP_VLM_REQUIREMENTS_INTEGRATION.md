# Workshop image-conditioned Qwen requirements

Workshop uses the same answer-free multimodal request described in
[`KITCHEN_LIVING_ROOM_VLM_REQUIREMENTS_INTEGRATION.md`](KITCHEN_LIVING_ROOM_VLM_REQUIREMENTS_INTEGRATION.md).

Qwen receives only the Workshop goal and raw initial-observation RGB images.
It decides the roles, object/region kind, counts, functions, properties, and
visible candidate objects. The request contains no `role_envelopes`, expected
driver/fastener list, relation names, or internal predicates.
The complete request text and schema are in
[`QWEN_VLM_EXACT_PROMPT.txt`](QWEN_VLM_EXACT_PROMPT.txt).

The reviewed Workshop ontology is consulted only after the response. Raw output
is always retained. If it cannot be normalized safely, the result reports
`ready_for_grounding: false` and records the error under
`reviewed_ontology_audit`; no action planner or execution is started.

After starting Qwen and the SSH tunnel as described in the shared runbook, run:

```bash
cd /home/naren/RA_iiith
export TAMP_FM_BASE_URL=http://127.0.0.1:18000/v1
export TAMP_FM_MODEL=qwen35-9b

.venv/bin/python -m mujoco_scenes.run_workshop_vlm_requirements \
  --image outputs/workshop_yoloworld_l_five_view_close/final_14_bright_profile_frozen_v3/F4_OBJECT_REGION_COUPLING/representative_visuals/stage_000_initial_workbench_front.jpg \
  --image outputs/workshop_yoloworld_l_five_view_close/final_14_bright_profile_frozen_v3/F4_OBJECT_REGION_COUPLING/representative_visuals/stage_000_initial_workbench_left.jpg \
  --image outputs/workshop_yoloworld_l_five_view_close/final_14_bright_profile_frozen_v3/F4_OBJECT_REGION_COUPLING/representative_visuals/stage_000_initial_workbench_right.jpg \
  --output outputs/workshop_vlm_requirements.json
```

Review the raw decision first:

```bash
jq '.raw_vlm_decomposition.functional_requirements' \
  outputs/workshop_vlm_requirements.json

jq '{ready_for_grounding, reviewed_ontology_audit,
     planning_started, execution_started}' \
  outputs/workshop_vlm_requirements.json
```

Use only raw RGB observations. Do not supply annotated detector images,
functional graphs, expected-witness visualizations, or filenames rendered into
the image that disclose answers.

The current integration is requirements-only. The earlier optional
grounding configuration was removed because it initialized Qwen before a real
initial image existed, which violated the corrected multimodal boundary.
