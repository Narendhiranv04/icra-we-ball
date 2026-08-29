# Phase 3 Interactive Pipeline + Phase 4 Handoff Plan

Repository audit basis: branch `naren/pipeline_check`, commit
`fa4b9527ca3c2a790a51d7f3fae3968c7973d6c7`. The worktree was not clean during
the audit: `mujoco_scenes/functional_tamp_pipeline/tests/test_evaluator_harness.py`
already had a user-owned modification. That file was not inspected as a proposed
change and must not be overwritten.

This is an implementation plan, not an authorization to change the frozen
perception, grounding, symbolic compilation, A*, replay, or execution logic.

## 1. Audit Verdict on Opus Plan

The Opus proposal is approved with substantial revisions, not as-is.

Confirmed:

- `mujoco_scenes/functional_tamp_pipeline/run.py::run_pipeline()` is the canonical
  integrated entry point.
- `FunctionalSpecProvider` is already the GT/VLM boundary, and all three domains
  already dispatch through `provider_for_mode()`.
- A callback shaped as `Callable[[str, dict[str, Any]], None]` is sufficient; no
  event bus is warranted.
- Workshop uses `search.py::search_until_satisfied()` and consumes
  `FunctionalRequirementGraph.region_ranking`.
- Living Room has no hidden-region loop. It performs one initial multi-view
  observation followed by global grounding and planning.
- `FunctionalRequirementGraph` is a frozen dataclass and could technically be
  copied with `dataclasses.replace()`.
- A small run manifest is missing and useful.
- The evaluator should continue to launch the canonical CLI in subprocesses.

Needs revision:

- Do not add `interactive.py` or `__main__.py`. Extend `run.py` and keep one CLI,
  one parser, and one runtime path.
- Do not mutate/copy G_F to change search order. Preserve the provider graph as
  emitted and pass a separate resolved `search_order` into the two search paths.
  This keeps specification source and search-order source scientifically
  independent.
- Use `provider` and `fixed` as the CLI terms, not `functional_ranked`. The GT
  order is manually authored, not FM-generated.
- Kitchen telemetry belongs in `sequential_inspection.py` and
  `domains/kitchen.py`, not `search.py`.
- The proposed OpenCV/matplotlib GUI thread is unsuitable. The repository states
  that its OpenCV build has no GUI support, matplotlib is not part of the current
  runtime, and MuJoCo scene access from a display thread is unsafe.
- The observer must receive camera snapshots captured or already saved by the
  pipeline thread. It must never own a scene reference or invoke `render_frame()`.
- `graph_grounding_result.json` is not a sufficient universal Phase 4 handoff.
  Kitchen plans contain opaque observed IDs, Living Room execution uses its
  payload/region registries and performs a separate entity-resolution step, and
  Workshop already translates observed tracks to physical handles.
- Observer failures must be reported to stderr and in `run_manifest.json`, while
  remaining non-fatal to planning.

Removed as unnecessary:

- `telemetry.py`, a `TelemetrySink` hierarchy, and a second JSONL logger.
- `interactive.py`, `__main__.py`, and `--pause-after-spec`.
- NetworkX/matplotlib as live-window dependencies.
- Search-policy handling in Living Room.

Verify during implementation:

- Exact Kitchen stage camera artifact selected for the live tile.
- ffplay window behavior in the user's desktop/SSH setup.
- Live VLM server availability and model compatibility.
- The exact Kitchen observed-ID-to-body resolver used in Phase 4.

## 2. Current Repository Architecture

`mujoco_scenes/functional_tamp_pipeline/run.py::run_pipeline()` resolves the
paper variant, creates `<output_root>/<domain>/<variant>/<mode>`, obtains G_F,
persists it, and dispatches by domain.

- Kitchen: `domains/kitchen.py::run_to_plan()` compiles a detector/task contract
  from G_F, then calls `sequential_inspection.py::run_sequential_inspection()`.
  `run_fixed_order_inspection()` performs the closed initial observation, calls
  `SequentialInspectionAdapter.inspect()` for each region, performs fresh
  perception, and invokes the Kitchen completion predicate. Only after that loop
  does `run_to_plan()` build canonical G_O, call `ground_graph()`, compile, run
  common A*, audit, and persist the final action sequence. `search.py` is not on
  this path.
- Workshop: `run.py` constructs `WorkshopDomainAdapter`, then calls
  `search.py::search_until_satisfied()`. The shared loop calls
  `observe_initial()`, `evaluate_satisfaction()`, and, until complete,
  `open_region()`, `observe_after_open()`, and grounding again. Exploratory OPEN
  uses the existing `WorkshopExecutionDispatcher` unless `--dry-run` requests
  direct articulation. Planning happens only after the search returns COMPLETE.
- Living Room: `domains/living_room.py::run_to_plan()` creates
  `IntegratedLivingRoomRegionRun`, whose inherited `run()` performs exactly one
  initial evidence capture, builds registries and compatibility, and globally
  allocates regions. There is no iterative OPEN or hidden-region selection.
- Specification: `spec_provider.py::provider_for_mode("gt" | "vlm")` returns
  `GTSpecProvider` or `VLMSpecProvider`. Both produce the same
  `FunctionalRequirementGraph` type, and the existing downstream dispatch does
  not branch on mode.
- Grounding and planning: `grounding.py::ground_graph()` produces phi* as a
  `GraphGroundingResult`; `planning.py::plan_with_common_astar()` produces the
  deterministic action sequence and independent replay data used by the domain.
- Evaluation: `scripts/evaluate_functional_tamp_variants.py` invokes
  `python -m mujoco_scenes.functional_tamp_pipeline.run` in a subprocess, then
  reads artifacts. It currently hard-codes `--mode gt --dry-run`.
- Display support: `mujoco_scenes/live_mosaic_viewer.py::LiveMosaicViewer` already
  streams RGB24 composites to an ffplay subprocess and explicitly documents why
  `cv2.imshow()` cannot be used in this environment.

## 3. Frozen Scientific Invariants

The implementation must preserve this dataflow:

```text
task + initial RGB -> GT/VLM G_F -> perception -> G_O
-> exploratory observation/OPEN where applicable -> growing G_O
-> graph grounding -> phi* -> symbolic compiler -> deterministic A*
-> independent replay -> persisted final action sequence
```

The following are no-change boundaries:

- No change to provider prompts, VLM normalization, detector vocabulary
  semantics, perception thresholds, tracking, geometry, grounding decisions,
  compiler semantics, A*, replay validation, audits, or benchmark truth.
- GT and VLM differ only before/at G_F production. The same search, grounding,
  compiler, planner, replay, and artifact code runs afterward.
- Search-order conditions may alter only the next-region order. They may not
  change candidates, stopping rules, OPEN mechanics, observations, grounding,
  or planning.
- Provider G_F must be persisted unchanged, including the provider's original
  `region_ranking`. The separately resolved order actually used must be recorded
  in the manifest.
- Exploratory OPEN operations are not copied into the final A* action list.
- The observer consumes snapshots and may not affect decisions, timing-dependent
  simulator state, or artifacts used as scientific evidence.
- Phase 3 ends at an independently replayed action sequence. Phase 4 execution
  consumes that sequence; execution failure is not grounding failure and must
  be stored separately.
- The default evaluator invocation remains the closed 32-variant GT/provider,
  direct-articulation sweep with the already validated expected statuses.

## 4. Final End-State Architecture

```text
 task + initial RGB
          |
          v
  +---------------------+       spec_mode = gt | vlm
  | FunctionalSpecProvider |<--------------------------+
  +----------+----------+                           |
             | unchanged provider G_F               |
             v                                      |
  +---------------------+       search_order_source |
  | resolve_search_order |<------ provider | fixed  |
  +----------+----------+       (N/A for Living Room)
             | G_F + separate order
             v
  +-----------------------------------------------------+
  | run.py::run_pipeline() -- the only canonical runner |
  |                                                     |
  | Kitchen: sequential_inspection                      |
  | Workshop: search_until_satisfied                    |
  | Living Room: one initial global observation         |
  |              -> G_O -> phi* -> compiler -> A*       |
  |              -> independent replay                  |
  +-------------+----------------------+----------------+
                |                      |
       immutable snapshots       canonical artifacts
                |                      |
                v                      v
  +---------------------------+  +----------------------+
  | optional callable observer|  | run_manifest.json    |
  | bounded queue             |  | G_F, G_O, phi*, plan |
  | Pillow/NumPy composite    |  | replay/audit         |
  | existing ffplay process  |  +----------+-----------+
  +---------------------------+             |
                                            | Phase 4 later
                                            v
                                  +----------------------+
                                  | domain execution     |
                                  | resolver + adapter   |
                                  | execution_result.json|
                                  +----------------------+
```

## 5. Canonical Run Configuration

The scientific condition is:

```text
(domain, variant, spec_mode, search_order_source, exploration_actuation)
```

- `domain`: `kitchen | living_room | workshop`.
- `variant`: paper label (`K1`...`K12`, `L1`...`L10`, `W1`...`W10`).
- `spec_mode`: `gt | vlm`.
- `search_order_source`: `provider | fixed`; effective value is
  `not_applicable` for Living Room.
- `exploration_actuation`: `direct_sim_articulation` for Kitchen,
  `robot_physical` or `direct_sim_articulation` for Workshop, and
  `not_applicable` for Living Room. This is recorded because it can affect the
  post-OPEN physical state.

`--visualize`, window size, display availability, and final-window hold behavior
are operational settings, not scientific condition identity. Phase 3 execution
state is always `planning_only`; a later Phase 4 run records execution separately.
For paired order comparisons, the G_F artifact SHA-256 is also a required pairing
key: both runs must consume the exact same graph bytes.

Every condition must use a fresh `--output-root`. The current directory layout
`<root>/<domain>/<variant>/<mode>` remains unchanged to protect evaluator and
artifact compatibility. Distinct provider/fixed comparisons therefore use
distinct output roots; they must never be run into the same directory.

## 6. Phase 3A — Evaluation Freeze

Phase 3A adds provenance and cheap contract tests before any GUI work. It does
not recalibrate the pipeline.

Canonical expected statuses remain:

- K1-K6, L1-L6, W1-W8: `ACTION_SEQUENCE_READY`.
- K7-K12, L7-L10, W9-W10: `INFEASIBLE`.

For a feasible run require `result.json`, G_F, G_O,
`graph_grounding_result.json`, the domain's final plan, and its existing replay
or validation artifact. Preserve Kitchen's grounding audit. For an infeasible
run require `result.json`, G_F, G_O, and an incomplete/infeasible grounding
artifact; absence of a plan is expected.

Add `run_manifest.json` for both successful and failed invocations. Do not copy
plan, graph, or diagnostics into it.

Flash may run unit tests and the compact smoke cases in Section 18. The user owns
the final 32-variant sweep. The freeze is accepted only when the user's fresh
default evaluator run is 32/32 exact, with all existing replay/audit rules still
passing.

## 7. Telemetry Design

Use one optional callback:

```python
EventCallback = Callable[[str, dict[str, Any]], None]
```

No base class, sink hierarchy, or new JSONL stream is needed. Existing Kitchen
and Living Room event logs remain authoritative domain diagnostics. Payloads
must contain snapshots (`to_dict()`, tuples/lists, copied RGB arrays, or artifact
paths), never live scene/domain objects.

Minimal events:

| Event | Required payload |
|---|---|
| `run_started` | domain, variant, spec_mode, requested order source, run directory |
| `stage_changed` | stage: `specification`, `perception`, `search_grounding`, `planning`, or `complete` |
| `spec_ready` | unchanged `graph`, provider source, provider ranking, resolved order |
| `observation_updated` | domain stage/index, G_O snapshot when available, inspected regions, copied `frame_rgb` or frame artifact path |
| `grounding_updated` | complete `GraphGroundingResult.to_dict()` snapshot |
| `search_region_selected` | region, zero-based rank index, total, order source |
| `search_region_opened` | region, success/result summary, exploratory flag `true` |
| `plan_ready` | final actions, planner statistics, replay/validation status |
| `run_finished` | terminal status, failure reason if any, artifact paths |

`run_failed` is emitted by the public wrapper only for an exception. Ordinary
scientific infeasibility is `run_finished(status="INFEASIBLE")`, not failure.

Emission points by domain:

Kitchen:

- `run.py`: run/stage/spec events after the provider returns.
- `run_fixed_order_inspection()`: initial `observation_updated` immediately after
  `observe("initial", None)`; selection immediately before `adapter.inspect()`;
  opened immediately after it returns; post-open observation immediately after
  the fresh `observe()` call.
- `domains/kitchen.py::kitchen_completion_predicate`: build the current G_O once,
  call existing `ground_graph()` once, emit both G_O and grounding snapshots,
  and return the existing `complete` Boolean. Do not perform an extra grounding
  call for telemetry.
- `domains/kitchen.py::run_to_plan()`: emit the authoritative final G_O/final
  grounding, then final plan/replay/audit status.

Living Room:

- Emit specification and `perception` stage before the single integrated run.
- Capture or copy one initial camera frame on the pipeline thread before the
  integrated perception call.
- After the integrated run, emit its final G_O and grounding. Do not emit search
  selection, OPEN, or search-order events.
- Emit the final plan and replay status.

Workshop:

- `search_until_satisfied()` emits initial observation/grounding, selection,
  OPEN result, post-open observation, and updated grounding around its existing
  calls. `WorkshopDomainAdapter` exposes a copied representative RGB view from
  `_capture_and_evaluate()` and a `graph.to_dict()` snapshot; it does not render
  again.
- `run.py` emits planning start and final plan/validation.

The callback passed into domain code is a guarded wrapper. A callback exception
is caught, summarized to stderr, appended to an in-memory observer-error list,
and planning continues. Repeated identical errors may be rate-limited in stderr
but must be counted in the manifest. Never silently swallow the first error.

## 8. Visualization Design

Use one composite RGB window rendered with Pillow/NumPy and displayed through
the existing `mujoco_scenes.live_mosaic_viewer.LiveMosaicViewer` ffplay process.
Do not use `cv2.imshow`, a matplotlib GUI backend, Qt, or a web server.

Lifecycle and concurrency:

1. `run.py::main()` constructs `LivePipelineVisualizer` only for `--visualize`.
2. The visualizer starts one worker thread and an ffplay subprocess. Its callback
   only snapshots the payload and performs non-blocking insertion into a bounded
   queue (size one or two). Superseded display frames may be dropped; scientific
   events may not be changed.
3. The worker updates display state, regenerates changed graph panels, composes
   the window, and writes RGB bytes to ffplay. ffplay, not a Python GUI toolkit,
   owns the native window/event loop.
4. After planning finishes, the final frame remains available until the user
   closes it or interrupts the command. This post-result hold is outside the
   recorded pipeline runtime. A non-interactive test path closes immediately.
5. If ffplay is missing, `$DISPLAY`/Wayland access is unavailable, or the window
   closes, record an observer error, warn, stop the worker, and allow planning to
   finish normally.

Composite panels:

- Camera: latest real RGB snapshot and camera/stage label.
- Status: run identity, current stage, terminal status, elapsed planning time,
  spec source, requested/effective order, and actuation mode.
- G_F: roles and required relations.
- G_O: discovered nodes grouped by source region and relevant relations.
- Search: resolved order, selected/opened regions, and current unresolved roles,
  relations, and constraints. Living Room shows `search: N/A`.
- Assignment: phi* and operation bindings when complete.
- Plan: exploratory trace clearly separated from final A* actions; final panel
  includes replay status and A* summary statistics.

Generate G_F only on `spec_ready`. Regenerate G_O only on
`observation_updated`. Text-only events redraw cheap panels without relaying out
graphs. The worker caps displayed node/edge counts and reports omitted counts.

Pillow graph rendering avoids another runtime dependency. G_F node labels show
role name, entity kind, required cardinality, binding policy, abbreviated
semantic categories, and required unary/numeric predicates. Edges show
`subject --predicate--> object`; operation groups are listed below rather than
rendered as a dense second edge type.

G_O node labels show generic instance ID, canonical category or `unknown`, entity
kind, source region, last-seen stage, and only predicates required by G_F.
Relations are filtered to predicates required by G_F/operation groups, with
assigned-node edges first and TRUE/UNKNOWN/FALSE visibly distinguished. Raw
geometry dictionaries and detector diagnostics stay out of the live graph.

## 9. Camera Frame Architecture

The observer must never receive a MuJoCo scene, model, data, renderer, domain
adapter, or controller.

`KitchenScene.render_frame()` calls `mj_forward()` and does not close its renderer;
Living Room also calls `mj_forward()`; all scene data are mutable and MuJoCo
render access is not safe to race against OPEN, perception, or future execution.
Therefore Opus's read-only scene-reference design is rejected.

Camera frames are produced only on the pipeline thread at safe boundaries:

- Kitchen: use a just-written stage RGB artifact from the existing virtual
  inspection capture. The event carries its path, and the worker loads it.
- Workshop: `_capture_and_evaluate()` already returns `ViewObservation` objects;
  copy one representative RGB array (or a small mosaic) before emitting.
- Living Room: render/copy one initial view on the same thread before the single
  perception run, then use `observed_grounding/observation/initial_scene_overview.png`
  when it becomes available.

In Phase 4, execution adapters likewise emit copied frames from their existing
step/recorder callbacks. The visualizer API does not change.

## 10. GT/VLM Mode

Already present:

- `provider_for_mode("gt")` returns reviewed static G_F for Kitchen, Living Room,
  and Workshop.
- `provider_for_mode("vlm")` returns one-shot VLM-generated G_F for all three.
- Kitchen and Workshop VLM modes also obtain a provider inspection order;
  Living Room has no region order.
- Initial RGB is persisted under `vlm_inputs/` when the user does not supply
  `--observation-image`.
- The generated graph, including most normalized/raw provider metadata, is
  persisted in the existing G_F artifacts.

The live backend is `FMAdapter` using an OpenAI-compatible `/chat/completions`
endpoint intended for Qwen on vLLM/SGLang. Configuration is resolved from:

```text
TAMP_FM_BASE_URL      fallback FM_BASE_URL
TAMP_FM_MODEL         fallback FM_MODEL
TAMP_FM_API_KEY       fallback FM_API_KEY (optional for an unauthenticated server)
TAMP_FM_TIMEOUT_SECONDS / FM_TIMEOUT_SECONDS (default 600)
TAMP_FM_MAX_TOKENS    / FM_MAX_TOKENS (default 4096)
```

The VLM receives the task, initial observation images, schemas/allowed ontology,
and generic candidate-region IDs/descriptions. The inspected repository path
does not pass variant labels, hidden contents, GT assignments, simulator body
names, or later observations into the VLM provider. Ontology normalization is a
reviewed boundary, not hidden scene truth.

Do not change prompts or providers. Add a CLI preflight error that names missing
`TAMP_FM_BASE_URL`/`TAMP_FM_MODEL`; never print the API key. Record the resolved
model name in the manifest and label G_F as GT or VLM in the UI. A server outage
is `run_failed`, not `INFEASIBLE`.

Add an optional provider-output replay boundary, `--specification-json PATH`.
It loads `FunctionalRequirementGraph.from_dict()`, validates the graph, requires
its domain and source to match the requested `--domain`/`--mode`, and then enters
the exact same downstream path used by a live provider. It is a cached provider
output, not a third specification mode. It must not accept a physical assignment,
G_O, or hidden scene data. This is necessary for a controlled provider-versus-
fixed comparison: invoking a stochastic VLM twice would otherwise change both
G_F and the order.

After G_F and search order are resolved, the same downstream functions must run
for GT and VLM. Tests should assert this dispatch property with mocked providers;
Flash must not make live remote calls.

## 11. Search-Order Policy (Revised Protocol: Three Scientific Regimes)

The scientific search protocol defines three real search regimes plus a mode-aware `auto` convenience resolution:

1. **`oracle`** (Privileged GT Minimum-Inspection Reference):
   Uses privileged GT variant configuration to place required hidden regions first, providing a minimum-inspection cost reference upper bound. Only valid in `gt` mode.
2. **`provider`** (FM/VLM-Guided Search):
   Uses the exact `specification.region_ranking` emitted by the specification provider without alteration.
3. **`random`** (Seeded Random Permutation Baseline):
   Deterministic seeded permutation of `specification.candidate_regions` using a local RNG (`random.Random(seed)`). Requires an explicit `--search-seed INT` (non-negative integer).
4. **`auto`** (Default Convenience Policy):
   Resolves `gt -> oracle` and `vlm -> provider`.
5. **`fixed`** (Deprecated Compatibility Alias):
   Resolves to `oracle`.

Living Room resolves search order to `()` and `not_applicable`. Explicit `oracle` or `random` on Living Room is rejected.

Search order is resolved with a pure helper that validates uniqueness and exact set equality against `specification.candidate_regions`.

The unchanged provider G_F is persisted. The manifest records `search_order_source_requested`, `search_order_source_effective`, `search_seed_requested`, `search_seed_effective`, `provider_region_ranking`, and `region_order_used`.

Scientific comparison protocol:

| Spec mode | Search policy | Effective source | Meaning |
|---|---|---|---|
| GT | auto / oracle | oracle | Privileged minimum-inspection GT reference |
| GT | provider | provider | GT specification with manual canonical region ranking |
| VLM | auto / provider | provider | VLM-generated G_F with VLM-produced ranking (FM-guided) |
| VLM | random (seed=N) | random | Paired comparison using the exact same saved VLM G_F with deterministic random seeds |

## 12. Interactive CLI

Extend the existing parser in `run.py`:

```text
--search-order {auto,oracle,provider,random,fixed}   default: auto
--search-seed INT                                    default: None (required for random)
--visualize                                          default: false
--specification-json PATH                            optional validated provider-output replay
```

Keep all existing flags and semantics. Do not add another runner.

For a controlled VLM order comparison, generate G_F once in the provider-order
run, then use that exact `functional_specification.json` for the fixed-order run.
The manifest must record identical specification hashes. Reusing the initial
images alone is insufficient because the live provider may be stochastic.

## 13. Run Manifest and Artifacts

The public `run_pipeline()` wrapper owns start/end time and writes the manifest
for programmatic and CLI calls. It may delegate the existing body to a private
implementation to avoid duplicating return-site writes. Window hold time is not
included. The evaluator retains its separate subprocess wall time.

Minimal schema:

```json
{
  "schema_version": 1,
  "domain": "kitchen",
  "variant": "K2",
  "internal_variant": "F1_HIDDEN_OBJECTS",
  "spec_mode": "vlm",
  "spec_provider_source": "VLM_FUNCTIONAL_SPEC",
  "spec_acquisition": "replayed_provider_output",
  "specification_sha256": "...",
  "specification_input": "/path/or/null",
  "provider_model": "Qwen/...",
  "search_order_source_requested": "fixed",
  "search_order_source_effective": "fixed",
  "provider_region_ranking": ["C2", "D1", "D2", "B1", "C1"],
  "region_order_used": ["D1", "D2", "C2", "B1", "C1"],
  "exploration_actuation": "direct_sim_articulation",
  "execution_state": "planning_only",
  "visualization_requested": true,
  "git_commit": "fa4b9527ca3c2a790a51d7f3fae3968c7973d6c7",
  "git_dirty": true,
  "started_at_utc": "...",
  "finished_at_utc": "...",
  "pipeline_runtime_seconds": 0.0,
  "terminal_status": "ACTION_SEQUENCE_READY",
  "observer_errors": [],
  "artifacts": {
    "functional_graph": "functional_requirement_graph.json",
    "observed_graph": "observed_scene_graph.json",
    "grounding": "graph_grounding_result.json",
    "result": "result.json",
    "plan": "action_sequence/action_plan.json"
  }
}
```

Use null/empty values when failure occurs before a field is available. Artifact
entries include only files that exist and are relative to the run directory.
Write atomically. Do not hash every large perception artifact in Phase 3.

Existing artifacts remain canonical. In particular, do not merge the three
domain-specific plan layouts merely for the UI. The observer consumes the
normalized `PipelineResult.plan`, while the manifest points at the native plan.

## 14. Evaluator Extension

Preserve subprocess invocation. It provides process isolation, return-code
checking, captured stdout/stderr, and the tested artifact boundary; importing
`run_pipeline()` directly offers no scientific benefit and increases leak risk.

Add:

```text
--mode {gt,vlm}                  default: gt
--search-order {provider,fixed} default: provider
--specification-root PATH        optional prior-run root for per-variant G_F replay
```

Thread these values into `evaluate_variant()`, the CLI command, run-directory
resolution, each result row, `summary.json`, and `summary.csv`. When a
specification root is supplied, resolve
`<root>/<domain>/<variant>/<mode>/functional_specification.json`, require it to
exist for every queued variant before starting any subprocess, and forward it as
`--specification-json`. Keep `--dry-run` in evaluator commands. Defaults must
generate the same command and expected status checks as today except for an
explicit redundant `--search-order provider` argument if desired.

Do not add visualization to evaluator runs. Do not change `EXPECTED`, artifact
validation, replay interpretation, audit interpretation, output-root freshness,
or default all-32 selection. Add unit tests for command construction and
mode-dependent run paths using mocks; do not launch MuJoCo in evaluator unit
tests.

The evaluator can technically run all four mode/order combinations, but reports
must label GT/provider and GT/fixed as identical-order conditions when their
recorded orders match. A controlled VLM/fixed sweep must use
`--specification-root` from the corresponding VLM/provider sweep and verify
equal per-variant graph hashes. Flash runs only selected variants; the user owns
full and remote-VLM experiments.

## 15. Phase 4 Handoff

Phase 4 receives immutable Phase 3 artifacts and reconstructs a fresh execution
scene for the recorded variant. It executes the exact persisted final plan in
order; it does not regenerate or substitute a GT plan.

Common required inputs:

- manifest and exact domain/variant configuration;
- native final action plan plus replay/validation result;
- `graph_grounding_result.json` and operation bindings;
- `observed_scene_graph.json` and domain-specific perception registries;
- actual exploratory opened-region list/state;
- a domain execution-entity resolution artifact produced by the execution
  adapter before the first task action.

Domain boundary facts:

- Workshop is closest to ready: `WorkshopDomainAdapter.evaluate_satisfaction()`
  already stores both observed track IDs and physical driver/fastener handles in
  its assignment, and its plan uses physical handles.
- Living Room already demonstrates the correct boundary in
  `living_room_mobile_execution.py::resolve_execution_entities()`: it maps
  generic object/region IDs to backend bodies/support geoms using semantic
  consistency and measured centroids, persists
  `execution_entity_resolution.json`, and only then refines actions. The
  canonical Phase 3 run already preserves `payload_registry.json`,
  `region_registry.json`, and `region_assignments.json` under
  `observed_grounding/`; a Phase 4 adapter must point the existing executor at
  these canonical paths rather than expecting GGR alone.
- Kitchen final plans use `object_####`. The original observation registry stores
  an opaque deterministic `instance_token` plus observed geometry/source region,
  but the current GT execution stack expects backend instance names. GGR alone
  cannot bridge this. Phase 4 needs a simulation-adapter-only resolver, analogous
  to Living Room, that maps every plan-referenced generic ID one-to-one to a
  physical body and rejects ambiguity before motion. First verify whether the
  existing token can be matched by enumerating scene instances with the same
  token function; otherwise use semantic consistency plus measured centroid and
  source region. Do not expose backend names to grounding or planning.

Phase 3 must retain the Kitchen `observed_search/phase1/object_registry.json`,
Living Room registries, Workshop track/physical assignment, and final opened
regions in the manifest's artifact pointers. Do not invent a universal
`ExecutionAdapter` protocol until these three existing boundaries are exercised.

Phase 4 writes a separate `execution_result.json` and emits optional
`execution_started`, `action_started`, `action_completed`, and
`execution_finished` events with copied camera frames. A failed physical action
stops or follows an explicitly selected execution policy, but never rewrites
G_F, G_O, phi*, the plan, or Phase 3 status.

## 16. File-Level Change Table

| Path | Existing/New | Exact change | Scientific risk | Implementation risk |
|---|---|---|---|---|
| `mujoco_scenes/functional_tamp_pipeline/run.py` | Existing | Keep sole CLI; add public run wrapper, search-order/observer args, guarded emission, manifest, visualization setup | Low if defaults remain GT/provider | Medium due three-domain dispatch |
| `mujoco_scenes/functional_tamp_pipeline/search_order.py` | New | Pure constants, validation, and order resolver; no scene access | Low | Low |
| `mujoco_scenes/functional_tamp_pipeline/search.py` | Existing | Optional explicit order and observer; emit around existing Workshop calls | Low | Low |
| `mujoco_scenes/functional_tamp_pipeline/domains/kitchen.py` | Existing | Accept order/observer; reuse completion grounding result for snapshots; emit final G_O/phi*/plan | Low | Medium |
| `mujoco_scenes/sequential_inspection.py` | Existing | Optional observer hook around initial observe and each select/open/observe boundary | Low | Medium because Kitchen path is calibrated |
| `mujoco_scenes/functional_tamp_pipeline/domains/workshop.py` | Existing | Accept/order diagnostics and expose copied capture snapshot; no decision change | Low | Medium due large controller state |
| `mujoco_scenes/functional_tamp_pipeline/domains/living_room.py` | Existing | Optional observer; same-thread initial frame; final G_O/grounding/plan snapshots | Low | Low |
| `mujoco_scenes/functional_tamp_pipeline/live_visualizer.py` | New | Callable bounded-queue state reducer and Pillow composite renderer using existing ffplay viewer | None to pipeline if isolated | Medium |
| `scripts/evaluate_functional_tamp_variants.py` | Existing | Add mode/order/specification-root parameters and report columns; preserve subprocess/defaults | Medium because evaluator is frozen | Low with command tests |
| `mujoco_scenes/functional_tamp_pipeline/tests/test_phase3_contract.py` | New | Pure order, manifest, callback-failure, dispatch, and CLI tests | None | Low |
| `mujoco_scenes/functional_tamp_pipeline/tests/test_live_visualizer.py` | New | State/render tests with fake viewer; no display/MuJoCo | None | Low |
| `mujoco_scenes/functional_tamp_pipeline/models.py` | Existing | No change | Frozen | None |
| `grounding.py`, `planning.py`, `audit.py`, providers, `scene_graph.py` | Existing | No change | Frozen | None |
| Existing execution modules | Existing | No Phase 3 change | Out of scope | None |

Do not create `interactive.py`, `__main__.py`, `telemetry.py`, or a second live
viewer implementation. Reuse `live_mosaic_viewer.py` without modification unless
a narrowly tested bug prevents reuse.

## 17. Final Implementation Passes for Flash

### Pass 3.0 — Run contract, independent order, and manifest

Goal: establish provenance and scientific condition separation without UI.

Files allowed: `run.py`, new `search_order.py`, new
`tests/test_phase3_contract.py`. Files forbidden: domain logic, search loop,
providers, models, perception, grounding, planning, execution, evaluator.

Actions:

- Add pure `provider|fixed` order resolution and exact candidate-set validation.
- Add the optional run arguments with current defaults, including validated
  provider-output replay through `--specification-json`.
- Preserve provider G_F, resolve order after provider return, and prepare it for
  domain dispatch without yet changing domain behavior.
- Wrap the current run body so success/exception manifests are atomic and timing
  excludes post-result UI hold.
- Add manifest tests with mocked domain dispatch/provider; include dirty Git and
  observer error fields.

Cheap tests Flash may run: new pure tests, `run.py --help`, existing architecture
tests not requiring MuJoCo. User tests: none.

Acceptance: old CLI arguments parse; default configuration is GT/provider when
selected by evaluator; G_F serialization is unchanged; fixed order is validated;
replayed G_F is source/domain checked and hash-recorded; failure manifest works
under a mocked exception.

Stop condition: do not wire domain loops or run any scene.

### Pass 3.1 — Domain-correct telemetry and order wiring

Goal: produce the minimal event stream and make the separate order the only
search difference.

Files allowed: `run.py`, `search.py`, `domains/kitchen.py`,
`domains/workshop.py`, `domains/living_room.py`, `sequential_inspection.py`, and
focused tests. Files forbidden: models, providers, perception algorithms,
grounding, compilers, A*, replay, execution.

Actions:

- Add the guarded callback and event emission points from Section 7.
- Wire explicit order into Kitchen and Workshop without altering G_F.
- Ensure Kitchen uses existing completion grounding once per predicate call.
- Ensure Living Room emits no search/OPEN events.
- Pass copied arrays/paths only; never a live scene.
- Record callback errors without propagating them.

Cheap tests Flash may run: fake Kitchen loop order, fake `SearchDomain` Workshop
event order, Living Room no-search assertion, raising observer test, all existing
functional pipeline unit tests.

Manual tests user should run: none yet.

Acceptance: with no callback and provider order, call sequence and results are
identical; fixed changes only iteration order; event payloads contain no scene or
controller; callback failure leaves the mocked result unchanged.

Stop condition: no GUI and no real detector/VLM calls.

### Pass 3.2 — Canonical interactive visualization

Goal: implement a usable live window without MuJoCo concurrency.

Files allowed: new `live_visualizer.py`, `run.py`, new visualizer tests. Files
forbidden: domain decisions, scene classes, render APIs, execution.

Actions:

- Implement callable state reducer, bounded non-blocking queue, worker, graph
  caching, Pillow composite, and existing ffplay viewer integration.
- Add `--visualize` to the existing parser and post-result lifecycle.
- Make display startup/worker/close failures visible and non-fatal.
- Use a fake viewer and synthetic G_F/G_O/camera arrays in tests.

Cheap tests Flash may run: headless renderer snapshot dimensions/content, queue
drop behavior, fake ffplay failure, `--help`. Do not open a real window in CI.

Manual tests user should run: K1 and K2 visual runs, then K7, L1, and W1 as time
permits.

Acceptance: one responsive window shows every required panel; K2's exploratory
trace is separated from A*; Living Room says search N/A; closing/failing the
window cannot change pipeline status; no non-pipeline thread touches MuJoCo.

Stop condition: do not add execution visualization yet.

### Pass 3.3 — Conservative evaluator extension

Goal: expose selected GT/VLM and provider/fixed runs while preserving the
validated default sweep.

Files allowed: `scripts/evaluate_functional_tamp_variants.py`, evaluator unit
tests. Files forbidden: pipeline/domain/scientific logic.

Actions:

- Add `--mode`, `--search-order`, and optional `--specification-root` with
  GT/provider/live-provider defaults.
- Preserve subprocesses, `--dry-run`, expected truth, return-code handling,
  output freshness, replay/audit logic, and all existing columns.
- Add spec mode/order and manifest path as new report fields.
- Unit-test exact default and alternate command construction.

Cheap tests Flash may run: evaluator helper/unit tests only. User tests: one or
two selected default variants after review.

Acceptance: the default generated command is semantically the existing command;
selected mode/order paths and per-variant replay specs are read correctly; a
missing replay spec fails before launching the queue; no 32-run sweep occurs.

Stop condition: Flash must not run the full evaluator.

### Pass 3.4 — GT/VLM and order-source manual wiring verification

Goal: confirm the already-existing provider boundary through the new runner.

Files allowed: tests/help text and only narrowly necessary fixes in Phase 3
files. Files forbidden: VLM prompts/provider normalization and all downstream
scientific logic.

Actions:

- Mock each domain's VLM provider and assert identical downstream dispatch.
- Document/preflight the FM environment variables.
- Confirm manifest captures provider order versus actual used order.
- Compare provider/fixed using one persisted validated G_F, not two independent
  VLM calls; assert equal manifest hashes.

Cheap tests Flash may run: provider mocks and malformed-order tests. Manual tests
user should run: one selected live VLM case when the server is available, then
the same saved G_F/images with fixed order.

Acceptance: remote failure is clear and non-infeasible; credentials are not
logged; no hidden simulator data reaches the provider; downstream code path is
mode-independent.

Stop condition: no prompt tuning, full VLM sweep, or scientific analysis.

### Pass 3.5 — Freeze and Phase 4 handoff audit

Goal: close Phase 3 and demonstrate that required execution inputs remain.

Files allowed: Phase 3 tests/docs and manifest artifact discovery. Files
forbidden: execution implementation and every frozen scientific module.

Actions:

- Verify manifest pointers for Kitchen registry/opened regions, Living Room
  registries, Workshop physical/track assignment, plan, and replay.
- Run the compact non-remote smoke matrix.
- Hand commands to the user for the 32-case default sweep and selected visual/VLM
  cases.
- Record Kitchen execution-ID resolution as a Phase 4 prerequisite, not a Phase
  3 workaround.

Cheap tests Flash may run: unit suite and at most the explicitly assigned smoke
cases. Manual tests user should run: visual cases and final 32 GT sweep.

Acceptance: user reports 32/32 default truth; selected UI paths are usable;
manifests are complete; no execution ran; repository diff contains only approved
Phase 3 files and preserves the pre-existing dirty evaluator test.

Stop condition: do not begin Phase 4.

## 18. Manual Validation Matrix

| Case | Configuration | Purpose | Runner |
|---|---|---|---|
| K1 | GT/provider, no window | initially satisfiable/plan path | Flash only if explicitly assigned; otherwise user |
| K2 | GT/provider, visual | Kitchen OPEN, growing G_O, early completion, final A* | User |
| K7 | GT/provider, visual | order exhaustion and INFEASIBLE UI | User |
| L1 | GT, visual, search N/A | one-observation G_F/G_O/phi*/plan | User |
| W1 | GT/provider, `--dry-run`, visual | shared Workshop loop and frame flow | User |
| 32 GT | GT/provider, evaluator defaults | final frozen regression | User only |
| Selected VLM | saved initial images, provider order | remote provider and common downstream path | User only |
| Same selected VLM | same graph/images, fixed order | meaningful order comparison | User only |

Flash may run pure/unit tests freely. It must not run the 32 variants, live remote
VLM calls, or long physical OPEN/execution experiments unless the user later
explicitly asks.

## 19. Risks / Failure Modes

| Risk | Mitigation |
|---|---|
| OpenCV GUI unavailable | Use existing ffplay viewer; Pillow/NumPy compose only |
| ffplay/display unavailable over SSH/headless | Warn, record observer error, continue planning; use artifacts/headless tests |
| ffplay pipe blocks | Only worker writes; bounded queue isolates pipeline callback |
| MuJoCo render/data race | UI receives copied frames/paths only; same-thread capture at safe boundaries |
| UI redraw is expensive | Cache G_F/G_O panels and redraw only on graph changes; cap nodes/edges |
| Callback raises | Guard centrally, stderr first error, manifest count/details, no propagation |
| Kitchen incorrectly instrumented through `search.py` | Hooks are explicitly in sequential inspection and Kitchen completion predicate |
| Living Room gets fake OPEN/policy events | Effective search order N/A and no search emission |
| Fixed order leaks scene truth | Constants are existing domain default/candidate order and variant-independent |
| GT/provider mislabelled as FM-ranked | Use `provider`; manifest records exact lists; GT comparison documented redundant |
| VLM server absent/slow | Preflight config, clear transport failure, user-owned live run; never label infeasible |
| Independent VLM calls confound order comparison | Generate once, replay the exact validated G_F, and require equal hashes |
| Artifact overwrite between order conditions | Require fresh output roots; retain existing directory schema |
| Evaluator regression | Defaults, subprocess boundary, EXPECTED, and validations frozen; command tests first |
| Kitchen generic IDs cannot execute directly | Phase 4 preflight resolver; fail on ambiguity before motion |
| Execution failure contaminates planning | Persist Phase 3 first; separate execution result/status; no automatic re-grounding |

## 20. Implementation-Time Verification Items

1. Select and test the exact Kitchen stage RGB path across initial and post-OPEN
   stage directory names.
2. Confirm the representative Workshop camera ID/mosaic that is consistently
   available from `_capture_and_evaluate()`.
3. Confirm ffplay's close/poll behavior on the user's actual desktop or SSH
   display and choose the final hold behavior accordingly.
4. Confirm live configured Qwen endpoint/model supports the repository's JSON
   schema response format; do not alter prompts during this phase.
5. In Phase 4, test whether Kitchen `instance_token` deterministically resolves
   by enumerating scene instance IDs before falling back to measured matching.

## 21. Flash Handoff Contract

```text
IMPLEMENT PHASE 3 ONLY, IN PASSES 3.0 THROUGH 3.5.

1. Preserve the frozen pipeline:
   task+RGB -> GT/VLM G_F -> perception -> G_O -> exploratory search/OPEN
   -> grounding -> phi* -> compiler -> deterministic A* -> independent replay
   -> persisted final plan.

2. Keep run.py as the only CLI/runtime entry. Do not create interactive.py,
   __main__.py, telemetry.py, a TelemetrySink hierarchy, or a second viewer.

3. Use search-order-source names `provider` and `fixed`. Persist provider G_F
   unchanged. Pass the resolved order separately. Validate exact equality with
   candidate_regions. Never call GT order learned/FM-ranked.

   For paired VLM order comparisons, generate once and load the exact persisted
   graph through --specification-json for the second run; require equal G_F
   hashes. Do not make two live VLM calls and call them a controlled comparison.

4. Kitchen telemetry must be wired through sequential_inspection.py and
   domains/kitchen.py. Workshop search telemetry belongs in search.py. Living
   Room has no search/OPEN/order events.

5. The callback is Callable[[str, dict[str, Any]], None]. Pass snapshots only.
   Catch observer failures, warn, record them in the manifest, and continue.

6. Implement the one live window with Pillow/NumPy plus the existing
   LiveMosaicViewer/ffplay subprocess and a bounded worker queue. Do not use
   cv2.imshow, a matplotlib GUI, Qt, or a browser server.

7. No observer or display thread may receive or access a MuJoCo scene/model/data.
   Capture/copy frames only on the pipeline thread at safe observation/execution
   boundaries.

8. Do not modify models.py, scene_graph.py, grounding.py, planning.py, audit.py,
   spec_provider.py, gt_spec_provider.py, vlm_spec_provider.py, provider prompts,
   perception/tracking/geometry logic, or execution modules.

9. Extend the evaluator conservatively with --mode and --search-order, defaulting
   to gt/provider, plus optional --specification-root for paired replay. Keep
   subprocess invocation, --dry-run, EXPECTED, paths, validation, and default
   32-case selection semantics.

10. Run only pure/unit tests and explicitly assigned compact smoke cases. Do not
    run the 32-variant sweep, remote VLM experiments, long physical experiments,
    or Phase 4 execution. The user owns those runs.

11. Phase 4 later must execute the exact persisted plan and resolve generic IDs
    at a simulation-adapter-only boundary. GGR alone is not a universal execution
    handoff. Execution status is separate and cannot rewrite planning status.

12. After every pass, show the diff, prove default behavior is unchanged, and
    stop if a frozen scientific interface would need modification. Do not commit
    or push. Preserve the user's pre-existing modification to
    test_evaluator_harness.py.
```
