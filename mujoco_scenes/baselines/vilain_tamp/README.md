# ViLaIn-TAMP baseline

This package is a **parallel ViLaIn-TAMP baseline**. It owns its observations,
PDDL problems, symbolic plans, refinement records, and execution projections.
It does not run through `G_F`, `G_O`, `ground_graph()`, or `phi*`, and it does
not consume their assignments, witnesses, search results, or action plans.

The baseline does not fabricate `Phase3Handoff` or Phase-4 handoff artifacts.
Safe generic infrastructure—raw benchmark observations, fixed scene mechanics,
generic geometry and motion facilities, and low-level controllers—may
eventually be shared through baseline-owned adapters. Proposed-method semantic
information, task-aware search, grounding results, roles, operation bindings,
functional witnesses, and planning audits may never be shared with this path.

Geometric refinement is a MuJoCo cloned-scene sequence preflight,
an adaptation of ViLaIn-TAMP's MoveIt Task Constructor refinement. It is not an
exact MoveIt Task Constructor reproduction.

The live observation boundary is implemented in `live_observations.py`. It
constructs the requested benchmark variant but exposes only five canonical
RGB-D views, metric camera calibration, and the fixed inspection order to the
baseline. Kitchen and workshop storage is opened through each scene's generic
`open_container` mechanism; the returned backend contents are discarded.
Living-room full inspection is its canonical initial multiview because that
domain has no closed storage. Variant IDs, simulator body identities,
segmentation, functional search vocabulary, ranked regions, and early stopping
are absent from the model-facing observation payload.

## Runner boundary

`mujoco_scenes.run_vilain_tamp_baseline` is the baseline-native CLI. It defaults
to planning-only and requires the explicit `--execute` flag before a runtime
adapter may start scored physical execution. The runner accepts no earlier
pipeline run directory or handoff. It records and locks repository, config,
domain, observation, problem, plan, refinement, and projection provenance before
execution; an inconsistent commit, branch, tracked tree, or artifact hash blocks
execution.

Runtime components are dependency-injected so importing the CLI and displaying
help cannot initialize MuJoCo, external planners, or foundation models. Recorded
offline model fixtures are supplied to the same component factory using
`--offline-model-fixtures`; they remain independent of proposed-method calls and
artifacts.

Example configuration-only validation, which performs no external calls:

```bash
python -m mujoco_scenes.run_vilain_tamp_baseline \
  --domain kitchen \
  --variant F0_ALL_VISIBLE \
  --planning-only \
  --dry-run
```

Scored execution is always opt-in:

```bash
python -m mujoco_scenes.run_vilain_tamp_baseline \
  --domain workshop \
  --variant F0_MANUAL_FIRST_ONE_REGION \
  --observation-mode fixed_full_inspection \
  --fast-downward /absolute/path/to/fast-downward.py \
  --val /absolute/path/to/Validate \
  --execute
```

The non-dry commands above are invocation templates for the experiment host,
which must inject baseline-owned capture, model, planning, refinement, and
execution adapters into `main(component_factory=...)`. The repository does not
silently construct live adapters when the module is invoked from a shell. This
keeps imports, `--help`, and `--dry-run` free of model, planner, and simulator
side effects. The Stage-15 test suite uses recorded synthetic components through
that same injection boundary; it is not a substitute for a scored experiment.

## Reproducible environment and external tools

Use a separate environment so the baseline's model stack cannot alter the
proposed method's environment. From the repository root, an operator may create
and populate it explicitly:

```bash
python -m venv .venv-vilain-tamp
. .venv-vilain-tamp/bin/activate
python -m pip install -r \
  mujoco_scenes/baselines/vilain_tamp/requirements-vilain-tamp.txt
```

No dependency, checkpoint, Fast Downward build, or VAL binary is installed by
the baseline. Install Fast Downward and VAL outside this repository and pass
absolute executable paths. Final paper-faithful runs expect Fast Downward
24.06 with `lama-first`; record the actual VAL version in the run configuration:

```bash
export VILAIN_FAST_DOWNWARD=/opt/vilain-tools/fast-downward-24.06/fast-downward.py
export VILAIN_VAL=/opt/vilain-tools/val/bin/Validate
test -x "$VILAIN_FAST_DOWNWARD" && test -x "$VILAIN_VAL"
```

The paper-faithful object-estimation condition uses an independently downloaded
`Qwen2.5-VL-7B-Instruct` checkpoint. Pin and record the exact local checkpoint
revision in the injected transport. The reasoning calls use the immutable
`gpt-4o-2024-08-06` snapshot. Supply API credentials only through the experiment
host's environment or secret manager; never place them in YAML, prompts, or run
artifacts. The optional model-matched condition similarly requires fresh,
independent baseline calls and the placeholders in `model_matched.yaml` must be
resolved before use.

## Live model transports

`live_fm.py` owns the concrete paper-faithful transports. Both are lazy: creating
the clients does not import Transformers, load a checkpoint, or create an OpenAI
client. A real call is rejected unless `.venv-vilain-tamp` is active. The local
Qwen transport requires a full 40-character Hugging Face commit, resolves every
image beneath the observation artifact root, and logs its model source, resolved
revision, device, dtype, and token counts. The GPT transport accepts only
`gpt-4o-2024-08-06`, checks that the provider returned that exact snapshot, and
logs the provider model and system fingerprint. `RecordedFMClient` retains the
sanitized request, raw generated text, and provider metadata for every call;
credentials stay in the process environment and are never added to requests.

Before connecting the full pipeline, make exactly one standalone object call
from a previously captured baseline observation manifest:

```bash
. .venv-vilain-tamp/bin/activate
python -m mujoco_scenes.baselines.vilain_tamp.live_fm \
  --observation-manifest /absolute/run/observations/observation_manifest.json \
  --task "Prepare the requested meal." \
  --domain kitchen \
  --model-source Qwen/Qwen2.5-VL-7B-Instruct \
  --revision FULL_40_CHARACTER_HUGGING_FACE_COMMIT \
  --output-directory /absolute/run/standalone_object_call
```

This command makes one local object-estimation call and writes `request.json`,
`raw_response.txt`, and `model_metadata.json`. It does not invoke GPT, planning,
refinement, execution, or the simulator. Automated tests inject inert fake
clients and never call either live model service.

## Experimental command matrix

First validate each resolved condition without loading any runtime:

```bash
python -m mujoco_scenes.run_vilain_tamp_baseline \
  --domain kitchen --variant F0_ALL_VISIBLE --planning-only --dry-run

python -m mujoco_scenes.run_vilain_tamp_baseline \
  --domain workshop --variant F0_MANUAL_FIRST_ONE_REGION \
  --observation-mode fixed_full_inspection --planning-only --dry-run
```

An experiment host using the injected component factory should apply these
arguments for a planning-only trial or scored execution, respectively:

```text
--domain kitchen --variant F0_ALL_VISIBLE --cp-limit 3 \
--fast-downward /opt/vilain-tools/fast-downward-24.06/fast-downward.py \
--val /opt/vilain-tools/val/bin/Validate --planning-only --seed 0

--domain living_room --variant F0_ALL_OBJECTS_IN_STAGING --cp-limit 3 \
--fast-downward /opt/vilain-tools/fast-downward-24.06/fast-downward.py \
--val /opt/vilain-tools/val/bin/Validate --execute --seed 0
```

Run five independently generated trials per benchmark instance, changing and
recording the seed/request identifiers for each trial. Keep initial-only and
fixed-full-inspection results separate. Before scored execution the runner
verifies the target branch, repository state, locked configuration/domain
hashes, and every material run artifact. `events.jsonl`, `metrics.json`, and
`baseline_manifest.json` provide the audit trail; the final manifest hashes all
material files recursively and excludes only itself.
