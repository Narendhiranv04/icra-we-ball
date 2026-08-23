# Final-paper GT execution and recording commands

## Outcome

The workflow below creates exactly one deliverable directory:

```text
FINAL_PAPER_GT_EXECUTIONS/
├── kitchen/<variant>/
├── living_room/<variant>/
├── workshop/<variant>/
├── manifest.json
└── README.md
```

Every variant directory contains:

```text
robot_execution_5cam.mp4
gt_actions.txt
gt_actions.json
function_object_assignments.txt
function_object_assignments.json       # Kitchen and Workshop
objects_and_regions.txt
timing.txt
timing.json
execution_summary.json
```

Living Room keeps its detailed assignment in
`function_object_assignments.txt` because its source contract is a collection
of region-function witness artifacts rather than one assignment JSON file.

The workflow covers all configured variants: 16 Kitchen, 10 Living Room, and
10 Workshop variants (36 total). Feasible variants execute the complete GT
task. Infeasible Kitchen and Workshop variants execute their available
inspection/rejection plan and terminate with the expected reason. Living Room
infeasible variants are rejected before Phase-2 physical planning, so those
videos contain an explicit rejection record rather than invented robot motion.
They are not falsely presented as successful task completions.

## Recording policy

- The robot executes the existing validated GT action dispatcher in every
  scene. The packager never creates object-only animations.
- These are deliberately **GT/oracle demonstrations**. They validate the
  expected assignment, action sequence, and physical execution; they do not
  claim that a production detector or VLM selected the assignment.
- Five synchronized views are merged into one 3-by-2 mosaic: left, right, top,
  front, detail/close, and a telemetry panel.
- Capture is at natural simulation time. No FFmpeg `setpts`, frame dropping for
  acceleration, or playback-speed transformation is used.
- Kitchen is explicitly fixed at `--speed 1.0`.
- Kitchen uses `--strict-robot-execution`; a missed physical grasp fails the
  variant instead of attaching or relocating the payload directly.
- Living Room uses `STRICT_PHYSICAL_POSTCONDITION`; the final-paper builder
  never passes `--assisted-suite`.
- Workshop retains its corrected slow motion profile, top-down drawer grasps,
  horizontal cabinet grasps/retrieval, and bilateral-contact grasp gate.
- Packaging is fail-closed. Any assisted Kitchen result, normal-execution
  Living Room object-qpos edit, Workshop non-robot action, or direct payload
  pose write prevents the variant from entering the final evidence folder.
- The default per-camera tile is `960x540` at 20 FPS, producing a 2880x1080
  mosaic. This is intentionally high quality and will require substantial time
  and storage.

## Run everything

From the repository root:

```bash
cd /home/naren/RA_iiith
./scripts/build_final_paper_gt_executions.sh
```

The script performs two complete passes for every variant:

1. Physics execution without recording, timed using a monotonic wall clock.
2. The same physics execution with natural-speed five-camera recording.

Raw run artifacts are created in a temporary directory and removed only after
successful packaging. Therefore, this command does not add another permanent
raw suite beneath `runs/`.

If `FINAL_PAPER_GT_EXECUTIONS` already exists, the script refuses to overwrite
it. Explicit replacement is:

```bash
./scripts/build_final_paper_gt_executions.sh --replace-final
```

The replacement flag is intentionally limited to the exact
`FINAL_PAPER_GT_EXECUTIONS` path. It cannot target the repository, `runs/`, or
`outputs/`.

## Run and package one variant at a time

Single-variant mode performs both passes for only the selected variant and
immediately appends its packaged evidence to `FINAL_PAPER_GT_EXECUTIONS`:

```bash
./scripts/build_final_paper_gt_executions.sh \
  --environment kitchen \
  --variant F0_REUSE_ONE
```

Valid environment names are `kitchen`, `living_room`, and `workshop`. The
script validates the variant against the current scene contract. Existing
completed variants are preserved. To rerun and replace only one variant:

```bash
./scripts/build_final_paper_gt_executions.sh \
  --environment kitchen \
  --variant F0_REUSE_ONE \
  --replace-variant
```

This mode materially reduces peak temporary storage because the raw timing and
recording runs are deleted after each variant is packaged.

Add `--show` to display the same five-view mosaic during both executions. The
first execution is live but saves only its elapsed time; the second execution
is live while simultaneously writing the MP4:

```bash
./scripts/build_final_paper_gt_executions.sh \
  --environment kitchen \
  --variant F0_REUSE_ONE \
  --replace-variant \
  --show
```

The command first opens a live timing-only pass without writing a video, then
opens the live mosaic again for the recording pass. Consequently, the
`without_recording` measurement includes the cost of live visualization when
`--show` is used. An interrupted or failed run is not packaged.

Live display is provided through `ffplay` because the project Conda
environment's OpenCV build has no GUI backend. Run `--show` from the graphical
desktop terminal, with `DISPLAY` or `WAYLAND_DISPLAY` available. If an
execution fails, the builder prints its captured log before cleaning temporary
files instead of returning silently.

## Kitchen: run every variant one by one, live and recorded

Run these from `/home/naren/RA_iiith` in this order. After each command
completes, inspect
`FINAL_PAPER_GT_EXECUTIONS/kitchen/<variant>/robot_execution_5cam.mp4`.

```bash
./scripts/build_final_paper_gt_executions.sh --environment kitchen --variant F0_REUSE_ONE --replace-variant --show
./scripts/build_final_paper_gt_executions.sh --environment kitchen --variant F1_INITIAL_COMPLETE --replace-variant --show
./scripts/build_final_paper_gt_executions.sh --environment kitchen --variant F2_DISTRIBUTED_COFFEE_TWO --replace-variant --show
./scripts/build_final_paper_gt_executions.sh --environment kitchen --variant F3_DISTRIBUTED_COFFEE_THREE --replace-variant --show
./scripts/build_final_paper_gt_executions.sh --environment kitchen --variant F4_EARLY_RELOCATION --replace-variant --show
./scripts/build_final_paper_gt_executions.sh --environment kitchen --variant F5_LATE_RELOCATION --replace-variant --show
./scripts/build_final_paper_gt_executions.sh --environment kitchen --variant F6_DECOY_HEAVY --replace-variant --show
./scripts/build_final_paper_gt_executions.sh --environment kitchen --variant F7_COUNT_SURPLUS --replace-variant --show
./scripts/build_final_paper_gt_executions.sh --environment kitchen --variant I0_MISSING_COFFEE_VESSEL --replace-variant --show
./scripts/build_final_paper_gt_executions.sh --environment kitchen --variant I1_MISSING_SOUP_VESSEL --replace-variant --show
./scripts/build_final_paper_gt_executions.sh --environment kitchen --variant I2_UNCOVERED_COFFEE_TARGET --replace-variant --show
./scripts/build_final_paper_gt_executions.sh --environment kitchen --variant I3_ONLY_TWO_SOUP_TOOLS --replace-variant --show
./scripts/build_final_paper_gt_executions.sh --environment kitchen --variant I4_SOUP_MATCHING_TRAP --replace-variant --show
./scripts/build_final_paper_gt_executions.sh --environment kitchen --variant I5_SEMANTIC_DECOY_GEOMETRY_FAILURE --replace-variant --show
./scripts/build_final_paper_gt_executions.sh --environment kitchen --variant P0_LAYOUT_BASE --replace-variant --show
./scripts/build_final_paper_gt_executions.sh --environment kitchen --variant P1_LAYOUT_SWAPPED --replace-variant --show
```

The six `I*` variants are intentionally infeasible. Their videos show the
robot executing the available physical inspection/search sequence and then
terminating with the expected infeasibility reason; they do not fabricate a
successful preparation task.

## Optional quality settings

The defaults are the recommended final-paper settings. To reduce storage while
retaining the same natural playback speed, lower only the spatial resolution:

```bash
TILE_RESOLUTION=640x360 FPS=20 \
  ./scripts/build_final_paper_gt_executions.sh --replace-final
```

Changing FPS affects temporal sampling, not robot motion speed. Do not use
FFmpeg speed filters on the resulting files.

## Monitoring a long run

The command prints the current environment, variant, and pass. Per-variant logs
are kept in the temporary workspace while the workflow is active. Interrupting
the script stops the current execution and cleans the temporary raw outputs;
the previous final folder is preserved unless replacement reached the final
atomic packaging step.

## Verify the final package

These commands are read-only:

```bash
jq '{total_variants, environment_counts, all_execution_runs_successful}' \
  FINAL_PAPER_GT_EXECUTIONS/manifest.json

find FINAL_PAPER_GT_EXECUTIONS -name robot_execution_5cam.mp4 | sort

find FINAL_PAPER_GT_EXECUTIONS -mindepth 3 -maxdepth 3 -type f \
  | sort
```

The expected manifest count is 36 and
`all_execution_runs_successful` must be `true`. Here, success includes an exact
expected infeasibility confirmation for deliberately infeasible variants.

## Optional old-video cleanup

No old evidence is deleted automatically. First preview every old MP4 under
`runs/` and `outputs/`:

```bash
find runs outputs -type f -name '*.mp4' -print | sort
```

Only after the final package has been verified, remove those old MP4 files with:

```bash
find runs outputs -type f -name '*.mp4' -delete
```

This does not touch the MP4s in `FINAL_PAPER_GT_EXECUTIONS`, because that folder
is outside both search roots. It also deliberately leaves JSON traces,
grounding results, detector outputs, and benchmark evidence intact; those are
not assumed redundant merely because they are inside `runs/` or `outputs/`.

The previous standardized export can be removed separately after inspection:

```bash
find GT_everything -depth -delete
```

Do not blanket-delete all of `runs/` or `outputs/`: many directories there are
grounding/detector evidence rather than replaceable video exports.
