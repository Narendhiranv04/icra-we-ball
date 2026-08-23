# YOLO-World L three-view adaptation

This report freezes separate YOLO-World-v2 L profiles for Kitchen and Living
Room without changing any camera pose, source resolution, scene, oracle, or
task decision rule. Kitchen used a fixed development/evaluation split. Living
Room used all 13 variants for calibration, so its 9/13 is a tuned-family score,
not a held-out generalization estimate.

## Frozen results

| Scene | Untuned L | Development | Evaluation | Combined | False feasible |
|---|---:|---:|---:|---:|---:|
| Kitchen | 6/16 | 4/8 | 5/8 | 9/16 (56.25%) | 0 |
| Living Room | 6/13 | — | — | 9/13 (69.23%) | 1 |

The prior medium baseline remains higher in Kitchen (11/16 versus 9/16).
Living Room L improves from 6/13 to 9/13. A two-view Living Room tie also
scored 9/13, but produced two false-feasible decisions rather than one, so the
one-view profile is the production-safe tie-breaker.

## Frozen parameters

Kitchen uses `yolov8l-worldv2.pt`, detector confidence 0.01, fusion minimum
mean confidence 0.01, minimum semantic support 1, and
`semantic_vocabulary_yoloworld_l_kitchen.yaml`. Living Room uses the same
checkpoint and confidence floors, semantic support 1, winning-score margin
0.015, and its existing integrated vocabulary. RGB-D geometry continues to
require multi-view evidence.

The search covered detector confidence 0.005–0.075, one/two-view semantic
support, confidence and margin sweeps, and concise/descriptive vocabularies.
Kitchen's visual-form prompts recovered a spoon case that L had confidently
called a fork. Living Room vocabulary rewrites reduced performance, so its
original vocabulary was retained.

## Reproduction

Run from the repository root with the EGL environment used by the canonical
benchmarks. `reproduction_commands.sh` contains both frozen commands. The
benchmark runners return status 2 when any variant disagrees with the oracle;
that is expected for the measured non-perfect results and does not mean the
run failed to produce artifacts.
