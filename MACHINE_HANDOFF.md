# Machine handoff: setting this repository up on a new host

Written for a move from an i5-11400H (12 threads) to an i9-12900HX (24
threads) + RTX 3080 Ti. Read `CLAUDE.md` and `CLAUDE_HANDOFF.md` first for
research intent and architecture; this file covers only the migration.

## What Git does not carry

`.gitignore` excludes several things this repository needs to run. Cloning is
not enough.

| Excluded | Size | How to restore |
|---|---|---|
| `runs/` | 5.0 GB | **Copy manually or lose the results.** See below. |
| `.venv/` | — | Recreate from `mujoco_scenes/requirements.txt` |
| `.paper_deps/pddlstream` | — | `bash vlm_tamp_baseline/setup_pddlstream.sh` |
| `weights/`, `mujoco_scenes/weights/` | — | `mujoco_scenes/scripts/prepare_semantic_models.py` |
| `semantic_model_cache/` | — | Repopulated on first retrieval run |

### Results are the one irreplaceable item

`runs/` holds every completed episode and is **not** in Git. Copy it before
wiping the old machine:

```bash
# from the NEW machine, pulling from the old one
rsync -av --progress old-host:~/Documents/RRC/LH_Extension/V1/runs/ \
  ~/Documents/RRC/LH_Extension/V1/runs/
```

Everything else in the table above can be rebuilt; these cannot.

## Setup on the new host

```bash
git clone https://github.com/Narendhiranv04/icra-we-ball.git
cd icra-we-ball
git checkout phase4_integration

python3.11 -m venv .venv
.venv/bin/pip install -r mujoco_scenes/requirements.txt
.venv/bin/pip install -r requirements-test.txt

bash vlm_tamp_baseline/setup_pddlstream.sh
.venv/bin/python mujoco_scenes/scripts/prepare_semantic_models.py
```

Confirm the pinned engine, because physical results are only comparable within
one MuJoCo build:

```bash
.venv/bin/python -c "import mujoco; print(mujoco.__version__)"   # expect 3.3.6
```

## Verify before running anything

```bash
# `env -u PYTHONPATH` matters if ROS is sourced: its pytest plugin hijacks
# collection and the suite exits 0 having tested nothing.
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  baseline_common/tests vlm_tamp_baseline/tests owl_tamp_baseline/tests \
  retrieval_baseline/tests mujoco_scenes/tests -q
```

Expect **7 failures, all Kitchen or robot-profile**, and everything else
passing. That is the known-good state, not a broken checkout:

- 4x kitchen serving allocator / ground-truth execution
- 1x phase-4 kitchen inspection
- 1x `test_only_the_physical_shoulder_mount_has_a_self_overlap_allowance`
  (asserts 2 self-collision allowances; the phase-4 port widened it to 5 --
  this is an open question, see "Open items")

Any *other* failure means the environment is wrong, not the code.

## Model server

The 9B is served remotely on `gvlab2`; nothing about that changes with the new
laptop. Open the tunnel and confirm the served name before a grid:

```bash
ssh -L 18000:127.0.0.1:8000 long-horizon@gvlab2.iiit.ac.in
# in that session, on the server:
tmux new -s vllm
cd ~/SearchTAMP && source .venv-qwen35/bin/activate.fish
vllm serve Qwen/Qwen3.5-9B --served-model-name qwen35-9b \
  --host 127.0.0.1 --port 8000 --dtype bfloat16 --max-model-len 32768 \
  --max-num-seqs 2 --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image":8}' --enable-prefix-caching \
  --generation-config vllm --reasoning-parser qwen3
# detach: Ctrl-b d
```

```bash
curl -s http://127.0.0.1:18000/v1/models | python3 -m json.tool
```

Only one tunnel may bind port 18000. If the forward fails, an old `ssh -N` is
probably still holding it: `pgrep -af "ssh.*18000"` then kill it.

### The 3080 Ti does not speed up execution

Physical execution is MuJoCo stepping, which is CPU-bound; ~14 s of the ~17 s
per action is physics. The 24-thread i9 is the win here, roughly halving
per-episode wall time against the i5. The GPU only matters if the model is
served locally, and 12 GB will not hold Qwen3.5-9B at bf16 (~18 GB) -- that
needs FP8/AWQ, which changes the model under test and must then be recorded as
a different condition.

## Running the grid

```bash
env -u PYTHONPATH .venv/bin/python -m baseline_common.run_baseline_execution_batch \
  --environment living_room --methods vlm_tamp,owl_tamp,retrieval \
  --variants L1,L2,L3,L4,L5,L6 \
  --camera-counts 3 --seeds 0,1,2,3,4,5,6,7,8,9 \
  --output-root runs/living_room/execution/<name> \
  --base-url http://127.0.0.1:18000/v1 --model qwen35-9b \
  --resume --continue-on-error
```

Re-run the identical command to resume; completed episodes are skipped and an
episode interrupted mid-write is moved aside and retried. Each episode is
bounded by `--episode-timeout` (default 3600 s) so one hang cannot stall the
grid.

Regenerate the paper tables at any point:

```bash
env -u PYTHONPATH .venv/bin/python -m baseline_common.make_paper_tables \
  runs/living_room/execution/<name>
```

## Do not mix hosts inside one reported grid

Execution artifacts now record `host_cpu`, `host_platform` and
`mujoco_version`. Contact-rich stepping is sensitive to host floating-point
behaviour, so episodes from two machines are not guaranteed reproducible
against each other.

The grid in `runs/living_room/execution/feasible_20260904` was produced on the
i5. **Recommendation: start a fresh grid on the new machine rather than
resuming that one.** It is ~6 h at the new machine's speed, and it buys a
single-host dataset instead of a split one that needs a caveat in the paper.
Keep the old run for comparison; do not pool the two.

## Open items

1. **Infeasible variants L7--L10 are not runnable as scored work yet.**
   `success` is physical goal satisfaction, which is unachievable on an
   infeasible variant by construction, so a method that correctly rejects the
   task scores identically to one that blunders through it. Before running
   them, carry `outcome_match` into `benchmark_execution_result.json` and make
   `summarize_execution_batch` partition feasible from infeasible variants.
2. **Goal coverage is recomputed, not recorded.** `make_paper_tables` derives
   it from `latest_observation.json` plus the private role map, because the
   goal verifier collapses a role-matching result to a boolean. Moving it into
   the verifier would make it a first-class metric.
3. **Self-collision allowances were widened by the phase-4 port** from 2 pairs
   to 5 (`link_forearm`, `link_wrist`, `link_gripper` at -0.030) in
   `mujoco_scenes/generic_manipulation.py`. This relaxes a physical validity
   criterion for the Google robot used in every Living Room episode, and the
   test asserting the old set was never updated. Decide whether the widening is
   intended and record it in `BASELINE_FIDELITY.md` either way.
4. **No runs of the proposed framework exist.** The "Ours" rows in the main
   results table have no data; only baselines have been executed.
5. **Kitchen and Workshop have no execution data**, and Kitchen carries the 7
   known test failures.

## Result classes must never be pooled

Three separate experiment types, three separate tables:

- **Oracle-evidence grounding** (`run_gt_evidence_ablation`) -- no VLM,
  detector, search, planner, or robot. Measures the grounding decision
  boundary only.
- **Planning-to-GT** (`run_plan_gt_batch`) -- plan compared against the
  ground-truth action sequence. No robot.
- **Physical execution** (`run_baseline_execution_batch`) -- goal verified on
  the simulated robot's final state.

A method can plan correctly and fail physically, or ground correctly and plan
badly. Never call a planning-only or oracle-grounding result an end-to-end
physical success.
