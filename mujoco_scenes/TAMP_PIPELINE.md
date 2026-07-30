# Functional-Alternative TAMP Pipeline

The pipeline separates model judgment from simulator authority:

1. `LivingRoomObserver` produces the currently observed objects, regions,
   robot state, and relations.
2. The function registry selects visible candidates for a simple predicate
   such as `can_store` or `can_clean`.
3. The remote foundation model returns the functional subset and its preferred
   order. Unknown, duplicate, and non-visible candidate IDs are rejected.
4. A deterministic planner expands one candidate into guarded simulator
   skills.
5. The executive runs one skill at a time and verifies the observed final
   effect. Recoverable candidate failures advance to the next ranked
   alternative.

The model never receives MuJoCo internals, the complete object catalogue, or
closed-container contents. For the living-room storage task it receives only
the visible game controller and visible storage regions. A closed drawer has
unknown occupancy until it is opened. Occupancy is checked again immediately
before placement.

## Run the living-room task

Start the vLLM workspace described in
[`../inference_server/README.md`](../inference_server/README.md), then export
the client settings:

```bash
export TAMP_FM_BASE_URL=http://inference-server:8000/v1
export TAMP_FM_MODEL=tamp-ranker
export TAMP_FM_API_KEY=replace-me
MUJOCO_GL=glfw uv run python -m mujoco_scenes.living_room_scene --viewer
```

Select **Store game controller** under **Functional task**. The current
alternatives are the right drawer, left drawer, and wall shelf. The robot
physically picks, navigates, opens or closes a drawer when needed, places, and
then verifies the symbolic storage relation.

For a local smoke test without an inference server:

```bash
TAMP_FM_BACKEND=fixed \
MUJOCO_GL=glfw uv run python -m mujoco_scenes.living_room_scene --viewer
```

The fixed backend is only for integration testing. It uses a static preference
order and does not replace the foundation-model judgment.

Set `TAMP_EVENT_LOG=artifacts/tamp-events.jsonl` to retain observations,
assessments, selected candidates, skill outcomes, failures, and final effects
for replay and evaluation.

## Execution workflow

For `store_game_controller`, execution follows this loop:

1. The task states the subject, required function (`can_store`), and desired
   relation (`stored_in`).
2. The observer takes a fresh snapshot. Closed drawer contents remain unknown
   until the drawer has been inspected.
3. Candidate generation selects only visible regions relevant to `can_store`
   and removes planner-only metadata from the model payload.
4. The foundation model identifies the functional subset and ranks it. It does
   not produce motor commands or a complete action plan.
5. The client rejects invented, duplicate, or non-visible IDs.
6. The deterministic planner expands the first candidate into grounded skills:
   pick, move, open when required, place, and close.
7. The dispatcher advances one physical controller at a time. Existing IK,
   RRT*, self-collision, furniture-collision, grasp-contact, and settling
   checks remain authoritative.
8. The observer refreshes after every skill. A newly observed occupied target
   produces a structured failure and the executive tries the next ranked
   alternative when the robot remains safe.
9. A final observation must contain the requested `stored_in` relation. A
   completed motion without the expected effect is treated as failure.

The generic executive supports discovery policies, but the current living-room
storage task does not need one because all three candidate storage regions are
visible at reset. Their contents may still be unknown.

## Grounded action file

Edit `configs/living_room_actions.txt`, putting one command on each line:

```text
move DESTINATION
open left
close left
pick OBJECT
place
place STORAGE_TARGET
task store_game_controller
state observed
gt
help
```

For example, this file tests the left drawer without model selection:

```text
pick game_controller
move drawer_left
open left
place media_console_left_drawer
close left
```

Press **Reload and run** in the Actions window. The runner reloads the file,
executes it from the first line, and waits for each physical action to finish
before starting the next one. Blank lines, lines beginning with `#`, and
inline comments are ignored. **Stop queue** discards actions that have not
started; an active physical action is allowed to finish safely.

Valid explicit storage targets are `media_console_left_drawer`,
`media_console_right_drawer`, and `media_shelf`. `place` without an argument
uses the ordinary object-specific manual destination. `task
store_game_controller` runs the complete foundation-model pipeline instead.

`state observed` prints the bounded observation used by the executive. `gt`
prints the simulator's symbolic object locations and drawer states. Ground
truth queries and direct commands are never added to a foundation-model
request. Direct commands bypass only candidate selection; all physical safety
and success checks still run.

Set `LIVING_ROOM_ACTION_FILE` to use a different action-file path. Buttons that
would conflict with a running file or active motion remain disabled.

## Extend it

- Add a simple predicate to `configs/functions.yaml`.
- Produce only observed candidate facts in the scene observer.
- Add a planner that maps a chosen candidate to existing guarded skills.
- Add an effect verifier based on a fresh observation.
- Test functional rejection, fallback, skill failure, and missing effects.

The pending point-cloud fit check belongs between visible candidate generation
and foundation-model assessment. It should add measured feasibility facts or
remove geometrically impossible candidates; it should not expose unobserved
objects. SQLite, a knowledge graph, BDI, and PRS are not required for this
loop. The typed in-memory scene state plus optional JSONL event log remains the
source of truth.
