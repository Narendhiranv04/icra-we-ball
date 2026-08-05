# Point-cloud perception

The kitchen keeps Google Robot and uses five calibrated simulator RGB-D
cameras for each inspection. Camera images, depth, intrinsics, extrinsics, and
known region bounds are treated as sensor/map inputs. MuJoCo object masks,
body names, and configured container contents are ground truth.

## Run the oracle baseline

Oracle mode is explicit so it cannot be mistaken for learned perception:

```bash
MUJOCO_GL=egl uv run python -m mujoco_scenes.inspect_geometry \
  --scene S1_coffee_missing_mug \
  --region INITIAL \
  --oracle
```

For a closed region, choose `D1`, `D2`, `C1`, `C2`, or `B1`. The runner
opens that region directly, captures five views, reconstructs stage-local point
clouds, extracts geometric properties, and updates `runs/geometry/observed_state`.

At 640×480, the calibrated rig has been checked with Google Robot present and
accepts the configured objects in all six regions. The rig temporarily moves
the five virtual cameras to the selected inspection pose and restores their
original poses afterwards.

## Learned segmentation

Start the separate SAM 3.1 service by following
[`perception_server/README.md`](../perception_server/README.md), then use the
built-in HTTP adapter:

```bash
MUJOCO_GL=egl uv run python -m mujoco_scenes.inspect_geometry \
  --region D1 \
  --segmenter mujoco_scenes.sam3_client:create_segmenter \
  --prompt utensil \
  --prompt container
```

Set `SAM3_BASE_URL` if the service is not available through localhost port
8010. The client sends RGB PNG data, camera labels, and text prompts only.
Depth, camera calibration, MuJoCo IDs, object poses, and configured container
contents remain local. IDs returned by the server need only be unique within
one frame; the geometry layer associates the five views from RGB-D centroids.

Each run writes `rgb.png`, `depth.png`, `masks_overlay.png`, `cloud.ply`, and
camera metadata below `runs/geometry/captures/<stage>/<camera>/`.

For a remote server reached through the SSH tunnel shown in the server guide:

```bash
export SAM3_BASE_URL=http://127.0.0.1:8010
curl "$SAM3_BASE_URL/health"
```

## Compare SAM 3.1 with the oracle

This is an evaluation command. It runs both paths on the same simulator state,
saves separate overlays, and reports one-to-one mask precision, recall, and IoU:

```bash
MUJOCO_GL=egl uv run python -m mujoco_scenes.compare_segmentation \
  --scene S1_coffee_missing_mug \
  --region INITIAL \
  --prompt spoon \
  --prompt jar
```

Inspect `runs/sam3_comparison/comparison.json`, `oracle/cameras/`, and
`sam3/cameras/`. Oracle masks are used only by this comparison process and are
never sent to SAM.

The large model stays in the separate server workspace. The primary backend is
[SAM 3.1](https://github.com/facebookresearch/sam3), which supports
open-vocabulary text/exemplar segmentation and multi-object tracking. A lighter fallback is
[Grounding DINO plus SAM 2](https://github.com/IDEA-Research/Grounded-SAM-2).
The simulator-facing interface is the same for either backend.

Do not use one narrow prompt list as the only discovery mechanism: it can hide
unanticipated alternatives. Use broad prompts or automatic masks, send the
visible crops to the foundation model for semantic/function ranking, and use
the point clouds only for deterministic geometric verification.

## Current geometric checks

- `ELONGATED_OBJECT`
- `OPEN_CAVITY`
- `PLANAR_SUPPORT`
- `INSERTABLE_IN`
- `REACHES_BOTTOM`

`ObservedGeometryState.select_first_compatible()` evaluates candidates in the
foundation model's order and stops at the first candidate for which every
required relation is `TRUE`. `UNKNOWN` requests more evidence; it is not
treated as success.

The current insertion checks compare observed dimensions. Motion execution
still needs a grasp-specific swept-volume, IK, and collision check.

## Region-order ablations

`inspection_policy.py` provides fixed, seeded-random, and validated ranked
policies. A foundation-model ranker receives only the currently available
region IDs and inspection history; invented or omitted regions are rejected.
All policies feed the same observation and geometry code, so the ablation
changes only region order.
