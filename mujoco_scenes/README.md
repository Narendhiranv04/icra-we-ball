# MuJoCo Kitchen Scene S1

This package builds the `S1_coffee_missing_mug` scene dynamically from the
shared kitchen XML, object library, and YAML configuration.

## Five camera views

- `left_shoulder_camera`
- `right_shoulder_camera`
- `overhead_camera`
- `wrist_camera`
- `front_camera`

Fetch also contributes `head_camera_rgb`. It is available from the same
`--camera` CLI option whenever the robot is enabled.

With the Fetch robot enabled, `wrist_camera` is attached to its gripper link.
The kitchen-only `--no-robot` mode uses a fixed placeholder with the same name.

The Fetch MJCF and mesh assets are supplied at runtime by the MIT-licensed
`gymnasium-robotics` package. Its Fetch model is based on Fetch Robotics assets
and was adapted by OpenAI/Farama. See [S1_ENVIRONMENT.md](S1_ENVIRONMENT.md) for
the region map, robot controls, and intended S1 episode.

## Run the interactive viewer in Docker (Linux/X11)

From the repository root:

```bash
docker build -t mujoco-kitchen-s1 -f mujoco_scenes/Dockerfile .
xhost +local:docker
docker run --rm -it \
  -e DISPLAY="$DISPLAY" \
  -e MUJOCO_GL=glfw \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  mujoco-kitchen-s1 \
  --scene S1_coffee_missing_mug --viewer --camera front_camera
xhost -local:docker
```

The same viewer can be started with Compose after allowing the local Docker
X11 connection:

```bash
xhost +local:docker
docker compose up --build kitchen-s1
```

Use the MuJoCo viewer's camera selector to switch between all five cameras.
To start with C1 already open, append `--open-container C1` to the `docker run`
command. The actuator controls in the viewer UI can also move every door,
drawer, and the box lid.

## Render without opening a GUI

```bash
docker run --rm \
  -e MUJOCO_GL=osmesa \
  -v "$PWD:/output" \
  mujoco-kitchen-s1 \
  --scene S1_coffee_missing_mug \
  --camera overhead_camera \
  --render /output/s1_overhead.png
```
