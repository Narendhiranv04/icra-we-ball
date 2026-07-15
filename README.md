# S1: Coffee Preparation with a Missing Mug

## Intended Workflow

The **S1 environment** is designed to demonstrate container-level missing-object search using the Fetch mobile manipulator.

The intended execution sequence is:

1. Fetch observes the closed workstation.

2. It detects the following visible objects on the countertop:

   * Kettle
   * Coffee jar
   * Sugar jar
   * Spoon

3. It determines that the required **mug is missing**.

4. Using its mug-location prior, Fetch selects container **C1** as the first search location.

5. Fetch navigates to a suitable manipulation pose in front of C1.

6. It opens C1 and observes the objects inside:

   ```text
   {mug, glass}
   ```

7. It selects the mug and transfers it to the countertop.

8. A later Task and Motion Planning (TAMP) system will:

   * Prepare the coffee.
   * Move the completed result to the serving area.

## Container-Opening Behaviour

Currently, `open_container()` directly commands the container actuator.

This implementation is intended as a **search and debugging action**, allowing the missing-object search pipeline to be developed independently of contact-based door manipulation.

In a later version, the Fetch gripper will physically open the same door joint through contact-based manipulation.

## Environment Documentation

The complete region definitions, object layout, and usage guide are available in:

[S1_ENVIRONMENT.md](S1_ENVIRONMENT.md)

## Cameras

### Existing Scene Cameras

The original five cameras remain available:

* `left_shoulder_camera`
* `right_shoulder_camera`
* `overhead_camera`
* `front_camera`
* `wrist_camera`

The `wrist_camera` is now mounted on the Fetch gripper.

### Fetch Camera

Fetch additionally provides:

* `head_camera_rgb`

## Running the Scene Locally

Navigate to the project directory:

```bash
cd /home/naren/RA_iiith
```

Launch the S1 environment using the front camera:

```bash
MUJOCO_GL=glfw /home/naren/miniconda3/bin/python \
  -m mujoco_scenes.scene_loader \
  --scene S1_coffee_missing_mug \
  --viewer \
  --camera front_camera
```
