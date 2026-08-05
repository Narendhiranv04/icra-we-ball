# Third-party model notice

The Fetch MJCF and mesh assets loaded by this project are distributed through
Farama Foundation's `gymnasium-robotics` package under the MIT License.

- Project: https://github.com/Farama-Foundation/Gymnasium-Robotics
- License: https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/main/LICENSE
- Asset notice: the Fetch model is based on models provided by Fetch Robotics
  and was adapted/refined by OpenAI.

The assets are not copied into this repository. They are installed as the
pinned Python dependency `gymnasium-robotics==1.4.2` in the Docker image and
composed into the kitchen model at runtime.

## Google Robot from MuJoCo Menagerie

The optional Google Robot backend loads the `google_robot` MJCF and mesh assets
from an external MuJoCo Menagerie checkout. The model describes the mobile
manipulator developed by Google and is distributed under Apache-2.0:

- Model: https://github.com/google-deepmind/mujoco_menagerie/tree/main/google_robot
- License: https://github.com/google-deepmind/mujoco_menagerie/blob/main/google_robot/LICENSE

The assets are not copied into this repository. At runtime, the kitchen
adapter namespaces the model, adds ideal planar base joints and robot-mounted
cameras, and composes it with the shared scene.

## Mink

Profile-driven inverse kinematics uses Mink, distributed under Apache-2.0:

- Project: https://github.com/kevinzakka/mink
- License: https://github.com/kevinzakka/mink/blob/main/LICENSE

Mink is installed as the pinned Python dependency `mink==1.0.0`. The original
project code is not copied into this repository.

## YCB Object and Model Set

This repository contains normalized and texture-downsampled models derived
from the YCB Object and Model Set:

- `002_master_chef_can`
- `004_sugar_box`
- `008_pudding_box`
- `024_bowl`
- `025_mug`
- `029_plate`
- `030_fork`
- `031_spoon`
- `032_knife`
- `040_large_marker`
- `065-a_cups`

Canonical project and download page:
https://www.ycbbenchmarks.com/object-models/

The preparation script retrieves an individually-addressable YCB mirror to
avoid repeatedly downloading the much larger official archives, while the
manifest records both the canonical official archive and preparation URL.
YCB's site provides these models for incorporation into manipulation and
simulation software but does not state a blanket software-style license on its
object-model page. Product names, packaging, and textures may retain their
respective owners' rights. Confirm redistribution requirements before making
a repository containing these assets public.

Recommended citation:

> Berk Calli et al. "Benchmarking in Manipulation Research: The YCB Object
> and Model Set and Benchmarking Protocols." IEEE Robotics & Automation
> Magazine, 2015.

## Ultralytics YOLO-World and CLIP

Joint semantic grounding optionally installs and runs:

- `ultralytics==8.4.112` with the pretrained
  `yolov8m-worldv2.pt` checkpoint;
- the Ultralytics CLIP fork pinned at commit
  `c4b6ea0932a2c0f39a0fa528af5ec4982ff15cab`;
- the pretrained OpenAI CLIP `ViT-B/32` text-encoder weights retrieved by
  that package.

The Ultralytics package metadata and source repository identify the software
as GNU Affero General Public License v3.0. The pinned Ultralytics CLIP fork
also contains an AGPL-3.0 licence:

- https://github.com/ultralytics/ultralytics/blob/main/LICENSE
- https://github.com/ultralytics/CLIP/blob/c4b6ea0932a2c0f39a0fa528af5ec4982ff15cab/LICENSE

Model files are not committed to this repository. The documented preparation
script downloads them into a user-owned ignored cache and verifies:

```text
yolov8m-worldv2.pt
b614d33aa35b8e61d988041ff6939dfb3ed627af88ccaf643e4cdb822eb41d71

ViT-B-32.pt
40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af
```

Review the upstream licences and any commercial licensing requirements before
redistributing a container that embeds the detector packages or checkpoints.

## Google Scanned Objects

This repository contains normalized OBJ/PNG assets derived from Google
Scanned Objects, released under the Creative Commons Attribution 4.0
International license. The Nescafe jar derivative also removes its scanned
lid faces so the simulated jar can be shown open with project-authored rim
and powder primitives:

- Project: https://research.google/pubs/google-scanned-objects-a-high-quality-dataset-of-3d-scanned-household-items/
- Public catalog: https://fuel.gazebosim.org/1.0/GoogleResearch/models
- License: https://creativecommons.org/licenses/by/4.0/

The preparation script uses Kevin Zakka's MuJoCo conversion as a compact
transport for the original GSO OBJ/PNG payloads:
https://github.com/kevinzakka/mujoco_scanned_objects

That conversion's MJCF is MIT licensed; the 3D assets remain CC-BY-4.0. Exact
model IDs and hashes are listed in `assets/objects/meshes/manifest.json`.

Recommended citation:

> Laura Downs et al. "Google Scanned Objects: A High-Quality Dataset of
> 3D-Scanned Household Items." ICRA 2022.
