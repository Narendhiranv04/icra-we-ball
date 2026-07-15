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
