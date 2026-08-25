"""Infer roles/properties/candidates from a goal and initial images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
import yaml

from .environment_vlm_requirements import (
    EnvironmentVLMRequirementProvider,
    SUPPORTED_ENVIRONMENTS,
)


DIRECT_SCENE_CAMERAS = {
    "kitchen": (
        "left_shoulder_camera",
        "right_shoulder_camera",
        "overhead_camera",
        "side_camera",
        "front_camera",
    ),
    "living_room": (
        "l2_camera_left",
        "l2_camera_right",
        "l2_camera_top",
        "l2_camera_front",
        "l2_camera_close",
    ),
}


def available_variants(environment: str) -> tuple[str, ...]:
    """Return the configured variant IDs without constructing a scene."""
    if environment == "kitchen":
        from .kitchen_feasibility_oracle import load_feasibility_benchmark_config

        variants = load_feasibility_benchmark_config()["variants"]
    elif environment == "living_room":
        from .living_room_variants import load_living_room_variants

        variants = load_living_room_variants()
    else:  # guarded by argparse/provider, retained for direct callers
        raise ValueError(f"Unsupported environment: {environment}")
    return tuple(variants)


def render_variant_initial_observation(
    environment: str,
    variant: str,
    output_dir: Path,
    *,
    width: int = 1280,
    height: int = 720,
) -> tuple[list[Path], str]:
    """Construct one no-robot variant and save fresh, unannotated RGB views."""
    valid_variants = available_variants(environment)
    if variant not in valid_variants:
        raise ValueError(
            f"Unknown {environment} variant {variant!r}; choose one of: "
            + ", ".join(valid_variants)
        )

    if environment == "kitchen":
        from .kitchen_feasibility_oracle import load_feasibility_benchmark_config
        from .scene_loader import KitchenScene

        scene_name = load_feasibility_benchmark_config()["variants"][variant][
            "scene_name"
        ]
        scene = KitchenScene(scene_name, include_robot=False, robot="none")
    else:
        from .living_room_region_scene import L2LivingRoomRegionScene
        from .living_room_variants import scene_name as living_room_scene_name

        scene_name = living_room_scene_name(variant)
        scene = L2LivingRoomRegionScene(scene_name, robot="none")

    output_dir.mkdir(parents=True, exist_ok=True)
    images: list[Path] = []
    for camera in DIRECT_SCENE_CAMERAS[environment]:
        path = output_dir / f"{camera}.png"
        frame = scene.render_frame(camera=camera, width=width, height=height)
        Image.fromarray(frame).save(path)
        images.append(path)
    return images, scene_name


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", choices=SUPPORTED_ENVIRONMENTS, required=True)
    parser.add_argument(
        "--variant",
        help=(
            "Configured scene variant to instantiate and render directly. "
            "Use this for normal variant-specific inference."
        ),
    )
    parser.add_argument("--instruction")
    parser.add_argument(
        "--image",
        type=Path,
        action="append",
        help=(
            "External initial-observation image override; repeat for multiple "
            "views (max 8). Cannot be combined with --variant."
        ),
    )
    parser.add_argument("--render-width", type=int, default=1280)
    parser.add_argument("--render-height", type=int, default=720)
    parser.add_argument(
        "--observation-dir",
        type=Path,
        help="Directory for raw RGB frames rendered by --variant",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--normalized-task-output", type=Path)
    arguments = parser.parse_args()

    if bool(arguments.variant) == bool(arguments.image):
        parser.error("provide exactly one of --variant or one/more --image overrides")

    variant_root = (
        Path("outputs/vlm_requirements") / arguments.environment / arguments.variant
        if arguments.variant
        else None
    )
    scene_name = None
    if arguments.variant:
        observation_dir = arguments.observation_dir or (
            variant_root / "initial_observation"
        )
        observation_images, scene_name = render_variant_initial_observation(
            arguments.environment,
            arguments.variant,
            observation_dir,
            width=arguments.render_width,
            height=arguments.render_height,
        )
    else:
        observation_images = arguments.image

    output = arguments.output or Path(
        variant_root / "requirements.json"
        if variant_root
        else f"outputs/{arguments.environment}_vlm_requirements.json"
    )
    task_output = arguments.normalized_task_output or Path(
        variant_root / "task_requirements.yaml"
        if variant_root
        else f"outputs/{arguments.environment}_vlm_task_requirements.yaml"
    )
    provider = EnvironmentVLMRequirementProvider(arguments.environment)
    result = provider.generate(
        arguments.instruction, observation_images=observation_images
    )
    if arguments.variant:
        result["scene_observation"] = {
            "source": "DIRECT_MUJOCO_INITIAL_RENDER",
            "environment": arguments.environment,
            "variant": arguments.variant,
            "scene_name": scene_name,
            "robot": "none",
            "width": arguments.render_width,
            "height": arguments.render_height,
            "cameras": list(DIRECT_SCENE_CAMERAS[arguments.environment]),
            "images": [str(path) for path in observation_images],
        }
    _atomic_text(output, json.dumps(result, indent=2, sort_keys=True) + "\n")
    normalized_task = result["normalized_task_contract"]
    if normalized_task is not None:
        _atomic_text(
            task_output,
            yaml.safe_dump(normalized_task, sort_keys=False),
        )
    elif task_output.exists():
        _atomic_text(
            task_output,
            yaml.safe_dump(
                {
                    "ready_for_grounding": False,
                    "reviewed_ontology_audit": result["reviewed_ontology_audit"],
                    "note": (
                        "This blocking marker replaced a stale normalized task. "
                        "It is not a grounding contract."
                    ),
                },
                sort_keys=False,
            ),
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"\nWrote {output}")
    if normalized_task is not None:
        print(f"Wrote {task_output}")
    else:
        print("No usable normalized task was written: ontology audit requires review")


if __name__ == "__main__":
    main()
