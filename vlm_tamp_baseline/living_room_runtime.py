"""Planning-only Living Room adapter for the VLM-TAMP baseline.

The VLM sees freshly rendered RGB views with semantic instance aliases and an
observable alias-to-planning-ID map. MuJoCo is never advanced by a planned
action: PDDLStream actions are applied to a small symbolic rollout solely so
their high-level sequence can be evaluated against the frozen GT catalogue.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from baseline_common.artifacts import write_json
from baseline_common.models import Action, ActionResult, Entity, Observation, Region
from mujoco_scenes.final_paper_variant_labels import (
    paper_variant_label,
    resolve_variant_name,
)
from mujoco_scenes.living_room_mobile_execution import resolve_execution_entities
from mujoco_scenes.living_room_recorder import L2_FIVE_CAMERAS
from mujoco_scenes.living_room_region_scene import L2LivingRoomRegionScene
from mujoco_scenes.living_room_variants import (
    load_living_room_variant_contract,
    scene_name,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE1_ROOT = (
    REPOSITORY_ROOT
    / "mujoco_scenes"
    / "benchmark_reports"
    / "living_room_region_feasibility_phase1"
    / "variants"
)
DEFAULT_EXPECTED_ROOT = REPOSITORY_ROOT / "EXPECTED_GT_ACTIONS" / "living_room"

LOGICAL_REGION_GEOMS = {
    "PERSONAL_TABLE_LEFT": "a2_personal_left_top",
    "PERSONAL_TABLE_RIGHT": "a2_personal_right_top",
    "SHARED_TABLE": "a2_control_table_top",
}
STAGING_REGION_ID = "staging_area"
STAGING_REGION_GEOM = "a2_staging_top"
LIVING_ROOM_CAMERA_SUBSETS = {
    1: ("l2_camera_top",),
    3: ("l2_camera_left", "l2_camera_right", "l2_camera_top"),
    5: L2_FIVE_CAMERAS,
}


def _unique_aliases(labels: Mapping[str, str]) -> dict[str, str]:
    totals: dict[str, int] = {}
    for label in labels.values():
        normalized = label.strip().lower().replace(" ", "_")
        totals[normalized] = totals.get(normalized, 0) + 1
    seen: dict[str, int] = {}
    result = {}
    for object_id, label in sorted(labels.items()):
        normalized = label.strip().lower().replace(" ", "_")
        seen[normalized] = seen.get(normalized, 0) + 1
        result[object_id] = (
            normalized
            if totals[normalized] == 1
            else f"{normalized}_{seen[normalized]}"
        )
    return result


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _body_geom_ids(model: mujoco.MjModel, root_body_name: str) -> frozenset[int]:
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body_name)
    if root < 0:
        raise ValueError(f"Missing Living Room payload body {root_body_name!r}")
    result: set[int] = set()
    pending = [root]
    while pending:
        body = pending.pop()
        start = int(model.body_geomadr[body])
        result.update(range(start, start + int(model.body_geomnum[body])))
        pending.extend(
            child
            for child in range(1, model.nbody)
            if int(model.body_parentid[child]) == body
        )
    return frozenset(result)


def _visible_support_geom_ids(
    model: mujoco.MjModel, marker_geom_id: int
) -> frozenset[int]:
    """Use rendered support geometry rather than an invisible logical marker."""
    body_id = int(model.geom_bodyid[marker_geom_id])
    visible = frozenset(
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) == body_id
        and float(model.geom_rgba[geom_id, 3]) > 0.0
    )
    return visible or frozenset({marker_geom_id})


@dataclass(frozen=True)
class ExpectedGT:
    variant: str
    internal_variant: str
    intended_outcome: str
    actions: tuple[Mapping[str, Any], ...]

    @classmethod
    def load(cls, root: Path, variant: str) -> "ExpectedGT":
        payload = _read(root / variant / "expected_gt_actions.json")
        return cls(
            variant=str(payload["variant"]),
            internal_variant=str(payload["internal_variant"]),
            intended_outcome=str(payload["intended_outcome"]),
            actions=tuple(payload.get("actions", ())),
        )


class LivingRoomSymbolicExecutor:
    """Validate and apply PICK/PLACE actions without moving MuJoCo."""

    def __init__(self, runtime: "LivingRoomPlanningRuntime"):
        self.runtime = runtime

    def execute(self, action: Action) -> ActionResult:
        skill = action.skill.upper()
        if skill == "PICK":
            object_id = action.arguments.get("object_id", "")
            if object_id not in self.runtime.object_roles:
                return ActionResult.failed("unknown_object", f"Unknown object {object_id}")
            if self.runtime.held_object is not None:
                return ActionResult.failed(
                    "gripper_occupied",
                    f"Already holding {self.runtime.held_object}",
                )
            self.runtime.held_object = object_id
            self.runtime.locations[object_id] = None
            self.runtime.revision += 1
            return ActionResult.succeeded(f"holding({object_id})")
        if skill == "PLACE":
            object_id = action.arguments.get("object_id", "")
            region_id = action.arguments.get("region_id", "")
            if self.runtime.held_object != object_id:
                return ActionResult.failed(
                    "not_holding_object", f"The robot is not holding {object_id}"
                )
            if region_id not in self.runtime.region_roles:
                return ActionResult.failed("unknown_region", f"Unknown region {region_id}")
            self.runtime.held_object = None
            self.runtime.locations[object_id] = region_id
            self.runtime.revision += 1
            return ActionResult.succeeded(f"placed({object_id},{region_id})")
        return ActionResult.failed(
            "unsupported_planning_action",
            f"Living Room planning-only mode cannot apply {action.skill}",
            recoverable=False,
        )


class LivingRoomPlanningRuntime:
    """Fresh five-view observation plus private symbolic evaluation state."""

    def __init__(
        self,
        variant: str,
        output_dir: Path,
        *,
        phase1_root: Path = DEFAULT_PHASE1_ROOT,
        expected_root: Path = DEFAULT_EXPECTED_ROOT,
        image_width: int = 960,
        image_height: int = 540,
        camera_count: int = 5,
        robot: str = "none",
        physical_execution: bool = False,
    ):
        internal = resolve_variant_name("living_room", variant)
        contract = load_living_room_variant_contract()
        if internal not in contract["variants"]:
            raise ValueError(f"Unknown Living Room variant {variant!r}")
        self.internal_variant = internal
        self.variant = paper_variant_label("living_room", internal)
        self.variant_spec = contract["variants"][internal]
        self.goal = str(contract["task"])
        self.output_dir = output_dir
        self.phase1_dir = phase1_root / internal
        self.expected = ExpectedGT.load(expected_root, self.variant)
        if self.expected.internal_variant != internal:
            raise ValueError("GT variant does not match the requested scene")
        self.payload_registry = _read(self.phase1_dir / "payload_registry.json")
        self.region_registry = _read(self.phase1_dir / "region_registry.json")
        self.scene = L2LivingRoomRegionScene(scene_name(internal), robot=robot)
        if camera_count not in LIVING_ROOM_CAMERA_SUBSETS:
            raise ValueError("Living Room camera_count must be one of 1, 3, or 5")
        self.camera_count = camera_count
        self.selected_cameras = LIVING_ROOM_CAMERA_SUBSETS[camera_count]
        self.image_width = image_width
        self.image_height = image_height
        self.renderer: mujoco.Renderer | None = None
        self._images_cache: tuple[dict[str, str], ...] | None = None
        self.revision = 0
        self.held_object: str | None = None

        resolution = resolve_execution_entities(
            self.scene.model,
            self.scene.data,
            self.payload_registry,
            self.region_registry,
        )
        self.resolution = resolution
        self.object_backends = {
            str(row["generic_object_id"]): str(row["backend_body"])
            for row in resolution["objects"]
        }
        self.region_backends = {
            str(row["generic_region_id"]): str(row["backend_support_geom"])
            for row in resolution["regions"]
        }
        registry_objects = self.payload_registry["objects"]
        self.object_roles = {
            object_id: str(registry_objects[object_id]["semantic_payload_role"])
            for object_id in self.object_backends
        }
        semantic_by_backend = {
            backend: logical for logical, backend in LOGICAL_REGION_GEOMS.items()
        }
        self.region_roles = {
            region_id: semantic_by_backend[backend]
            for region_id, backend in self.region_backends.items()
            if backend in semantic_by_backend
        }
        if len(self.object_backends) != len(registry_objects):
            raise RuntimeError("Not every observed payload resolved to a MuJoCo body")
        if len(self.region_roles) != len(self.region_registry["regions"]):
            raise RuntimeError("Not every observed region resolved to a task support")
        self.region_roles[STAGING_REGION_ID] = "STAGING_AREA"
        self.object_aliases = _unique_aliases(self.object_roles)
        self.region_aliases = {
            region_id: role.lower() for region_id, role in self.region_roles.items()
        }

        object_by_backend = {value: key for key, value in self.object_backends.items()}
        region_by_logical = {
            logical: next(
                region_id
                for region_id, backend in self.region_backends.items()
                if backend == backend_geom
            )
            for logical, backend_geom in LOGICAL_REGION_GEOMS.items()
            if backend_geom in self.region_backends.values()
        }
        self.locations: dict[str, str | None] = {
            object_id: "staging_area" for object_id in self.object_backends
        }
        for backend_object, logical_region in self.variant_spec.get(
            "object_locations", {}
        ).items():
            object_id = object_by_backend.get(str(backend_object))
            region_id = region_by_logical.get(str(logical_region))
            if object_id is None or region_id is None:
                raise RuntimeError(
                    f"Cannot resolve initial placement {backend_object} -> {logical_region}"
                )
            self.locations[object_id] = region_id

        self.object_geom_ids = {
            object_id: _body_geom_ids(self.scene.model, backend)
            for object_id, backend in self.object_backends.items()
        }
        self.region_geom_ids = {}
        for region_id, backend in self.region_backends.items():
            marker = mujoco.mj_name2id(
                self.scene.model, mujoco.mjtObj.mjOBJ_GEOM, backend
            )
            self.region_geom_ids[region_id] = _visible_support_geom_ids(
                self.scene.model, marker
            )
        staging_marker = mujoco.mj_name2id(
            self.scene.model, mujoco.mjtObj.mjOBJ_GEOM, STAGING_REGION_GEOM
        )
        self.region_geom_ids[STAGING_REGION_ID] = _visible_support_geom_ids(
            self.scene.model, staging_marker
        )
        if any(next(iter(ids)) < 0 for ids in self.region_geom_ids.values()):
            raise RuntimeError("A Living Room support annotation geom is missing")

        self.inventory = self._build_private_inventory()
        write_json(
            output_dir / "shared_observation_contract.json",
            {
                "scene": "living_room",
                "variant": self.variant,
                "internal_variant": internal,
                "camera_count": self.camera_count,
                "camera_ids": list(self.selected_cameras),
                "camera_ablation": "NESTED_TOP_LEFT_RIGHT_THEN_ALL",
                "rgb_annotation": "UNIQUE_SEMANTIC_ALIASES_ONLY",
                "semantic_labels_exposed": True,
                "alias_to_planning_id_map_exposed": True,
                "physical_execution": bool(physical_execution),
            },
        )
        write_json(
            output_dir / "_private_evaluation" / "adapter_resolution.json",
            resolution,
        )
        write_json(
            output_dir / "_private_evaluation" / "expected_gt_actions.json",
            {
                "variant": self.expected.variant,
                "intended_outcome": self.expected.intended_outcome,
                "actions": list(self.expected.actions),
            },
        )

    def _build_private_inventory(self) -> dict[str, Any]:
        rows = []
        for object_id, record in sorted(self.payload_registry["objects"].items()):
            geometry = record["geometry"]
            rows.append(
                {
                    "generic_object_id": object_id,
                    "semantic_label": record["semantic_payload_role"],
                    "observed_dimensions_m": {
                        "length": geometry["footprint_length_m"]["value"],
                        "width": geometry["footprint_width_m"]["value"],
                    },
                    "source_context": {
                        "observed_source_region": self.locations[object_id],
                        "required_workspace": "home",
                    },
                }
            )
        return {"objects": rows}

    @property
    def aliases(self) -> dict[str, str]:
        aliases = dict(self.object_aliases)
        aliases.update(self.region_aliases)
        return aliases

    def goal_verifier(self, _observation: Observation | None = None) -> bool:
        """Check the task relation, not the catalogue's canonical ID order."""
        shared = [
            region for region, role in self.region_roles.items() if role == "SHARED_TABLE"
        ]
        personal = [
            region
            for region, role in self.region_roles.items()
            if role in {"PERSONAL_TABLE_LEFT", "PERSONAL_TABLE_RIGHT"}
        ]
        if len(shared) != 1 or len(personal) != 2:
            return False
        for region in personal:
            roles = {
                self.object_roles[object_id]
                for object_id, location in self.locations.items()
                if location == region
            }
            if not {"cup", "saucer"} <= roles:
                return False
        return any(
            self.object_roles[object_id] == "tv_remote" and location == shared[0]
            for object_id, location in self.locations.items()
        )

    def observe_state(self) -> Observation:
        entities = tuple(
            Entity(
                object_id,
                "object",
                self.object_aliases[object_id],
                (
                    {"location": "held"}
                    if object_id == self.held_object
                    else {"region_id": str(self.locations[object_id])}
                ),
            )
            for object_id in sorted(self.object_roles)
        )
        regions = tuple(
            Region(region_id, self.region_aliases[region_id], "open", True)
            for region_id in sorted(self.region_roles)
        )
        observation = Observation(
            "living_room",
            self.revision,
            entities,
            regions,
            {"workspace": "home", "holding": self.held_object},
            self.goal_verifier(),
        )
        write_json(
            self.output_dir / "latest_observation.json",
            observation.as_semantic_neutral_prompt_dict(),
        )
        write_json(
            self.output_dir / "_private_evaluation" / "latest_observation.json",
            observation.as_prompt_dict(),
        )
        return observation

    def observe(self) -> tuple[Observation, tuple[dict[str, str], ...]]:
        return self.observe_state(), self.images()

    @staticmethod
    def _box(
        segmentation: np.ndarray, geom_ids: frozenset[int]
    ) -> tuple[int, int, int, int, int] | None:
        mask = (
            (segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
            & np.isin(segmentation[:, :, 0], tuple(geom_ids))
        )
        rows, columns = np.nonzero(mask)
        if len(rows) < 6:
            return None
        return (
            int(columns.min()),
            int(rows.min()),
            int(columns.max()),
            int(rows.max()),
            int(len(rows)),
        )

    @staticmethod
    def _draw_label(
        draw: ImageDraw.ImageDraw,
        box: tuple[int, int, int, int, int],
        identifier: str,
        color: tuple[int, int, int],
        font: ImageFont.ImageFont,
        *,
        image_size: tuple[int, int],
        occupied: list[tuple[int, int, int, int]],
        forbidden: tuple[tuple[int, int, int, int], ...],
        avoid_target: bool,
    ) -> None:
        left, top, right, bottom, _ = box
        draw.rectangle((left, top, right, bottom), outline=color, width=2)
        bounds = draw.textbbox((left, top), identifier, font=font)
        width = bounds[2] - bounds[0] + 4
        height = bounds[3] - bounds[1] + 4
        image_width, image_height = image_size
        inside_candidates = [
            ((left + right - width) // 2, top + 3),
            ((left + right - width) // 2, bottom - height - 3),
        ]
        outside_candidates = []
        for gap in (2, height + 4, 2 * height + 6):
            outside_candidates.extend(
                [
                    (left, top - height - gap),
                    (right - width, top - height - gap),
                    (left, bottom + gap),
                    (right - width, bottom + gap),
                    (right + gap, top),
                    (left - width - gap, top),
                ]
            )
        candidates = (
            outside_candidates + inside_candidates
            if avoid_target
            else inside_candidates + outside_candidates
        )

        def area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
            return max(0, min(a[2], b[2]) - max(a[0], b[0]) + 1) * max(
                0, min(a[3], b[3]) - max(a[1], b[1]) + 1
            )

        target = (left, top, right, bottom)
        ranked = []
        for order, (x, y) in enumerate(candidates):
            x = min(max(0, x), max(0, image_width - width - 1))
            y = min(max(0, y), max(0, image_height - height - 1))
            label_box = (x, y, x + width, y + height)
            ranked.append(
                (
                    (area(label_box, target) if avoid_target else 0) * 10_000
                    + sum(area(label_box, row) for row in forbidden) * 5_000
                    + sum(area(label_box, row) for row in occupied) * 5_000
                    + order,
                    label_box,
                )
            )
        _, label_box = min(ranked, key=lambda row: row[0])
        occupied.append(label_box)
        draw.rectangle(label_box, fill="black")
        draw.text((label_box[0] + 2, label_box[1] + 2), identifier, fill=color, font=font)

    def _annotate(
        self, frame: np.ndarray, segmentation: np.ndarray
    ) -> tuple[Image.Image, dict[str, Any]]:
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.load_default(size=11)
        except TypeError:
            font = ImageFont.load_default()
        manifest: dict[str, Any] = {"objects": [], "regions": []}
        region_rows = []
        for region_id, geom_ids in sorted(self.region_geom_ids.items()):
            box = self._box(segmentation, geom_ids)
            if box:
                region_rows.append((region_id, box))
        object_rows = []
        for object_id, geom_ids in sorted(self.object_geom_ids.items()):
            box = self._box(segmentation, geom_ids)
            if box:
                object_rows.append((object_id, box))
        object_boxes = tuple(tuple(box[:4]) for _, box in object_rows)
        occupied: list[tuple[int, int, int, int]] = []
        for region_id, box in region_rows:
            role = self.region_aliases.get(region_id, region_id).replace("_", " ")
            self._draw_label(
                draw, box, role, (255, 190, 40), font,
                image_size=image.size,
                occupied=occupied,
                forbidden=object_boxes,
                avoid_target=False,
            )
            manifest["regions"].append(
                {
                    "id": region_id,
                    "semantic_label": role,
                    "bbox_xyxy": list(box[:4]),
                    "pixel_count": box[4],
                }
            )
        for object_id, box in object_rows:
            role = self.object_aliases[object_id].replace("_", " ")
            self._draw_label(
                draw, box, role, (40, 235, 255), font,
                image_size=image.size,
                occupied=occupied,
                forbidden=object_boxes,
                avoid_target=True,
            )
            manifest["objects"].append(
                {
                    "id": object_id,
                    "semantic_label": role,
                    "bbox_xyxy": list(box[:4]),
                    "pixel_count": box[4],
                }
            )
        return image, manifest

    def images(self) -> tuple[dict[str, str], ...]:
        if self._images_cache is not None:
            return self._images_cache
        self.renderer = mujoco.Renderer(
            self.scene.model, height=self.image_height, width=self.image_width
        )
        directory = self.output_dir / "observations" / "initial"
        directory.mkdir(parents=True, exist_ok=True)
        manifest: dict[str, Any] = {
            "semantics_exposed": True,
            "object_annotation": "UNIQUE_SEMANTIC_ALIAS_ONLY",
            "region_annotation": "UNIQUE_SEMANTIC_ALIAS_ONLY",
            "cameras": {},
        }
        images: list[dict[str, str]] = []
        mujoco.mj_forward(self.scene.model, self.scene.data)
        for camera in self.selected_cameras:
            self.renderer.update_scene(self.scene.data, camera=camera)
            frame = self.renderer.render().copy()
            Image.fromarray(frame).save(directory / f"raw_{camera}.png")
            self.renderer.enable_segmentation_rendering()
            segmentation = self.renderer.render().copy()
            self.renderer.disable_segmentation_rendering()
            annotated, rows = self._annotate(frame, segmentation)
            buffer = BytesIO()
            annotated.save(buffer, format="PNG")
            content = buffer.getvalue()
            (directory / f"{camera}.png").write_bytes(content)
            manifest["cameras"][camera] = rows
            images.append(
                {
                    "camera": camera,
                    "data_url": "data:image/png;base64,"
                    + base64.b64encode(content).decode("ascii"),
                }
            )
        write_json(directory / "annotations.json", manifest)
        self._images_cache = tuple(images)
        self.latest_annotation_manifest = manifest
        return self._images_cache

    def invalidate_images(self) -> None:
        """Force fresh rendered evidence after physical state changes."""
        self._images_cache = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


def canonical_actions(history: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in history:
        if not row.get("success"):
            continue
        action = row.get("action", {})
        skill = str(action.get("skill", "")).upper()
        arguments = action.get("arguments", {})
        if skill == "PICK":
            values = [str(arguments["object_id"])]
        elif skill == "PLACE":
            values = [str(arguments["object_id"]), str(arguments["region_id"])]
        else:
            continue
        result.append({"operator": skill, "arguments": values})
    return result


def compare_action_sequences(
    predicted: Sequence[Mapping[str, Any]], expected: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    def token(row: Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
        return str(row["operator"]), tuple(map(str, row.get("arguments", ())))

    def scores(left: Sequence[Any], right: Sequence[Any]) -> tuple[int, float, float, float]:
        table = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
        for i, item in enumerate(left, start=1):
            for j, other in enumerate(right, start=1):
                table[i][j] = (
                    table[i - 1][j - 1] + 1
                    if item == other
                    else max(table[i - 1][j], table[i][j - 1])
                )
        lcs = table[-1][-1]
        precision = lcs / len(left) if left else float(not right)
        recall = lcs / len(right) if right else float(not left)
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        return lcs, precision, recall, f1

    left = [token(row) for row in predicted]
    right = [token(row) for row in expected]
    lcs, precision, recall, f1 = scores(left, right)
    operator_lcs, operator_precision, operator_recall, operator_f1 = scores(
        [item[0] for item in left], [item[0] for item in right]
    )
    return {
        "exact_sequence_match": left == right,
        "predicted_action_count": len(left),
        "expected_action_count": len(right),
        "lcs_action_count": lcs,
        "ordered_precision": precision,
        "ordered_recall": recall,
        "ordered_f1": f1,
        "operator_lcs_action_count": operator_lcs,
        "operator_ordered_precision": operator_precision,
        "operator_ordered_recall": operator_recall,
        "operator_ordered_f1": operator_f1,
        "predicted_actions": list(predicted),
        "expected_actions": list(expected),
    }


__all__ = [
    "DEFAULT_EXPECTED_ROOT",
    "DEFAULT_PHASE1_ROOT",
    "ExpectedGT",
    "LivingRoomPlanningRuntime",
    "LivingRoomSymbolicExecutor",
    "LIVING_ROOM_CAMERA_SUBSETS",
    "canonical_actions",
    "compare_action_sequences",
]
