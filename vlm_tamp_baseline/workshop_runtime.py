"""Planning-only Workshop adapter shared by the VLM-TAMP and OWL-TAMP checks.

The adapter renders the closed workcell, exposes only the frame joint and
labelled storage regions, and keeps each variant's storage contents private
until a symbolic ``INSPECT`` action succeeds.  It never advances MuJoCo for a
baseline action; the symbolic rollout exists only for planning-to-GT scoring.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml

from baseline_common.artifacts import write_json
from baseline_common.models import Action, ActionResult, Entity, Observation, Region
from mujoco_scenes.final_paper_variant_labels import paper_variant_label, resolve_variant_name
from mujoco_scenes.scene_loader import ROBOT_NONE
from mujoco_scenes.workshop_scene import (
    WORKSHOP_ASSETS_DIR,
    WORKSHOP_CAMERAS,
    build_workshop_xml,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPECTED_ROOT = REPOSITORY_ROOT / "EXPECTED_GT_ACTIONS" / "workshop"
VARIANT_CONFIG = REPOSITORY_ROOT / "mujoco_scenes" / "configs" / "workshop_variants.yaml"

WORKSHOP_CAMERA_SUBSETS = {
    1: ("workshop_camera_top",),
    3: ("workshop_camera_left", "workshop_camera_right", "workshop_camera_top"),
    5: WORKSHOP_CAMERAS,
}
STORAGE_BACKENDS = ("LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET")
WORK_SURFACE = "MAIN_WORKBENCH_ZONE"
OBJECT_BACKENDS = (
    "workshop_frame_joint",
    "workshop_long_phillips_driver",
    "workshop_power_driver",
    "workshop_medium_phillips_screw",
    "workshop_wooden_hammer",
)
OBJECT_LABELS = {
    "workshop_frame_joint": "frame joint",
    "workshop_long_phillips_driver": "manual screwdriver",
    "workshop_power_driver": "power screwdriver",
    "workshop_medium_phillips_screw": "screw",
    "workshop_wooden_hammer": "wooden hammer",
}
REGION_LABELS = {
    "LEFT_DRAWER": "left drawer",
    "RIGHT_DRAWER": "right drawer",
    "TOOL_CABINET": "tool cabinet",
    WORK_SURFACE: "main workbench",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("variants"), dict):
        raise ValueError(f"Invalid Workshop variant configuration: {path}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _body_geom_ids(model: mujoco.MjModel, name: str) -> frozenset[int]:
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if root < 0:
        raise ValueError(f"Missing Workshop body {name!r}")
    result: set[int] = set()
    pending = [root]
    while pending:
        body = pending.pop()
        start = int(model.body_geomadr[body])
        result.update(range(start, start + int(model.body_geomnum[body])))
        pending.extend(
            child for child in range(1, model.nbody) if int(model.body_parentid[child]) == body
        )
    return frozenset(result)


def _add_closed_right_drawer(worldbody: ET.Element) -> None:
    """Add the third closed storage unit missing from the legacy W1 XML.

    The planning benchmark observes the cell before storage inspection.  This
    static visual counterpart gives every W1--W10 variant the same visible,
    labelled three-storage workcell without fabricating hidden contents.
    """
    # The legacy W1 XML has a solid right cabinet where the later benchmark
    # uses a second drawer.  Position this closed drawer just forward of that
    # cabinet so it remains visibly distinguishable in every camera subset.
    drawer = ET.SubElement(worldbody, "body", {"name": "baseline_right_drawer", "pos": "0.58 -0.05 0.38"})
    common = {"material": "cabinet_body_mat", "contype": "0", "conaffinity": "0"}
    for name, attrs in (
        ("baseline_right_drawer_floor", {"type": "box", "size": "0.38 0.30 0.025"}),
        ("baseline_right_drawer_front", {"type": "box", "pos": "0 -0.31 0.09", "size": "0.38 0.025 0.11", "material": "cabinet_face_mat"}),
        ("baseline_right_drawer_back", {"type": "box", "pos": "0 0.29 0.07", "size": "0.38 0.018 0.07"}),
        ("baseline_right_drawer_left", {"type": "box", "pos": "-0.37 0 0.07", "size": "0.018 0.29 0.07"}),
        ("baseline_right_drawer_right", {"type": "box", "pos": "0.37 0 0.07", "size": "0.018 0.29 0.07"}),
        ("baseline_right_drawer_lid", {"type": "box", "pos": "0 0 0.155", "size": "0.38 0.30 0.018"}),
    ):
        ET.SubElement(drawer, "geom", {"name": name, **common, **attrs})


def _add_visual_proxy(
    worldbody: ET.Element,
    backend: str,
    position: tuple[float, float, float],
) -> None:
    """Add a static visual instance for one private storage item.

    The legacy Workshop scene has one fixed tool arrangement.  The benchmark
    variants need different contents, so their items are lightweight visual
    proxies.  They are hidden while the source storage is closed and become
    visible only after the symbolic inspection transition.
    """
    body = ET.SubElement(
        worldbody,
        "body",
        {"name": f"baseline_proxy_{backend}", "pos": " ".join(map(str, position))},
    )
    common = {"contype": "0", "conaffinity": "0"}
    if backend == "workshop_long_phillips_driver":
        ET.SubElement(body, "geom", {
            "type": "capsule", "fromto": "-0.13 0 0 0.02 0 0", "size": "0.034",
            "material": "screwdrivers_visual_mat", **common,
        })
        ET.SubElement(body, "geom", {
            "type": "capsule", "fromto": "0.02 0 0 0.22 0 0", "size": "0.007",
            "material": "polished_steel", **common,
        })
    elif backend == "workshop_power_driver":
        ET.SubElement(body, "geom", {
            "type": "capsule", "fromto": "-0.12 0 0 0.10 0 0", "size": "0.048",
            "material": "drill_visual_mat", **common,
        })
        ET.SubElement(body, "geom", {
            "type": "capsule", "fromto": "-0.03 0 0 0.01 0 -0.11", "size": "0.028",
            "material": "dark_steel", **common,
        })
        ET.SubElement(body, "geom", {
            "type": "capsule", "fromto": "0.12 0 0 0.25 0 0", "size": "0.006",
            "material": "polished_steel", **common,
        })
    elif backend == "workshop_medium_phillips_screw":
        ET.SubElement(body, "geom", {
            "type": "cylinder", "quat": "0.7071 0 0.7071 0", "size": "0.012 0.052",
            "material": "polished_steel", **common,
        })
        ET.SubElement(body, "geom", {
            "type": "cylinder", "pos": "0.06 0 0", "quat": "0.7071 0 0.7071 0",
            "size": "0.023 0.008", "material": "screwdrivers_visual_mat", **common,
        })
    elif backend == "workshop_wooden_hammer":
        ET.SubElement(body, "geom", {
            "type": "capsule", "fromto": "-0.13 0 0 0.12 0 0", "size": "0.018",
            "material": "wood_frame_mat", **common,
        })
        ET.SubElement(body, "geom", {
            "type": "box", "pos": "0.11 0 0", "size": "0.05 0.085 0.040",
            "material": "dark_steel", **common,
        })
    else:
        raise ValueError(f"No visual proxy shape for {backend!r}")


def _closed_workshop_model(
    storage_contents: Mapping[str, Sequence[str]],
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    root = ET.fromstring(build_workshop_xml(ROBOT_NONE))
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("Workshop XML has no worldbody")
    # Storage lives below the workbench in the scene XML, so it is not a
    # direct child of ``worldbody``.  Search the complete XML tree; using
    # ``findall('body')`` here made every Workshop baseline episode fail
    # before it could render its first observation.
    left = next(
        (body for body in worldbody.iter("body") if body.get("name") == "left_tool_drawer"),
        None,
    )
    if left is None:
        raise RuntimeError("Workshop XML has no left drawer")
    # In the legacy scene the drawer is entirely beneath the workbench top, so
    # its region label never reaches the cameras.  Bring its closed front into
    # the shared observable workspace; this changes no private contents.
    left.set("pos", "-0.60 -0.22 0.38")
    ET.SubElement(
        left,
        "geom",
        {
            "name": "baseline_left_drawer_lid",
            "type": "box",
            "pos": "0 0 0.155",
            "size": "0.38 0.30 0.018",
            "material": "cabinet_body_mat",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    _add_closed_right_drawer(worldbody)
    slots = {
        # These are the visible positions of an opened drawer/cabinet.  The
        # contents start transparent behind their closed storage cover, then
        # appear on the exposed tray after INSPECT.  Keeping them above the
        # workbench avoids an occluded-but-textually-revealed observation.
        "LEFT_DRAWER": ((-0.78, 0.05, 0.65), (-0.52, 0.05, 0.65)),
        "RIGHT_DRAWER": ((0.48, 0.05, 0.65), (0.76, 0.05, 0.65)),
        "TOOL_CABINET": ((0.55, 0.68, 0.90), (0.88, 0.68, 0.90)),
    }
    for region, contents in storage_contents.items():
        if len(contents) > len(slots[region]):
            raise ValueError(f"Workshop visual proxy capacity exceeded for {region}")
        for backend, position in zip(contents, slots[region]):
            _add_visual_proxy(worldbody, str(backend), position)
    # ``from_xml_string`` has no base directory for mesh/texture paths.  Feed
    # the Workshop assets explicitly, matching the real scene constructor.
    assets = {
        f"workshop_realistic/{path.name}": path.read_bytes()
        for path in WORKSHOP_ASSETS_DIR.iterdir()
        if path.suffix.lower() in {".obj", ".png", ".mtl"}
    }
    model = mujoco.MjModel.from_xml_string(
        ET.tostring(root, encoding="unicode"), assets=assets
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


@dataclass(frozen=True)
class ExpectedGT:
    variant: str
    internal_variant: str
    intended_outcome: str
    actions: tuple[Mapping[str, Any], ...]

    @classmethod
    def load(cls, root: Path, variant: str) -> "ExpectedGT":
        payload = _load_json(root / variant / "expected_gt_actions.json")
        return cls(
            variant=str(payload["variant"]),
            internal_variant=str(payload["internal_variant"]),
            intended_outcome=str(payload["intended_outcome"]),
            actions=tuple(payload.get("actions", ())),
        )


class WorkshopSymbolicExecutor:
    def __init__(self, runtime: "WorkshopPlanningRuntime"):
        self.runtime = runtime

    def execute(self, action: Action) -> ActionResult:
        skill = action.skill.upper()
        args = action.arguments
        if skill == "INSPECT":
            region_id = str(args.get("region_id", ""))
            if region_id not in self.runtime.storage_region_ids:
                return ActionResult.failed("unknown_region", f"Unknown storage region {region_id}")
            if self.runtime.held_object is not None:
                return ActionResult.failed("gripper_occupied", "Inspect requires an empty gripper")
            self.runtime.inspected.add(region_id)
            self.runtime.revision += 1
            self.runtime._set_render_visibility()
            self.runtime.invalidate_images()
            return ActionResult.succeeded(f"inspected({region_id})")
        if skill == "PICK":
            object_id = str(args.get("object_id", ""))
            if object_id not in self.runtime.visible_object_ids:
                return ActionResult.failed("unknown_object", f"Object {object_id} is not observed")
            if object_id == self.runtime.target_object_id:
                return ActionResult.failed("fixed_target", "The frame joint is not movable")
            if self.runtime.held_object is not None:
                return ActionResult.failed("gripper_occupied", "The gripper already holds an object")
            self.runtime.held_object = object_id
            self.runtime.locations[object_id] = "held"
            self.runtime.revision += 1
            return ActionResult.succeeded(f"holding({object_id})")
        if skill == "PLACE":
            object_id = str(args.get("object_id", ""))
            destination = str(args.get("region_id", ""))
            if self.runtime.held_object != object_id:
                return ActionResult.failed("not_holding_object", f"The robot is not holding {object_id}")
            if destination not in self.runtime.destination_ids:
                return ActionResult.failed("unknown_destination", f"Unknown destination {destination}")
            self.runtime.held_object = None
            self.runtime.locations[object_id] = destination
            self.runtime.revision += 1
            return ActionResult.succeeded(f"placed({object_id},{destination})")
        if skill == "INSERT":
            fastener = str(args.get("fastener_id", ""))
            target = str(args.get("target_id", ""))
            if self.runtime.held_object != fastener:
                return ActionResult.failed("not_holding_fastener", "Insert requires the held fastener")
            if target != self.runtime.target_object_id or not self.runtime.is_fastener(fastener):
                return ActionResult.failed("invalid_insertion", "Only the observed compatible screw fits the frame joint")
            self.runtime.held_object = None
            self.runtime.locations[fastener] = target
            self.runtime.inserted = (fastener, target)
            self.runtime.revision += 1
            return ActionResult.succeeded(f"inserted({fastener},{target})")
        if skill == "FASTEN":
            tool = str(args.get("tool_id", ""))
            fastener = str(args.get("fastener_id", ""))
            target = str(args.get("target_id", ""))
            if self.runtime.held_object != tool:
                return ActionResult.failed("not_holding_tool", "Fasten requires the held driver")
            if self.runtime.inserted != (fastener, target):
                return ActionResult.failed("fastener_not_inserted", "The selected screw is not inserted")
            if not self.runtime.is_driver(tool) or target != self.runtime.target_object_id:
                return ActionResult.failed("invalid_driver", "The held object cannot fasten this joint")
            self.runtime.fastened = (tool, fastener, target)
            self.runtime.revision += 1
            return ActionResult.succeeded(f"fastened({tool},{fastener},{target})")
        return ActionResult.failed("unsupported_planning_action", f"Workshop cannot apply {action.skill}", recoverable=False)


class WorkshopPlanningRuntime:
    """Closed-cell observation and private symbolic Workshop state."""

    def __init__(
        self,
        variant: str,
        output_dir: Path,
        *,
        expected_root: Path = DEFAULT_EXPECTED_ROOT,
        image_width: int = 960,
        image_height: int = 540,
        camera_count: int = 5,
    ):
        internal = resolve_variant_name("workshop", variant)
        config = _load_yaml(VARIANT_CONFIG)
        variants = config["variants"]
        if internal not in variants:
            raise ValueError(f"Unknown Workshop variant {variant!r}")
        if camera_count not in WORKSHOP_CAMERA_SUBSETS:
            raise ValueError("Workshop camera_count must be one of 1, 3, or 5")
        self.internal_variant = internal
        self.variant = paper_variant_label("workshop", internal)
        self.variant_spec = dict(variants[internal])
        self.goal = str(config["canonical_task_instruction"])
        self.output_dir = output_dir
        self.expected = ExpectedGT.load(expected_root, self.variant)
        if self.expected.internal_variant != internal:
            raise ValueError("Expected GT does not match requested Workshop variant")
        self.camera_count = camera_count
        self.selected_cameras = WORKSHOP_CAMERA_SUBSETS[camera_count]
        self.image_width = image_width
        self.image_height = image_height
        self.model, self.data = _closed_workshop_model(self.variant_spec["storage_contents"])
        self.renderer: mujoco.Renderer | None = None
        self._images_cache: tuple[dict[str, str], ...] | None = None
        self.revision = 0
        self.held_object: str | None = None
        self.inspected: set[str] = set()
        self.inserted: tuple[str, str] | None = None
        self.fastened: tuple[str, str, str] | None = None

        self.backend_by_object_id = {backend: f"object_{index:04d}" for index, backend in enumerate(OBJECT_BACKENDS, 1)}
        self.object_by_backend = {value: key for key, value in self.backend_by_object_id.items()}
        self.backend_by_region_id = {backend: f"region_{index:04d}" for index, backend in enumerate((*STORAGE_BACKENDS, WORK_SURFACE), 1)}
        self.region_by_backend = {value: key for key, value in self.backend_by_region_id.items()}
        self.target_object_id = self.backend_by_object_id["workshop_frame_joint"]
        self.storage_region_ids = frozenset(self.backend_by_region_id[name] for name in STORAGE_BACKENDS)
        self.destination_ids = frozenset((*self.backend_by_region_id.values(), self.target_object_id))
        self.object_aliases = {self.backend_by_object_id[name]: label for name, label in OBJECT_LABELS.items()}
        self.region_aliases = {self.backend_by_region_id[name]: label for name, label in REGION_LABELS.items()}
        self.locations: dict[str, str] = {self.target_object_id: self.backend_by_region_id[WORK_SURFACE]}
        self._hidden_by_region = {
            self.backend_by_region_id[region]: tuple(
                self.backend_by_object_id[item] for item in contents
            )
            for region, contents in self.variant_spec["storage_contents"].items()
        }
        for region_id, object_ids in self._hidden_by_region.items():
            for object_id in object_ids:
                self.locations[object_id] = region_id
        self.inventory = self._build_private_inventory()
        self._region_geom_ids = {
            self.backend_by_region_id["LEFT_DRAWER"]: _body_geom_ids(self.model, "left_tool_drawer"),
            self.backend_by_region_id["RIGHT_DRAWER"]: _body_geom_ids(self.model, "baseline_right_drawer"),
            self.backend_by_region_id["TOOL_CABINET"]: _body_geom_ids(self.model, "tool_cabinet"),
            self.backend_by_region_id[WORK_SURFACE]: _body_geom_ids(self.model, "workbench"),
        }
        self._object_geom_ids = {self.target_object_id: _body_geom_ids(self.model, "workshop_frame_joint")}
        for backend, object_id in self.backend_by_object_id.items():
            if backend == "workshop_frame_joint":
                continue
            if object_id in self.locations:
                self._object_geom_ids[object_id] = _body_geom_ids(
                    self.model, f"baseline_proxy_{backend}"
                )
        # The visual benchmark uses per-variant proxy bodies.  Older scene
        # revisions also contained a fixed set of legacy tool bodies; retain
        # them only when present so an asset-only scene revision remains
        # runnable.
        self._legacy_item_geom_ids = frozenset(
            geom_id
            for name in (
                "workshop_manual_driver",
                "workshop_short_screw",
                "workshop_power_driver",
                "workshop_long_screw",
            )
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0
            for geom_id in _body_geom_ids(self.model, name)
        )
        self._storage_cover_geom_ids = {
            self.backend_by_region_id["LEFT_DRAWER"]: frozenset({
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "baseline_left_drawer_lid")
            }),
            self.backend_by_region_id["RIGHT_DRAWER"]: frozenset({
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "baseline_right_drawer_lid")
            }),
            self.backend_by_region_id["TOOL_CABINET"]: _body_geom_ids(self.model, "tool_cabinet_door"),
        }
        if any(geom_id < 0 for ids in self._storage_cover_geom_ids.values() for geom_id in ids):
            raise RuntimeError("Workshop storage cover geometry is missing")
        self._set_render_visibility()
        write_json(output_dir / "shared_observation_contract.json", {
            "scene": "workshop",
            "variant": self.variant,
            "internal_variant": internal,
            "camera_count": camera_count,
            "camera_ids": list(self.selected_cameras),
            "camera_ablation": "NESTED_TOP_LEFT_RIGHT_THEN_ALL",
            "initial_observation": "CLOSED_STORAGE_ONLY",
            "semantic_labels_exposed": True,
            "alias_to_planning_id_map_exposed": True,
            "physical_execution": False,
            "gt_visible_to_model": False,
        })
        write_json(output_dir / "_private_evaluation" / "expected_gt_actions.json", {
            "variant": self.expected.variant,
            "intended_outcome": self.expected.intended_outcome,
            "actions": list(self.expected.actions),
        })
        write_json(output_dir / "_private_evaluation" / "variant_state.json", {
            "internal_variant": internal,
            "storage_contents": self.variant_spec["storage_contents"],
        })

    def _build_private_inventory(self) -> dict[str, Any]:
        rows = []
        for backend, object_id in self.backend_by_object_id.items():
            if backend == "workshop_frame_joint" or object_id not in self.locations:
                continue
            rows.append({
                "generic_object_id": object_id,
                "semantic_label": OBJECT_LABELS[backend],
                "source_context": {"observed_source_region": self.locations[object_id]},
            })
        return {"objects": rows}

    @property
    def aliases(self) -> dict[str, str]:
        return {**self.object_aliases, **self.region_aliases}

    @property
    def visible_object_ids(self) -> frozenset[str]:
        visible = {self.target_object_id}
        for region_id in self.inspected:
            visible.update(self._hidden_by_region.get(region_id, ()))
        return frozenset(visible)

    def is_driver(self, object_id: str) -> bool:
        return self.object_by_backend.get(object_id) in {
            "workshop_long_phillips_driver", "workshop_power_driver"
        }

    def is_fastener(self, object_id: str) -> bool:
        return self.object_by_backend.get(object_id) == "workshop_medium_phillips_screw"

    def infeasibility_proven(self) -> bool:
        """Whether the observed state proves that no repair pair exists.

        This is an evaluator-side conclusion only.  It is never supplied to a
        model prompt, and it requires every storage region to have been
        inspected before a missing item can be treated as evidence.
        """
        if not self.storage_region_ids <= self.inspected:
            return False
        return not any(self.is_driver(item) for item in self.visible_object_ids) or not any(
            self.is_fastener(item) for item in self.visible_object_ids
        )

    def _set_alpha(self, geom_ids: Sequence[int], alpha: float) -> None:
        for geom_id in geom_ids:
            self.model.geom_rgba[geom_id, 3] = alpha

    def _set_render_visibility(self) -> None:
        """Synchronize visual evidence with the observable symbolic state."""
        self._set_alpha(self._legacy_item_geom_ids, 0.0)
        for object_id, geom_ids in self._object_geom_ids.items():
            if object_id == self.target_object_id:
                continue
            self._set_alpha(geom_ids, 1.0 if object_id in self.visible_object_ids else 0.0)
        for region_id, geom_ids in self._storage_cover_geom_ids.items():
            self._set_alpha(geom_ids, 0.0 if region_id in self.inspected else 1.0)

    def invalidate_images(self) -> None:
        self._images_cache = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def goal_verifier(self, _observation: Observation | None = None) -> bool:
        if self.fastened is None:
            return False
        tool, _fastener, _target = self.fastened
        return (
            self.held_object is None
            and self.locations.get(tool) == self.backend_by_region_id[WORK_SURFACE]
        )

    def observe_state(self) -> Observation:
        entities = []
        for object_id in sorted(self.visible_object_ids):
            facts: dict[str, Any]
            if object_id == self.held_object:
                facts = {"location": "held"}
            else:
                facts = {"region_id": self.locations[object_id]}
            if object_id == self.target_object_id:
                if self.inserted is not None:
                    facts["inserted_fasteners"] = [self.inserted[0]]
                if self.fastened is not None:
                    facts["fastened_with"] = self.fastened[0]
            entities.append(Entity(object_id, "object", self.object_aliases[object_id], facts))
        regions = tuple(
            Region(
                region_id,
                self.region_aliases[region_id],
                "open"
                if region_id in self.inspected
                or self.region_by_backend[region_id] == WORK_SURFACE
                else "closed",
                region_id in self.inspected
                or self.region_by_backend[region_id] == WORK_SURFACE,
            )
            for region_id in sorted(self.region_aliases)
        )
        observation = Observation("workshop", self.revision, tuple(entities), regions, {"workspace": "workbench", "holding": self.held_object}, self.goal_verifier())
        write_json(self.output_dir / "latest_observation.json", observation.as_semantic_neutral_prompt_dict())
        write_json(self.output_dir / "_private_evaluation" / "latest_observation.json", observation.as_prompt_dict())
        return observation

    def observe(self) -> tuple[Observation, tuple[dict[str, str], ...]]:
        return self.observe_state(), self.images()

    @staticmethod
    def _box(segmentation: np.ndarray, geom_ids: frozenset[int]) -> tuple[int, int, int, int, int] | None:
        mask = (segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(segmentation[:, :, 0], tuple(geom_ids))
        rows, columns = np.nonzero(mask)
        if len(rows) < 6:
            return None
        return int(columns.min()), int(rows.min()), int(columns.max()), int(rows.max()), int(len(rows))

    @staticmethod
    def _label(
        draw: ImageDraw.ImageDraw,
        box: tuple[int, int, int, int, int],
        text: str,
        color: tuple[int, int, int],
        font: ImageFont.ImageFont,
        occupied: list[tuple[int, int, int, int]],
    ) -> tuple[int, int, int, int]:
        left, top, right, bottom, _ = box
        draw.rectangle((left, top, right, bottom), outline=color, width=2)
        bounds = draw.textbbox((left, top), text, font=font)
        width, height = bounds[2] - bounds[0] + 4, bounds[3] - bounds[1] + 4
        image_width, image_height = draw._image.size
        raw_positions = (
            (left, top - height - 2),
            (left, bottom + 2),
            (right + 2, top),
            (left - width - 2, top),
        )
        candidates = []
        for raw_x, raw_y in raw_positions:
            x = min(max(0, raw_x), max(0, image_width - width - 1))
            y = min(max(0, raw_y), max(0, image_height - height - 1))
            candidates.append((x, y, x + width, y + height))
        x, y, label_right, label_bottom = next(
            (
                candidate
                for candidate in candidates
                if all(
                    candidate[2] < existing[0]
                    or existing[2] < candidate[0]
                    or candidate[3] < existing[1]
                    or existing[3] < candidate[1]
                    for existing in occupied
                )
            ),
            candidates[0],
        )
        draw.rectangle((x, y, x + width, y + height), fill="black")
        draw.text((x + 2, y + 2), text, fill=color, font=font)
        label_box = (x, y, label_right, label_bottom)
        occupied.append(label_box)
        return label_box

    def images(self) -> tuple[dict[str, str], ...]:
        if self._images_cache is not None:
            return self._images_cache
        self.renderer = mujoco.Renderer(self.model, height=self.image_height, width=self.image_width)
        stage = "initial" if self.revision == 0 else f"revision_{self.revision:03d}"
        directory = self.output_dir / "observations" / stage
        directory.mkdir(parents=True, exist_ok=True)
        manifest: dict[str, Any] = {"semantic_labels_exposed": True, "cameras": {}}
        images: list[dict[str, str]] = []
        mujoco.mj_forward(self.model, self.data)
        try:
            font = ImageFont.load_default(size=11)
        except TypeError:
            font = ImageFont.load_default()
        for camera in self.selected_cameras:
            self.renderer.update_scene(self.data, camera=camera)
            frame = self.renderer.render().copy()
            Image.fromarray(frame).save(directory / f"raw_{camera}.png")
            self.renderer.enable_segmentation_rendering()
            segmentation = self.renderer.render().copy()
            self.renderer.disable_segmentation_rendering()
            annotated = Image.fromarray(frame)
            draw = ImageDraw.Draw(annotated)
            rows: dict[str, list[dict[str, Any]]] = {"objects": [], "regions": []}
            occupied_labels: list[tuple[int, int, int, int]] = []
            for object_id in sorted(self.visible_object_ids):
                geom_ids = self._object_geom_ids[object_id]
                box = self._box(segmentation, geom_ids)
                if box:
                    label_box = self._label(draw, box, self.object_aliases[object_id], (45, 235, 255), font, occupied_labels)
                    rows["objects"].append({"id": object_id, "semantic_label": self.object_aliases[object_id], "bbox_xyxy": list(box[:4]), "label_bbox_xyxy": list(label_box), "pixel_count": box[4]})
            for region_id, geom_ids in self._region_geom_ids.items():
                box = self._box(segmentation, geom_ids)
                if box:
                    label_box = self._label(draw, box, self.region_aliases[region_id], (255, 190, 40), font, occupied_labels)
                    rows["regions"].append({"id": region_id, "semantic_label": self.region_aliases[region_id], "bbox_xyxy": list(box[:4]), "label_bbox_xyxy": list(label_box), "pixel_count": box[4]})
            buffer = BytesIO()
            annotated.save(buffer, format="PNG")
            content = buffer.getvalue()
            (directory / f"{camera}.png").write_bytes(content)
            manifest["cameras"][camera] = rows
            images.append({"camera": camera, "data_url": "data:image/png;base64," + base64.b64encode(content).decode("ascii")})
        write_json(directory / "annotations.json", manifest)
        self._images_cache = tuple(images)
        return self._images_cache

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


def canonical_workshop_actions(history: Sequence[Mapping[str, Any]], backend_by_id: Mapping[str, str]) -> list[dict[str, Any]]:
    argument_order = {
        "INSPECT": ("region_id",), "PICK": ("object_id",), "PLACE": ("object_id", "region_id"),
        "INSERT": ("fastener_id", "target_id"), "FASTEN": ("tool_id", "fastener_id", "target_id"),
    }
    rows = []
    for record in history:
        if not record.get("success"):
            continue
        action = record.get("action", {})
        operator = str(action.get("skill", "")).upper()
        if operator not in argument_order:
            continue
        args = action.get("arguments", {})
        rows.append({"operator": operator, "arguments": [backend_by_id.get(str(args[key]), str(args[key])) for key in argument_order[operator]]})
    return rows


def normalize_workshop_actions(actions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    mapping = {
        "INSPECT_STORAGE": "INSPECT", "PLACE_ON_SURFACE": "PLACE",
        "INSERT_FASTENER": "INSERT", "DRIVE_FASTENER": "FASTEN",
    }
    ignored = {"MOVE_TO", "OPEN_STORAGE", "CLOSE_STORAGE", "VERIFY_REPAIR", "TERMINATE_INFEASIBLE"}
    rows = []
    for row in actions:
        operator = str(row["operator"]).upper()
        if operator in ignored:
            continue
        operator = mapping.get(operator, operator)
        arguments = list(map(str, row.get("arguments", ())))
        if operator == "PICK":
            # The expected executor records a source region; the planning
            # action signature contains only the selected object.
            arguments = arguments[:1]
        rows.append({"operator": operator, "arguments": arguments})

    # The expected traces include temporary workbench staging imposed by the
    # physical single-arm executor (pick -> stage -> later re-pick).  A
    # planning-only baseline is not evaluated on those drawer-management
    # details.  Retain the causal task skeleton: inspections, the final pick
    # before insertion/fastening, the insert/fasten relations, and the final
    # returned driver placement required by the task.
    retained: list[dict[str, Any]] = [row for row in rows if row["operator"] == "INSPECT"]
    for index, row in enumerate(rows):
        if row["operator"] == "INSERT":
            fastener = row["arguments"][0]
            previous = next(
                (
                    candidate for candidate in reversed(rows[:index])
                    if candidate["operator"] == "PICK" and candidate["arguments"][0] == fastener
                ),
                None,
            )
            if previous is not None:
                retained.append(previous)
            retained.append(row)
        elif row["operator"] == "FASTEN":
            tool = row["arguments"][0]
            previous = next(
                (
                    candidate for candidate in reversed(rows[:index])
                    if candidate["operator"] == "PICK" and candidate["arguments"][0] == tool
                ),
                None,
            )
            if previous is not None:
                retained.append(previous)
            retained.append(row)
            final_place = next(
                (
                    candidate for candidate in rows[index + 1:]
                    if candidate["operator"] == "PLACE" and candidate["arguments"][0] == tool
                ),
                None,
            )
            if final_place is not None:
                retained.append(final_place)
    return retained


def compare_workshop_actions(predicted: Sequence[Mapping[str, Any]], expected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    from .living_room_runtime import compare_action_sequences
    return {
        "raw_execution_vocabulary": compare_action_sequences(predicted, expected),
        "shared_task_vocabulary": compare_action_sequences(normalize_workshop_actions(predicted), normalize_workshop_actions(expected)),
        "normalization": {
            "INSPECT_STORAGE": "INSPECT", "PLACE_ON_SURFACE": "PLACE", "INSERT_FASTENER": "INSERT", "DRIVE_FASTENER": "FASTEN",
            "MOVE_TO|OPEN_STORAGE|CLOSE_STORAGE|VERIFY_REPAIR|TERMINATE_INFEASIBLE": "excluded_execution_detail",
        },
    }


__all__ = [
    "DEFAULT_EXPECTED_ROOT", "WORKSHOP_CAMERA_SUBSETS", "WorkshopPlanningRuntime", "WorkshopSymbolicExecutor",
    "canonical_workshop_actions", "compare_workshop_actions", "normalize_workshop_actions",
]
