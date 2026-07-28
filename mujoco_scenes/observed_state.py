"""Persistent observed-object registry, graph, events, and stage visualizations."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mujoco_scenes.geometry_checker import (
    GeometryChecker,
    PointCloudRun,
    read_ply,
    voxel_downsample,
    write_ply,
)
from mujoco_scenes.geometry_properties import (
    candidate_functions,
    category_family,
    extract_object_properties,
    load_semantics_config,
    pairwise_relation_status,
)


SCHEMA_VERSION = 1
STATUS_COLORS = {
    "previous": (61, 116, 184),
    "new": (45, 166, 85),
    "updated": (230, 126, 34),
}
RELATION_COLORS = {
    "TRUE": (45, 166, 85),
    "FALSE": (210, 62, 62),
    "UNKNOWN": (135, 135, 135),
    "SEMANTIC": (125, 84, 170),
    "OBSERVED": (45, 85, 140),
}


def _font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size=size)
    except OSError:
        return ImageFont.load_default()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    write_ply(temporary, points, colors)
    temporary.replace(path)


def _safe_run_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not safe:
        raise ValueError("run_id must contain at least one letter or digit")
    return safe


def _property_number(record: dict[str, Any] | None) -> float | None:
    if not record or record.get("value") is None:
        return None
    try:
        value = float(record["value"])
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _dimension_values(record: dict[str, Any]) -> list[float | None]:
    dimensions = record.get("dimensions_m", {})
    return [
        _property_number(dimensions.get(name))
        for name in ("length", "width", "height")
    ]


class ObservedStateRun:
    """One persistent registry and growing graph across observation stages."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        scene_name: str,
        region_ids: Iterable[str],
        voxel_size: float = 0.003,
        semantics_config: str | Path | None = None,
        run_config: dict[str, Any] | None = None,
    ):
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = self.run_dir.name
        self.scene_name = scene_name
        self.voxel_size = voxel_size
        self.config = load_semantics_config(
            semantics_config
        ) if semantics_config else load_semantics_config()
        self.registry_path = self.run_dir / "object_registry.json"
        self.graph_path = self.run_dir / "observed_graph.json"
        self.events_path = self.run_dir / "events.jsonl"
        self.stages_dir = self.run_dir / "stages"
        self.stages_dir.mkdir(exist_ok=True)
        self.region_ids = tuple(region_ids)

        if self.registry_path.exists():
            self.registry = json.loads(self.registry_path.read_text())
        else:
            self.registry = {
                "schema_version": SCHEMA_VERSION,
                "run_id": self.run_id,
                "scene_name": scene_name,
                "voxel_size_m": voxel_size,
                "current_stage": -1,
                "instance_index": {},
                "objects": {},
            }
        self.next_stage = int(self.registry.get("current_stage", -1)) + 1
        config_payload = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "scene_name": scene_name,
            "created_at": datetime.now().astimezone().isoformat(),
            "voxel_size_m": voxel_size,
            "regions": ["countertop", *self.region_ids],
            "semantic_config": str(
                Path(semantics_config).resolve()
                if semantics_config
                else "configs/observed_state_semantics.yaml"
            ),
            **(run_config or {}),
        }
        if not (self.run_dir / "run_config.json").exists():
            _atomic_json(self.run_dir / "run_config.json", config_payload)

    @classmethod
    def create_for_scene(
        cls,
        scene,
        *,
        runs_root: str | Path = "runs",
        run_id: str | None = None,
        voxel_size: float = 0.003,
        run_config: dict[str, Any] | None = None,
    ) -> "ObservedStateRun":
        if run_id is None:
            timestamp = datetime.now().astimezone().strftime(
                "%Y%m%d_%H%M%S_%f"
            )
            run_id = f"{scene.scene_name}_{timestamp}"
        run_id = _safe_run_id(run_id)
        region_ids = tuple(scene.get_region_observation_states().keys())
        return cls(
            Path(runs_root) / run_id,
            scene_name=scene.scene_name,
            region_ids=region_ids,
            voxel_size=voxel_size,
            run_config=run_config,
        )

    def observe_scene(
        self,
        scene,
        *,
        stage_label: str,
        region_opened: str | None = None,
        width: int = 640,
        height: int = 480,
    ) -> tuple[PointCloudRun, Path]:
        """Run five-view reconstruction and update every persistent output."""
        cloud_run = GeometryChecker(
            scene,
            width=width,
            height=height,
            voxel_size=self.voxel_size,
        ).run()
        stage_dir = self.update_from_point_cloud_run(
            scene,
            cloud_run,
            stage_label=stage_label,
            region_opened=region_opened,
        )
        return cloud_run, stage_dir

    def _source_region(self, scene, instance_id: str) -> str:
        if hasattr(scene, "get_instance_source_region"):
            source = scene.get_instance_source_region(instance_id)
            return source if source is not None else "countertop"
        sources = getattr(scene, "instance_source_regions", {})
        return sources.get(instance_id, "countertop")

    def _region_states(self, scene) -> dict[str, dict[str, Any]]:
        if hasattr(scene, "get_region_observation_states"):
            return scene.get_region_observation_states()
        provided = getattr(scene, "region_states", {})
        return {
            region_id: {
                "region_id": region_id,
                "open": bool(provided.get(region_id, {}).get("open", False)),
                "inspected": bool(
                    provided.get(region_id, {}).get("inspected", False)
                ),
            }
            for region_id in self.region_ids
        }

    def _new_object_id(self, category: str) -> str:
        existing = sum(
            record.get("category") == category
            for record in self.registry["objects"].values()
        )
        return f"{category}_{existing + 1:02d}"

    def _load_cumulative(
        self, record: dict[str, Any] | None
    ) -> tuple[np.ndarray, np.ndarray]:
        if not record:
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.uint8),
            )
        path = self.run_dir / record["cumulative_cloud_path"]
        if not path.exists():
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.uint8),
            )
        return read_ply(path)

    def update_from_point_cloud_run(
        self,
        scene,
        cloud_run: PointCloudRun,
        *,
        stage_label: str,
        region_opened: str | None = None,
    ) -> Path:
        """Merge one observation into the registry, graph, and stage history."""
        stage = self.next_stage
        safe_label = _safe_run_id(stage_label)
        stage_dir = self.stages_dir / f"{stage:03d}_{safe_label}"
        if stage_dir.exists():
            raise RuntimeError(f"Stage output already exists: {stage_dir}")
        stage_dir.mkdir(parents=True)

        events: list[dict[str, Any]] = []
        if region_opened is not None:
            events.append(
                {
                    "stage": stage,
                    "event": "REGION_OPENED",
                    "region_id": region_opened,
                }
            )

        stage_changes = {
            object_id: "previous"
            for object_id in self.registry["objects"]
        }
        for instance_id, cloud in cloud_run.clouds.items():
            object_id = self.registry["instance_index"].get(instance_id)
            existing = (
                self.registry["objects"].get(object_id)
                if object_id is not None
                else None
            )
            is_new = existing is None
            has_visible_samples = len(cloud.points) > 0 or any(
                count > 0 for count in cloud.pixels_by_camera.values()
            )
            if not is_new and not has_visible_samples:
                # The symbolic catalogue retains previously inspected objects
                # after a region closes. Persistence belongs in the registry;
                # a zero-pixel mask is not a new observation of that instance.
                continue
            if is_new:
                object_id = self._new_object_id(cloud.object_kind)
                self.registry["instance_index"][instance_id] = object_id
                events.append(
                    {
                        "stage": stage,
                        "event": "OBJECT_DISCOVERED",
                        "object_id": object_id,
                        "instance_id": instance_id,
                    }
                )

            previous_points, previous_colors = self._load_cumulative(existing)
            merged_points = np.concatenate((previous_points, cloud.points))
            merged_colors = np.concatenate((previous_colors, cloud.colors))
            merged_points, merged_colors = voxel_downsample(
                merged_points, merged_colors, self.voxel_size
            )
            relative_cloud_path = (
                Path("objects") / object_id / "cumulative.ply"
            )
            _atomic_ply(
                self.run_dir / relative_cloud_path,
                merged_points,
                merged_colors,
            )

            camera_count = sum(
                count > 0 for count in cloud.pixels_by_camera.values()
            )
            measured = extract_object_properties(
                merged_points,
                category=cloud.object_kind,
                contributing_camera_count=camera_count,
                config=self.config,
            )
            source_region = (
                self._source_region(scene, instance_id)
                if is_new
                else existing["source_region"]
            )
            record = {
                "object_id": object_id,
                "instance_id": instance_id,
                "category": cloud.object_kind,
                "object_family": category_family(cloud.object_kind, self.config),
                "source_region": source_region,
                "first_seen_stage": stage if is_new else existing["first_seen_stage"],
                "last_seen_stage": stage,
                "observation_count": (
                    1 if is_new else int(existing["observation_count"]) + 1
                ),
                "cumulative_cloud_path": relative_cloud_path.as_posix(),
                **measured,
            }
            self.registry["objects"][object_id] = record
            _atomic_json(
                self.run_dir / "objects" / object_id / "properties.json",
                record,
            )
            stage_changes[object_id] = "new" if is_new else "updated"
            events.append(
                {
                    "stage": stage,
                    "event": "PROPERTY_UPDATED",
                    "object_id": object_id,
                    "point_count": len(merged_points),
                }
            )

        region_states = self._region_states(scene)
        graph = self._build_graph(region_states, stage_changes)
        self.registry["current_stage"] = stage
        _atomic_json(self.registry_path, self.registry)
        _atomic_json(self.graph_path, graph)
        _atomic_json(stage_dir / "properties.json", self.registry)
        _atomic_json(stage_dir / "graph.json", graph)

        all_points, all_colors = self._all_cumulative_clouds()
        _atomic_ply(stage_dir / "combined_cloud.ply", all_points, all_colors)
        self._append_events(events)
        self._render_stage(stage_dir, graph, stage_changes)
        self._update_growth_gif()
        self.next_stage += 1
        return stage_dir

    def _append_events(self, events: list[dict[str, Any]]) -> None:
        if not events:
            return
        with self.events_path.open("a", encoding="utf-8") as output:
            for event in events:
                output.write(json.dumps(event, sort_keys=True) + "\n")

    def _all_cumulative_clouds(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        all_points, all_colors = [], []
        for record in self.registry["objects"].values():
            points, colors = self._load_cumulative(record)
            if len(points):
                all_points.append(points)
                all_colors.append(colors)
        return (
            np.concatenate(all_points)
            if all_points
            else np.empty((0, 3), dtype=np.float32),
            np.concatenate(all_colors)
            if all_colors
            else np.empty((0, 3), dtype=np.uint8),
        )

    def _build_graph(
        self,
        region_states: dict[str, dict[str, Any]],
        stage_changes: dict[str, str],
    ) -> dict[str, Any]:
        nodes, edges = [], []
        all_regions = {
            "countertop": {
                "region_id": "countertop",
                "open": True,
                "inspected": True,
            },
            **region_states,
        }
        for region_id, state in all_regions.items():
            contents = sorted(
                object_id
                for object_id, record in self.registry["objects"].items()
                if record["source_region"] == region_id
            )
            inspected = bool(state.get("inspected", False))
            nodes.append(
                {
                    "id": f"region:{region_id}",
                    "type": "region",
                    "attributes": {
                        "region_id": region_id,
                        "open": bool(state.get("open", False)),
                        "inspected": inspected,
                        "contents": contents if inspected else "UNKNOWN",
                    },
                }
            )

        function_ids = set()
        for object_id, record in self.registry["objects"].items():
            functions = candidate_functions(record["category"], self.config)
            dimensions = _dimension_values(record)
            centroid_record = record.get("centroid_world_m", {})
            nodes.append(
                {
                    "id": f"object:{object_id}",
                    "type": "object",
                    "attributes": {
                        "object_id": object_id,
                        "instance_id": record["instance_id"],
                        "category": record["category"],
                        "object_family": record["object_family"],
                        "centroid_world_m": centroid_record.get("value"),
                        "dimensions_m": dimensions,
                        "candidate_functions": functions,
                        "stage_state": stage_changes.get(object_id, "previous"),
                    },
                }
            )
            edges.append(
                {
                    "source": f"object:{object_id}",
                    "target": f"region:{record['source_region']}",
                    "relation": "OBSERVED_IN",
                    "status": "OBSERVED",
                }
            )
            for function in functions:
                function_ids.add(function)
                edges.append(
                    {
                        "source": f"object:{object_id}",
                        "target": f"function:{function}",
                        "relation": "CANDIDATE_FOR",
                        "status": "SEMANTIC",
                    }
                )
        for function in sorted(function_ids):
            nodes.append(
                {
                    "id": f"function:{function}",
                    "type": "function",
                    "attributes": {"function_id": function},
                }
            )

        utensils = [
            (object_id, record)
            for object_id, record in self.registry["objects"].items()
            if record["object_family"] == "utensil"
        ]
        receptacles = [
            (object_id, record)
            for object_id, record in self.registry["objects"].items()
            if record["object_family"] == "receptacle"
        ]
        for utensil_id, utensil in utensils:
            for receptacle_id, receptacle in receptacles:
                for relation in ("INSERTABLE_IN", "REACHES_BOTTOM"):
                    edges.append(
                        {
                            "source": f"object:{utensil_id}",
                            "target": f"object:{receptacle_id}",
                            "relation": relation,
                            "status": pairwise_relation_status(
                                relation,
                                utensil,
                                receptacle,
                                self.config,
                            ),
                        }
                    )
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "stage": self.next_stage,
            "nodes": nodes,
            "edges": edges,
        }

    def _render_stage(
        self,
        stage_dir: Path,
        graph: dict[str, Any],
        stage_changes: dict[str, str],
    ) -> None:
        pointcloud_image = self._render_pointcloud(stage_changes)
        graph_image = self._render_graph(graph)
        pointcloud_image.save(stage_dir / "pointcloud.png")
        graph_image.save(stage_dir / "graph.png")

        panel_height = max(pointcloud_image.height, graph_image.height)
        overview = Image.new(
            "RGB",
            (pointcloud_image.width + graph_image.width, panel_height + 70),
            "white",
        )
        draw = ImageDraw.Draw(overview)
        draw.text(
            (24, 18),
            f"Observed-state growth · {self.scene_name} · stage {self.next_stage:03d}",
            fill=(25, 35, 50),
            font=_font(28, bold=True),
        )
        overview.paste(pointcloud_image, (0, 70))
        overview.paste(graph_image, (pointcloud_image.width, 70))
        overview.save(stage_dir / "overview.png")

    def _render_pointcloud(
        self, stage_changes: dict[str, str]
    ) -> Image.Image:
        width, height = 850, 1100
        image = Image.new("RGB", (width, height), (248, 250, 252))
        draw = ImageDraw.Draw(image)
        draw.text(
            (28, 24),
            "Cumulative observed object clouds",
            fill=(25, 35, 50),
            font=_font(25, bold=True),
        )
        clouds = []
        for object_id, record in self.registry["objects"].items():
            points, _colors = self._load_cumulative(record)
            if len(points):
                clouds.append((object_id, points))
        if not clouds:
            draw.text(
                (width // 2, height // 2),
                "No finite observed points",
                anchor="mm",
                fill=(100, 100, 100),
                font=_font(22),
            )
            return image

        projected_sets = []
        for object_id, points in clouds:
            finite = points[np.all(np.isfinite(points), axis=1)]
            projected = np.column_stack(
                (
                    0.866 * (finite[:, 0] - finite[:, 1]),
                    0.46 * (finite[:, 0] + finite[:, 1]) - 1.15 * finite[:, 2],
                )
            )
            projected_sets.append((object_id, projected))
        combined = np.concatenate([projected for _, projected in projected_sets])
        lower = np.percentile(combined, 1.0, axis=0)
        upper = np.percentile(combined, 99.0, axis=0)
        span = np.maximum(upper - lower, 1e-6)
        margin = 65
        scale = min(
            (width - 2 * margin) / span[0],
            (height - 190) / span[1],
        )
        for object_id, projected in projected_sets:
            if len(projected) > 5000:
                projected = projected[:: math.ceil(len(projected) / 5000)]
            pixels = (projected - lower) * scale
            pixels[:, 0] += margin
            pixels[:, 1] += 105
            pixels[:, 1] = height - 85 - pixels[:, 1]
            color = STATUS_COLORS[stage_changes.get(object_id, "previous")]
            for x, y in pixels:
                if 4 <= x < width - 4 and 80 <= y < height - 80:
                    draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=color)

        legend = (
            ("Previously observed", "previous"),
            ("Newly discovered", "new"),
            ("Measurements updated", "updated"),
        )
        for index, (label, status) in enumerate(legend):
            x = 28 + index * 260
            draw.rectangle(
                (x, height - 46, x + 18, height - 28),
                fill=STATUS_COLORS[status],
            )
            draw.text(
                (x + 26, height - 49),
                label,
                fill=(45, 50, 58),
                font=_font(14),
            )
        return image

    @staticmethod
    def _dashed_line(
        draw: ImageDraw.ImageDraw,
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        fill: tuple[int, int, int],
        width: int = 2,
        dash: int = 8,
    ) -> None:
        x1, y1 = start
        x2, y2 = end
        distance = math.hypot(x2 - x1, y2 - y1)
        if distance == 0:
            return
        ux, uy = (x2 - x1) / distance, (y2 - y1) / distance
        position = 0.0
        while position < distance:
            segment_end = min(position + dash, distance)
            draw.line(
                (
                    x1 + ux * position,
                    y1 + uy * position,
                    x1 + ux * segment_end,
                    y1 + uy * segment_end,
                ),
                fill=fill,
                width=width,
            )
            position += 2 * dash

    def _render_graph(self, graph: dict[str, Any]) -> Image.Image:
        object_nodes = [node for node in graph["nodes"] if node["type"] == "object"]
        region_nodes = [node for node in graph["nodes"] if node["type"] == "region"]
        function_nodes = [
            node for node in graph["nodes"] if node["type"] == "function"
        ]
        rows = max(6, math.ceil(len(object_nodes) / 2))
        width, height = 1750, max(1100, 180 + rows * 145)
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        draw.text(
            (28, 24),
            "Growing observed graph",
            fill=(25, 35, 50),
            font=_font(25, bold=True),
        )

        positions: dict[str, tuple[float, float]] = {}
        for index, node in enumerate(region_nodes):
            y = 115 + index * (height - 250) / max(1, len(region_nodes) - 1)
            positions[node["id"]] = (155, y)
        for index, node in enumerate(object_nodes):
            column = index % 2
            row = index // 2
            positions[node["id"]] = (560 + column * 470, 125 + row * 145)
        for index, node in enumerate(function_nodes):
            y = 125 + index * (height - 270) / max(1, len(function_nodes) - 1)
            positions[node["id"]] = (1540, y)

        # Draw edges first so node labels remain readable.
        for edge in graph["edges"]:
            if edge["source"] not in positions or edge["target"] not in positions:
                continue
            start, end = positions[edge["source"]], positions[edge["target"]]
            color = RELATION_COLORS.get(edge["status"], (120, 120, 120))
            if edge["status"] == "UNKNOWN" or edge["relation"] == "REACHES_BOTTOM":
                self._dashed_line(draw, start, end, fill=color, width=1, dash=6)
            else:
                draw.line((start, end), fill=color, width=2)

        for node in region_nodes:
            x, y = positions[node["id"]]
            attrs = node["attributes"]
            inspected = attrs["inspected"]
            fill = (48, 82, 130) if inspected else (225, 227, 230)
            outline = (36, 61, 98) if inspected else (130, 130, 130)
            box = (x - 120, y - 42, x + 120, y + 42)
            if inspected:
                draw.rounded_rectangle(box, radius=10, fill=fill, outline=outline, width=3)
            else:
                draw.rounded_rectangle(box, radius=10, fill=fill)
                self._dashed_line(
                    draw,
                    (box[0], box[1]),
                    (box[2], box[1]),
                    fill=outline,
                    width=2,
                )
                self._dashed_line(
                    draw,
                    (box[0], box[3]),
                    (box[2], box[3]),
                    fill=outline,
                    width=2,
                )
            state = "OPEN" if attrs["open"] else "CLOSED"
            contents = attrs["contents"]
            content_label = (
                f"{len(contents)} observed"
                if isinstance(contents, list)
                else "contents UNKNOWN"
            )
            text_color = "white" if inspected else (55, 55, 55)
            draw.text(
                (x, y - 12),
                f"{attrs['region_id']} · {state}",
                anchor="mm",
                fill=text_color,
                font=_font(17, bold=True),
            )
            draw.text(
                (x, y + 14),
                content_label,
                anchor="mm",
                fill=text_color,
                font=_font(13),
            )

        for node in object_nodes:
            x, y = positions[node["id"]]
            attrs = node["attributes"]
            state = attrs.get("stage_state", "previous")
            fill = STATUS_COLORS[state]
            box = (x - 185, y - 59, x + 185, y + 59)
            draw.rounded_rectangle(
                box, radius=10, fill=(250, 250, 250), outline=fill, width=5
            )
            dimensions = attrs["dimensions_m"]
            dimension_label = (
                " × ".join(
                    "?" if value is None else f"{value:.3f}"
                    for value in dimensions
                )
                + " m"
            )
            functions = ", ".join(attrs["candidate_functions"]) or "none"
            lines = [
                attrs["object_id"],
                f"category: {attrs['category']}",
                dimension_label,
                f"functions: {functions}",
            ]
            for offset, line in zip((-39, -14, 11, 36), lines):
                draw.text(
                    (x, y + offset),
                    line,
                    anchor="mm",
                    fill=(30, 35, 42),
                    font=_font(12, bold=offset == -39),
                )

        for node in function_nodes:
            x, y = positions[node["id"]]
            function_id = node["attributes"]["function_id"]
            box = (x - 145, y - 33, x + 145, y + 33)
            draw.rounded_rectangle(
                box,
                radius=24,
                fill=(239, 230, 247),
                outline=RELATION_COLORS["SEMANTIC"],
                width=3,
            )
            draw.text(
                (x, y),
                function_id,
                anchor="mm",
                fill=(65, 40, 85),
                font=_font(16, bold=True),
            )

        legend_y = height - 75
        legend = [
            ("TRUE", "solid"),
            ("FALSE", "solid"),
            ("UNKNOWN", "dashed"),
            ("candidate function", "semantic"),
        ]
        for index, (label, style) in enumerate(legend):
            x = 390 + index * 290
            color = (
                RELATION_COLORS["SEMANTIC"]
                if style == "semantic"
                else RELATION_COLORS[label]
            )
            if style == "dashed":
                self._dashed_line(
                    draw, (x, legend_y), (x + 65, legend_y), fill=color
                )
            else:
                draw.line((x, legend_y, x + 65, legend_y), fill=color, width=3)
            draw.text(
                (x + 78, legend_y),
                label,
                anchor="lm",
                fill=(45, 50, 58),
                font=_font(14),
            )
        return image

    def _update_growth_gif(self) -> None:
        frames = []
        for path in sorted(self.stages_dir.glob("*/overview.png")):
            with Image.open(path) as image:
                frame = image.copy()
                frame.thumbnail((1600, 900), Image.Resampling.LANCZOS)
                canvas = Image.new("RGB", (1600, 900), "white")
                canvas.paste(
                    frame,
                    ((canvas.width - frame.width) // 2, 0),
                )
                frames.append(canvas)
        if not frames:
            return
        frames[0].save(
            self.run_dir / "graph_growth.gif",
            save_all=True,
            append_images=frames[1:],
            duration=1100,
            loop=0,
        )
