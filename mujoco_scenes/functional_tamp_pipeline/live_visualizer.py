"""Non-intrusive live pipeline visualizer for Phase 3 functional TAMP."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mujoco_scenes.live_mosaic_viewer import LiveMosaicViewer

# Maximum items to display on graphical panels before truncating for readability
MAX_GF_NODES = 12
MAX_GO_NODES = 16
MAX_RELATIONS = 18
MAX_ACTIONS = 12
MAX_EXPLORATORY_OPEN = 10


@dataclass
class _VisualizerState:
    domain: str = "unknown"
    variant: str = "unknown"
    spec_mode: str = "unknown"
    stage: str = "initializing"
    terminal_status: str = "IN_PROGRESS"
    is_terminal: bool = False
    is_exception: bool = False
    exception_type: str | None = None
    exception_message: str | None = None
    run_dir: str | None = None
    started_monotonic: float = field(default_factory=time.monotonic)
    frozen_elapsed_seconds: float | None = None

    # G_F
    spec_graph: dict[str, Any] | None = None
    spec_source: str | None = None
    provider_region_ranking: list[str] = field(default_factory=list)
    search_order_source_requested: str = "auto"
    search_order_source_effective: str | None = None
    search_seed_requested: int | None = None
    search_seed_effective: int | None = None
    resolved_region_order: list[str] = field(default_factory=list)
    exploration_actuation: str = "unknown"

    # Camera / Observation
    latest_frame: np.ndarray | None = None
    latest_stage_label: str = "initial"

    # G_O
    scene_graph: dict[str, Any] | None = None

    # Grounding & Assignment
    grounding: dict[str, Any] | None = None
    satisfied: bool = False
    grounding_status: str = "UNINITIALIZED"
    assignment: dict[str, Any] | None = None
    operation_bindings: list[dict[str, Any]] = field(default_factory=list)
    missing_roles: list[str] = field(default_factory=list)
    unsatisfied_relations: list[str] = field(default_factory=list)
    unresolved_constraints: list[str] = field(default_factory=list)

    # Search & Inspection
    current_selected_region: str | None = None
    inspected_regions: list[str] = field(default_factory=list)
    exploratory_open_trace: list[dict[str, Any]] = field(default_factory=list)

    # Plan
    plan_actions: list[dict[str, Any]] = field(default_factory=list)
    search_statistics: dict[str, Any] = field(default_factory=list)

    # Telemetry health
    dropped_display_updates: int = 0
    display_errors: list[dict[str, Any]] = field(default_factory=list)


def _get_default_font(size: int = 14) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        for font_name in ("DejaVuSansMono.ttf", "DejaVuSans.ttf", "LiberationMono-Regular.ttf", "FreeMono.ttf"):
            try:
                return ImageFont.truetype(font_name, size)
            except OSError:
                continue
        return ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()


def _truncate_text(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


class LivePipelineVisualizer:
    """Non-blocking live pipeline monitor rendering to LiveMosaicViewer."""

    def __init__(
        self,
        viewer_factory: Callable[[int, int, int, str], Any] = LiveMosaicViewer,
        width: int = 1600,
        height: int = 960,
        fps: int = 15,
        title: str = "Functional TAMP Pipeline Live Monitor",
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.title = title
        self._viewer_factory = viewer_factory
        self._viewer: Any | None = None
        self._viewer_disabled = False

        self._state = _VisualizerState()
        self._queue: queue.Queue[tuple[str, dict[str, Any]] | None] = queue.Queue(maxsize=100)
        self._stop_event = threading.Event()

        # Cached renderings
        self._cached_gf_key: str | None = None
        self._cached_gf_image: Image.Image | None = None
        self._cached_go_key: str | None = None
        self._cached_go_image: Image.Image | None = None

        self._font_title = _get_default_font(18)
        self._font_heading = _get_default_font(15)
        self._font_body = _get_default_font(13)
        self._font_small = _get_default_font(11)

        try:
            self._viewer = self._viewer_factory(self.width, self.height, self.fps, self.title)
        except Exception as error:
            print(
                f"VISUALIZER WARNING: Viewer initialization failed ({type(error).__name__}: {error}). "
                "Visualization will run in headless mode.",
                file=sys.stderr,
                flush=True,
            )
            self._viewer = None
            self._viewer_disabled = True
            self._state.display_errors.append({
                "event": "viewer_init",
                "type": type(error).__name__,
                "message": str(error),
            })

        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def __call__(self, event_type: str, payload: dict[str, Any]) -> None:
        """Pipeline callback: non-blocking, safe snapshot enqueue."""
        if self._stop_event.is_set():
            return
        snapshot = self._safe_copy_payload(payload)
        try:
            self._queue.put_nowait((event_type, snapshot))
        except queue.Full:
            self._state.dropped_display_updates += 1

    def _safe_copy_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        copied: dict[str, Any] = {}
        for k, v in payload.items():
            if isinstance(v, np.ndarray):
                copied[k] = np.array(v, copy=True)
            elif isinstance(v, (dict, list, tuple, str, int, float, bool)) or v is None:
                copied[k] = v
            else:
                copied[k] = str(v)
        return copied

    def _reduce_event(self, event_type: str, payload: dict[str, Any]) -> None:
        s = self._state
        if event_type == "run_started":
            s.domain = payload.get("domain", s.domain)
            s.variant = payload.get("variant", s.variant)
            s.spec_mode = payload.get("spec_mode", s.spec_mode)
            s.search_order_source_requested = payload.get("search_order_source_requested", s.search_order_source_requested)
            s.search_seed_requested = payload.get("search_seed_requested", s.search_seed_requested)
            s.run_dir = payload.get("run_dir", s.run_dir)
            s.started_monotonic = time.monotonic()

        elif event_type == "stage_changed":
            s.stage = payload.get("stage", s.stage)

        elif event_type == "spec_ready":
            s.spec_graph = payload.get("graph", s.spec_graph)
            s.spec_source = payload.get("source", s.spec_source)
            s.provider_region_ranking = list(payload.get("provider_region_ranking", s.provider_region_ranking))
            s.search_order_source_effective = payload.get("search_order_source_effective", s.search_order_source_effective)
            s.resolved_region_order = list(payload.get("region_order_used", s.resolved_region_order))
            s.search_seed_effective = payload.get("search_seed_effective", s.search_seed_effective)

        elif event_type == "observation_updated":
            s.latest_stage_label = payload.get("stage", s.latest_stage_label)
            if "inspected_regions" in payload:
                s.inspected_regions = list(payload["inspected_regions"])
            if "scene_graph" in payload and payload["scene_graph"] is not None:
                s.scene_graph = payload["scene_graph"]
            if "frame_rgb" in payload and payload["frame_rgb"] is not None:
                s.latest_frame = np.array(payload["frame_rgb"], copy=True)
            elif "frame_path" in payload and payload["frame_path"]:
                frame_path = Path(payload["frame_path"])
                if frame_path.exists():
                    try:
                        with Image.open(frame_path) as img:
                            s.latest_frame = np.array(img.convert("RGB"), dtype=np.uint8)
                    except Exception as error:
                        s.display_errors.append({
                            "event": "load_frame_path",
                            "type": type(error).__name__,
                            "message": str(error),
                        })

        elif event_type == "grounding_updated":
            s.grounding = payload.get("grounding", s.grounding)
            s.satisfied = bool(payload.get("satisfied", s.satisfied))
            s.grounding_status = payload.get("status", s.grounding_status)
            if s.grounding:
                s.assignment = s.grounding.get("assignment", s.assignment)
                s.operation_bindings = list(s.grounding.get("operation_bindings", s.operation_bindings))
                s.missing_roles = list(s.grounding.get("missing_roles", []))
                s.unsatisfied_relations = list(s.grounding.get("unsatisfied_relations", []))
                s.unresolved_constraints = list(s.grounding.get("unresolved_constraints", []))
            if "scene_graph" in payload and payload["scene_graph"] is not None:
                s.scene_graph = payload["scene_graph"]

        elif event_type == "search_region_selected":
            s.current_selected_region = payload.get("region")

        elif event_type == "search_region_opened":
            region = payload.get("region")
            if region:
                if region not in s.inspected_regions:
                    s.inspected_regions.append(region)
                s.exploratory_open_trace.append({
                    "region": region,
                    "success": payload.get("success", True),
                    "exploratory": payload.get("exploratory", True),
                })

        elif event_type == "plan_ready":
            s.plan_actions = list(payload.get("actions", s.plan_actions))
            s.search_statistics = dict(payload.get("search_statistics", s.search_statistics))

        elif event_type == "run_finished":
            s.is_terminal = True
            s.terminal_status = payload.get("terminal_status", s.terminal_status)
            s.stage = "complete"
            s.frozen_elapsed_seconds = time.monotonic() - s.started_monotonic

        elif event_type == "run_failed":
            s.is_terminal = True
            s.is_exception = True
            s.terminal_status = "PIPELINE_EXCEPTION"
            s.exception_type = payload.get("error_type", "UnknownError")
            s.exception_message = payload.get("error_message", "Pipeline failed with exception")
            s.stage = "failed"
            s.frozen_elapsed_seconds = time.monotonic() - s.started_monotonic

    def _worker_loop(self) -> None:
        last_render_time = 0.0
        frame_interval = 1.0 / max(1, self.fps)

        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=frame_interval)
                if item is None:
                    break
                event_type, payload = item
                self._reduce_event(event_type, payload)

                # Drain pending events to reduce state completely before rendering
                while True:
                    try:
                        next_item = self._queue.get_nowait()
                        if next_item is None:
                            return
                        ev, py = next_item
                        self._reduce_event(ev, py)
                    except queue.Empty:
                        break

                now = time.monotonic()
                if now - last_render_time >= frame_interval:
                    self._render_and_show()
                    last_render_time = now

            except queue.Empty:
                if not self._state.is_terminal:
                    now = time.monotonic()
                    if now - last_render_time >= frame_interval:
                        self._render_and_show()
                        last_render_time = now

            except Exception as error:
                print(
                    f"VISUALIZER WORKER ERROR: {type(error).__name__}: {error}",
                    file=sys.stderr,
                    flush=True,
                )
                self._state.display_errors.append({
                    "event": "worker_loop",
                    "type": type(error).__name__,
                    "message": str(error),
                })
                self._viewer_disabled = True

    def _render_and_show(self) -> None:
        if self._viewer_disabled or self._viewer is None:
            return
        try:
            frame_rgb = self.render_composite_frame()
            self._viewer.show(frame_rgb)
        except Exception as error:
            print(
                f"VISUALIZER DISPLAY ERROR: {type(error).__name__}: {error}",
                file=sys.stderr,
                flush=True,
            )
            self._state.display_errors.append({
                "event": "viewer_show",
                "type": type(error).__name__,
                "message": str(error),
            })
            self._viewer_disabled = True

    def render_composite_frame(self) -> np.ndarray:
        """Render the 1600x960 6-panel composite display image."""
        composite = Image.new("RGB", (self.width, self.height), (24, 24, 27))

        # Row 1: Camera (0..560, 0..400), Status+Search (560..1600, 0..400)
        camera_img = self._render_camera_panel(560, 400)
        status_img = self._render_status_and_search_panel(1040, 400)
        composite.paste(camera_img, (0, 0))
        composite.paste(status_img, (560, 0))

        # Row 2: G_F (0..800, 400..680), G_O (800..1600, 400..680)
        gf_img = self._get_or_render_gf_panel(800, 280)
        go_img = self._get_or_render_go_panel(800, 280)
        composite.paste(gf_img, (0, 400))
        composite.paste(go_img, (800, 400))

        # Row 3: Grounding/Assignment (0..800, 680..960), Plan (800..1600, 680..960)
        assignment_img = self._render_assignment_panel(800, 280)
        plan_img = self._render_plan_panel(800, 280)
        composite.paste(assignment_img, (0, 680))
        composite.paste(plan_img, (800, 680))

        return np.array(composite, dtype=np.uint8)

    def _render_camera_panel(self, w: int, h: int) -> Image.Image:
        panel = Image.new("RGB", (w, h), (30, 30, 36))
        draw = ImageDraw.Draw(panel)
        draw.rectangle([(2, 2), (w - 3, h - 3)], outline=(63, 63, 70), width=1)
        draw.text((12, 10), "CAMERA / LATEST OBSERVATION", fill=(244, 244, 245), font=self._font_heading)
        draw.text((w - 180, 12), f"Stage: {self._state.latest_stage_label}", fill=(161, 161, 170), font=self._font_small)

        view_w, view_h = w - 24, h - 45
        view_x, view_y = 12, 35

        if self._state.latest_frame is not None and self._state.latest_frame.size > 0:
            try:
                frame_pil = Image.fromarray(self._state.latest_frame)
                frame_pil.thumbnail((view_w, view_h), Image.Resampling.BILINEAR)
                px = view_x + (view_w - frame_pil.width) // 2
                py = view_y + (view_h - frame_pil.height) // 2
                panel.paste(frame_pil, (px, py))
            except Exception:
                draw.rectangle([(view_x, view_y), (view_x + view_w, view_y + view_h)], fill=(18, 18, 20))
                draw.text((view_x + 20, view_y + view_h // 2), "Frame display error", fill=(248, 113, 113), font=self._font_body)
        else:
            draw.rectangle([(view_x, view_y), (view_x + view_w, view_y + view_h)], fill=(18, 18, 20))
            msg = "Waiting for observation..." if not self._state.is_terminal else "No observation frame available"
            draw.text((view_x + 40, view_y + view_h // 2 - 10), msg, fill=(161, 161, 170), font=self._font_body)

        return panel

    def _render_status_and_search_panel(self, w: int, h: int) -> Image.Image:
        panel = Image.new("RGB", (w, h), (34, 34, 40))
        draw = ImageDraw.Draw(panel)
        draw.rectangle([(2, 2), (w - 3, h - 3)], outline=(63, 63, 70), width=1)

        domain_name = self._state.domain.upper()
        variant_name = self._state.variant.upper()
        spec_mode = self._state.spec_mode.upper()
        draw.text((16, 10), f"SYSTEM STATUS — {domain_name} {variant_name} ({spec_mode})", fill=(255, 255, 255), font=self._font_heading)

        elapsed = self._state.frozen_elapsed_seconds if self._state.frozen_elapsed_seconds is not None else (time.monotonic() - self._state.started_monotonic)
        draw.text((w - 160, 10), f"Time: {elapsed:05.1f}s", fill=(56, 189, 248), font=self._font_heading)

        y = 42
        draw.text((16, y), "Stage:", fill=(161, 161, 170), font=self._font_body)
        draw.text((70, y), self._state.stage.upper(), fill=(228, 228, 231), font=self._font_body)

        status_col = (74, 222, 128) if self._state.terminal_status == "ACTION_SEQUENCE_READY" else (251, 191, 36) if self._state.terminal_status == "INFEASIBLE" else (248, 113, 113) if self._state.is_exception else (161, 161, 170)
        draw.text((260, y), "Status:", fill=(161, 161, 170), font=self._font_body)
        draw.text((320, y), self._state.terminal_status, fill=status_col, font=self._font_body)

        if self._state.is_exception and self._state.exception_type:
            y += 24
            draw.text((16, y), f"Error: [{self._state.exception_type}] {_truncate_text(self._state.exception_message or '', 90)}", fill=(248, 113, 113), font=self._font_small)

        y += 28
        draw.line([(16, y), (w - 16, y)], fill=(63, 63, 70), width=1)

        y += 10
        draw.text((16, y), "SEARCH REGIME & SEQUENTIAL PROGRESSION", fill=(244, 244, 245), font=self._font_heading)

        y += 24
        if self._state.domain == "living_room":
            draw.text((16, y), "Search Policy: N/A (single-stage global grounding, no region search)", fill=(161, 161, 170), font=self._font_body)
        else:
            eff_policy = self._state.search_order_source_effective or self._state.search_order_source_requested
            policy_label = "GT ORACLE (Privileged)" if eff_policy == "oracle" else "FM-GUIDED (VLM Ranking)" if (eff_policy == "provider" and self._state.spec_mode == "vlm") else "PROVIDER (Manual Canonical)" if eff_policy == "provider" else f"SEEDED RANDOM (Seed={self._state.search_seed_effective or self._state.search_seed_requested})" if eff_policy == "random" else str(eff_policy).upper()

            draw.text((16, y), f"Policy: {policy_label}", fill=(56, 189, 248), font=self._font_body)
            draw.text((450, y), f"Requested: {self._state.search_order_source_requested}", fill=(161, 161, 170), font=self._font_body)

            y += 24
            order_str = " -> ".join(self._state.resolved_region_order) if self._state.resolved_region_order else "None"
            draw.text((16, y), f"Resolved Order: {order_str}", fill=(228, 228, 231), font=self._font_body)

            y += 24
            inspected_str = ", ".join(self._state.inspected_regions) if self._state.inspected_regions else "None"
            draw.text((16, y), f"Inspected: [{inspected_str}]", fill=(74, 222, 128), font=self._font_body)

            current_sel = self._state.current_selected_region or "None"
            draw.text((450, y), f"Selected Region: {current_sel}", fill=(251, 191, 36), font=self._font_body)

        y += 28
        draw.line([(16, y), (w - 16, y)], fill=(63, 63, 70), width=1)
        y += 10
        draw.text((16, y), f"Actuation: {self._state.exploration_actuation}  |  Dropped updates: {self._state.dropped_display_updates}  |  UI Errors: {len(self._state.display_errors)}", fill=(113, 113, 122), font=self._font_small)

        return panel

    def _get_or_render_gf_panel(self, w: int, h: int) -> Image.Image:
        key = str(self._state.spec_graph) if self._state.spec_graph else "None"
        if self._cached_gf_key == key and self._cached_gf_image is not None:
            return self._cached_gf_image

        panel = Image.new("RGB", (w, h), (28, 28, 32))
        draw = ImageDraw.Draw(panel)
        draw.rectangle([(2, 2), (w - 3, h - 3)], outline=(63, 63, 70), width=1)
        draw.text((12, 8), "FUNCTIONAL REQUIREMENT GRAPH  G_F", fill=(244, 244, 245), font=self._font_heading)

        if self._state.spec_source:
            draw.text((w - 240, 10), f"Source: {self._state.spec_source}", fill=(161, 161, 170), font=self._font_small)

        if not self._state.spec_graph:
            draw.text((20, h // 2), "Waiting for functional specification...", fill=(161, 161, 170), font=self._font_body)
            self._cached_gf_key = key
            self._cached_gf_image = panel
            return panel

        nodes = self._state.spec_graph.get("nodes", {})
        relations = self._state.spec_graph.get("edges", self._state.spec_graph.get("relations", []))
        op_groups = self._state.spec_graph.get("operation_groups", [])

        y = 34
        draw.text((12, y), f"Roles ({len(nodes)}):", fill=(56, 189, 248), font=self._font_body)
        y += 18
        displayed_nodes = 0
        for name, node_data in list(nodes.items())[:MAX_GF_NODES]:
            kind = node_data.get("entity_kind", "OBJECT")
            count = node_data.get("count", 1)
            cats = node_data.get("semantic_categories", [])
            cats_str = f"[{', '.join(cats[:2])}]" if cats else ""
            line = f"• {name} ({kind}, n={count}) {cats_str}"
            draw.text((18, y), _truncate_text(line, 55), fill=(228, 228, 231), font=self._font_small)
            y += 15
            displayed_nodes += 1
            if y > h - 80:
                break
        if len(nodes) > displayed_nodes:
            draw.text((18, y), f"... +{len(nodes) - displayed_nodes} roles omitted", fill=(113, 113, 122), font=self._font_small)
            y += 15

        rx = w // 2 + 10
        ry = 34
        draw.text((rx, ry), f"Relations ({len(relations)}):", fill=(56, 189, 248), font=self._font_body)
        ry += 18
        displayed_rels = 0
        for rel in list(relations)[:MAX_RELATIONS // 2]:
            sub = rel.get("subject", "")
            pred = rel.get("predicate", "")
            obj = rel.get("object", "")
            draw.text((rx + 6, ry), _truncate_text(f"{sub} --{pred}--> {obj}", 45), fill=(228, 228, 231), font=self._font_small)
            ry += 15
            displayed_rels += 1
            if ry > h - 80:
                break

        if op_groups:
            ry = max(ry + 4, h - 70)
            draw.text((rx, ry), f"Operation Groups ({len(op_groups)}):", fill=(251, 191, 36), font=self._font_small)
            ry += 14
            for op in op_groups[:2]:
                fn = op.get("function", "")
                tool = op.get("tool_role", "")
                target = op.get("target_role", "")
                draw.text((rx + 6, ry), _truncate_text(f"• {fn}({tool} -> {target})", 45), fill=(228, 228, 231), font=self._font_small)
                ry += 14

        self._cached_gf_key = key
        self._cached_gf_image = panel
        return panel

    def _get_or_render_go_panel(self, w: int, h: int) -> Image.Image:
        key = str(self._state.scene_graph) if self._state.scene_graph else "None"
        if self._cached_go_key == key and self._cached_go_image is not None:
            return self._cached_go_image

        panel = Image.new("RGB", (w, h), (28, 28, 32))
        draw = ImageDraw.Draw(panel)
        draw.rectangle([(2, 2), (w - 3, h - 3)], outline=(63, 63, 70), width=1)
        draw.text((12, 8), "OBSERVED SCENE GRAPH  G_O", fill=(244, 244, 245), font=self._font_heading)

        if not self._state.scene_graph:
            draw.text((20, h // 2), "Waiting for observed scene evidence...", fill=(161, 161, 170), font=self._font_body)
            self._cached_go_key = key
            self._cached_go_image = panel
            return panel

        nodes = self._state.scene_graph.get("nodes", {})
        edges = self._state.scene_graph.get("edges", [])

        y = 34
        draw.text((12, y), f"Observed Nodes ({len(nodes)}):", fill=(56, 189, 248), font=self._font_body)
        y += 18
        displayed_nodes = 0
        for node_id, node_data in list(nodes.items())[:MAX_GO_NODES]:
            cat = node_data.get("canonical_category") or "unknown"
            region = node_data.get("source_region") or ""
            kind = node_data.get("entity_kind", "OBJECT")
            line = f"• {node_id}: {cat} ({kind}, {region})"
            draw.text((18, y), _truncate_text(line, 55), fill=(228, 228, 231), font=self._font_small)
            y += 15
            displayed_nodes += 1
            if y > h - 30:
                break
        if len(nodes) > displayed_nodes:
            draw.text((18, y), f"... +{len(nodes) - displayed_nodes} nodes omitted", fill=(113, 113, 122), font=self._font_small)

        rx = w // 2 + 10
        ry = 34
        draw.text((rx, ry), f"Relations ({len(edges)}):", fill=(56, 189, 248), font=self._font_body)
        ry += 18
        displayed_edges = 0
        for edge in list(edges)[:MAX_RELATIONS]:
            sub = edge.get("subject", "")
            pred = edge.get("predicate", "")
            obj = edge.get("object", "")
            val = edge.get("value", True)
            val_str = "TRUE" if val is True else "FALSE" if val is False else "UNK"
            val_col = (74, 222, 128) if val is True else (248, 113, 113) if val is False else (251, 191, 36)
            line = f"{sub} --{pred}--> {obj} [{val_str}]"
            draw.text((rx + 6, ry), _truncate_text(line, 45), fill=val_col, font=self._font_small)
            ry += 15
            displayed_edges += 1
            if ry > h - 30:
                break

        self._cached_go_key = key
        self._cached_go_image = panel
        return panel

    def _render_assignment_panel(self, w: int, h: int) -> Image.Image:
        panel = Image.new("RGB", (w, h), (30, 30, 36))
        draw = ImageDraw.Draw(panel)
        draw.rectangle([(2, 2), (w - 3, h - 3)], outline=(63, 63, 70), width=1)
        draw.text((12, 8), "CONSTRAINT GROUNDING & ASSIGNMENT  φ*", fill=(244, 244, 245), font=self._font_heading)

        y = 34
        status_str = "SATISFIED (φ* Complete)" if self._state.satisfied else f"INCOMPLETE ({self._state.grounding_status})" if self._state.grounding_status != "INFEASIBLE" else "INFEASIBLE (No valid grounding)"
        status_col = (74, 222, 128) if self._state.satisfied else (248, 113, 113) if self._state.grounding_status == "INFEASIBLE" else (251, 191, 36)
        draw.text((12, y), f"Status: {status_str}", fill=status_col, font=self._font_body)

        y += 24
        if self._state.assignment:
            draw.text((12, y), "Role Bindings:", fill=(56, 189, 248), font=self._font_body)
            y += 18
            for role, bound_obj in self._state.assignment.items():
                bound_str = str(bound_obj)
                draw.text((18, y), f"• {role}  ->  {bound_str}", fill=(228, 228, 231), font=self._font_small)
                y += 15
                if y > h - 60:
                    break
        elif self._state.missing_roles:
            draw.text((12, y), "Missing Roles:", fill=(248, 113, 113), font=self._font_body)
            y += 18
            for r in self._state.missing_roles[:4]:
                draw.text((18, y), f"• {r}", fill=(228, 228, 231), font=self._font_small)
                y += 15

        if self._state.operation_bindings:
            rx = w // 2 + 10
            ry = 34
            draw.text((rx, ry), "Operation Bindings:", fill=(56, 189, 248), font=self._font_body)
            ry += 18
            for ob in self._state.operation_bindings[:4]:
                fn = ob.get("function", "")
                tool_inst = ob.get("tool_instance", "")
                tgt_inst = ob.get("target_instance", "")
                draw.text((rx + 6, ry), f"• {fn}({tool_inst} -> {tgt_inst})", fill=(228, 228, 231), font=self._font_small)
                ry += 15

        return panel

    def _render_plan_panel(self, w: int, h: int) -> Image.Image:
        panel = Image.new("RGB", (w, h), (30, 30, 36))
        draw = ImageDraw.Draw(panel)
        draw.rectangle([(2, 2), (w - 3, h - 3)], outline=(63, 63, 70), width=1)
        draw.text((12, 8), "EXPLORATION TRACE & FINAL A* PLAN", fill=(244, 244, 245), font=self._font_heading)

        y = 34
        draw.text((12, y), f"Exploratory OPENs ({len(self._state.exploratory_open_trace)}):", fill=(251, 191, 36), font=self._font_body)
        y += 18
        if self._state.exploratory_open_trace:
            for item in self._state.exploratory_open_trace[:MAX_EXPLORATORY_OPEN]:
                reg = item.get("region", "")
                draw.text((18, y), f"• OPEN({reg}) [pre-TAMP]", fill=(228, 228, 231), font=self._font_small)
                y += 15
                if y > h - 30:
                    break
        else:
            msg = "None (no search)" if self._state.domain == "living_room" else "None"
            draw.text((18, y), msg, fill=(161, 161, 170), font=self._font_small)

        rx = w // 2 + 10
        ry = 34
        draw.text((rx, ry), f"Final A* Plan ({len(self._state.plan_actions)} actions):", fill=(74, 222, 128), font=self._font_body)
        ry += 18
        if self._state.plan_actions:
            for act in self._state.plan_actions[:MAX_ACTIONS]:
                idx = act.get("action_index", 0)
                op = act.get("operator", "")
                args = act.get("arguments", [])
                draw.text((rx + 6, ry), f"{idx:02d}. {op}({', '.join(args)})", fill=(228, 228, 231), font=self._font_small)
                ry += 15
                if ry > h - 40:
                    break
            if self._state.search_statistics:
                exp = self._state.search_statistics.get("expansions", 0)
                st = self._state.search_statistics.get("search_time_sec", 0.0)
                draw.text((rx + 6, h - 22), f"A* expansions: {exp} | search: {st:.3f}s", fill=(113, 113, 122), font=self._font_small)
        elif self._state.terminal_status == "INFEASIBLE":
            draw.text((rx + 6, ry), "NO FINAL PLAN (Task is infeasible)", fill=(251, 191, 36), font=self._font_small)
        elif self._state.is_exception:
            draw.text((rx + 6, ry), "NO PLAN (Pipeline exception occurred)", fill=(248, 113, 113), font=self._font_small)
        else:
            draw.text((rx + 6, ry), "Waiting for planning...", fill=(161, 161, 170), font=self._font_small)

        return panel

    def hold_until_closed(self, poll_interval_sec: float = 0.1) -> None:
        """Hold window open after run completion until user closes ffplay or interrupts."""
        if self._viewer is None or getattr(self._viewer, "closed", False):
            return
        process = getattr(self._viewer, "process", None)
        if process is None:
            return
        try:
            while process.poll() is None and not self._stop_event.is_set():
                time.sleep(poll_interval_sec)
        except (KeyboardInterrupt, SystemExit):
            pass

    def drain_errors(self) -> list[dict[str, Any]]:
        """Return and clear accumulated display/worker errors."""
        errs = list(self._state.display_errors)
        self._state.display_errors.clear()
        return errs

    def close(self) -> None:
        """Clean shutdown of worker thread and LiveMosaicViewer."""
        self._stop_event.set()
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None
