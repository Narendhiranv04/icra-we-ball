"""Non-intrusive live pipeline visualizer for Phase 3 functional TAMP."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import os
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Any, Callable, Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mujoco_scenes.live_mosaic_viewer import LiveMosaicViewer

# Display item caps to keep panels legible
MAX_GF_ROLES = 10
MAX_GF_RELATIONS = 12
MAX_GF_OP_GROUPS = 4
MAX_GO_NODES = 14
MAX_GO_RELATIONS = 16
MAX_ACTIONS = 12
MAX_EXPLORATORY_OPEN = 10
MAX_UNSATISFIED_RELS = 6


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
    latest_frame_path: str | None = None
    latest_stage_label: str = "initial"

    # G_O
    scene_graph: dict[str, Any] | None = None

    # Grounding & Assignment
    grounding: dict[str, Any] | None = None
    satisfied: bool = False
    grounding_status: str = "UNINITIALIZED"
    assignment: dict[str, Any] | None = None
    operation_bindings: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    missing_roles: list[str] = field(default_factory=list)
    unsatisfied_relations: list[Any] = field(default_factory=list)
    unresolved_constraints: list[str] = field(default_factory=list)

    # Search & Inspection
    current_selected_region: str | None = None
    inspected_regions: list[str] = field(default_factory=list)
    exploratory_open_trace: list[dict[str, Any]] = field(default_factory=list)

    # Plan
    plan_actions: list[dict[str, Any]] = field(default_factory=list)
    search_statistics: dict[str, Any] = field(default_factory=dict)

    # Telemetry health
    dropped_display_updates: int = 0
    display_errors: list[dict[str, Any]] = field(default_factory=list)


def _safe_copy_value(val: Any) -> Any:
    """Recursively make defensive copy of json-compatible and numpy structures."""
    if val is None or isinstance(val, (bool, int, float, str)):
        return val
    if isinstance(val, Path):
        return str(val)
    if isinstance(val, np.ndarray):
        return val.copy()
    if isinstance(val, dict):
        return {str(k): _safe_copy_value(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        copied_list = [_safe_copy_value(v) for v in val]
        return copied_list if isinstance(val, list) else tuple(copied_list)
    raise TypeError(f"Unsupported live object in visualization payload: {type(val).__name__}")


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


def _format_gf_role(role_data: Mapping[str, Any]) -> str:
    name = role_data.get("name", "")
    kind = role_data.get("entity_kind", "OBJECT")
    count = role_data.get("count", 1)
    min_c = role_data.get("min_count")
    max_c = role_data.get("max_count")
    if min_c is not None or max_c is not None:
        count_str = f"n={min_c or count}..{max_c or count}"
    else:
        count_str = f"n={count}"
    policy = role_data.get("binding_policy", "DISTINCT")
    cats = role_data.get("semantic_categories", [])
    cats_str = f" sem:[{', '.join(str(c) for c in cats[:2])}]" if cats else ""
    return f"{name} ({kind}, {count_str}, {policy}){cats_str}"


def _format_gf_relation(rel_data: Mapping[str, Any]) -> str:
    sub = rel_data.get("subject_role", rel_data.get("subject", ""))
    pred = rel_data.get("predicate", "")
    obj = rel_data.get("object_role", rel_data.get("object", ""))
    return f"{sub} --{pred}--> {obj}"


def _format_operation_group(op_data: Mapping[str, Any]) -> str:
    gid = op_data.get("id", "")
    fn = op_data.get("function", "")
    tool = op_data.get("tool_role", "")
    tgt = op_data.get("target_role", "")
    cnt = op_data.get("required_target_count", 1)
    return f"{gid}: {fn}({tool} -> {tgt}, n={cnt})"


def _format_go_relation(rel_data: Mapping[str, Any]) -> tuple[str, str]:
    sub = rel_data.get("subject_id", rel_data.get("subject", ""))
    pred = rel_data.get("predicate", "")
    obj = rel_data.get("object_id", rel_data.get("object", ""))
    status = str(rel_data.get("status", "UNKNOWN")).upper()
    if status not in {"TRUE", "FALSE", "UNKNOWN"}:
        status = "UNKNOWN"
    return f"{sub} --{pred}--> {obj}", status


def _format_operation_binding(group_id: str, binding: Mapping[str, Any]) -> str:
    items = []
    for k, v in sorted(binding.items()):
        items.append(f"{k}={v}")
    return f"{group_id}: " + (", ".join(items) if items else "{}")


def _format_unsatisfied_relation(rel: Any) -> str:
    if isinstance(rel, dict):
        pred = rel.get("predicate", "REL")
        sub = rel.get("subject_role", rel.get("subject", "?"))
        obj = rel.get("object_role", rel.get("object", "?"))
        return f"{pred}({sub}, {obj})"
    return str(rel)


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

        self._state_lock = threading.Lock()
        self._state = _VisualizerState()
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=2)
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
        """Pipeline callback: defensively copies payload, reduces under lock, non-blocking wakeup."""
        if self._stop_event.is_set():
            return
        snapshot = _safe_copy_value(payload)
        with self._state_lock:
            self._reduce_event(event_type, snapshot)
        try:
            self._queue.put_nowait("REDRAW")
        except queue.Full:
            with self._state_lock:
                self._state.dropped_display_updates += 1

    def _reduce_event(self, event_type: str, payload: dict[str, Any]) -> None:
        s = self._state
        if event_type == "run_started":
            s.domain = payload.get("domain", s.domain)
            s.variant = payload.get("variant", s.variant)
            s.spec_mode = payload.get("spec_mode", s.spec_mode)
            s.search_order_source_requested = payload.get("search_order_source_requested", s.search_order_source_requested)
            s.search_seed_requested = payload.get("search_seed_requested", s.search_seed_requested)
            s.exploration_actuation = payload.get("exploration_actuation", s.exploration_actuation)
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
                s.latest_frame = payload["frame_rgb"]
                s.latest_frame_path = None
            elif "frame_path" in payload and payload["frame_path"]:
                s.latest_frame_path = payload["frame_path"]

        elif event_type == "grounding_updated":
            s.grounding = payload.get("grounding", s.grounding)
            s.satisfied = bool(payload.get("satisfied", s.satisfied))
            s.grounding_status = payload.get("status", s.grounding_status)
            if s.grounding:
                s.assignment = s.grounding.get("assignment", s.assignment)
                bindings_raw = s.grounding.get("operation_bindings")
                if isinstance(bindings_raw, dict):
                    s.operation_bindings = bindings_raw
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
                token = self._queue.get(timeout=frame_interval)
                if token is None:
                    break

                # Drain any extra queued tokens to coalesce redraws
                while True:
                    try:
                        next_token = self._queue.get_nowait()
                        if next_token is None:
                            return
                    except queue.Empty:
                        break

                now = time.monotonic()
                if now - last_render_time >= frame_interval:
                    self._render_and_show()
                    last_render_time = now

            except queue.Empty:
                with self._state_lock:
                    is_term = self._state.is_terminal
                if not is_term:
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
                with self._state_lock:
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
            with self._state_lock:
                self._state.display_errors.append({
                    "event": "viewer_show",
                    "type": type(error).__name__,
                    "message": str(error),
                })
            self._viewer_disabled = True

    def _snapshot_state_for_render(self) -> _VisualizerState:
        with self._state_lock:
            snap = copy.deepcopy(self._state)
        # Load frame_path on worker thread if needed
        if snap.latest_frame is None and snap.latest_frame_path:
            frame_p = Path(snap.latest_frame_path)
            if frame_p.exists():
                try:
                    with Image.open(frame_p) as img:
                        snap.latest_frame = np.array(img.convert("RGB"), dtype=np.uint8)
                except Exception as error:
                    with self._state_lock:
                        self._state.display_errors.append({
                            "event": "load_frame_path",
                            "type": type(error).__name__,
                            "message": str(error),
                        })
        return snap

    def render_composite_frame(self) -> np.ndarray:
        """Render the 1600x960 6-panel composite display image from a thread-safe snapshot."""
        s = self._snapshot_state_for_render()
        composite = Image.new("RGB", (self.width, self.height), (24, 24, 27))

        # Row 1: Camera (0..560, 0..400), Status+Search (560..1600, 0..400)
        camera_img = self._render_camera_panel(s, 560, 400)
        status_img = self._render_status_and_search_panel(s, 1040, 400)
        composite.paste(camera_img, (0, 0))
        composite.paste(status_img, (560, 0))

        # Row 2: G_F (0..800, 400..680), G_O (800..1600, 400..680)
        gf_img = self._get_or_render_gf_panel(s, 800, 280)
        go_img = self._get_or_render_go_panel(s, 800, 280)
        composite.paste(gf_img, (0, 400))
        composite.paste(go_img, (800, 400))

        # Row 3: Grounding/Assignment (0..800, 680..960), Plan (800..1600, 680..960)
        assignment_img = self._render_assignment_panel(s, 800, 280)
        plan_img = self._render_plan_panel(s, 800, 280)
        composite.paste(assignment_img, (0, 680))
        composite.paste(plan_img, (800, 680))

        return np.array(composite, dtype=np.uint8)

    def _render_camera_panel(self, s: _VisualizerState, w: int, h: int) -> Image.Image:
        panel = Image.new("RGB", (w, h), (30, 30, 36))
        draw = ImageDraw.Draw(panel)
        draw.rectangle([(2, 2), (w - 3, h - 3)], outline=(63, 63, 70), width=1)
        draw.text((12, 10), "CAMERA / LATEST OBSERVATION", fill=(244, 244, 245), font=self._font_heading)
        draw.text((w - 180, 12), f"Stage: {s.latest_stage_label}", fill=(161, 161, 170), font=self._font_small)

        view_w, view_h = w - 24, h - 45
        view_x, view_y = 12, 35

        if s.latest_frame is not None and s.latest_frame.size > 0:
            try:
                frame_pil = Image.fromarray(s.latest_frame)
                frame_pil.thumbnail((view_w, view_h), Image.Resampling.BILINEAR)
                px = view_x + (view_w - frame_pil.width) // 2
                py = view_y + (view_h - frame_pil.height) // 2
                panel.paste(frame_pil, (px, py))
            except Exception:
                draw.rectangle([(view_x, view_y), (view_x + view_w, view_y + view_h)], fill=(18, 18, 20))
                draw.text((view_x + 20, view_y + view_h // 2), "Frame display error", fill=(248, 113, 113), font=self._font_body)
        else:
            draw.rectangle([(view_x, view_y), (view_x + view_w, view_y + view_h)], fill=(18, 18, 20))
            msg = "Waiting for observation..." if not s.is_terminal else "No observation frame available"
            draw.text((view_x + 40, view_y + view_h // 2 - 10), msg, fill=(161, 161, 170), font=self._font_body)

        return panel

    def _render_status_and_search_panel(self, s: _VisualizerState, w: int, h: int) -> Image.Image:
        panel = Image.new("RGB", (w, h), (34, 34, 40))
        draw = ImageDraw.Draw(panel)
        draw.rectangle([(2, 2), (w - 3, h - 3)], outline=(63, 63, 70), width=1)

        domain_name = s.domain.upper()
        variant_name = s.variant.upper()
        spec_mode = s.spec_mode.upper()
        draw.text((16, 10), f"SYSTEM STATUS — {domain_name} {variant_name} ({spec_mode})", fill=(255, 255, 255), font=self._font_heading)

        elapsed = s.frozen_elapsed_seconds if s.frozen_elapsed_seconds is not None else (time.monotonic() - s.started_monotonic)
        draw.text((w - 160, 10), f"Time: {elapsed:05.1f}s", fill=(56, 189, 248), font=self._font_heading)

        y = 42
        draw.text((16, y), "Stage:", fill=(161, 161, 170), font=self._font_body)
        draw.text((70, y), s.stage.upper(), fill=(228, 228, 231), font=self._font_body)

        status_col = (74, 222, 128) if s.terminal_status == "ACTION_SEQUENCE_READY" else (251, 191, 36) if s.terminal_status == "INFEASIBLE" else (248, 113, 113) if s.is_exception else (161, 161, 170)
        draw.text((260, y), "Status:", fill=(161, 161, 170), font=self._font_body)
        draw.text((320, y), s.terminal_status, fill=status_col, font=self._font_body)

        if s.is_exception and s.exception_type:
            y += 24
            draw.text((16, y), f"Error: [{s.exception_type}] {_truncate_text(s.exception_message or '', 90)}", fill=(248, 113, 113), font=self._font_small)

        y += 28
        draw.line([(16, y), (w - 16, y)], fill=(63, 63, 70), width=1)

        y += 10
        draw.text((16, y), "SEARCH REGIME & SEQUENTIAL PROGRESSION", fill=(244, 244, 245), font=self._font_heading)

        y += 24
        if s.domain == "living_room":
            draw.text((16, y), "Search Policy: N/A (single-stage global grounding, no region search)", fill=(161, 161, 170), font=self._font_body)
        else:
            eff_policy = s.search_order_source_effective or s.search_order_source_requested
            seed_val = s.search_seed_effective if s.search_seed_effective is not None else s.search_seed_requested
            policy_label = "GT ORACLE (Privileged)" if eff_policy == "oracle" else "FM-GUIDED (VLM Ranking)" if (eff_policy == "provider" and s.spec_mode == "vlm") else "PROVIDER (Manual Canonical)" if eff_policy == "provider" else f"SEEDED RANDOM (Seed={seed_val})" if eff_policy == "random" else str(eff_policy).upper()

            draw.text((16, y), f"Policy: {policy_label}", fill=(56, 189, 248), font=self._font_body)
            draw.text((450, y), f"Requested: {s.search_order_source_requested}", fill=(161, 161, 170), font=self._font_body)

            y += 24
            order_str = " -> ".join(s.resolved_region_order) if s.resolved_region_order else "None"
            draw.text((16, y), f"Resolved Order: {order_str}", fill=(228, 228, 231), font=self._font_body)

            y += 24
            inspected_str = ", ".join(s.inspected_regions) if s.inspected_regions else "None"
            draw.text((16, y), f"Inspected: [{inspected_str}]", fill=(74, 222, 128), font=self._font_body)

            current_sel = s.current_selected_region or "None"
            draw.text((450, y), f"Selected Region: {current_sel}", fill=(251, 191, 36), font=self._font_body)

        y += 28
        draw.line([(16, y), (w - 16, y)], fill=(63, 63, 70), width=1)
        y += 10
        draw.text((16, y), f"Actuation: {s.exploration_actuation}  |  Dropped redraws: {s.dropped_display_updates}  |  UI Errors: {len(s.display_errors)}", fill=(113, 113, 122), font=self._font_small)

        return panel

    def _get_or_render_gf_panel(self, s: _VisualizerState, w: int, h: int) -> Image.Image:
        key = str(s.spec_graph) if s.spec_graph else "None"
        if self._cached_gf_key == key and self._cached_gf_image is not None:
            return self._cached_gf_image

        panel = Image.new("RGB", (w, h), (28, 28, 32))
        draw = ImageDraw.Draw(panel)
        draw.rectangle([(2, 2), (w - 3, h - 3)], outline=(63, 63, 70), width=1)
        draw.text((12, 8), "FUNCTIONAL REQUIREMENT GRAPH  G_F", fill=(244, 244, 245), font=self._font_heading)

        if s.spec_source:
            draw.text((w - 240, 10), f"Source: {s.spec_source}", fill=(161, 161, 170), font=self._font_small)

        if not s.spec_graph:
            draw.text((20, h // 2), "Waiting for functional specification...", fill=(161, 161, 170), font=self._font_body)
            self._cached_gf_key = key
            self._cached_gf_image = panel
            return panel

        nodes = s.spec_graph.get("nodes", {})
        if isinstance(nodes, list):
            nodes = {r.get("name", f"role_{i}"): r for i, r in enumerate(nodes)}
        relations = s.spec_graph.get("relations", s.spec_graph.get("edges", []))
        op_groups = s.spec_graph.get("operation_groups", [])

        # Left column: Roles
        y = 34
        draw.text((12, y), f"Roles ({len(nodes)}):", fill=(56, 189, 248), font=self._font_body)
        y += 18
        displayed_nodes = 0
        for name, node_data in list(nodes.items())[:MAX_GF_ROLES]:
            line = _format_gf_role(node_data)
            draw.text((18, y), _truncate_text(line, 55), fill=(228, 228, 231), font=self._font_small)
            y += 15

            # Compact secondary line for unary / numeric constraints if present
            unary = node_data.get("unary_predicates", node_data.get("unary_properties", []))
            num_constraints = node_data.get("numeric_constraints", [])
            extra_items = []
            if unary:
                extra_items.append(f"unary:[{', '.join(str(u) for u in unary)}]")
            for num in num_constraints:
                extra_items.append(f"{num.get('property_name')}{num.get('operator')}{num.get('threshold')}{num.get('unit')}")
            if extra_items:
                draw.text((26, y), _truncate_text(" ".join(extra_items), 55), fill=(161, 161, 170), font=self._font_small)
                y += 14

            displayed_nodes += 1
            if y > h - 40:
                break

        if len(nodes) > displayed_nodes:
            draw.text((18, y), f"... +{len(nodes) - displayed_nodes} roles omitted", fill=(113, 113, 122), font=self._font_small)

        # Right column: Relations and Operation Groups
        rx = w // 2 + 10
        ry = 34
        draw.text((rx, ry), f"Relations ({len(relations)}):", fill=(56, 189, 248), font=self._font_body)
        ry += 18
        displayed_rels = 0
        for rel in list(relations)[:MAX_GF_RELATIONS]:
            line = _format_gf_relation(rel)
            draw.text((rx + 6, ry), _truncate_text(line, 45), fill=(228, 228, 231), font=self._font_small)
            ry += 15
            displayed_rels += 1
            if ry > h - 85:
                break

        if op_groups:
            ry = max(ry + 4, h - 75)
            draw.text((rx, ry), f"Operation Groups ({len(op_groups)}):", fill=(251, 191, 36), font=self._font_small)
            ry += 15
            for op in list(op_groups)[:MAX_GF_OP_GROUPS]:
                line = _format_operation_group(op)
                draw.text((rx + 6, ry), _truncate_text(line, 45), fill=(228, 228, 231), font=self._font_small)
                ry += 14

        self._cached_gf_key = key
        self._cached_gf_image = panel
        return panel

    def _get_or_render_go_panel(self, s: _VisualizerState, w: int, h: int) -> Image.Image:
        key = str(s.scene_graph) if s.scene_graph else "None"
        if self._cached_go_key == key and self._cached_go_image is not None:
            return self._cached_go_image

        panel = Image.new("RGB", (w, h), (28, 28, 32))
        draw = ImageDraw.Draw(panel)
        draw.rectangle([(2, 2), (w - 3, h - 3)], outline=(63, 63, 70), width=1)
        draw.text((12, 8), "OBSERVED SCENE GRAPH  G_O", fill=(244, 244, 245), font=self._font_heading)

        if not s.scene_graph:
            draw.text((20, h // 2), "Waiting for observed scene evidence...", fill=(161, 161, 170), font=self._font_body)
            self._cached_go_key = key
            self._cached_go_image = panel
            return panel

        nodes_raw = s.scene_graph.get("nodes", s.scene_graph.get("objects", {}))
        if isinstance(nodes_raw, list):
            nodes = {n.get("instance_id", f"node_{i}"): n for i, n in enumerate(nodes_raw)}
        elif isinstance(nodes_raw, dict):
            nodes = nodes_raw
        else:
            nodes = {}

        relations_raw = s.scene_graph.get("relations", s.scene_graph.get("edges", []))
        if isinstance(relations_raw, dict):
            relations = list(relations_raw.values())
        elif isinstance(relations_raw, list):
            relations = relations_raw
        else:
            relations = []

        # Deterministic sorting: source_region -> entity_kind -> instance_id
        sorted_nodes = sorted(
            nodes.items(),
            key=lambda item: (
                str(item[1].get("source_region") or item[1].get("region") or ""),
                str(item[1].get("entity_kind") or ""),
                str(item[0]),
            ),
        )

        y = 34
        draw.text((12, y), f"Observed Nodes ({len(sorted_nodes)}):", fill=(56, 189, 248), font=self._font_body)
        y += 18
        displayed_nodes = 0
        for node_id, node_data in sorted_nodes[:MAX_GO_NODES]:
            cat = node_data.get("canonical_category") or "unknown"
            region = node_data.get("source_region") or node_data.get("region") or ""
            kind = node_data.get("entity_kind", "OBJECT")
            region_str = f", {region}" if region else ""
            line = f"• {node_id}: {cat} ({kind}{region_str})"
            draw.text((18, y), _truncate_text(line, 55), fill=(228, 228, 231), font=self._font_small)
            y += 15
            displayed_nodes += 1
            if y > h - 30:
                break
        if len(sorted_nodes) > displayed_nodes:
            draw.text((18, y), f"... +{len(sorted_nodes) - displayed_nodes} nodes omitted", fill=(113, 113, 122), font=self._font_small)

        # Right column: Relations (filtered / prioritized)
        rx = w // 2 + 10
        ry = 34
        draw.text((rx, ry), f"Relations ({len(relations)}):", fill=(56, 189, 248), font=self._font_body)
        ry += 18

        # Prioritize relations matching G_F predicates or current assignment
        gf_preds = set()
        if s.spec_graph:
            for r in s.spec_graph.get("relations", []):
                if isinstance(r, dict) and "predicate" in r:
                    gf_preds.add(str(r["predicate"]))
        assigned_insts = set(s.assignment.values()) if s.assignment else set()

        def _rel_priority(rel_dict: Mapping[str, Any]) -> int:
            sub = rel_dict.get("subject_id", rel_dict.get("subject", ""))
            obj = rel_dict.get("object_id", rel_dict.get("object", ""))
            pred = rel_dict.get("predicate", "")
            is_gf_pred = pred in gf_preds
            is_assigned = (sub in assigned_insts) or (obj in assigned_insts)
            if is_gf_pred and is_assigned:
                return 0
            if is_gf_pred:
                return 1
            if is_assigned:
                return 2
            return 3

        sorted_relations = sorted(relations, key=_rel_priority)

        displayed_edges = 0
        for rel in sorted_relations[:MAX_GO_RELATIONS]:
            line, status = _format_go_relation(rel)
            val_col = (74, 222, 128) if status == "TRUE" else (248, 113, 113) if status == "FALSE" else (251, 191, 36)
            full_line = f"{line} [{status}]"
            draw.text((rx + 6, ry), _truncate_text(full_line, 45), fill=val_col, font=self._font_small)
            ry += 15
            displayed_edges += 1
            if ry > h - 30:
                break

        if len(relations) > displayed_edges:
            draw.text((rx + 6, ry), f"... +{len(relations) - displayed_edges} relations omitted", fill=(113, 113, 122), font=self._font_small)

        self._cached_go_key = key
        self._cached_go_image = panel
        return panel

    def _render_assignment_panel(self, s: _VisualizerState, w: int, h: int) -> Image.Image:
        panel = Image.new("RGB", (w, h), (30, 30, 36))
        draw = ImageDraw.Draw(panel)
        draw.rectangle([(2, 2), (w - 3, h - 3)], outline=(63, 63, 70), width=1)
        draw.text((12, 8), "CONSTRAINT GROUNDING & ASSIGNMENT  φ*", fill=(244, 244, 245), font=self._font_heading)

        y = 34
        status_str = "SATISFIED (φ* Complete)" if s.satisfied else f"INCOMPLETE ({s.grounding_status})" if s.grounding_status != "INFEASIBLE" else "INFEASIBLE (No valid grounding)"
        status_col = (74, 222, 128) if s.satisfied else (248, 113, 113) if s.grounding_status == "INFEASIBLE" else (251, 191, 36)
        draw.text((12, y), f"Status: {status_str}", fill=status_col, font=self._font_body)

        y += 24
        if s.assignment:
            draw.text((12, y), "Role Bindings:", fill=(56, 189, 248), font=self._font_body)
            y += 18
            for role, bound_obj in list(s.assignment.items()):
                draw.text((18, y), f"• {role}  ->  {bound_obj}", fill=(228, 228, 231), font=self._font_small)
                y += 15
                if y > h - 60:
                    break
        else:
            if s.missing_roles:
                draw.text((12, y), f"Missing Roles ({len(s.missing_roles)}):", fill=(248, 113, 113), font=self._font_body)
                y += 18
                for r in s.missing_roles[:3]:
                    draw.text((18, y), f"• {r}", fill=(228, 228, 231), font=self._font_small)
                    y += 15

            if s.unsatisfied_relations:
                y += 4
                draw.text((12, y), f"Unsatisfied Relations ({len(s.unsatisfied_relations)}):", fill=(248, 113, 113), font=self._font_small)
                y += 15
                for rel in s.unsatisfied_relations[:MAX_UNSATISFIED_RELS // 2]:
                    draw.text((18, y), _truncate_text(f"• {_format_unsatisfied_relation(rel)}", 48), fill=(228, 228, 231), font=self._font_small)
                    y += 14

            if s.unresolved_constraints:
                y += 4
                draw.text((12, y), f"Unresolved Constraints ({len(s.unresolved_constraints)}):", fill=(251, 191, 36), font=self._font_small)
                y += 15
                for c in s.unresolved_constraints[:2]:
                    draw.text((18, y), _truncate_text(f"• {c}", 48), fill=(228, 228, 231), font=self._font_small)
                    y += 14

        # Right column: Operation Bindings
        rx = w // 2 + 10
        ry = 34
        if s.operation_bindings and isinstance(s.operation_bindings, dict):
            draw.text((rx, ry), "Operation Bindings:", fill=(56, 189, 248), font=self._font_body)
            ry += 18
            displayed_bindings = 0
            for group_id, bindings_list in list(s.operation_bindings.items())[:3]:
                if isinstance(bindings_list, list):
                    for b in bindings_list[:2]:
                        if isinstance(b, dict):
                            line = _format_operation_binding(group_id, b)
                            draw.text((rx + 6, ry), _truncate_text(line, 45), fill=(228, 228, 231), font=self._font_small)
                            ry += 15
                            displayed_bindings += 1
                elif isinstance(bindings_list, dict):
                    line = _format_operation_binding(group_id, bindings_list)
                    draw.text((rx + 6, ry), _truncate_text(line, 45), fill=(228, 228, 231), font=self._font_small)
                    ry += 15
                    displayed_bindings += 1
                if ry > h - 30:
                    break

        return panel

    def _render_plan_panel(self, s: _VisualizerState, w: int, h: int) -> Image.Image:
        panel = Image.new("RGB", (w, h), (30, 30, 36))
        draw = ImageDraw.Draw(panel)
        draw.rectangle([(2, 2), (w - 3, h - 3)], outline=(63, 63, 70), width=1)
        draw.text((12, 8), "EXPLORATION TRACE & FINAL A* PLAN", fill=(244, 244, 245), font=self._font_heading)

        # Left column: Exploratory OPENs
        y = 34
        draw.text((12, y), f"Exploratory OPENs ({len(s.exploratory_open_trace)}):", fill=(251, 191, 36), font=self._font_body)
        y += 18
        if s.exploratory_open_trace:
            for item in s.exploratory_open_trace[:MAX_EXPLORATORY_OPEN]:
                reg = item.get("region", "")
                draw.text((18, y), f"• OPEN({reg}) [pre-TAMP]", fill=(228, 228, 231), font=self._font_small)
                y += 15
                if y > h - 30:
                    break
        else:
            msg = "None (no search)" if s.domain == "living_room" else "None"
            draw.text((18, y), msg, fill=(161, 161, 170), font=self._font_small)

        # Right column: Final A* Plan
        rx = w // 2 + 10
        ry = 34
        draw.text((rx, ry), f"Final A* Plan ({len(s.plan_actions)} actions):", fill=(74, 222, 128), font=self._font_body)
        ry += 18
        if s.plan_actions:
            for act in s.plan_actions[:MAX_ACTIONS]:
                idx = act.get("action_index", 0)
                op = act.get("operator", "")
                args = act.get("arguments", [])
                draw.text((rx + 6, ry), f"{idx:02d}. {op}({', '.join(str(a) for a in args)})", fill=(228, 228, 231), font=self._font_small)
                ry += 15
                if ry > h - 40:
                    break
            if s.search_statistics:
                exp = s.search_statistics.get("expansions", 0)
                st = s.search_statistics.get("search_time_sec", 0.0)
                draw.text((rx + 6, h - 22), f"A* expansions: {exp} | search: {st:.3f}s", fill=(113, 113, 122), font=self._font_small)
        elif s.terminal_status == "INFEASIBLE":
            draw.text((rx + 6, ry), "NO FINAL PLAN (Task is infeasible)", fill=(251, 191, 36), font=self._font_small)
        elif s.is_exception:
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
        with self._state_lock:
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
