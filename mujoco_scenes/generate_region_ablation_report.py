"""Presentation report for the L2 living-room region ablation."""

from __future__ import annotations

import argparse
import html
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mujoco_scenes.geometry_checker import read_ply


STATUS_COLORS = {
    "TRUE": "#2eaa65",
    "FALSE": "#df4b4b",
    "UNKNOWN": "#9aa2ad",
}
MODE_TITLES = {
    "geometry_only": "Geometry-only",
    "semantic_only": "Semantic-only",
    "joint": "Joint semantic–geometric",
}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _font(size: int, bold: bool = False):
    names = (
        ("DejaVuSans-Bold.ttf", "Arial Bold.ttf")
        if bold
        else ("DejaVuSans.ttf", "Arial.ttf")
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _status_color(value: str) -> str:
    return STATUS_COLORS.get(value, "#9aa2ad")


def _region_label(row: dict[str, Any]) -> str:
    semantic = row.get("parent_semantic_label") or "unknown"
    return f"{row['region_id']}\n{semantic}"


def _render_matrix(rows: list[dict[str, Any]], output: Path) -> None:
    columns = [
        ("semantic_role_status", "Serving\nsemantics"),
        ("PLANAR_SUPPORT", "Planar\nsupport"),
        ("FITS_ON", "Tray\nfits"),
        ("NEAR_SEATING_AREA", "Near\nsofa"),
        ("geometry_only_status", "Geometry\nonly"),
        ("semantic_only_status", "Semantic\nonly"),
        ("joint_status", "Joint"),
    ]
    fig, axis = plt.subplots(
        figsize=(12.5, max(3.7, 1.15 * len(rows) + 1.8))
    )
    axis.set_xlim(0, len(columns))
    axis.set_ylim(0, len(rows))
    for row_index, row in enumerate(rows):
        y = len(rows) - row_index - 1
        for column_index, (key, _label) in enumerate(columns):
            value = row.get(key, "UNKNOWN")
            axis.add_patch(
                plt.Rectangle(
                    (column_index, y),
                    1,
                    1,
                    facecolor=_status_color(value),
                    edgecolor="white",
                    linewidth=2,
                )
            )
            axis.text(
                column_index + 0.5,
                y + 0.5,
                value,
                ha="center",
                va="center",
                color="white",
                fontsize=10,
                fontweight="bold",
            )
    axis.set_xticks(
        np.arange(len(columns)) + 0.5,
        [label for _key, label in columns],
        fontsize=10,
    )
    axis.set_yticks(
        np.arange(len(rows)) + 0.5,
        [_region_label(row) for row in reversed(rows)],
        fontsize=10,
    )
    axis.tick_params(length=0)
    axis.set_title(
        "Region–function compatibility matrix\n"
        "Green = TRUE · red = FALSE · grey = UNKNOWN",
        fontsize=15,
        fontweight="bold",
        pad=18,
    )
    for spine in axis.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _render_ablation(
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    output: Path,
) -> None:
    stage_count = max(len(rows), 1)
    fig, axes = plt.subplots(3, 1, figsize=(12, 7.6), sharex=True)
    for axis, mode in zip(axes, MODE_TITLES):
        result = summary["modes"][mode]
        axis.set_xlim(-0.35, stage_count - 0.65)
        axis.set_ylim(0, 1)
        axis.set_yticks([])
        axis.grid(axis="x", alpha=0.18)
        axis.set_title(MODE_TITLES[mode], loc="left", fontweight="bold")
        completion = result.get("completion_stage")
        for row in rows:
            stage = row["discovery_stage"]
            key = f"{mode}_status"
            status = row[key]
            axis.scatter(
                stage,
                0.5,
                s=650,
                c=_status_color(status),
                edgecolors="white",
                linewidths=2,
                zorder=3,
            )
            axis.text(
                stage,
                0.5,
                str(stage),
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
            )
        if completion is not None:
            axis.axvline(completion, color="#17233b", linestyle="--")
            axis.text(
                completion + 0.04,
                0.08,
                f"selected {result['selected_region_id']}",
                fontsize=9,
                color="#17233b",
            )
        else:
            axis.text(
                stage_count - 0.75,
                0.08,
                "EXHAUSTED",
                ha="right",
                color="#a52b2b",
                fontweight="bold",
            )
        for spine in axis.spines.values():
            spine.set_visible(False)
    axes[-1].set_xticks(
        range(stage_count),
        [
            f"{row['discovery_stage']}\n"
            f"{row.get('parent_semantic_label') or 'unknown'}"
            for row in rows
        ],
    )
    fig.suptitle(
        "Same evidence, different acceptance logic",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _render_graph(
    rows: list[dict[str, Any]],
    output: Path,
    *,
    maximum_stage: int | None = None,
) -> None:
    graph = nx.DiGraph()
    graph.add_node(
        "payload",
        label="payload\nobject_0001\nrefreshment\ntray",
        kind="payload",
    )
    graph.add_node(
        "function",
        label="function\nPLACE\nREFRESHMENT\nTRAY",
        kind="function",
    )
    graph.add_node(
        "sofa", label="observed\nsofa context", kind="context"
    )
    selected = None
    for row in rows:
        if maximum_stage is not None and row["discovery_stage"] > maximum_stage:
            continue
        region_id = row["region_id"]
        label = (
            f"{region_id}\n{row.get('parent_semantic_label') or 'UNKNOWN'}\n"
            f"{row['support_length_m']:.2f} × "
            f"{row['support_width_m']:.2f} m"
        )
        graph.add_node(region_id, label=label, kind="region")
        graph.add_edge(
            region_id,
            "function",
            label=f"SEM {row['semantic_role_status']}",
            status=row["semantic_role_status"],
        )
        graph.add_edge(
            "payload",
            region_id,
            label=f"FITS {row['FITS_ON']}",
            status=row["FITS_ON"],
        )
        graph.add_edge(
            region_id,
            "sofa",
            label=f"NEAR {row['NEAR_SEATING_AREA']}",
            status=row["NEAR_SEATING_AREA"],
        )
        if row["joint_status"] == "TRUE" and selected is None:
            selected = region_id
            graph[region_id]["function"]["label"] = "SEM TRUE\nASSIGNED"
    positions = {
        "payload": (-1.55, 0.0),
        "function": (1.55, 0.62),
        "sofa": (1.55, -0.62),
    }
    region_nodes = [
        node
        for node, data in graph.nodes(data=True)
        if data["kind"] == "region"
    ]
    for index, node in enumerate(region_nodes):
        positions[node] = (
            -0.18 + index * 0.18,
            0.66 - index * 0.66,
        )
    fig, axis = plt.subplots(figsize=(12, 7))
    node_colors = {
        "payload": "#e67e22",
        "function": "#7547b8",
        "context": "#3978b8",
        "region": "#2b8b72",
    }
    nx.draw_networkx_nodes(
        graph,
        positions,
        node_color=[
            node_colors[data["kind"]] for _node, data in graph.nodes(data=True)
        ],
        node_size=[
            6600 if data["kind"] in {"payload", "function"} else 5600
            for _node, data in graph.nodes(data=True)
        ],
        edgecolors="white",
        linewidths=2,
        ax=axis,
    )
    nx.draw_networkx_labels(
        graph,
        positions,
        labels={
            node: data["label"] for node, data in graph.nodes(data=True)
        },
        font_color="white",
        font_size=8,
        font_weight="bold",
        ax=axis,
    )
    for status in ("TRUE", "FALSE", "UNKNOWN"):
        edges = [
            (source, target)
            for source, target, data in graph.edges(data=True)
            if data["status"] == status
        ]
        if edges:
            nx.draw_networkx_edges(
                graph,
                positions,
                edgelist=edges,
                edge_color=_status_color(status),
                width=2.6,
                style="dashed" if status == "UNKNOWN" else "solid",
                arrows=True,
                arrowsize=17,
                connectionstyle="arc3,rad=0.08",
                ax=axis,
            )
    edge_labels = {
        (source, target): data["label"]
        for source, target, data in graph.edges(data=True)
    }
    nx.draw_networkx_edge_labels(
        graph,
        positions,
        edge_labels=edge_labels,
        font_size=8,
        rotate=False,
        label_pos=0.55,
        ax=axis,
    )
    axis.set_title(
        "Observed region-function graph",
        fontsize=16,
        fontweight="bold",
    )
    axis.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _render_point_cloud(stage_dir: Path, output: Path) -> None:
    region_points, region_colors = read_ply(
        stage_dir / "region_evidence" / "fused.ply"
    )
    payload_path = stage_dir / "payload_evidence" / "fused.ply"
    payload_points = np.empty((0, 3))
    payload_colors = np.empty((0, 3))
    if payload_path.exists():
        payload_points, payload_colors = read_ply(payload_path)
    fig = plt.figure(figsize=(8.2, 6.2))
    axis = fig.add_subplot(111, projection="3d")
    maximum = 18000
    if len(region_points) > maximum:
        indices = np.linspace(0, len(region_points) - 1, maximum).astype(int)
        region_points = region_points[indices]
        region_colors = region_colors[indices]
    axis.scatter(
        region_points[:, 0],
        region_points[:, 1],
        region_points[:, 2],
        c=region_colors / 255.0,
        s=1.2,
        alpha=0.9,
        label="stage-local support evidence",
    )
    metadata = _load(stage_dir / "inspection_metadata.json")
    volume = metadata["inspection_volume"]
    minimum = np.asarray(volume["minimum_world_m"], float)
    maximum_bounds = np.asarray(volume["maximum_world_m"], float)
    corners = np.asarray(
        [
            [x, y, z]
            for x in (minimum[0], maximum_bounds[0])
            for y in (minimum[1], maximum_bounds[1])
            for z in (minimum[2], maximum_bounds[2])
        ]
    )
    for first in range(8):
        for second in range(first + 1, 8):
            if np.count_nonzero(corners[first] != corners[second]) == 1:
                axis.plot(
                    *zip(corners[first], corners[second]),
                    color="#6552a5",
                    linewidth=0.9,
                    alpha=0.55,
                )
    target = np.asarray(metadata["target_world_m"], float)
    for camera in metadata["camera_poses"].values():
        position = np.asarray(camera["position_world_m"], float)
        direction = target - position
        direction /= max(np.linalg.norm(direction), 1e-9)
        axis.scatter(*position, c="#182033", marker="^", s=35)
        axis.quiver(
            *position,
            *(0.18 * direction),
            color="#182033",
            linewidth=1.1,
        )
    if len(payload_points):
        axis.scatter(
            payload_points[:, 0],
            payload_points[:, 1],
            payload_points[:, 2],
            c=payload_colors / 255.0,
            s=2,
            alpha=0.9,
            label="payload evidence",
        )
    axis.set_xlabel("world X (m)")
    axis.set_ylabel("world Y (m)")
    axis.set_zlabel("world Z (m)")
    axis.set_title(
        "Fresh stage-local evidence · five cameras · selection volume"
    )
    axis.view_init(elev=27, azim=-55)
    axis.legend(loc="upper left", fontsize=8)
    axis.set_box_aspect((1.3, 1.0, 0.45))
    fig.tight_layout()
    fig.savefig(output, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    copy = image.copy()
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(
        copy, ((size[0] - copy.width) // 2, (size[1] - copy.height) // 2)
    )
    return canvas


def _stage_overview(
    *,
    stage_dir: Path,
    stage_output: Path,
    row: dict[str, Any],
    graph_path: Path,
    pointcloud_path: Path,
) -> None:
    width, panel_height = 1800, 620
    canvas = Image.new("RGB", (width, 2 * panel_height + 150), "#f1f4f8")
    semantic = Image.open(stage_dir / "semantic_overview.png").convert("RGB")
    pointcloud = Image.open(pointcloud_path).convert("RGB")
    graph = Image.open(graph_path).convert("RGB")
    front_mask = Image.open(
        stage_dir / "cameras" / "inspection_front" / "evidence_masks.png"
    ).convert("RGB")
    canvas.paste(_fit_image(semantic, (900, panel_height)), (0, 70))
    canvas.paste(_fit_image(pointcloud, (900, panel_height)), (900, 70))
    canvas.paste(_fit_image(graph, (1050, panel_height)), (0, 690))
    canvas.paste(_fit_image(front_mask, (750, panel_height)), (1050, 690))
    draw = ImageDraw.Draw(canvas)
    title = (
        f"Stage {row['discovery_stage']:03d} · "
        f"{row.get('parent_semantic_label') or 'UNKNOWN'} · {row['region_id']}"
    )
    draw.text((28, 18), title, fill="#182033", font=_font(29, True))
    values = (
        f"semantic={row['semantic_role_status']}   "
        f"planar={row['PLANAR_SUPPORT']}   fits={row['FITS_ON']}   "
        f"near={row['NEAR_SEATING_AREA']}   joint={row['joint_status']}   "
        f"fit margin={row['fit_margin_m']:+.3f} m"
    )
    draw.rectangle((0, 1310, width, 1390), fill="#17233b")
    draw.text((28, 1332), values, fill="white", font=_font(22, True))
    canvas.save(stage_output)


def _copy_json_artifacts(run_dir: Path, report_dir: Path) -> None:
    names = (
        "run_config.json",
        "task_requirements.json",
        "region_registry.json",
        "payload_registry.json",
        "observed_graph.json",
        "region_function_evaluations.json",
        "region_compatibility_matrix.json",
        "region_compatibility_matrix.csv",
        "region_ablation_summary.json",
        "offline_region_ablation_evaluation.json",
        "region_ablation_validation.json",
        "verified_region_handoff.json",
        "events.jsonl",
    )
    for name in names:
        source = run_dir / name
        if source.exists():
            shutil.copy2(source, report_dir / name)


def _make_progression(
    frames: list[Path], gif_path: Path, mp4_path: Path
) -> dict[str, Any]:
    images = [Image.open(frame).convert("RGB") for frame in frames]
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=1800,
        loop=0,
        optimize=True,
    )
    command = [
        "ffmpeg",
        "-y",
        "-framerate",
        "1/2",
        "-i",
        str(frames[0].parent / "frame_%03d.png"),
        "-vf",
        "scale=1800:-2:flags=lanczos,format=yuv420p",
        "-r",
        "30",
        str(mp4_path),
    ]
    completed = subprocess.run(
        command, capture_output=True, text=True, check=False
    )
    return {
        "gif": str(gif_path),
        "mp4": str(mp4_path) if completed.returncode == 0 else None,
        "ffmpeg_returncode": completed.returncode,
        "ffmpeg_error": (
            completed.stderr[-1200:] if completed.returncode else None
        ),
    }


def _table_html(rows: list[dict[str, Any]]) -> str:
    output = []
    for row in rows:
        output.append(
            "<tr>"
            f"<td>{html.escape(row['region_id'])}</td>"
            f"<td>{html.escape(str(row.get('parent_semantic_label')))}</td>"
            f"<td>{row['semantic_confidence']:.3f}</td>"
            f"<td>{row['semantic_supporting_views']}</td>"
            f"<td>{row['support_length_m']:.3f} × "
            f"{row['support_width_m']:.3f}</td>"
            f"<td>{row['fit_margin_m']:+.3f}</td>"
            f"<td>{row['semantic_role_status']}</td>"
            f"<td>{row['PLANAR_SUPPORT']}</td>"
            f"<td>{row['FITS_ON']}</td>"
            f"<td>{row['NEAR_SEATING_AREA']}</td>"
            f"<td>{row['joint_status']}</td>"
            f"<td>{html.escape(str(row.get('rejection_reason')))}</td>"
            "</tr>"
        )
    return "\n".join(output)


def _value(record: dict[str, Any], name: str, default: Any = None) -> Any:
    value = record.get(name, default)
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "UNKNOWN"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return html.escape(str(value))


def _status_badge(status: str) -> str:
    normalized = str(status or "UNKNOWN").upper()
    return (
        f"<span class='status status-{normalized.lower()}'>"
        f"{html.escape(normalized)}</span>"
    )


def _mode_frame(
    source: Path,
    output: Path,
    *,
    mode: str,
    row: dict[str, Any],
    result: dict[str, Any],
) -> None:
    image = Image.open(source).convert("RGB")
    banner_height = 108
    canvas = Image.new(
        "RGB", (image.width, image.height + banner_height), "#0f172a"
    )
    canvas.paste(image, (0, banner_height))
    draw = ImageDraw.Draw(canvas)
    status = row[f"{mode}_status"]
    selected = (
        result.get("completion_stage") == row["discovery_stage"]
        and result.get("selected_region_id") == row["region_id"]
    )
    draw.text(
        (28, 18),
        f"{MODE_TITLES[mode]} · stage {row['discovery_stage']:03d} · "
        f"{row.get('parent_semantic_label') or 'UNKNOWN'}",
        fill="white",
        font=_font(28, True),
    )
    detail = f"candidate={status}"
    if selected:
        detail += f" · SELECTED {row['region_id']}"
    elif result.get("completion_stage") is not None:
        detail += f" · completion stage={result['completion_stage']}"
    draw.text(
        (30, 61),
        detail,
        fill=STATUS_COLORS.get(status, "#cbd5e1"),
        font=_font(23, True),
    )
    canvas.save(output)


def _write_html(
    *,
    report_dir: Path,
    run_config: dict[str, Any],
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    stage_assets: list[dict[str, Any]],
    region_registry: dict[str, Any],
    payload_registry: dict[str, Any],
    handoff: dict[str, Any] | None,
    mode_animations: dict[str, dict[str, Any]],
) -> None:
    expected = {
        "geometry_only": "Incorrect diagnostic: rug",
        "semantic_only": "Incorrect diagnostic: undersized side table",
        "joint": "Correct: coffee table",
    }
    outcomes = []
    for mode, title in MODE_TITLES.items():
        result = summary["modes"][mode]
        selected = result.get("selected_region_id") or "none"
        outcomes.append(
            "<tr>"
            f"<td><b>{html.escape(title)}</b></td>"
            f"<td><code>{html.escape(selected)}</code></td>"
            f"<td>{_fmt(result.get('completion_stage'))}</td>"
            f"<td>{html.escape(expected[mode])}</td>"
            f"<td>{'YES' if mode == 'joint' else 'NO'}</td>"
            "</tr>"
        )

    progression_rows = []
    for row in rows:
        progression_rows.append(
            "<tr>"
            f"<td>{row['discovery_stage']:03d}</td>"
            f"<td>{html.escape(str(row.get('parent_semantic_label') or 'UNKNOWN'))}</td>"
            f"<td>{_status_badge(row['geometry_only_status'])}</td>"
            f"<td>{_status_badge(row['semantic_only_status'])}</td>"
            f"<td>{_status_badge(row['joint_status'])}</td>"
            f"<td>{html.escape(str(row.get('rejection_reason') or 'accepted'))}</td>"
            "</tr>"
        )

    mode_explanations = {
        "geometry_only": (
            "Checks planar support, payload fit, and distance to the sofa, but "
            "ignores what the RGB detector says the surface is. The rug therefore "
            "becomes an intentional false positive at stage 0."
        ),
        "semantic_only": (
            "Accepts a detector-supported serving-surface category, but ignores "
            "measured payload fit. It therefore stops incorrectly at the small "
            "side table in stage 1."
        ),
        "joint": (
            "Requires semantic compatibility AND PLANAR_SUPPORT AND FITS_ON AND "
            "NEAR_SEATING_AREA. It rejects both counterexamples and selects the "
            "coffee table at stage 2."
        ),
    }
    mode_cards = []
    for mode in MODE_TITLES:
        animation = mode_animations[mode]
        video = (
            f"<video controls muted loop playsinline poster='{stage_assets[-1]['overview']}'>"
            f"<source src='{animation['mp4']}' type='video/mp4'></video>"
            if animation.get("mp4")
            else f"<img class='wide' src='{animation['gif']}' alt='{mode} animation'>"
        )
        mode_cards.append(
            f"<article class='mode mode-{mode}'><h3>{MODE_TITLES[mode]}</h3>"
            f"<p>{mode_explanations[mode]}</p>{video}"
            f"<details><summary>Open GIF and outcome data</summary><div class='inside'>"
            f"<img class='wide' src='{animation['gif']}' alt='{mode} GIF'>"
            f"<p>Selected <code>{summary['modes'][mode].get('selected_region_id')}</code> "
            f"at stage {_fmt(summary['modes'][mode].get('completion_stage'))}.</p>"
            "</div></details></article>"
        )

    registry_regions = region_registry.get("regions", {})
    measurement_rows = []
    for row in rows:
        region = registry_regions.get(row["region_id"], {})
        geometry = region.get("geometric_properties", {})
        provenance = region.get("provenance", {})
        relations = region.get("functional_evaluations", {})
        near = relations.get("NEAR_SEATING_AREA", {})
        measurement_rows.append(
            "<tr>"
            f"<td><code>{row['region_id']}</code></td>"
            f"<td>{html.escape(str(row.get('parent_semantic_label') or 'UNKNOWN'))}</td>"
            f"<td>{row['semantic_confidence']:.3f} / {row['semantic_supporting_views']}</td>"
            f"<td>{row['support_length_m']:.3f} × {row['support_width_m']:.3f}</td>"
            f"<td>{_fmt(_value(geometry, 'support_area_m2'))}</td>"
            f"<td>{_fmt(_value(geometry, 'planarity_score'))}</td>"
            f"<td>{_fmt(provenance.get('point_count'))} / "
            f"{len(provenance.get('contributing_camera_ids', []))}</td>"
            f"<td>{_fmt(near.get('measured_distance_m'))}</td>"
            f"<td>{row['fit_margin_m']:+.3f}</td>"
            f"<td><code>{html.escape(str(provenance.get('measurement_cloud_path', 'UNKNOWN')))}</code></td>"
            "</tr>"
        )

    payloads = payload_registry.get("objects", {})
    payload = next(iter(payloads.values()), {})
    payload_geometry = payload.get("geometric_properties", {})
    payload_quality = payload_geometry.get("measurement_quality", {})
    payload_html = (
        "<table><thead><tr><th>ID</th><th>RGB label</th><th>footprint L×W</th>"
        "<th>area</th><th>points / cameras</th><th>MeasurementEvidence</th></tr></thead><tbody><tr>"
        f"<td><code>{html.escape(str(payload.get('identity', {}).get('object_id', 'UNKNOWN')))}</code></td>"
        f"<td>{html.escape(str(payload.get('semantic_context', {}).get('canonical_label', 'UNKNOWN')))}</td>"
        f"<td>{_fmt(_value(payload_geometry, 'footprint_length_m'))} × "
        f"{_fmt(_value(payload_geometry, 'footprint_width_m'))} m</td>"
        f"<td>{_fmt(_value(payload_geometry, 'footprint_area_m2'))} m²</td>"
        f"<td>{_fmt(payload_quality.get('point_count'))} / "
        f"{_fmt(payload_quality.get('contributing_camera_count'))}</td>"
        f"<td><code>{html.escape(str(payload.get('provenance', {}).get('measurement_cloud_path', 'UNKNOWN')))}</code></td>"
        "</tr></tbody></table>"
    )

    handoff_html = "<p>No verified handoff was produced.</p>"
    if handoff:
        handoff_html = (
            "<div class='callout success'><b>Verified destination:</b> "
            f"<code>{html.escape(str(handoff.get('selected_region_id')))}</code> "
            f"at stage {handoff.get('completion_stage')} "
            f"({html.escape(str(handoff.get('completion_inspection_label')))})</div>"
            "<table><thead><tr><th>Relation</th><th>Inputs</th><th>margin</th><th>result</th></tr></thead><tbody>"
            f"<tr><td>FITS_ON</td><td>payload {_fmt(handoff['FITS_ON'].get('payload_length_m'))} × "
            f"{_fmt(handoff['FITS_ON'].get('payload_width_m'))} m; support "
            f"{_fmt(handoff['FITS_ON'].get('region_usable_length_m'))} × "
            f"{_fmt(handoff['FITS_ON'].get('region_usable_width_m'))} m</td>"
            f"<td>{_fmt(handoff['FITS_ON'].get('signed_fit_margin_m'))} m</td>"
            f"<td>{_status_badge(handoff['FITS_ON'].get('status'))}</td></tr>"
            f"<tr><td>NEAR_SEATING_AREA</td><td>distance "
            f"{_fmt(handoff['NEAR_SEATING_AREA'].get('measured_distance_m'))} m; maximum "
            f"{_fmt(handoff['NEAR_SEATING_AREA'].get('maximum_distance_m'))} m</td>"
            f"<td>{_fmt(handoff['NEAR_SEATING_AREA'].get('signed_margin_m'))} m</td>"
            f"<td>{_status_badge(handoff['NEAR_SEATING_AREA'].get('status'))}</td></tr>"
            "</tbody></table>"
        )

    stage_html = []
    for index, asset in enumerate(stage_assets):
        cameras = "".join(
            f"<figure><img class='wide' src='{camera['consensus']}' "
            f"alt='{camera['camera_id']} detector overlay'><figcaption>"
            f"{html.escape(camera['camera_id'])} · "
            f"<a href='{camera['raw']}'>raw detector boxes</a>"
            "</figcaption></figure>"
            for camera in asset["camera_overlays"]
        )
        stage_html.append(
            f"<details {'open' if index == 0 else ''}><summary>{html.escape(asset['title'])}</summary>"
            "<div class='inside'>"
            f"<img class='wide' src='{asset['overview']}' alt='complete stage overview'>"
            "<div class='component-grid'>"
            f"<figure><figcaption>YOLO-World semantic overview</figcaption><img class='wide' src='{asset['semantic']}'></figure>"
            f"<figure><figcaption>Fresh stage-local point cloud</figcaption><img class='wide' src='{asset['pointcloud']}'></figure>"
            f"<figure><figcaption>Observed graph through this stage</figcaption><img class='wide' src='{asset['graph']}'></figure>"
            f"<figure><figcaption>Front-view region/payload masks</figcaption><img class='wide' src='{asset['mask']}'></figure>"
            "</div><details><summary>Five RGB detector and association overlays</summary>"
            f"<div class='inside camera-grid'>{cameras}</div></details>"
            f"<p class='downloads'><a href='{asset['semantic']}'>semantic PNG</a> · "
            f"<a href='{asset['pointcloud']}'>point-cloud PNG</a> · "
            f"<a href='{asset['graph']}'>graph PNG</a></p>"
            "</div></details>"
        )

    detector = run_config.get("detector", {})
    css = """*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#eef2f7;color:#111827;font-family:Inter,system-ui,-apple-system,Segoe UI,sans-serif;line-height:1.55}nav{position:sticky;top:0;z-index:10;background:#0f172a;padding:12px 24px;box-shadow:0 3px 12px #0004}nav a{color:#f8fafc;text-decoration:none;font-weight:800;margin-right:22px}main{max-width:1560px;margin:auto;padding:24px}.hero{background:linear-gradient(125deg,#0f172a,#253b67);color:white}.card,.mode,section{background:white;border-radius:17px;padding:27px;margin:20px 0;box-shadow:0 4px 18px #0f172a14}h1{font-size:43px;line-height:1.12;margin:.25em 0}.lede{font-size:20px;color:#dbeafe;max-width:1050px}.pill{display:inline-block;padding:6px 12px;background:#dcfce7;color:#166534;border-radius:999px;font-weight:800;margin:5px}.wide{display:block;width:100%;height:auto;border:1px solid #dbe3ef;border-radius:11px;background:white}video{display:block;width:100%;max-height:760px;border-radius:11px;background:#111}table{width:100%;border-collapse:collapse;font-size:13px}th,td{border:1px solid #dbe3ef;padding:9px;text-align:left;vertical-align:top}th{background:#e9eef5;position:sticky;top:47px;z-index:2}.scroll{overflow:auto}.two{display:grid;grid-template-columns:1fr 1fr;gap:18px}.modes{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}.mode{margin:0;border-top:8px solid #64748b}.mode-geometry_only{border-color:#7c3aed}.mode-semantic_only{border-color:#ea580c}.mode-joint{border-color:#16a34a}details{border:1px solid #dbe3ef;border-radius:11px;margin:14px 0;overflow:hidden}summary{padding:14px 16px;background:#f8fafc;font-weight:850;cursor:pointer}.inside{padding:16px}.component-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px}.camera-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}figure{margin:0}figcaption{font-weight:800;margin:0 0 7px}.status{display:inline-block;color:white;border-radius:999px;padding:3px 9px;font-weight:900;font-size:11px}.status-true{background:#239456}.status-false{background:#cf3e3e}.status-unknown{background:#7c8593}.callout{padding:16px;border-radius:10px;margin:12px 0}.success{background:#dcfce7;color:#14532d;border-left:6px solid #16a34a}code{background:#e8edf4;padding:2px 5px;border-radius:4px;overflow-wrap:anywhere}.pipeline{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.pipeline span{padding:10px 13px;background:#e9eef5;border-radius:9px;font-weight:800}.pipeline b{color:#64748b}.downloads a{font-weight:750}@media(max-width:1100px){.modes{grid-template-columns:1fr}.two,.component-grid{grid-template-columns:1fr}}@media(max-width:760px){nav{position:static}nav a{display:inline-block;margin:4px 12px 4px 0}main{padding:10px}section,.card,.mode{padding:17px}h1{font-size:32px}.camera-grid{grid-template-columns:1fr}}"""
    document = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Living-room Region Ablation 1</title><style>{css}</style></head><body>
<nav><a href="#summary">Summary</a><a href="#ablations">Ablations</a><a href="#matrix">Matrix</a><a href="#measurements">Measurements</a><a href="#stages">Scene audit</a><a href="#provenance">Provenance</a></nav><main>
<section id="summary" class="hero"><small>PRESENTATION REPORT · LIVING-ROOM REGION ABLATION 1</small><h1>Semantic–geometric destination-region grounding</h1><p class="lede"><b>Goal:</b> {html.escape(run_config['natural_language_goal'])}</p><span class="pill">Scene: {html.escape(run_config['scene_name'])}</span><span class="pill">3 observed regions</span><span class="pill">5 RGB-D cameras per stage</span><span class="pill">Same evidence for all modes</span><span class="pill">No robot / FM / TAMP</span><img class="wide" src="region_ablation_comparison.png" alt="same-evidence ablation comparison"></section>
<section><h2>What was run</h2><div class="pipeline"><span>Rendered RGB-D + masks</span><b>→</b><span>five-view world cloud</span><b>→</b><span>stage-local region evidence</span><b>→</b><span>YOLO-World semantics</span><b>→</b><span>geometry relations</span><b>→</b><span>joint verified handoff</span></div><p>The benchmark searches for a destination surface for one measured refreshment tray. The three acceptance modes reuse the exact same saved observations; only their acceptance logic changes.</p></section>
<section><h2>Headline outcomes</h2><div class="scroll"><table><thead><tr><th>Mode</th><th>Selected region</th><th>Stage</th><th>Interpretation</th><th>Production-valid?</th></tr></thead><tbody>{''.join(outcomes)}</tbody></table></div><h3>Stage-by-stage decisions</h3><div class="scroll"><table><thead><tr><th>Stage</th><th>RGB parent</th><th>Geometry-only</th><th>Semantic-only</th><th>Joint</th><th>Joint reason</th></tr></thead><tbody>{''.join(progression_rows)}</tbody></table></div></section>
<section id="ablations"><h2>Three individual policy ablations</h2><div class="modes">{''.join(mode_cards)}</div></section>
<section id="matrix"><h2>Compatibility matrix</h2><p>Each row is a generic persistent region. Green means the saved evidence supports that gate; red means it refutes it. Joint acceptance requires every required column to be TRUE.</p><img class="wide" src="region_compatibility_matrix.png" alt="region compatibility matrix"><div class="scroll"><table><thead><tr><th>ID</th><th>RGB parent</th><th>confidence</th><th>views</th><th>support L×W (m)</th><th>fit margin (m)</th><th>semantics</th><th>planar</th><th>fits</th><th>near</th><th>joint</th><th>rejection</th></tr></thead><tbody>{_table_html(rows)}</tbody></table></div></section>
<section id="measurements"><h2>Payload MeasurementEvidence</h2>{payload_html}<h2>Region MeasurementEvidence</h2><div class="scroll"><table><thead><tr><th>ID</th><th>RGB label</th><th>confidence / views</th><th>support L×W (m)</th><th>area (m²)</th><th>planarity</th><th>points / cameras</th><th>sofa distance (m)</th><th>fit margin (m)</th><th>stage-local evidence path</th></tr></thead><tbody>{''.join(measurement_rows)}</tbody></table></div></section>
<section><h2>How the geometric checks are defined</h2><div class="two"><div><h3>PLANAR_SUPPORT(region)</h3><p>Uses only the fresh region point cloud. A dominant horizontal plane must have enough support, sufficient planarity, acceptable normal alignment to gravity, and valid multi-view coverage.</p><h3>FITS_ON(payload, region)</h3><p>The measured payload footprint plus edge clearance is tested against the measured usable support extents at 0° and 90°. The minimum signed dimension margin determines TRUE/FALSE.</p></div><div><h3>NEAR_SEATING_AREA(region, sofa)</h3><p>The measured region centroid is compared with the sofa context reconstructed from current visible evidence. It passes when the measured distance is within the configured maximum.</p><h3>Joint gate</h3><p><code>semantic compatibility AND PLANAR_SUPPORT AND FITS_ON AND NEAR_SEATING_AREA</code>. Required UNKNOWN or FALSE evidence cannot complete the task.</p></div></div></section>
<section><h2>Verified production handoff</h2>{handoff_html}<img class="wide" src="region_assignment_graph.png" alt="final observed region-function graph"></section>
<section><h2>Full progression animation</h2><video controls muted loop playsinline><source src="region_progression.mp4" type="video/mp4"></video><details><summary>Open the GIF version</summary><div class="inside"><img class="wide" src="region_progression.gif" alt="complete region progression GIF"></div></details></section>
<section id="stages"><h2>Rendered scene and component audit</h2><p>Expand every stage to inspect the scene, five RGB detections, association overlays, stage-local point cloud, region mask, and graph. These are the actual artifacts used by the report.</p>{''.join(stage_html)}</section>
<section id="provenance"><h2>Provenance and evidence boundary</h2><p><b>Detector:</b> {html.escape(str(detector.get('name', 'UNKNOWN')))}; <b>checkpoint:</b> {html.escape(str(detector.get('checkpoint', 'UNKNOWN')))}; <b>version:</b> {html.escape(str(detector.get('version', 'UNKNOWN')))}; <b>device:</b> {html.escape(str(detector.get('device', 'UNKNOWN')))}; <b>capture:</b> {html.escape(str(run_config.get('capture_resolution')))}.</p><p>Geometry consumed typed, fresh <code>RegionMeasurementEvidence</code> and <code>PayloadMeasurementEvidence</code> only. Cumulative visualization clouds, complete-room combined clouds, configured geom sizes, hidden simulator geometry, and detector labels are not geometric measurement inputs. RGB semantics come from rendered pixels and are associated through visible masks. No placement, robot motion, navigation, FM, LLM, VLM, planning, or TAMP execution occurs.</p><p class="downloads"><a href="offline_region_ablation_evaluation.json">offline evaluation JSON</a> · <a href="region_compatibility_matrix.json">matrix JSON</a> · <a href="region_compatibility_matrix.csv">matrix CSV</a> · <a href="region_registry.json">region registry</a> · <a href="payload_registry.json">payload registry</a> · <a href="verified_region_handoff.json">verified handoff</a> · <a href="events.jsonl">events</a></p></section>
</main></body></html>"""
    (report_dir / "presentation_report.html").write_text(
        document, encoding="utf-8"
    )
    (report_dir / "ablation_report.html").write_text(
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<meta http-equiv=\"refresh\" "
        "content=\"0; url=presentation_report.html\">"
        "<title>Living-room ablation presentation</title></head><body>"
        "<p>Open <a href=\"presentation_report.html\">"
        "presentation_report.html</a>.</p></body></html>\n",
        encoding="utf-8",
    )


def generate_report(run_dir: Path, report_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    report_dir = report_dir.resolve()
    if report_dir.exists():
        raise RuntimeError(f"Report directory already exists: {report_dir}")
    report_dir.mkdir(parents=True)
    run_config = _load(run_dir / "run_config.json")
    matrix = _load(run_dir / "region_compatibility_matrix.json")
    rows = matrix["rows"]
    summary = _load(run_dir / "offline_region_ablation_evaluation.json")
    region_registry = _load(run_dir / "region_registry.json")
    payload_registry = _load(run_dir / "payload_registry.json")
    handoff_path = run_dir / "verified_region_handoff.json"
    handoff = _load(handoff_path) if handoff_path.exists() else None
    _copy_json_artifacts(run_dir, report_dir)
    _render_matrix(rows, report_dir / "region_compatibility_matrix.png")
    _render_ablation(
        summary, rows, report_dir / "region_ablation_comparison.png"
    )
    _render_graph(rows, report_dir / "region_assignment_graph.png")
    stages_output = report_dir / "stages"
    stages_output.mkdir()
    frames_dir = report_dir / "frames"
    frames_dir.mkdir()
    stage_assets = []
    frame_paths = []
    for row, stage_dir in zip(rows, sorted((run_dir / "stages").iterdir())):
        stage_target = stages_output / stage_dir.name
        stage_target.mkdir()
        pointcloud = stage_target / "region_pointcloud.png"
        graph = stage_target / "graph.png"
        overview = stage_target / "overview.png"
        _render_point_cloud(stage_dir, pointcloud)
        _render_graph(
            rows,
            graph,
            maximum_stage=int(row["discovery_stage"]),
        )
        semantic_target = stage_target / "semantic_overview.png"
        mask_target = stage_target / "evidence_masks.png"
        shutil.copy2(stage_dir / "semantic_overview.png", semantic_target)
        shutil.copy2(
            stage_dir
            / "cameras"
            / "inspection_front"
            / "evidence_masks.png",
            mask_target,
        )
        cameras_target = stage_target / "cameras"
        cameras_target.mkdir()
        camera_overlays = []
        semantic_cameras = stage_dir / "semantics" / "cameras"
        for camera_dir in sorted(semantic_cameras.iterdir()):
            camera_target = cameras_target / camera_dir.name
            camera_target.mkdir()
            consensus_target = camera_target / "consensus_overlay.png"
            raw_target = camera_target / "overlay.png"
            shutil.copy2(
                camera_dir / "consensus_overlay.png", consensus_target
            )
            shutil.copy2(camera_dir / "overlay.png", raw_target)
            camera_overlays.append(
                {
                    "camera_id": camera_dir.name,
                    "consensus": consensus_target.relative_to(
                        report_dir
                    ).as_posix(),
                    "raw": raw_target.relative_to(report_dir).as_posix(),
                }
            )
        _stage_overview(
            stage_dir=stage_dir,
            stage_output=overview,
            row=row,
            graph_path=graph,
            pointcloud_path=pointcloud,
        )
        frame_path = frames_dir / f"frame_{int(row['discovery_stage']):03d}.png"
        shutil.copy2(overview, frame_path)
        frame_paths.append(frame_path)
        stage_assets.append(
            {
                "title": (
                    f"Stage {row['discovery_stage']:03d}: "
                    f"{row.get('parent_semantic_label') or 'UNKNOWN'}"
                ),
                "overview": overview.relative_to(report_dir).as_posix(),
                "semantic": semantic_target.relative_to(report_dir).as_posix(),
                "pointcloud": pointcloud.relative_to(report_dir).as_posix(),
                "graph": graph.relative_to(report_dir).as_posix(),
                "mask": mask_target.relative_to(report_dir).as_posix(),
                "camera_overlays": camera_overlays,
            }
        )
    animation = _make_progression(
        frame_paths,
        report_dir / "region_progression.gif",
        report_dir / "region_progression.mp4",
    )
    mode_animations: dict[str, dict[str, Any]] = {}
    for mode in MODE_TITLES:
        mode_frames_dir = report_dir / f".{mode}_frames"
        mode_frames_dir.mkdir()
        mode_frames = []
        for row, stage_asset in zip(rows, stage_assets):
            frame = mode_frames_dir / (
                f"frame_{int(row['discovery_stage']):03d}.png"
            )
            _mode_frame(
                report_dir / stage_asset["overview"],
                frame,
                mode=mode,
                row=row,
                result=summary["modes"][mode],
            )
            mode_frames.append(frame)
        mode_animation = _make_progression(
            mode_frames,
            report_dir / f"{mode}_progression.gif",
            report_dir / f"{mode}_progression.mp4",
        )
        mode_animations[mode] = {
            **mode_animation,
            "gif": Path(mode_animation["gif"]).relative_to(
                report_dir
            ).as_posix(),
            "mp4": (
                Path(mode_animation["mp4"])
                .relative_to(report_dir)
                .as_posix()
                if mode_animation["mp4"]
                else None
            ),
        }
        shutil.rmtree(mode_frames_dir)
    shutil.rmtree(frames_dir)
    report_data = {
        "run_directory": str(run_dir),
        "report_directory": str(report_dir),
        "run_config": run_config,
        "ablation_summary": summary,
        "compatibility_matrix": matrix,
        "stage_assets": stage_assets,
        "animation": animation,
        "mode_animations": mode_animations,
    }
    (report_dir / "report_data.json").write_text(
        json.dumps(report_data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_html(
        report_dir=report_dir,
        run_config=run_config,
        summary=summary,
        rows=rows,
        stage_assets=stage_assets,
        region_registry=region_registry,
        payload_registry=payload_registry,
        handoff=handoff,
        mode_animations=mode_animations,
    )
    numeric_rows = "\n".join(
        "| {region_id} | {semantic} | {length:.3f} | {width:.3f} | "
        "{margin:+.3f} | {semantic_status} | {fits} | {joint} |".format(
            region_id=row["region_id"],
            semantic=row.get("parent_semantic_label") or "UNKNOWN",
            length=row["support_length_m"],
            width=row["support_width_m"],
            margin=row["fit_margin_m"],
            semantic_status=row["semantic_role_status"],
            fits=row["FITS_ON"],
            joint=row["joint_status"],
        )
        for row in rows
    )
    readme = f"""# L2 living-room Region Ablation 1

Goal: **{run_config['natural_language_goal']}**

Kitchen grounding searches for functional objects. This report searches for a
functional destination region. A large planar patch is not automatically a
valid serving region, and a semantically suitable table is not automatically
large enough for the measured payload.

| Mode | Primary selected region | Stage | Correct? |
|---|---|---:|---|
| Geometry-only | `{summary['modes']['geometry_only']['selected_region_id']}` (rug) | {summary['modes']['geometry_only']['completion_stage']} | No |
| Semantic-only | `{summary['modes']['semantic_only']['selected_region_id']}` (small side table) | {summary['modes']['semantic_only']['completion_stage']} | No |
| Joint | `{summary['modes']['joint']['selected_region_id']}` (coffee table) | {summary['modes']['joint']['completion_stage']} | Yes |

| Region | YOLO parent | support L (m) | support W (m) | fit margin (m) | semantic role | FITS_ON | joint |
|---|---|---:|---:|---:|---|---|---|
{numeric_rows}

The rug is the geometry-only false positive. The undersized side table is the
semantic-only false positive. The coffee table is the joint solution. The
manually supplied future-FM-style ranking proposes inspection order; it does
not prove suitability or override a failed verifier.

Every mode reuses the SHA-256-identified RGB, depth, segmentation, region
masks, detector outputs, semantic associations, region clouds, payload cloud,
and sofa evidence listed in `offline_region_ablation_evaluation.json`.

Open `presentation_report.html` for the complete visual report.

Function requirements and ranking are manually configured. No FM, LLM, VLM,
planning, placement, robot execution, or TAMP execution occurs. The successful
output is only a verified destination-region handoff.
"""
    (report_dir / "README.md").write_text(readme, encoding="utf-8")
    return report_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--report-dir", type=Path, required=True)
    arguments = parser.parse_args()
    report = generate_report(arguments.run_dir, arguments.report_dir)
    print(
        json.dumps(
            {
                "report_directory": report["report_directory"],
                "html": str(
                    Path(report["report_directory"])
                    / "presentation_report.html"
                ),
                "gif": report["animation"]["gif"],
                "mp4": report["animation"]["mp4"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
