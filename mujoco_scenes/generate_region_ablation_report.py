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


def _write_html(
    *,
    report_dir: Path,
    run_config: dict[str, Any],
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    stage_assets: list[dict[str, str]],
) -> None:
    outcomes = []
    for mode, title in MODE_TITLES.items():
        result = summary["modes"][mode]
        selected = result["selected_region_id"] or "none"
        correct = mode == "joint" and result["status"] == "COMPLETE"
        outcomes.append(
            "<tr>"
            f"<td>{title}</td><td>{selected}</td>"
            f"<td>{result['completion_stage']}</td>"
            f"<td>{'Yes' if correct else 'No — diagnostic false result'}</td>"
            "</tr>"
        )
    stage_html = []
    for asset in stage_assets:
        stage_html.append(
            "<section class='card'>"
            f"<h3>{html.escape(asset['title'])}</h3>"
            f"<img src='{asset['overview']}' alt='stage overview'>"
            "<div class='links'>"
            f"<a href='{asset['semantic']}'>semantic overlay</a>"
            f"<a href='{asset['pointcloud']}'>point cloud</a>"
            f"<a href='{asset['graph']}'>graph</a>"
            f"<a href='{asset['mask']}'>region mask</a>"
            "</div></section>"
        )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>L2 Region Ablation 1</title>
<style>
body{{margin:0;background:#eef1f5;color:#192132;font:16px/1.5 Arial,sans-serif}}
main{{max-width:1500px;margin:auto;padding:30px}} .hero{{background:#17233b;color:white;
padding:34px;border-radius:18px}} .card{{background:white;margin:22px 0;padding:26px;
border-radius:16px;box-shadow:0 4px 18px #17233b18}} img{{max-width:100%;
height:auto;border:1px solid #dbe0e8;border-radius:10px}} table{{width:100%;
border-collapse:collapse;font-size:14px}} th,td{{padding:10px;border-bottom:1px solid #ddd;
text-align:left}} th{{background:#edf1f7;position:sticky;top:0}} .grid{{display:grid;
grid-template-columns:1fr 1fr;gap:18px}} .links a{{display:inline-block;margin:10px 12px 0 0}}
code{{background:#eef1f5;padding:2px 5px;border-radius:4px}}
@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<section class="hero"><h1>Living-room Region Ablation 1</h1>
<p><strong>{html.escape(run_config['natural_language_goal'])}</strong></p>
<p>This benchmark grounds a functional destination region. It does not execute
placement, planning, TAMP, an FM, LLM, or VLM.</p></section>
<section class="card"><h2>What the ablation demonstrates</h2>
<ol><li>A large planar rug passes geometry but is semantically inappropriate.</li>
<li>A side table is semantically suitable but too small for the measured tray.</li>
<li>Only the coffee table passes semantics, planar support, tray fit, and sofa context.</li>
<li>Candidate ranking proposes an inspection order; it never overrides verification.</li>
<li>All three modes read exactly the same saved RGB-D, detections, masks, and clouds.</li></ol>
<table><thead><tr><th>Mode</th><th>Selected generic region</th>
<th>Completion stage</th><th>Correct?</th></tr></thead>
<tbody>{''.join(outcomes)}</tbody></table></section>
<section class="card"><h2>Compatibility matrix</h2>
<img src="region_compatibility_matrix.png" alt="compatibility matrix"></section>
<section class="grid"><section class="card"><h2>Ablation progression</h2>
<img src="region_ablation_comparison.png" alt="ablation comparison"></section>
<section class="card"><h2>Observed graph</h2>
<img src="region_assignment_graph.png" alt="assignment graph"></section></section>
<section class="card"><h2>Measured evidence table</h2><div style="overflow:auto">
<table><thead><tr><th>ID</th><th>RGB parent</th><th>confidence</th><th>views</th>
<th>support L×W (m)</th><th>fit margin (m)</th><th>semantics</th><th>planar</th>
<th>fits</th><th>near</th><th>joint</th><th>rejection</th></tr></thead>
<tbody>{_table_html(rows)}</tbody></table></div></section>
<section class="card"><h2>Progression animation</h2>
<video controls loop muted style="max-width:100%" src="region_progression.mp4"></video>
<p><a href="region_progression.gif">Open GIF</a> ·
<a href="offline_region_ablation_evaluation.json">Offline evaluation JSON</a> ·
<a href="region_compatibility_matrix.csv">Matrix CSV</a></p></section>
<h2>Stage and component audit</h2>{''.join(stage_html)}
<section class="card"><h2>Evidence boundary</h2>
<p>Geometry uses typed fresh <code>RegionMeasurementEvidence</code> and
<code>PayloadMeasurementEvidence</code>. Cumulative, full-room, configured-size,
and hidden simulator geometry are rejected as measurement inputs. RGB semantics
come from YOLO-World and are associated through visible projected evidence.</p>
</section></main></body></html>"""
    (report_dir / "presentation_report.html").write_text(
        document, encoding="utf-8"
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
            }
        )
    animation = _make_progression(
        frame_paths,
        report_dir / "region_progression.gif",
        report_dir / "region_progression.mp4",
    )
    shutil.rmtree(frames_dir)
    report_data = {
        "run_directory": str(run_dir),
        "report_directory": str(report_dir),
        "run_config": run_config,
        "ablation_summary": summary,
        "compatibility_matrix": matrix,
        "stage_assets": stage_assets,
        "animation": animation,
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
