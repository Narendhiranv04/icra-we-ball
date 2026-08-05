"""Self-contained presentation report for living-room Region Ablation 2."""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import mimetypes
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mujoco_scenes.geometry_checker import read_ply


POLICY_TITLES = {
    "always_shared": "Always shared",
    "always_distinct": "Always distinct",
    "function_aware": "Function-aware",
}
POLICY_COLORS = {
    "always_shared": "#7c3aed",
    "always_distinct": "#ea580c",
    "function_aware": "#16a34a",
}
STATUS_COLORS = {"TRUE": "#269b59", "FALSE": "#d44747", "UNKNOWN": "#87909d"}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _font(size: int, bold: bool = False):
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(filename, size)
    except OSError:
        return ImageFont.load_default()


def _data_uri(root: Path, relative: str) -> str:
    path = root / relative
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return (
        f"data:{mime};base64,"
        + base64.b64encode(path.read_bytes()).decode("ascii")
    )


def _copy_artifacts(run_dir: Path, report_dir: Path) -> None:
    for name in (
        "run_config.json",
        "task_requirements.json",
        "payload_registry.json",
        "region_registry.json",
        "seating_registry.json",
        "observed_graph.json",
        "events.jsonl",
        "drink_region_compatibility.json",
        "drink_region_compatibility.csv",
        "control_region_compatibility.json",
        "control_region_compatibility.csv",
        "policy_evaluations.json",
        "region_assignments.json",
        "distinct_region_counts.json",
        "offline_region_policy_ablation_evaluation.json",
        "region_ablation2_summary.json",
        "region_ablation2_validation.json",
        "verified_region_allocation_handoff.json",
    ):
        source = run_dir / name
        if source.exists():
            shutil.copy2(source, report_dir / name)
    for name in (
        "initial_scene_overview.png",
        "semantic_overview.png",
        "region_masks_overview.png",
    ):
        shutil.copy2(run_dir / "observation" / name, report_dir / name)
    cameras = report_dir / "cameras"
    cameras.mkdir()
    for source in sorted(
        (run_dir / "observation" / "semantics" / "cameras").iterdir()
    ):
        target = cameras / source.name
        target.mkdir()
        shutil.copy2(
            source / "association_overlay.png",
            target / "association_overlay.png",
        )
        shutil.copy2(
            run_dir
            / "observation"
            / "cameras"
            / source.name
            / "evidence_masks.png",
            target / "evidence_masks.png",
        )


def _render_measurements(
    run_dir: Path,
    registry: dict[str, Any],
    *,
    kind: str,
    output: Path,
) -> None:
    records = registry["objects" if kind == "payload" else "regions"]
    columns = min(3, len(records))
    rows = int(np.ceil(len(records) / columns))
    fig = plt.figure(figsize=(5.2 * columns, 4.3 * rows))
    for index, (entity_id, record) in enumerate(sorted(records.items()), 1):
        axis = fig.add_subplot(rows, columns, index, projection="3d")
        evidence = (
            record["provenance"]["measurement_cloud_path"]
            if kind == "payload"
            else record["provenance"]["measurement_cloud_path"]
        )
        points, colors = read_ply(run_dir / evidence)
        if len(points) > 12000:
            selected = np.linspace(0, len(points) - 1, 12000).astype(int)
            points, colors = points[selected], colors[selected]
        axis.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=colors / 255.0,
            s=1.4,
        )
        geometry = record["geometry"]
        if kind == "payload":
            dimensions = (
                geometry["footprint_length_m"]["value"],
                geometry["footprint_width_m"]["value"],
            )
            label = record["semantics"].get("canonical_label")
        else:
            dimensions = (
                geometry["support_length_m"]["value"],
                geometry["support_width_m"]["value"],
            )
            label = record["semantics"].get("canonical_label")
        axis.set_title(
            f"{entity_id} · {label or 'UNKNOWN'}\n"
            f"{dimensions[0]:.3f} × {dimensions[1]:.3f} m · "
            f"{record['provenance']['point_count']:,} points",
            fontsize=10,
            fontweight="bold",
        )
        axis.view_init(elev=30, azim=-55)
        axis.set_box_aspect((1.2, 1.0, 0.35))
        axis.tick_params(labelsize=7)
    fig.suptitle(
        f"Measured {kind} point clouds · one initial observation",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _render_drink_matrix(rows: list[dict[str, Any]], output: Path) -> None:
    slots = sorted({row["slot_id"] for row in rows})
    regions = sorted({row["region_id"] for row in rows})
    fig, axes = plt.subplots(
        1, len(slots), figsize=(7 * len(slots), 4.8), squeeze=False
    )
    for axis, slot in zip(axes[0], slots):
        selected = {row["region_id"]: row for row in rows if row["slot_id"] == slot}
        for index, region_id in enumerate(regions):
            row = selected[region_id]
            status = row["compatibility_status"]
            axis.add_patch(
                plt.Rectangle(
                    (0, len(regions) - index - 1),
                    1,
                    1,
                    color=STATUS_COLORS[status],
                    ec="white",
                    lw=2,
                )
            )
            axis.text(
                0.5,
                len(regions) - index - 0.5,
                f"{status}\nfit {row['fit_margin_m']:+.3f} m\n"
                f"seat {row['near_seat_margin_m']:+.3f} m",
                ha="center",
                va="center",
                color="white",
                fontsize=9,
                fontweight="bold",
            )
        first = next(row for row in rows if row["slot_id"] == slot)
        axis.set_title(
            f"{slot}\n{first['payload_id']} → "
            f"{first['seating_target_id']}",
            fontweight="bold",
        )
        axis.set_xlim(0, 1)
        axis.set_ylim(0, len(regions))
        axis.set_xticks([])
        axis.set_yticks(
            np.arange(len(regions)) + 0.5, list(reversed(regions))
        )
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
    fig.suptitle(
        "Personal drink target-specific compatibility",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _render_control_matrix(rows: list[dict[str, Any]], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(11, max(4.5, len(rows) * 0.9)))
    columns = [
        ("semantic_role_status", "control\nsemantics"),
        ("PLANAR_SUPPORT", "planar"),
        ("FITS_SET_ON", "set fits"),
        ("ACCESSIBLE_FROM_VIEWING_AREA", "accessible"),
        ("compatibility_status", "final"),
    ]
    for row_index, row in enumerate(rows):
        y = len(rows) - row_index - 1
        for column_index, (key, _label) in enumerate(columns):
            status = row[key]
            axis.add_patch(
                plt.Rectangle(
                    (column_index, y),
                    1,
                    1,
                    color=STATUS_COLORS[status],
                    ec="white",
                    lw=2,
                )
            )
            text = status
            if key == "FITS_SET_ON" and row["packing_margin_m"] is not None:
                text += f"\n{row['packing_margin_m']:+.3f} m"
            if (
                key == "ACCESSIBLE_FROM_VIEWING_AREA"
                and row["accessibility_margin_m"] is not None
            ):
                text += f"\n{row['accessibility_margin_m']:+.3f} m"
            axis.text(
                column_index + 0.5,
                y + 0.5,
                text,
                ha="center",
                va="center",
                color="white",
                fontsize=9,
                fontweight="bold",
            )
    axis.set_xticks(
        np.arange(len(columns)) + 0.5,
        [label for _key, label in columns],
    )
    axis.set_yticks(
        np.arange(len(rows)) + 0.5,
        [
            f"{row['region_id']} · {row['region_semantic_label']}"
            for row in reversed(rows)
        ],
    )
    axis.set_xlim(0, len(columns))
    axis.set_ylim(0, len(rows))
    axis.tick_params(length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_title(
        "Shared controls set-packing compatibility",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _render_policy_graph(
    policy: str,
    result: dict[str, Any],
    payloads: dict[str, Any],
    seats: dict[str, Any],
    regions: dict[str, Any],
    output: Path,
) -> None:
    graph = nx.DiGraph()
    assignments = result.get("assignments", [])
    selected_payloads = {item["payload_id"] for item in assignments}
    selected_regions = {item["region_id"] for item in assignments}
    selected_seats = {
        item["target_id"]
        for item in assignments
        if item["target_id"] in seats
    }
    for object_id in selected_payloads or payloads:
        label = payloads[object_id]["semantics"].get("canonical_label")
        graph.add_node(
            f"p:{object_id}",
            label=f"{object_id}\n{label or 'UNKNOWN'}",
            kind="payload",
        )
    for region_id in selected_regions:
        label = regions[region_id]["semantics"].get("canonical_label")
        graph.add_node(
            f"r:{region_id}",
            label=f"{region_id}\n{label or 'UNKNOWN'}",
            kind="region",
        )
    for seat_id in selected_seats:
        graph.add_node(
            f"s:{seat_id}",
            label=seat_id,
            kind="seat",
        )
    for assignment in assignments:
        graph.add_edge(
            f"p:{assignment['payload_id']}",
            f"r:{assignment['region_id']}",
            label=assignment["function_group"],
            color=(
                "#2878b8"
                if assignment["function_group"] == "personal_drinks"
                else "#7c3aed"
            ),
        )
        if assignment["target_id"] in seats:
            graph.add_edge(
                f"r:{assignment['region_id']}",
                f"s:{assignment['target_id']}",
                label="NEAR_SEAT",
                color="#2a9d61",
            )
    fig, axis = plt.subplots(figsize=(11, 6.5))
    if graph.nodes:
        positions: dict[str, tuple[float, float]] = {}
        drink_assignments = [
            item
            for item in assignments
            if item["function_group"] == "personal_drinks"
        ]
        control_assignments = [
            item
            for item in assignments
            if item["function_group"] == "shared_controls"
        ]
        for index, item in enumerate(drink_assignments):
            y = 0.80 - 0.23 * index
            positions[f"p:{item['payload_id']}"] = (0.10, y)
            positions[f"r:{item['region_id']}"] = (0.50, y)
            if item["target_id"] in seats:
                positions[f"s:{item['target_id']}"] = (0.90, y)
        for index, item in enumerate(control_assignments):
            y = 0.29 - 0.18 * index
            positions[f"p:{item['payload_id']}"] = (0.10, y)
            positions[f"r:{item['region_id']}"] = (0.58, 0.20)
        unplaced = [node for node in graph.nodes if node not in positions]
        for index, node in enumerate(unplaced):
            positions[node] = (
                0.18 + 0.64 * (index % 2),
                0.72 - 0.42 * (index // 2),
            )
        colors = {
            "payload": "#e67e22",
            "region": "#238b76",
            "seat": "#3575b8",
        }
        nx.draw_networkx_nodes(
            graph,
            positions,
            node_color=[
                colors[graph.nodes[node]["kind"]] for node in graph.nodes
            ],
            node_size=5000,
            edgecolors="white",
            linewidths=2,
            ax=axis,
        )
        nx.draw_networkx_labels(
            graph,
            positions,
            labels={
                node: "\n".join(
                    textwrap.wrap(
                        graph.nodes[node]["label"].replace("\n", " "),
                        width=17,
                    )
                )
                for node in graph.nodes
            },
            font_color="white",
            font_size=7,
            font_weight="bold",
            ax=axis,
        )
        nx.draw_networkx_edges(
            graph,
            positions,
            edge_color=[
                graph.edges[edge]["color"] for edge in graph.edges
            ],
            width=2.5,
            arrows=True,
            arrowsize=18,
            ax=axis,
        )
        nx.draw_networkx_edge_labels(
            graph,
            positions,
            edge_labels={
                edge: graph.edges[edge]["label"] for edge in graph.edges
            },
            font_size=8,
            rotate=False,
            label_pos=0.55,
            ax=axis,
        )
    else:
        axis.text(
            0.5,
            0.55,
            "NO COMPLETE ASSIGNMENT",
            ha="center",
            va="center",
            fontsize=24,
            fontweight="bold",
            color="#c43d3d",
        )
        axis.text(
            0.5,
            0.42,
            "\n".join(result.get("failed_constraints", [])),
            ha="center",
            va="center",
            fontsize=12,
        )
    axis.set_title(
        f"{POLICY_TITLES[policy]} · {result['status']} · "
        f"{result['classification']}\n"
        f"distinct physical regions: "
        f"{result.get('distinct_physical_region_count')}",
        fontsize=15,
        fontweight="bold",
    )
    axis.margins(0.14)
    axis.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _comparison_and_animation(
    report_dir: Path, graph_paths: dict[str, Path]
) -> dict[str, Any]:
    images = [
        Image.open(graph_paths[policy]).convert("RGB") for policy in POLICY_TITLES
    ]
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    canvas = Image.new("RGB", (3 * width, height), "#eef2f7")
    for index, image in enumerate(images):
        copy = image.copy()
        copy.thumbnail((width, height), Image.Resampling.LANCZOS)
        canvas.paste(
            copy,
            (
                index * width + (width - copy.width) // 2,
                (height - copy.height) // 2,
            ),
        )
    comparison = report_dir / "policy_ablation_comparison.png"
    canvas.save(comparison)
    normalized = []
    for policy, image in zip(POLICY_TITLES, images):
        frame = Image.new("RGB", (width, height + 74), "#0f172a")
        frame.paste(
            image,
            ((width - image.width) // 2, 74 + (height - image.height) // 2),
        )
        draw = ImageDraw.Draw(frame)
        draw.text(
            (24, 18),
            f"Same evidence · {POLICY_TITLES[policy]} allocation",
            fill="white",
            font=_font(28, True),
        )
        normalized.append(frame)
    gif = report_dir / "policy_ablation_comparison.gif"
    normalized[0].save(
        gif,
        save_all=True,
        append_images=normalized[1:],
        duration=1900,
        loop=0,
        optimize=True,
    )
    frames = report_dir / ".policy_frames"
    frames.mkdir()
    for index, image in enumerate(normalized):
        image.save(frames / f"frame_{index:03d}.png")
    mp4 = report_dir / "policy_ablation_comparison.mp4"
    completed = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            "1/2",
            "-i",
            str(frames / "frame_%03d.png"),
            "-vf",
            f"scale={width}:-2:flags=lanczos,format=yuv420p",
            "-r",
            "30",
            str(mp4),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    shutil.rmtree(frames)
    return {
        "comparison": comparison.name,
        "gif": gif.name,
        "mp4": mp4.name if completed.returncode == 0 else None,
        "ffmpeg_returncode": completed.returncode,
        "ffmpeg_error": completed.stderr[-1000:] if completed.returncode else None,
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "UNKNOWN"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return html.escape(str(value))


def _status(status: str) -> str:
    return (
        f"<span class='status status-{status.lower()}'>"
        f"{html.escape(status)}</span>"
    )


def _write_html(
    report_dir: Path,
    *,
    run_config: dict[str, Any],
    task: dict[str, Any],
    payloads: dict[str, Any],
    regions: dict[str, Any],
    seats: dict[str, Any],
    drink_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    policies: dict[str, Any],
    animation: dict[str, Any],
) -> None:
    outcome_rows = []
    descriptions = {
        "always_shared": (
            "same region for both drink targets",
            "one shared control region",
            "false positive",
        ),
        "always_distinct": (
            "two distinct drink regions",
            "remote/controller forced apart",
            "false negative",
        ),
        "function_aware": (
            "target-specific distinct regions",
            "one jointly packed shared region",
            "correct",
        ),
    }
    for policy in POLICY_TITLES:
        result = policies[policy]
        drink, controls, interpretation = descriptions[policy]
        count = (
            result["distinct_physical_region_count"]
            if result["distinct_physical_region_count"] is not None
            else "4 needed; no solution"
        )
        outcome_rows.append(
            "<tr>"
            f"<td><b>{POLICY_TITLES[policy]}</b></td>"
            f"<td>{drink}</td><td>{controls}</td><td>{count}</td>"
            f"<td>{result['status']} · {interpretation}</td></tr>"
        )
    payload_rows = []
    for object_id, record in sorted(payloads.items()):
        geometry = record["geometry"]
        semantic = record["semantics"]
        payload_rows.append(
            "<tr>"
            f"<td><code>{object_id}</code></td>"
            f"<td>{semantic.get('canonical_label') or 'UNKNOWN'}</td>"
            f"<td>{_fmt(semantic.get('confidence'))}</td>"
            f"<td>{semantic.get('supporting_view_count')}</td>"
            f"<td>{record.get('semantic_payload_role') or 'UNKNOWN'}</td>"
            f"<td>{_fmt(geometry['footprint_length_m'].get('value'))} × "
            f"{_fmt(geometry['footprint_width_m'].get('value'))}</td>"
            f"<td>{record['provenance']['point_count']:,}</td>"
            f"<td><code>{record['provenance']['measurement_cloud_path']}</code></td>"
            "</tr>"
        )
    region_rows = []
    for region_id, record in sorted(regions.items()):
        geometry = record["geometry"]
        semantic = record["semantics"]
        region_rows.append(
            "<tr>"
            f"<td><code>{region_id}</code></td><td>{record['candidate_rank']}</td>"
            f"<td>{semantic.get('canonical_label') or 'UNKNOWN'}</td>"
            f"<td>{_fmt(semantic.get('confidence'))}</td>"
            f"<td>{semantic.get('supporting_view_count')}</td>"
            f"<td>{_fmt(geometry['support_length_m'].get('value'))} × "
            f"{_fmt(geometry['support_width_m'].get('value'))}</td>"
            f"<td>{_fmt(geometry['planarity_score'].get('value'))}</td>"
            f"<td>{record['provenance']['point_count']:,}</td>"
            f"<td><code>{record['provenance']['measurement_cloud_path']}</code></td>"
            "</tr>"
        )
    assignment_sections = []
    for policy in POLICY_TITLES:
        result = policies[policy]
        assignment_rows = "".join(
            "<tr>"
            f"<td>{item['function_group']}</td>"
            f"<td><code>{item['payload_id']}</code></td>"
            f"<td>{item['target_id']}</td>"
            f"<td><code>{item['region_id']}</code></td>"
            f"<td>{item['signed_margin_m']:+.3f}</td></tr>"
            for item in result.get("assignments", [])
        )
        if not assignment_rows:
            assignment_rows = (
                "<tr><td colspan='5'>No complete assignment: "
                + ", ".join(result.get("failed_constraints", []))
                + "</td></tr>"
            )
        graph_name = (
            "function_aware_assignment_graph.png"
            if policy == "function_aware"
            else "always_shared_assignment_graph.png"
            if policy == "always_shared"
            else "always_distinct_failure_graph.png"
        )
        assignment_sections.append(
            f"<article class='mode' style='--accent:{POLICY_COLORS[policy]}'>"
            f"<h3>{POLICY_TITLES[policy]}</h3>"
            f"<p><b>{result['status']} · {result['classification']}</b></p>"
            f"<img class='wide' src='{_data_uri(report_dir, graph_name)}'>"
            "<div class='scroll'><table><thead><tr><th>Group</th>"
            "<th>Payload</th><th>Target</th><th>Region</th><th>margin m</th>"
            f"</tr></thead><tbody>{assignment_rows}</tbody></table></div></article>"
        )
    drink_table = "".join(
        "<tr>"
        f"<td>{row['slot_id']}</td><td>{row['payload_id']}</td>"
        f"<td>{row['seating_target_id']}</td><td>{row['region_id']}</td>"
        f"<td>{row['region_semantic_label']}</td>"
        f"<td>{_status(row['semantic_role_status'])}</td>"
        f"<td>{_status(row['PLANAR_SUPPORT'])}</td>"
        f"<td>{_status(row['FITS_SET_ON'])} ({_fmt(row['fit_margin_m'])})</td>"
        f"<td>{_status(row['NEAR_SEAT'])} ({_fmt(row['near_seat_margin_m'])})</td>"
        f"<td>{_status(row['compatibility_status'])}</td></tr>"
        for row in drink_rows
    )
    control_table = "".join(
        "<tr>"
        f"<td>{row['region_id']}</td><td>{row['region_semantic_label']}</td>"
        f"<td>{_status(row['semantic_role_status'])}</td>"
        f"<td>{_status(row['PLANAR_SUPPORT'])}</td>"
        f"<td>{_status(row['FITS_SET_ON'])} "
        f"({_fmt(row['packing_margin_m'])})</td>"
        f"<td>{_status(row['ACCESSIBLE_FROM_VIEWING_AREA'])} "
        f"({_fmt(row['accessibility_margin_m'])})</td>"
        f"<td>{_status(row['compatibility_status'])}</td></tr>"
        for row in control_rows
    )
    camera_html = "".join(
        f"<figure><img class='wide' src='{_data_uri(report_dir, f'cameras/{camera_id}/association_overlay.png')}'><figcaption>{camera_id} · <a href='cameras/{camera_id}/association_overlay.png'>PNG</a></figcaption></figure>"
        for camera_id in (
            "inspection_left",
            "inspection_right",
            "inspection_top",
            "inspection_front",
            "inspection_close",
        )
    )
    detector = run_config["detector"]
    css = """*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#eef2f7;color:#111827;font-family:Inter,system-ui,sans-serif;line-height:1.5}nav{position:sticky;top:0;z-index:10;background:#0f172a;padding:12px 22px}nav a{color:white;text-decoration:none;font-weight:800;margin-right:20px}main{max-width:1560px;margin:auto;padding:24px}section,.mode{background:white;border-radius:17px;padding:27px;margin:20px 0;box-shadow:0 4px 18px #0f172a14}.hero{background:linear-gradient(125deg,#0f172a,#2b4272);color:white}.hero .lede{color:#dbeafe}.mode{border-top:8px solid var(--accent)}h1{font-size:43px;line-height:1.12}.lede{font-size:20px}.pill{display:inline-block;background:#dcfce7;color:#166534;padding:6px 11px;border-radius:999px;font-weight:800;margin:4px}.wide{display:block;width:100%;height:auto;border:1px solid #dbe3ef;border-radius:10px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.modes{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.cams{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}figure{margin:0}figcaption{font-weight:800;margin-top:6px}table{width:100%;border-collapse:collapse;font-size:12px}th,td{border:1px solid #dbe3ef;padding:8px;text-align:left;vertical-align:top}th{background:#e9eef5;position:sticky;top:45px}.scroll{overflow:auto}.status{display:inline-block;color:white;padding:2px 7px;border-radius:999px;font-size:10px;font-weight:900}.status-true{background:#269b59}.status-false{background:#d44747}.status-unknown{background:#87909d}details{border:1px solid #dbe3ef;border-radius:10px;margin:12px 0;overflow:hidden}summary{background:#f8fafc;padding:13px;font-weight:850;cursor:pointer}.inside{padding:14px}code{background:#e8edf4;padding:2px 5px;border-radius:4px;overflow-wrap:anywhere}@media(max-width:1100px){.modes,.grid{grid-template-columns:1fr}}@media(max-width:760px){nav{position:static}.cams{grid-template-columns:1fr}main{padding:10px}section,.mode{padding:16px}h1{font-size:31px}}"""
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Living-room Region Ablation 2</title><style>{css}</style></head><body><nav><a href="#summary">Summary</a><a href="#policies">Policies</a><a href="#evidence">Evidence</a><a href="#matrices">Matrices</a><a href="#camera">Cameras</a><a href="#boundary">Boundary</a></nav><main>
<section id="summary" class="hero"><small>PRESENTATION REPORT · LIVING-ROOM REGION ABLATION 2</small><h1>Function-dependent region sharing and physical-region count</h1><p class="lede"><b>Goal:</b> {html.escape(run_config['natural_language_goal'])}</p><span class="pill">One initial observation</span><span class="pill">{len(payloads)} payloads</span><span class="pill">{len(regions)} candidate regions</span><span class="pill">{len(seats)} seating targets</span><span class="pill">Same evidence for 3 policies</span><span class="pill">No FM / robot / TAMP</span><img class="wide" src="{_data_uri(report_dir,'policy_ablation_comparison.png')}"></section>
<section><h2>Scientific result</h2><p>A destination region is not inherently reusable or non-reusable. Sharing and distinctness are constraints of each function group. Personal drink-and-snack sets require two target-specific distinct regions. The remote and handheld game device require one common region that packs both without overlap. Cross-function sharing is disabled. The resulting physical-region count is derived from the assignment, not hard-coded.</p><div class="scroll"><table><thead><tr><th>Policy</th><th>Refreshment-set allocation</th><th>Entertainment allocation</th><th>Regions</th><th>Outcome</th></tr></thead><tbody>{''.join(outcome_rows)}</tbody></table></div><img class="wide" src="{_data_uri(report_dir,'policy_ablation_comparison.gif')}"><p><a href="policy_ablation_comparison.gif">download GIF</a> · <a href="policy_ablation_comparison.mp4">download MP4</a></p></section>
<section id="policies"><h2>Policy-specific allocations from identical evidence</h2><div class="modes">{''.join(assignment_sections)}</div></section>
<section id="evidence"><h2>Measured payload evidence</h2><img class="wide" src="{_data_uri(report_dir,'payload_measurements.png')}"><div class="scroll"><table><thead><tr><th>ID</th><th>RGB label</th><th>confidence</th><th>views</th><th>role</th><th>L×W m</th><th>points</th><th>MeasurementEvidence</th></tr></thead><tbody>{''.join(payload_rows)}</tbody></table></div><h2>Measured candidate-region evidence</h2><img class="wide" src="{_data_uri(report_dir,'region_measurements.png')}"><div class="scroll"><table><thead><tr><th>ID</th><th>rank</th><th>RGB label</th><th>confidence</th><th>views</th><th>L×W m</th><th>planarity</th><th>points</th><th>MeasurementEvidence</th></tr></thead><tbody>{''.join(region_rows)}</tbody></table></div></section>
<section id="matrices"><h2>Personal refreshment set × seating-target matrix</h2><p>Every cell combines RGB region-role semantics, PLANAR_SUPPORT, the assigned mug-and-bowl set's measured FITS_SET_ON relation, and the target-specific NEAR_SEAT relation. Only TRUE cells enter allocation.</p><img class="wide" src="{_data_uri(report_dir,'drink_target_compatibility_matrix.png')}"><div class="scroll"><table><thead><tr><th>slot</th><th>payload set</th><th>seat</th><th>region</th><th>RGB label</th><th>semantics</th><th>planar</th><th>set fits (margin)</th><th>near (margin)</th><th>final</th></tr></thead><tbody>{drink_table}</tbody></table></div><h2>Shared entertainment set-fit matrix</h2><p>FITS_SET_ON tests both payload rotations, both region axes, edge clearance, inter-payload clearance, and non-overlap. Total area alone is insufficient.</p><img class="wide" src="{_data_uri(report_dir,'control_set_compatibility_matrix.png')}"><div class="scroll"><table><thead><tr><th>region</th><th>RGB label</th><th>semantics</th><th>planar</th><th>set fits (margin)</th><th>accessible (margin)</th><th>final</th></tr></thead><tbody>{control_table}</tbody></table></div></section>
<section id="camera"><h2>One initial five-view RGB semantic observation</h2><p>All payloads, seating targets, and candidate regions are visible here. The policies do not rerender, rediscover, or rerun YOLO-World.</p><img class="wide" src="{_data_uri(report_dir,'initial_scene_overview.png')}"><h3>Detector and generic-ID association overlays</h3><div class="cams">{camera_html}</div><h3>Evidence-selection masks</h3><img class="wide" src="{_data_uri(report_dir,'region_masks_overview.png')}"></section>
<section><h2>Household-asset provenance</h2><table><thead><tr><th>role</th><th>dataset / ID</th><th>prepared visual</th></tr></thead><tbody><tr><td>drink A / bowl A</td><td>YCB · 025_mug / 024_bowl</td><td>assets/objects/meshes/ycb</td></tr><tr><td>drink B</td><td>GSO · Cole_Hardware_Mug_Classic_Blue</td><td>gso/living_room_mug</td></tr><tr><td>bowl B</td><td>GSO · Room_Essentials_Bowl_Turquiose</td><td>gso/living_room_snack_bowl</td></tr><tr><td>handheld device</td><td>GSO · BlackBlack_Nintendo_3DSXL</td><td>gso/living_room_game_console</td></tr><tr><td>TV remote / furniture</td><td>project-authored textured meshes</td><td>assets/movie_night</td></tr></tbody></table><p>GSO has no catalogued TV remote. Exact scan URLs, texture paths, scale, collision representation, and hashes are in the object manifest and third-party notice.</p></section>
<section id="boundary"><h2>Provenance and research boundary</h2><p><b>Detector:</b> {html.escape(str(detector.get('name')))}; <b>checkpoint:</b> {html.escape(str(detector.get('checkpoint')))}; <b>version:</b> {html.escape(str(detector.get('version')))}; <b>device:</b> {html.escape(str(detector.get('device')))}; <b>resolution:</b> {html.escape(str(run_config['capture_resolution']))}.</p><p>Geometry consumes typed fresh stage-0 RegionMeasurementEvidence and PayloadMeasurementEvidence only. Region volumes select evidence but never manufacture dimensions. Simulator names, geom sizes, hidden poses, configured expected IDs, cumulative clouds, and full-room combined clouds do not control measurement or allocation. Requirements are manual. No FM call, task ordering, navigation, placement, manipulation, or TAMP execution occurs. The successful artifact is a verified allocation handoff for future TAMP.</p><p><a href="verified_region_allocation_handoff.json">verified handoff</a> · <a href="policy_evaluations.json">policy JSON</a> · <a href="drink_region_compatibility.csv">drink CSV</a> · <a href="control_region_compatibility.csv">controls CSV</a> · <a href="events.jsonl">events</a></p></section>
</main></body></html>"""
    (report_dir / "presentation_report.html").write_text(page, encoding="utf-8")
    (report_dir / "ablation_report.html").write_text(
        "<!doctype html><meta charset='utf-8'><meta http-equiv='refresh' "
        "content='0;url=presentation_report.html'><a "
        "href='presentation_report.html'>Open presentation</a>\n",
        encoding="utf-8",
    )


def generate_report(run_dir: str | Path, report_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    report_dir = Path(report_dir).resolve()
    if report_dir.exists():
        raise RuntimeError(f"Report directory already exists: {report_dir}")
    report_dir.mkdir(parents=True)
    _copy_artifacts(run_dir, report_dir)
    run_config = _load(run_dir / "run_config.json")
    task = _load(run_dir / "task_requirements.json")
    payloads = _load(run_dir / "payload_registry.json")["objects"]
    regions = _load(run_dir / "region_registry.json")["regions"]
    seats = _load(run_dir / "seating_registry.json")["seating_targets"]
    drink_rows = _load(run_dir / "drink_region_compatibility.json")["rows"]
    control_rows = _load(run_dir / "control_region_compatibility.json")["rows"]
    policies = _load(run_dir / "policy_evaluations.json")["policies"]
    _render_measurements(
        run_dir,
        {"objects": payloads},
        kind="payload",
        output=report_dir / "payload_measurements.png",
    )
    _render_measurements(
        run_dir,
        {"regions": regions},
        kind="region",
        output=report_dir / "region_measurements.png",
    )
    _render_drink_matrix(
        drink_rows, report_dir / "drink_target_compatibility_matrix.png"
    )
    _render_control_matrix(
        control_rows, report_dir / "control_set_compatibility_matrix.png"
    )
    graph_paths = {}
    for policy, filename in (
        ("always_shared", "always_shared_assignment_graph.png"),
        ("always_distinct", "always_distinct_failure_graph.png"),
        ("function_aware", "function_aware_assignment_graph.png"),
    ):
        path = report_dir / filename
        _render_policy_graph(
            policy, policies[policy], payloads, seats, regions, path
        )
        graph_paths[policy] = path
    animation = _comparison_and_animation(report_dir, graph_paths)
    _write_html(
        report_dir,
        run_config=run_config,
        task=task,
        payloads=payloads,
        regions=regions,
        seats=seats,
        drink_rows=drink_rows,
        control_rows=control_rows,
        policies=policies,
        animation=animation,
    )
    report_data = {
        "schema_version": 1,
        "run_directory": str(run_dir),
        "report_directory": str(report_dir),
        "animation": animation,
        "policy_results": policies,
        "self_contained_html": True,
    }
    (report_dir / "report_data.json").write_text(
        json.dumps(report_data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    readme = f"""# Living-room Region Ablation 2

Goal: **{run_config['natural_language_goal']}**

All candidate regions, payloads, and seating targets are visible in one
five-view observation. The three policies reuse identical saved evidence.

| Policy | Status | Distinct regions | Classification |
|---|---|---:|---|
| Always shared | {policies['always_shared']['status']} | {policies['always_shared']['distinct_physical_region_count']} | {policies['always_shared']['classification']} |
| Always distinct | {policies['always_distinct']['status']} | {policies['always_distinct']['distinct_physical_region_count']} | {policies['always_distinct']['classification']} |
| Function-aware | {policies['function_aware']['status']} | {policies['function_aware']['distinct_physical_region_count']} | {policies['function_aware']['classification']} |

Open `presentation_report.html`; its displayed PNGs and animated GIF are
embedded and portable. MP4, JSON, CSV, and raw PNG files remain downloadable.
"""
    (report_dir / "README.md").write_text(readme, encoding="utf-8")
    return report_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
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
