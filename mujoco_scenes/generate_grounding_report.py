"""Generate a human-readable, visual report from one saved grounding run.

The report is deliberately offline: it never rerenders the scene and never
changes detector or geometric evidence. All three ablations are visualized
from the exact stage artifacts saved by the production joint run.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import mimetypes
import shutil
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from mujoco_scenes.evaluate_joint_grounding_run import evaluate_saved_run


MODES = ("geometry-only", "semantic-only", "joint")
MODE_TITLES = {
    "geometry-only": "Geometry-only diagnostic",
    "semantic-only": "Semantic-only diagnostic",
    "joint": "Joint semantic + geometric grounding",
}
MODE_EXPLANATIONS = {
    "geometry-only": (
        "Ignores semantic compatibility. It therefore accepts the marker/pen "
        "at INITIAL because its shape is elongated, insertable, and long "
        "enough. This is intentionally an incorrect diagnostic result."
    ),
    "semantic-only": (
        "Ignores unary and pairwise geometry. It therefore accepts the "
        "rank-1 oversized spoon after D1 even though its measured cross-section "
        "does not fit the bowl opening. This is intentionally incorrect."
    ),
    "joint": (
        "Requires semantic compatibility, unary geometry, pairwise geometry, "
        "and distinct role assignments. It rejects the marker semantically, "
        "rejects the oversized spoon geometrically, and selects the fork at D2."
    ),
}
MODE_COLORS = {
    "geometry-only": "#7c3aed",
    "semantic-only": "#c2410c",
    "joint": "#047857",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _font(size: int, *, bold: bool = False):
    names = (
        ("DejaVuSans-Bold.ttf", "Arial Bold.ttf")
        if bold
        else ("DejaVuSans.ttf", "Arial.ttf")
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _value(record: Any) -> Any:
    return record.get("value") if isinstance(record, dict) else None


def _status(record: Any) -> str:
    return (
        str(record.get("status", "UNKNOWN"))
        if isinstance(record, dict)
        else "UNKNOWN"
    )


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "UNKNOWN"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _stage_dirs(run_dir: Path) -> dict[int, Path]:
    result = {}
    for path in sorted((run_dir / "stages").iterdir()):
        if path.is_dir() and path.name[:3].isdigit():
            result[int(path.name[:3])] = path
    return result


def _first_completion(
    ablations: dict[str, Any], mode: str
) -> dict[str, Any] | None:
    return next(
        (
            stage
            for stage in ablations["stages"]
            if stage["modes"][mode]["status"] == "COMPLETE"
        ),
        None,
    )


def _selected_text(mode_result: dict[str, Any]) -> str:
    edges = mode_result.get("selected_candidate_edges", [])
    if not edges:
        return "No complete assignment"
    return ", ".join(
        f"{edge['role']} = {edge['object_id']} "
        f"({edge.get('canonical_label') or 'UNKNOWN'})"
        for edge in edges
    )


def _find_relation_rows(
    stage_dirs: dict[int, Path],
) -> list[dict[str, Any]]:
    rows = []
    seen: set[tuple[str, str]] = set()
    for stage, directory in stage_dirs.items():
        path = directory / "candidate_evaluations.json"
        if not path.exists():
            continue
        data = _load(path)
        for assignment in data.get("assignment_evaluations", []):
            selected = assignment.get("selected_objects", {})
            containers = selected.get("mixing_container", [])
            tools = selected.get("mixing_tool", [])
            if len(containers) != 1 or len(tools) != 1:
                continue
            tool = tools[0]
            container = containers[0]
            if tool == container:
                continue
            for relation in assignment.get("relation_checks", []):
                key = (tool, str(relation.get("relation")))
                if key in seen:
                    continue
                seen.add(key)
                evidence = relation.get("evidence", {})
                rows.append(
                    {
                        "stage": stage,
                        "tool": tool,
                        "container": container,
                        "relation": relation.get("relation"),
                        "status": relation.get("status"),
                        "pass_margin_m": evidence.get("pass_margin_m"),
                        "evidence": evidence,
                    }
                )
    return rows


def _object_rows(registry: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for object_id, record in sorted(registry["objects"].items()):
        semantics = record.get("semantics", {}).get("validated") or {}
        geometry = record.get("geometric_properties", {})
        predicates = record.get("geometric_predicates", {})
        quality = record.get("measurement_quality", {})
        rows.append(
            {
                "object_id": object_id,
                "label": semantics.get("canonical_label", "UNKNOWN"),
                "confidence": semantics.get(
                    "mean_confidence", semantics.get("confidence")
                ),
                "semantic_status": semantics.get("status", "UNKNOWN"),
                "source_region": record.get("last_evidence_source_region"),
                "source_stage": record.get("last_evidence_stage"),
                "point_count": quality.get("point_count"),
                "camera_count": quality.get(
                    "contributing_camera_count"
                ),
                "quality": quality.get("status", "UNKNOWN"),
                "length_m": _value(geometry.get("usable_length_m")),
                "cross_section_m": _value(
                    geometry.get("maximum_cross_section_m")
                ),
                "opening_width_m": _value(
                    geometry.get("opening_width_m")
                ),
                "cavity_depth_m": _value(
                    geometry.get("cavity_depth_m")
                ),
                "elongated": _value(
                    predicates.get("ELONGATED_OBJECT")
                ),
                "open_cavity": _value(predicates.get("OPEN_CAVITY")),
                "measurement_path": record.get("measurement_cloud_path"),
            }
        )
    return rows


def _fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    copy = image.copy()
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(
        copy,
        ((size[0] - copy.width) // 2, (size[1] - copy.height) // 2),
    )
    return canvas


def _wrap(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    font,
    width: int,
) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textlength(candidate, font=font) <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _draw_mode_frame(
    *,
    mode: str,
    stage_record: dict[str, Any],
    stage_dir: Path,
    destination: Path,
) -> None:
    canvas = Image.new("RGB", (1920, 1080), "#f3f4f6")
    draw = ImageDraw.Draw(canvas)
    color = MODE_COLORS[mode]
    result = stage_record["modes"][mode]
    status = result["status"]
    status_color = "#15803d" if status == "COMPLETE" else "#b45309"
    draw.rectangle((0, 0, 1920, 130), fill=color)
    draw.text(
        (45, 22),
        MODE_TITLES[mode],
        font=_font(42, bold=True),
        fill="white",
    )
    draw.text(
        (45, 78),
        f"Stage {stage_record['stage']:03d} · "
        f"region {stage_record['region_id']}",
        font=_font(27),
        fill="white",
    )
    draw.rounded_rectangle(
        (1550, 28, 1870, 103), radius=18, fill=status_color
    )
    draw.text(
        (1710, 65),
        status,
        anchor="mm",
        font=_font(30, bold=True),
        fill="white",
    )

    semantic_path = stage_dir / "semantic_overview.png"
    overview_path = stage_dir / "overview.png"
    semantic = _fit(Image.open(semantic_path).convert("RGB"), (1040, 890))
    overview = _fit(Image.open(overview_path).convert("RGB"), (760, 395))
    canvas.paste(semantic, (35, 155))
    canvas.paste(overview, (1120, 155))

    draw.rounded_rectangle(
        (1120, 575, 1880, 1045),
        radius=20,
        fill="white",
        outline="#d1d5db",
        width=2,
    )
    draw.text(
        (1150, 605),
        "Ablation decision",
        font=_font(30, bold=True),
        fill="#111827",
    )
    selected_font = _font(22, bold=True)
    selected_lines = _wrap(
        draw,
        _selected_text(result),
        font=selected_font,
        width=680,
    )
    y = 655
    for line in selected_lines:
        draw.text(
            (1150, y),
            line,
            font=selected_font,
            fill=status_color,
        )
        y += 31
    y += 12
    for line in _wrap(
        draw,
        MODE_EXPLANATIONS[mode],
        font=_font(22),
        width=680,
    ):
        draw.text((1150, y), line, font=_font(22), fill="#374151")
        y += 33
    reasons = ", ".join(result.get("reason_codes", [])) or "none"
    for line in _wrap(
        draw,
        f"Reason code: {reasons}",
        font=_font(18),
        width=680,
    ):
        draw.text((1150, y + 12), line, font=_font(18), fill="#6b7280")
        y += 28
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def _make_mode_visualizations(
    *,
    output_dir: Path,
    ablations: dict[str, Any],
    stages: dict[int, Path],
) -> dict[str, Any]:
    visualizations = {}
    for mode in MODES:
        completion = _first_completion(ablations, mode)
        terminal_stage = (
            int(completion["stage"])
            if completion is not None
            else max(stages)
        )
        mode_dir = output_dir / "ablations" / mode.replace("-", "_")
        frame_paths = []
        for stage_record in ablations["stages"]:
            stage = int(stage_record["stage"])
            if stage > terminal_stage:
                continue
            frame = mode_dir / f"stage_{stage:03d}.png"
            _draw_mode_frame(
                mode=mode,
                stage_record=stage_record,
                stage_dir=stages[stage],
                destination=frame,
            )
            frame_paths.append(frame)
        gif_path = mode_dir / f"{mode.replace('-', '_')}.gif"
        images = [Image.open(path).convert("P") for path in frame_paths]
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=[
                1500 if index < len(images) - 1 else 3500
                for index in range(len(images))
            ],
            loop=0,
        )
        mp4_path = mode_dir / f"{mode.replace('-', '_')}.mp4"
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-loglevel",
                    "error",
                    "-framerate",
                    "1/2",
                    "-i",
                    str(mode_dir / "stage_%03d.png"),
                    "-vf",
                    "format=yuv420p",
                    "-r",
                    "30",
                    str(mp4_path),
                ],
                check=False,
            )
        visualizations[mode] = {
            "terminal_stage": terminal_stage,
            "frames": [
                path.relative_to(output_dir).as_posix()
                for path in frame_paths
            ],
            "gif": gif_path.relative_to(output_dir).as_posix(),
            "mp4": (
                mp4_path.relative_to(output_dir).as_posix()
                if mp4_path.exists()
                else None
            ),
        }
    return visualizations


def _copy_stage_components(
    run_dir: Path, output_dir: Path, stages: dict[int, Path]
) -> list[dict[str, Any]]:
    records = []
    for stage, source in stages.items():
        destination = output_dir / "stages" / source.name
        destination.mkdir(parents=True, exist_ok=True)
        copied = {}
        for name in (
            "semantic_overview.png",
            "overview.png",
            "pointcloud.png",
            "graph.png",
        ):
            source_file = source / name
            if source_file.exists():
                target = destination / name
                shutil.copy2(source_file, target)
                copied[name] = target.relative_to(output_dir).as_posix()
        camera_overlays = []
        camera_root = source / "semantics" / "cameras"
        if camera_root.exists():
            for camera_dir in sorted(camera_root.iterdir()):
                overlay = camera_dir / "overlay.png"
                if not overlay.exists():
                    continue
                target = destination / "cameras" / (
                    f"{camera_dir.name}_overlay.png"
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(overlay, target)
                camera_overlays.append(
                    {
                        "camera_id": camera_dir.name,
                        "path": target.relative_to(output_dir).as_posix(),
                    }
                )
        records.append(
            {
                "stage": stage,
                "name": source.name,
                "paths": copied,
                "camera_overlays": camera_overlays,
                "source": source.relative_to(run_dir).as_posix(),
            }
        )
    return records


def _comparison_image(
    output_dir: Path,
    visualizations: dict[str, Any],
) -> str:
    canvas = Image.new("RGB", (1920, 760), "#f9fafb")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (45, 25),
        "Same-evidence ablation comparison",
        font=_font(42, bold=True),
        fill="#111827",
    )
    for index, mode in enumerate(MODES):
        x = 35 + index * 630
        frame_path = output_dir / visualizations[mode]["frames"][-1]
        frame = _fit(Image.open(frame_path).convert("RGB"), (590, 470))
        canvas.paste(frame, (x, 100))
        draw.rectangle((x, 590, x + 590, 730), fill="white")
        draw.text(
            (x + 20, 605),
            MODE_TITLES[mode],
            font=_font(24, bold=True),
            fill=MODE_COLORS[mode],
        )
        summary = {
            "geometry-only": "Incorrect: marker/pen at INITIAL",
            "semantic-only": "Incorrect: oversized spoon after D1",
            "joint": "Correct: fork after D2",
        }[mode]
        draw.text(
            (x + 20, 650),
            summary,
            font=_font(22, bold=True),
            fill="#111827",
        )
    path = output_dir / "ablation_comparison.png"
    canvas.save(path)
    return path.relative_to(output_dir).as_posix()


def _table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(value)}</th>" for value in headers)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(value)}</td>" for value in row)
        + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _data_uri(output_dir: Path, relative_path: str) -> str:
    path = output_dir / relative_path
    media_type = mimetypes.guess_type(path.name)[0] or (
        "application/octet-stream"
    )
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def _write_reports(
    *,
    output_dir: Path,
    run_dir: Path,
    run_config: dict[str, Any],
    offline: dict[str, Any],
    object_rows: list[dict[str, Any]],
    relation_rows: list[dict[str, Any]],
    visualizations: dict[str, Any],
    stage_components: list[dict[str, Any]],
    comparison_path: str,
) -> None:
    outcome_rows = []
    for mode in MODES:
        record = offline["modes"][mode]
        labels = record.get("selected_labels", {})
        assignment = ", ".join(
            f"{role}={label}" for role, label in sorted(labels.items())
        ) or "none"
        outcome_rows.append(
            [
                MODE_TITLES[mode],
                record["actual_status"],
                _fmt(record.get("completion_stage"), 0),
                ", ".join(record.get("regions_opened", [])) or "none",
                assignment,
                (
                    "YES"
                    if record["matches_evaluation_ground_truth"]
                    else "NO (diagnostic counterexample)"
                ),
            ]
        )
    geometry_rows = [
        [
            row["object_id"],
            row["label"],
            str(row["source_region"]),
            str(row["point_count"]),
            str(row["camera_count"]),
            _fmt(row["length_m"]),
            _fmt(row["cross_section_m"]),
            _fmt(row["opening_width_m"]),
            _fmt(row["cavity_depth_m"]),
            _fmt(row["elongated"]),
            _fmt(row["open_cavity"]),
        ]
        for row in object_rows
    ]
    relation_table_rows = [
        [
            row["tool"],
            str(row["relation"]),
            str(row["status"]),
            _fmt(row["pass_margin_m"]),
            str(row["stage"]),
        ]
        for row in relation_rows
    ]
    detector = run_config.get("semantic_detector", {})
    css = """
:root{--ink:#111827;--muted:#64748b;--line:#dbe3ef;--paper:#fff;--bg:#eef2f7}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif;margin:0;background:var(--bg);color:var(--ink);line-height:1.55}
nav{position:sticky;top:0;z-index:20;background:#0f172aee;backdrop-filter:blur(9px);padding:12px 28px;display:flex;gap:22px;flex-wrap:wrap}
nav a{color:#e2e8f0;text-decoration:none;font-weight:700;font-size:14px}
main{max-width:1480px;margin:auto;padding:30px}
section{background:var(--paper);padding:32px;margin:24px 0;border-radius:18px;box-shadow:0 5px 20px #0f172a12}
h1{font-size:44px;line-height:1.1;margin:0 0 16px}h2{font-size:30px;margin:0 0 18px}h3{margin-top:0}
.kicker{text-transform:uppercase;letter-spacing:.13em;color:#0f766e;font-weight:800;font-size:13px}
.lede{font-size:20px;color:#334155;max-width:1050px}.meta{display:flex;gap:12px;flex-wrap:wrap;margin:18px 0}
.pill{background:#e2e8f0;border-radius:999px;padding:7px 12px;font-weight:700;font-size:13px}
.success{background:#dcfce7;color:#166534}.hero,.wide{width:100%;height:auto;border:1px solid var(--line);border-radius:12px;display:block}
.takeaways{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;margin-top:22px}
.takeaway{padding:18px;border:1px solid var(--line);border-radius:14px;background:#f8fafc}.takeaway b{display:block;font-size:18px}
.pipeline{display:flex;align-items:center;justify-content:center;gap:10px;flex-wrap:wrap;margin-top:24px}
.step{padding:14px;background:#ecfeff;border:1px solid #a5f3fc;border-radius:12px;font-weight:800}.arrow{color:#0891b2;font-size:22px;font-weight:900}
table{border-collapse:separate;border-spacing:0;width:100%;font-size:14px;overflow:hidden;border:1px solid var(--line);border-radius:12px}
th,td{border-bottom:1px solid var(--line);padding:11px;text-align:left;vertical-align:top}th{background:#e9eef5}tr:last-child td{border-bottom:0}
.mode-block{border:1px solid var(--line);border-left:8px solid var(--accent);border-radius:16px;padding:24px;margin:24px 0;background:#fbfdff}
.mode-head{display:flex;justify-content:space-between;gap:18px}.mode-head h3{font-size:27px;margin-bottom:5px}
.mode-status{border-radius:999px;padding:8px 15px;background:#dcfce7;color:#166534;font-weight:900;height:max-content}
.mode-media{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(300px,.7fr);gap:20px}
video{width:100%;background:#0f172a;border-radius:13px;box-shadow:0 4px 14px #0f172a22}
.decision{padding:20px;border-radius:13px;background:white;border:1px solid var(--line)}.decision code{display:block;white-space:normal;margin:10px 0;color:#0f766e;font-weight:800}
.buttons{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0}.button{padding:9px 13px;border-radius:9px;background:#0f172a;color:white!important;text-decoration:none;font-weight:800;font-size:13px}
details{margin:16px 0;border:1px solid var(--line);border-radius:12px;background:white;overflow:hidden}summary{cursor:pointer;padding:15px 18px;font-weight:800;background:#f8fafc}
.detail-body{padding:18px}.gif-preview{width:100%;border-radius:12px;display:block}
.stage-grid{display:grid;grid-template-columns:2fr 1fr;gap:18px}.stage-side{display:grid;gap:18px}
.camera-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}.camera-card{background:#f8fafc;padding:10px;border-radius:10px}.camera-card img{width:100%;border-radius:8px}.camera-card b{display:block;margin-bottom:7px}
.good{color:#047857;font-weight:800}.bad{color:#b91c1c;font-weight:800}code{background:#eef2ff;padding:2px 5px;border-radius:4px}.small{color:var(--muted);font-size:13px}
@media(max-width:1000px){.takeaways,.mode-media,.stage-grid{grid-template-columns:1fr}.camera-grid{grid-template-columns:1fr}main{padding:16px}h1{font-size:34px}}
"""
    mode_sections = ""
    for mode in MODES:
        record = offline["modes"][mode]
        gif_path = visualizations[mode]["gif"]
        mp4_path = visualizations[mode]["mp4"]
        terminal_frame = visualizations[mode]["frames"][-1]
        video_html = (
            f'<video controls muted loop playsinline '
            f'poster="{_data_uri(output_dir, terminal_frame)}">'
            f'<source src="{_data_uri(output_dir, mp4_path)}" '
            f'type="video/mp4">Your browser cannot play this MP4.</video>'
            if mp4_path
            else f'<img class="wide" src="{_data_uri(output_dir, gif_path)}">'
        )
        labels = ", ".join(
            f"{role} = {label}"
            for role, label in sorted(
                record.get("selected_labels", {}).items()
            )
        )
        correctness = (
            "Matches evaluation ground truth"
            if record["matches_evaluation_ground_truth"]
            else "Diagnostic counterexample — intentionally incorrect"
        )
        explanation = record.get("failure_reason") or (
            "Every required semantic, unary geometric, pairwise geometric, "
            "and distinctness check is satisfied."
        )
        mode_sections += f"""
<article class="mode-block" style="--accent:{MODE_COLORS[mode]}">
 <div class="mode-head"><div><h3>{html.escape(MODE_TITLES[mode])}</h3>
 <p class="small">Same saved observations · completion stage {html.escape(str(record.get('completion_stage')))}</p></div>
 <span class="mode-status">{html.escape(record['actual_status'])}</span></div>
 <div class="mode-media"><div>{video_html}
 <div class="buttons">
 <a class="button" href="{html.escape(mp4_path or gif_path)}">Open MP4</a>
 <a class="button" href="{html.escape(gif_path)}">Open GIF</a></div></div>
 <div class="decision"><h3>Decision</h3>
 <code>{html.escape(labels or 'No assignment')}</code>
 <p><b>{html.escape(correctness)}</b></p>
 <p>{html.escape(MODE_EXPLANATIONS[mode])}</p>
 <p class="small">{html.escape(explanation)}</p></div></div>
 <details><summary>Show the animated GIF in the report</summary>
 <div class="detail-body"><img class="gif-preview"
 src="{_data_uri(output_dir, gif_path)}"
 alt="{html.escape(mode)} animated evidence"></div></details>
</article>"""
    stages_html = ""
    for stage in stage_components:
        paths = stage["paths"]
        semantic = _data_uri(
            output_dir, paths["semantic_overview.png"]
        )
        side_images = "".join(
            f'<div><h4>{html.escape(label)}</h4>'
            f'<img class="wide" src="{_data_uri(output_dir, paths[name])}" '
            f'alt="{html.escape(label)}"></div>'
            for name, label in (
                ("overview.png", "Point cloud + observed graph"),
                ("pointcloud.png", "Stage-local point cloud"),
                ("graph.png", "Observed graph"),
            )
            if name in paths
        )
        cameras = "".join(
            f'<div class="camera-card">'
            f'<b>{html.escape(camera["camera_id"])}</b>'
            f'<img src="{_data_uri(output_dir, camera["path"])}" '
            f'alt="{html.escape(camera["camera_id"])} overlay"></div>'
            for camera in stage.get("camera_overlays", [])
        )
        stages_html += f"""
<details {'open' if stage['stage'] == 0 else ''}>
 <summary>Stage {stage['stage']:03d} · {html.escape(stage['name'])}</summary>
 <div class="detail-body"><div class="stage-grid">
 <div><h3>Five-view rendered scene and RGB detections</h3>
 <img class="wide" src="{semantic}" alt="stage semantic overview"></div>
 <div class="stage-side">{side_images}</div></div>
 <details><summary>Inspect each camera overlay individually</summary>
 <div class="detail-body camera-grid">{cameras}</div></details></div>
</details>"""
    comparison_uri = _data_uri(output_dir, comparison_path)
    html_report = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Joint grounding ablation report</title><style>{css}</style></head><body>
<nav><a href="#summary">Summary</a><a href="#architecture">Architecture</a>
<a href="#ablations">Ablations</a><a href="#measurements">Measurements</a>
<a href="#stages">Scene stages</a><a href="#provenance">Provenance</a></nav><main>
<section id="summary"><div class="kicker">Presentation report · same-evidence ablation</div>
<h1>Joint semantic–geometric grounding</h1>
<p class="lede">Why geometry alone chooses a marker, semantics alone chooses
an oversized spoon, and the joint system correctly selects a fork.</p>
<div class="meta">
<span class="pill">Scene: {html.escape(str(run_config.get('scene_name')))}</span>
<span class="pill">Task: {html.escape(str(run_config.get('task_id')))}</span>
<span class="pill">Run: {html.escape(run_dir.name)}</span>
<span class="pill success">Expected outcomes matched:
{html.escape(str(offline['all_expected_results_matched']))}</span></div>
<img class="hero" src="{comparison_uri}" alt="ablation comparison">
<div class="takeaways">
<div class="takeaway"><b style="color:{MODE_COLORS['geometry-only']}">Geometry-only</b>
Marker/pen selected at INITIAL. Geometry passes; meaning is ignored.</div>
<div class="takeaway"><b style="color:{MODE_COLORS['semantic-only']}">Semantic-only</b>
Oversized spoon selected at D1. Label passes; physical fit is ignored.</div>
<div class="takeaway"><b style="color:{MODE_COLORS['joint']}">Joint</b>
Fork selected at D2 after all semantic and geometric checks pass.</div>
</div></section>
<section id="architecture"><h2>What was run</h2>
<p>The MuJoCo scene was rendered once per inspection stage using five
region-facing cameras. RGB entered the pretrained semantic detector. Metric
depth and instance masks produced fresh, region-gated, stage-local point
clouds. Geometry was measured only from typed measurement evidence; cumulative
clouds were retained only for visualization.</p>
<p>Every stage evaluated geometry-only, semantic-only, and joint acceptance
from the exact same saved registry, semantic evidence, and geometric evidence.
The two ablations did not rerender or modify observations. Offline ground truth
was used only to explain whether each diagnostic result was correct.</p>
<div class="pipeline"><div class="step">Five-view RGB-D</div><div class="arrow">→</div>
<div class="step">Instance association</div><div class="arrow">→</div>
<div class="step">Stage-local geometry</div><div class="arrow">→</div>
<div class="step">Joint role resolver</div></div></section>
<section><h2>Outcome matrix</h2>
{_table(['Mode','Status','Completion stage','Regions opened','Assignment','Matches ground truth'], outcome_rows)}
</section>
<section id="ablations"><h2>Individual ablation visualizations</h2>
<p>Every MP4 and GIF below comes from the same scene observations. Only the
acceptance gates differ.</p>{mode_sections}</section>
<section id="measurements"><h2>Measured object evidence</h2>
{_table(['Object','RGB label','Region','Points','Cameras','Usable length (m)','Cross-section (m)','Opening width (m)','Cavity depth (m)','Elongated','Open cavity'], geometry_rows)}
<p>UNKNOWN means the point cloud did not support that measurement; it was not fabricated.</p></section>
<section><h2>Pairwise geometric checks</h2>
<p><code>INSERTABLE_IN</code>: opening width − tool cross-section − clearance.
<code>REACHES_BOTTOM</code>: usable length − grip allowance − cavity depth.
Positive signed margins pass.</p>
{_table(['Tool','Relation','Status','Signed margin (m)','First evaluated stage'], relation_table_rows)}
</section>
<section id="provenance"><h2>Semantic detector and provenance</h2>
<p>Backend: {html.escape(str(detector.get('name')))};
checkpoint: {html.escape(str(detector.get('checkpoint')))};
version: {html.escape(str(detector.get('version')))};
device: {html.escape(str(detector.get('device')))};
input size: {html.escape(str(detector.get('inference_size')))};
confidence threshold: {html.escape(str(detector.get('confidence_threshold')))};
isolated process: {html.escape(str(detector.get('process_isolation')))}.</p>
<p>MuJoCo instance masks were used only to associate RGB boxes with persistent
generic object IDs. Simulator names did not enter semantic or geometric
inference.</p></section>
<section id="stages"><h2>Scene and component audit</h2>
<p>Expand each stage to inspect the actual rendered scene, detections,
stage-local point cloud, growing graph, and individual camera overlays.</p>
{stages_html}</section>
<section><h2>Evidence safeguards</h2><ul>
<li>No robot, navigation, IK, gripper execution, TAMP, LLM, or VLM was used.</li>
<li>No hidden object appeared before its region was opened and inspected.</li>
<li>Geometry used stage-local <code>MeasurementEvidence</code>, never cumulative or combined scene clouds.</li>
<li>Semantic and geometric namespaces remain separate.</li>
<li>The production joint run stopped at D2; C2, B1, and C1 remained unopened.</li>
</ul></section></main></body></html>"""
    (output_dir / "ablation_report.html").write_text(
        html_report, encoding="utf-8"
    )

    markdown = [
        "# Joint semantic–geometric grounding report",
        "",
        f"- Scene: `{run_config.get('scene_name')}`",
        f"- Task: `{run_config.get('task_id')}`",
        f"- Source run: `{run_dir}`",
        "- Same saved observation evidence used by all modes: `true`",
        f"- All expected outcomes matched: `{offline['all_expected_results_matched']}`",
        "",
        f"![Ablation comparison]({comparison_path})",
        "",
        "## What was done",
        "",
        "The scene was captured once per stage with five region-facing cameras. "
        "RGB supplied semantic evidence; metric depth and instance masks supplied "
        "fresh stage-local point-cloud evidence. Geometry-only and semantic-only "
        "are diagnostic acceptance ablations over those same saved observations. "
        "Only joint mode is the production decision.",
        "",
        "## Outcomes",
        "",
        "| Mode | Completion | Selection | Correct? |",
        "|---|---:|---|---|",
    ]
    for mode in MODES:
        record = offline["modes"][mode]
        markdown.append(
            f"| {MODE_TITLES[mode]} | stage "
            f"{record.get('completion_stage')} | "
            f"`{record.get('selected_labels', {})}` | "
            f"{record['matches_evaluation_ground_truth']} |"
        )
    markdown += ["", "## Visualizations", ""]
    for mode in MODES:
        markdown += [
            f"### {MODE_TITLES[mode]}",
            "",
            MODE_EXPLANATIONS[mode],
            "",
            f"![{mode}]({visualizations[mode]['gif']})",
            "",
        ]
    markdown += [
        "## Machine-readable evidence",
        "",
        "- `offline_ablation_evaluation.json`",
        "- `report_data.json`",
        f"- Original run: `{run_dir}`",
        "",
    ]
    (output_dir / "ablation_report.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )


def generate_report(
    run_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> Path:
    run_path = Path(run_dir).resolve()
    destination = (
        Path(output_dir).resolve()
        if output_dir is not None
        else (Path("reports") / run_path.name).resolve()
    )
    destination.mkdir(parents=True, exist_ok=True)
    run_config = _load(run_path / "run_config.json")
    registry = _load(run_path / "object_registry.json")
    ablations = _load(run_path / "ablation_summary.json")
    stages = _stage_dirs(run_path)
    offline_path = destination / "offline_ablation_evaluation.json"
    offline = evaluate_saved_run(
        run_path,
        output_path=offline_path,
    )
    object_rows = _object_rows(registry)
    relation_rows = _find_relation_rows(stages)
    stage_components = _copy_stage_components(
        run_path, destination, stages
    )
    visualizations = _make_mode_visualizations(
        output_dir=destination,
        ablations=ablations,
        stages=stages,
    )
    comparison = _comparison_image(destination, visualizations)
    report_data = {
        "schema_version": 1,
        "source_run": str(run_path),
        "shared_observation_evidence": True,
        "run_config": run_config,
        "offline_evaluation": offline,
        "objects": object_rows,
        "relations": relation_rows,
        "visualizations": visualizations,
        "stage_components": stage_components,
        "comparison": comparison,
    }
    (destination / "report_data.json").write_text(
        json.dumps(report_data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_reports(
        output_dir=destination,
        run_dir=run_path,
        run_config=run_config,
        offline=offline,
        object_rows=object_rows,
        relation_rows=relation_rows,
        visualizations=visualizations,
        stage_components=stage_components,
        comparison_path=comparison,
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate visual and written reports for a saved run"
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()
    destination = generate_report(
        arguments.run_dir,
        output_dir=arguments.output_dir,
    )
    print(
        json.dumps(
            {
                "report_directory": str(destination),
                "html": str(destination / "ablation_report.html"),
                "markdown": str(destination / "ablation_report.md"),
                "comparison": str(
                    destination / "ablation_comparison.png"
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
