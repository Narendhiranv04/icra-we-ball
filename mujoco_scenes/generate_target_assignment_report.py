"""Generate the self-contained Ablation 3 presentation package."""

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

from PIL import Image, ImageDraw

from mujoco_scenes.evaluate_target_assignment_run import (
    ASSIGNMENT_MODES,
    DEFAULT_EVALUATION_CONFIG,
    evaluate_saved_target_assignment_run,
)
from mujoco_scenes.generate_grounding_report import (
    _copy_stage_components,
    _fit,
    _font,
    _stage_dirs,
    _table,
    _wrap,
)


MODE_TITLES = {
    "semantic-only": "Semantic-only target assignment",
    "geometry-only": "Geometry-only target-specific assignment",
    "joint-target-agnostic-count": "Joint target-agnostic count",
    "joint-target-specific": "Joint target-specific production",
}
MODE_COLORS = {
    "semantic-only": "#c2410c",
    "geometry-only": "#7c3aed",
    "joint-target-agnostic-count": "#b45309",
    "joint-target-specific": "#047857",
}
MODE_EXPLANATIONS = {
    "semantic-only": (
        "False positive: label and count satisfaction ignores whether the "
        "assigned spoon fits or reaches each particular container."
    ),
    "geometry-only": (
        "False positive: target identity and all pairwise measurements are "
        "retained, but utensil-category semantics are removed, so a fork can "
        "fill the missing spoon slot."
    ),
    "joint-target-agnostic-count": (
        "False positive: candidates valid somewhere are counted without "
        "proving reusable all-target coverage or a dedicated matching."
    ),
    "joint-target-specific": (
        "Correct: every selected edge passes semantics, unary geometry, "
        "INSERTABLE_IN, and REACHES_BOTTOM for that exact target, followed "
        "by function-scoped reuse and distinctness constraints."
    ),
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _uri(output_dir: Path, relative: str) -> str:
    path = output_dir / relative
    media = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return "data:" + media + ";base64," + base64.b64encode(
        path.read_bytes()
    ).decode("ascii")


def _stages(run_dir: Path) -> list[dict[str, Any]]:
    summary = _load(run_dir / "assignment_ablation_summary.json")
    records = []
    for item in summary["stages"]:
        stage = int(item["stage"])
        stage_dir = next(
            path
            for number, path in _stage_dirs(run_dir).items()
            if number == stage
        )
        comparison = _load(stage_dir / "assignment_mode_comparison.json")
        records.append(
            {
                **item,
                "modes": comparison["modes"],
            }
        )
    return sorted(records, key=lambda item: int(item["stage"]))


def _completion(stages: list[dict[str, Any]], mode: str) -> int | None:
    return next(
        (
            int(item["stage"])
            for item in stages
            if item["modes"][mode]["status"] == "COMPLETE"
        ),
        None,
    )


def _cell_color(status: str) -> str:
    return {"TRUE": "#dcfce7", "FALSE": "#fee2e2"}.get(
        status, "#e5e7eb"
    )


def _matrix_axes(matrix: dict[str, Any]) -> tuple[list[str], list[str]]:
    tools = sorted({cell["tool_object_id"] for cell in matrix["cells"]})
    targets = sorted({cell["target_object_id"] for cell in matrix["cells"]})
    return tools, targets


def _draw_matrix(matrix: dict[str, Any], destination: Path) -> None:
    tools, targets = _matrix_axes(matrix)
    index = {
        (cell["tool_object_id"], cell["target_object_id"]): cell
        for cell in matrix["cells"]
    }
    width = 470 + 305 * max(1, len(targets))
    height = 190 + 92 * max(1, len(tools))
    canvas = Image.new("RGB", (width, height), "#f8fafc")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (24, 18),
        "Measured tool–target compatibility matrix",
        font=_font(30, bold=True),
        fill="#111827",
    )
    draw.text(
        (24, 61),
        "Each cell is independently grounded from saved target-specific evidence",
        font=_font(17),
        fill="#475569",
    )
    for column, target in enumerate(targets):
        x = 450 + column * 305
        sample = next(
            cell for cell in matrix["cells"] if cell["target_object_id"] == target
        )
        label = (
            sample.get("target_fused_semantic_label")
            or sample.get("target_semantic_label")
            or "UNKNOWN"
        )
        draw.rectangle((x, 105, x + 295, 176), fill="#dbeafe")
        draw.text((x + 10, 116), target, font=_font(18, bold=True), fill="#1e3a8a")
        draw.text((x + 10, 143), label, font=_font(16), fill="#334155")
    for row, tool in enumerate(tools):
        y = 184 + row * 92
        sample = next(
            cell for cell in matrix["cells"] if cell["tool_object_id"] == tool
        )
        label = (
            sample.get("tool_fused_semantic_label")
            or sample.get("tool_semantic_label")
            or "UNKNOWN"
        )
        draw.rectangle((20, y, 440, y + 82), fill="#e2e8f0")
        draw.text((32, y + 12), tool, font=_font(19, bold=True), fill="#111827")
        draw.text((32, y + 43), label, font=_font(16), fill="#475569")
        for column, target in enumerate(targets):
            x = 450 + column * 305
            cell = index.get((tool, target))
            status = (
                cell["target_specific_compatibility_status"]
                if cell is not None
                else "N/A"
            )
            draw.rectangle(
                (x, y, x + 295, y + 82),
                fill=_cell_color(status),
                outline="#cbd5e1",
            )
            if cell is None:
                text = "N/A"
            else:
                insertion = cell.get("insertable_in_pass_margin_m")
                reach = cell.get("reaches_bottom_pass_margin_m")
                text = (
                    f"{status} · {cell['function_group_id']}\n"
                    f"insert {insertion:+.4f} m · reach {reach:+.4f} m"
                    if insertion is not None and reach is not None
                    else f"{status} · {cell.get('rejection_reason') or ''}"
                )
            for line_index, line in enumerate(text.splitlines()):
                draw.text(
                    (x + 9, y + 12 + line_index * 27),
                    line,
                    font=_font(14, bold=line_index == 0),
                    fill="#111827",
                )
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def _assignment_lines(result: dict[str, Any]) -> list[str]:
    lines = []
    for group in result.get("function_group_evaluations", []):
        counts = group["counts"]
        lines.append(
            f"{group['function_group_id']}: {group['status']} · "
            f"{counts['satisfied_target_slots']}/"
            f"{counts['required_target_slots']} targets"
        )
        for item in group.get("selected_assignments", []):
            flags = []
            if item.get("reused_assignment"):
                flags.append("reused")
            if item.get("dedicated_assignment"):
                flags.append("dedicated")
            if item.get("cross_group_reused_assignment"):
                flags.append("cross-group")
            suffix = f" ({', '.join(flags)})" if flags else ""
            lines.append(
                f"  {item['utensil_object_id']} → "
                f"{item['target_object_id']}{suffix}"
            )
    return lines or ["No complete target assignment"]


def _draw_mode_frame(
    mode: str,
    stage: dict[str, Any],
    stage_dir: Path,
    destination: Path,
) -> None:
    result = stage["modes"][mode]
    canvas = Image.new("RGB", (1920, 1080), "#eef2f7")
    draw = ImageDraw.Draw(canvas)
    color = MODE_COLORS[mode]
    status_color = "#15803d" if result["status"] == "COMPLETE" else "#b45309"
    draw.rectangle((0, 0, 1920, 125), fill=color)
    draw.text((38, 18), MODE_TITLES[mode], font=_font(38, bold=True), fill="white")
    draw.text(
        (38, 74),
        f"Stage {int(stage['stage']):03d} · {stage['region_id']} · same saved RGB-D evidence",
        font=_font(23),
        fill="white",
    )
    draw.rounded_rectangle((1585, 25, 1880, 98), radius=16, fill=status_color)
    draw.text(
        (1732, 61), result["status"], anchor="mm", font=_font(27, bold=True), fill="white"
    )
    semantic = _fit(
        Image.open(stage_dir / "semantic_overview.png").convert("RGB"),
        (1030, 900),
    )
    canvas.paste(semantic, (22, 145))
    matrix_path = destination.with_name(destination.stem + "_matrix.png")
    _draw_matrix(_load(stage_dir / "compatibility_matrix.json"), matrix_path)
    matrix_image = _fit(Image.open(matrix_path).convert("RGB"), (820, 510))
    canvas.paste(matrix_image, (1080, 145))
    matrix_path.unlink(missing_ok=True)
    draw.rounded_rectangle((1080, 680, 1900, 1045), radius=16, fill="white")
    y = 700
    for raw in _assignment_lines(result):
        for line in _wrap(draw, raw, font=_font(17, bold=True), width=770):
            draw.text((1100, y), line, font=_font(17, bold=True), fill="#111827")
            y += 25
    y += 8
    for line in _wrap(draw, MODE_EXPLANATIONS[mode], font=_font(16), width=770):
        draw.text((1100, y), line, font=_font(16), fill="#475569")
        y += 23
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def _animations(
    output_dir: Path, run_dir: Path, stages: list[dict[str, Any]]
) -> dict[str, Any]:
    stage_dirs = _stage_dirs(run_dir)
    artifacts = {}
    for mode in ASSIGNMENT_MODES:
        terminal = _completion(stages, mode)
        terminal = terminal if terminal is not None else max(stage_dirs)
        mode_dir = output_dir / "ablations" / mode.replace("-", "_")
        frames = []
        for stage in stages:
            number = int(stage["stage"])
            if number > terminal:
                continue
            frame = mode_dir / f"stage_{number:03d}.png"
            _draw_mode_frame(mode, stage, stage_dirs[number], frame)
            frames.append(frame)
        gif_path = mode_dir / f"{mode.replace('-', '_')}.gif"
        images = [Image.open(path).convert("P") for path in frames]
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=[1800] * (len(images) - 1) + [3600],
            loop=0,
        )
        mp4_path = mode_dir / f"{mode.replace('-', '_')}.mp4"
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            manifest = mode_dir / "frames.txt"
            content = []
            for frame in frames:
                content.extend((f"file '{frame.name}'", "duration 2.0"))
            content.append(f"file '{frames[-1].name}'")
            manifest.write_text("\n".join(content) + "\n", encoding="utf-8")
            subprocess.run(
                [
                    ffmpeg, "-y", "-loglevel", "error", "-f", "concat",
                    "-safe", "0", "-i", manifest.name,
                    "-vf", "fps=30,format=yuv420p", mp4_path.name,
                ],
                cwd=mode_dir,
                check=True,
            )
            manifest.unlink()
        artifacts[mode] = {
            "terminal_stage": terminal,
            "frames": [p.relative_to(output_dir).as_posix() for p in frames],
            "gif": gif_path.relative_to(output_dir).as_posix(),
            "mp4": mp4_path.relative_to(output_dir).as_posix() if mp4_path.exists() else None,
        }
    return artifacts


def _comparison(
    output_dir: Path, animations: dict[str, Any], offline: dict[str, Any]
) -> str:
    canvas = Image.new("RGB", (1920, 680), "#f8fafc")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (35, 18), "Ablation 3 · same-evidence assignment comparison",
        font=_font(38, bold=True), fill="#111827",
    )
    for index, mode in enumerate(ASSIGNMENT_MODES):
        x = 22 + index * 475
        frame = _fit(
            Image.open(output_dir / animations[mode]["frames"][-1]).convert("RGB"),
            (450, 430),
        )
        canvas.paste(frame, (x, 82))
        item = offline["modes"][mode]
        draw.rectangle((x, 522, x + 450, 655), fill="white")
        draw.text((x + 12, 537), MODE_TITLES[mode], font=_font(17, bold=True), fill=MODE_COLORS[mode])
        result = f"{item['actual_status']} at stage {item['completion_stage']}"
        verdict = "intentionally incorrect" if item["intentionally_incorrect"] else "correct production"
        draw.text((x + 12, 574), result, font=_font(17, bold=True), fill="#111827")
        draw.text((x + 12, 608), verdict, font=_font(15), fill="#475569")
    path = output_dir / "assignment_ablation_comparison.png"
    canvas.save(path)
    return path.relative_to(output_dir).as_posix()


def _fmt(value: Any) -> str:
    return "—" if value is None else f"{float(value):.4f}" if isinstance(value, (int, float)) else str(value)


def _matrix_rows(matrix: dict[str, Any]) -> list[list[str]]:
    return [
        [
            cell["function_group_id"], cell["tool_object_id"],
            cell.get("tool_fused_semantic_label")
            or cell.get("tool_semantic_label")
            or "UNKNOWN",
            cell["target_object_id"],
            cell.get("target_fused_semantic_label")
            or cell.get("target_semantic_label")
            or "UNKNOWN",
            _fmt(cell.get("maximum_cross_section_m")), _fmt(cell.get("usable_length_m")),
            _fmt(cell.get("opening_width_m")), _fmt(cell.get("cavity_depth_m")),
            _fmt(cell.get("insertable_in_pass_margin_m")), cell["insertable_in_status"],
            _fmt(cell.get("reaches_bottom_pass_margin_m")), cell["reaches_bottom_status"],
            cell["target_specific_compatibility_status"], cell.get("rejection_reason") or "—",
        ]
        for cell in matrix["cells"]
    ]


def _object_rows(registry: dict[str, Any]) -> list[list[str]]:
    rows = []
    for object_id, record in sorted(registry["objects"].items()):
        semantic = record.get("semantics", {}).get("validated") or {}
        geometry = record.get("geometric_properties", {})
        predicate = record.get("geometric_predicates", {})
        value = lambda name: geometry.get(name, {}).get("value")
        rows.append([
            object_id, semantic.get("canonical_label") or "UNKNOWN",
            _fmt(semantic.get("mean_confidence")), str(semantic.get("supporting_view_count", 0)),
            str(record.get("last_property_source_region")), str(record.get("point_count")),
            _fmt(value("usable_length_m")), _fmt(value("maximum_cross_section_m")),
            _fmt(value("opening_width_m")), _fmt(value("cavity_depth_m")),
            str(predicate.get("ELONGATED_OBJECT", {}).get("status", "UNKNOWN")),
            str(predicate.get("OPEN_CAVITY", {}).get("status", "UNKNOWN")),
            str(record.get("measurement_cloud_path")),
        ])
    return rows


def _target_rows(matrix: dict[str, Any]) -> list[list[str]]:
    rows = []
    for target_id in matrix["target_object_ids"]:
        cell = next(
            item
            for item in matrix["cells"]
            if item["target_object_id"] == target_id
        )
        rows.append(
            [
                target_id,
                cell.get("target_fused_semantic_label")
                or cell.get("target_semantic_label")
                or "UNKNOWN",
                cell["target_role"],
                _fmt(cell.get("opening_width_m")),
                _fmt(cell.get("cavity_depth_m")),
                cell["open_cavity_status"],
                str(cell.get("target_source_stage")),
                str(cell.get("target_source_region")),
                str(cell.get("target_geometry_evidence_path")),
            ]
        )
    return rows


def _tool_rows(matrix: dict[str, Any]) -> list[list[str]]:
    rows = []
    for tool_id in matrix["tool_object_ids"]:
        cell = next(
            item
            for item in matrix["cells"]
            if item["tool_object_id"] == tool_id
        )
        rows.append(
            [
                tool_id,
                cell.get("tool_fused_semantic_label")
                or cell.get("tool_semantic_label")
                or "UNKNOWN",
                _fmt(cell.get("maximum_cross_section_m")),
                _fmt(cell.get("usable_length_m")),
                cell["elongated_object_status"],
                str(cell.get("tool_source_stage")),
                str(cell.get("tool_source_region")),
                str(cell.get("tool_geometry_evidence_path")),
                str(cell.get("tool_semantic_evidence_path")),
            ]
        )
    return rows


def _write_report(
    output_dir: Path,
    run_dir: Path,
    offline: dict[str, Any],
    animations: dict[str, Any],
    components: list[dict[str, Any]],
    comparison: str,
) -> None:
    run_config = _load(run_dir / "run_config.json")
    registry = _load(run_dir / "object_registry.json")
    matrix = _load(run_dir / "compatibility_matrix.json")
    stages = _stages(run_dir)
    witness = _load(run_dir / "latest_witness.json")
    matrix_png = "compatibility_matrix.png"
    _draw_matrix(matrix, output_dir / matrix_png)
    outcome_rows = [
        [
            MODE_TITLES[mode], offline["modes"][mode]["actual_status"],
            str(offline["modes"][mode]["completion_stage"]),
            "No" if offline["modes"][mode]["intentionally_incorrect"] else "Yes",
            "Yes" if offline["modes"][mode]["matches_expected_result"] else "No",
        ]
        for mode in ASSIGNMENT_MODES
    ]
    stage_rows = [
        [
            f"{int(item['stage']):03d}",
            str(item["region_id"]),
            *[
                item["modes"][mode]["status"]
                for mode in ASSIGNMENT_MODES
            ],
        ]
        for item in stages
    ]
    assignment_rows = [
        [
            item["function_group_id"],
            item["utensil_object_id"],
            item["target_object_id"],
            "YES" if item.get("reused_assignment") else "NO",
            "YES" if item.get("dedicated_assignment") else "NO",
            "YES"
            if item.get("cross_group_reused_assignment")
            else "NO",
            ", ".join(
                f"{check['relation']}={check['status']} "
                f"({_fmt(check.get('evidence', {}).get('pass_margin_m'))} m)"
                for check in item.get("relation_checks", [])
            ),
        ]
        for item in witness.get("operation_assignments", [])
    ]
    mode_html = ""
    for mode in ASSIGNMENT_MODES:
        viz = animations[mode]
        video = (
            f'<video controls muted loop playsinline poster="{_uri(output_dir, viz["frames"][-1])}">'
            f'<source type="video/mp4" src="{_uri(output_dir, viz["mp4"])}"></video>'
            if viz["mp4"] else f'<img class="wide" src="{_uri(output_dir, viz["gif"])}">'
        )
        mode_html += f'''<article class="mode" style="--accent:{MODE_COLORS[mode]}"><h3>{html.escape(MODE_TITLES[mode])}</h3>
<p>{html.escape(MODE_EXPLANATIONS[mode])}</p>{video}
<details><summary>GIF and individual frames</summary><div class="inside"><img class="wide" src="{_uri(output_dir, viz['gif'])}"><p>{', '.join(viz['frames'])}</p></div></details></article>'''
    stage_html = ""
    for item in components:
        paths = item["paths"]
        images = "".join(
            f'<div><b>{name}</b><img class="wide" src="{_uri(output_dir, path)}"></div>'
            for name, path in paths.items()
        )
        overlays = "".join(
            f'<div><b>{camera["camera_id"]}</b><img class="wide" src="{_uri(output_dir, camera["path"])}"></div>'
            for camera in item["camera_overlays"]
        )
        stage_html += f'''<details {'open' if item['stage']==0 else ''}><summary>Stage {item['stage']:03d} · {html.escape(item['name'])}</summary><div class="inside grid">{images}</div><details><summary>Five RGB detector overlays</summary><div class="inside cams">{overlays}</div></details></details>'''
    detector = run_config["semantic_detector"]
    css = """*{box-sizing:border-box}body{margin:0;background:#eef2f7;color:#111827;font-family:Inter,system-ui,sans-serif;line-height:1.5}nav{position:sticky;top:0;z-index:5;background:#0f172a;padding:12px 22px}nav a{color:white;margin-right:20px;text-decoration:none;font-weight:800}main{max-width:1560px;margin:auto;padding:24px}section,.mode{background:white;border-radius:17px;padding:27px;margin:20px 0;box-shadow:0 4px 17px #0f172a14}.mode{border-left:8px solid var(--accent)}h1{font-size:42px}.lede{font-size:20px;color:#334155}.wide{width:100%;height:auto;border:1px solid #dbe3ef;border-radius:10px;display:block}video{width:100%;border-radius:10px;background:#111}table{width:100%;border-collapse:collapse;font-size:12px}th,td{border:1px solid #dbe3ef;padding:8px;text-align:left;vertical-align:top}th{background:#e9eef5;position:sticky;top:44px}details{border:1px solid #dbe3ef;border-radius:10px;margin:12px 0;overflow:hidden}summary{padding:13px;background:#f8fafc;font-weight:800;cursor:pointer}.inside{padding:14px}.grid{display:grid;grid-template-columns:2fr 1fr;gap:14px}.cams{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}.pill{display:inline-block;padding:6px 11px;background:#dcfce7;color:#166534;border-radius:999px;font-weight:800;margin:4px}@media(max-width:900px){.grid,.cams{grid-template-columns:1fr}main{padding:12px}h1{font-size:32px}}"""
    document = f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Ablation 3 · multi-target assignment</title><style>{css}</style></head><body><nav><a href="#summary">Summary</a><a href="#modes">Modes</a><a href="#matrix">Matrix</a><a href="#objects">Evidence</a><a href="#stages">Scene</a></nav><main>
<section id="summary"><small>PRESENTATION REPORT · ABLATION 3</small><h1>Target-specific semantic–geometric assignment</h1><p class="lede">A utensil is not globally valid. It is valid for a declared function and a specific target only when semantic compatibility, scale-independent unary geometry, and both measured pairwise relations pass. Reuse and distinctness are constraints on each task-level function group.</p><span class="pill">Scene: {html.escape(run_config['scene_name'])}</span><span class="pill">Same perception for all modes</span><span class="pill">Expected results matched: {offline['all_expected_results_matched']}</span><img class="wide" src="{_uri(output_dir, comparison)}"></section>
<section><h2>Headline outcomes</h2>{_table(['Mode','Outcome','Completion stage','Scientifically correct?','Expected matched?'],outcome_rows)}<p>Semantic-only, geometry-only, and target-agnostic count are intentional diagnostic false positives. Only joint target-specific assignment controls runtime stopping and the verified handoff.</p></section>
<section><h2>Stage progression</h2>{_table(['Stage','Region',*[MODE_TITLES[mode] for mode in ASSIGNMENT_MODES]],stage_rows)}<p>Only the production column controls inspection. It remains incomplete at INITIAL and D1, becomes complete at D2, and stops before C2, B1, or C1.</p></section>
<section><h2>Where reusability is attached</h2><p><code>coffee_stirring</code> declares sequential reuse and requires the same physical spoon to pass both cup/mug target edges. <code>soup_serving</code> declares dedicated-per-target matching, so the two bowl assignments must use distinct persistent IDs. Cross-group reuse is separately allowed. These are task-level function constraints, not permanent properties of spoons or forks.</p></section>
<section id="modes"><h2>Four individual ablation visualizations</h2>{mode_html}</section>
<section><h2>Selected production assignment</h2>{_table(['Function group','Tool ID','Target ID','Reused in group','Dedicated','Cross-group reused','Selected relation evidence'],assignment_rows)}</section>
<section><h2>Target-container measurements</h2>{_table(['Target ID','RGB label','Role','Opening width m','Cavity depth m','Open cavity','Stage','Region','MeasurementEvidence path'],_target_rows(matrix))}</section>
<section><h2>Utensil measurements</h2>{_table(['Tool ID','RGB label','Cross-section m','Usable length m','Elongated','Stage','Region','MeasurementEvidence path','Semantic evidence path'],_tool_rows(matrix))}</section>
<section id="matrix"><h2>Complete compatibility matrix</h2><img class="wide" src="{_uri(output_dir,matrix_png)}">{_table(['Group','Tool','Tool label','Target','Target label','Cross-section m','Usable length m','Opening m','Depth m','Insert margin','Insert','Reach margin','Reach','Final','Reason'],_matrix_rows(matrix))}</section>
<section id="objects"><h2>Measured stage-local object evidence</h2>{_table(['Object','YOLO label','Confidence','Views','Region','Points','Usable L','Cross-section','Opening','Depth','Elongated','Open cavity','MeasurementEvidence path'],_object_rows(registry))}</section>
<section id="stages"><h2>Rendered scene and component audit</h2>{stage_html}</section>
<section><h2>Provenance and boundary</h2><p>Detector: {html.escape(str(detector['name']))}; checkpoint: {html.escape(str(detector['checkpoint']))}; version: {html.escape(str(detector['version']))}; device: {html.escape(str(detector['device']))}; input size: {html.escape(str(detector['inference_size']))}; isolated process: {html.escape(str(detector['process_isolation']))}.</p><p>Geometry consumed typed stage-local MeasurementEvidence only. Cumulative and combined clouds were visualization-only. Requirements were manually authored; no FM, TAMP, robot, navigation, IK, grasping, or execution was used. Evaluation annotations were offline-only.</p></section></main></body></html>'''
    for name in ("presentation_report.html", "ablation_report.html"):
        (output_dir / name).write_text(document, encoding="utf-8")
    markdown_rows = "\n".join("| " + " | ".join(row) + " |" for row in outcome_rows)
    links = "\n".join(
        f"- [{MODE_TITLES[mode]} GIF]({animations[mode]['gif']}) · [MP4]({animations[mode]['mp4']})"
        for mode in ASSIGNMENT_MODES
    )
    readme = f'''# Ablation 3: multi-target semantic–geometric assignment

Compatibility is evaluated as `VALID_FOR(tool, function, target)`, not as a global `VALID_TOOL(tool)` flag. Reuse/distinctness belongs to each task-level function group.

![Same-evidence comparison]({comparison})

| Mode | Outcome | Completion stage | Scientifically correct? | Expected matched? |
|---|---:|---:|---:|---:|
{markdown_rows}

![Compatibility matrix]({matrix_png})

## Animations

{links}

Open `presentation_report.html` for the self-contained report with all stage views, detector overlays, point clouds, graph images, numeric measurements, pairwise margins, assignments, GIFs, and MP4s.
'''
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    (output_dir / "ablation_report.md").write_text(readme, encoding="utf-8")


def generate_target_assignment_report(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    evaluation_config: str | Path = DEFAULT_EVALUATION_CONFIG,
) -> dict[str, Any]:
    run_path = Path(run_dir).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    offline = evaluate_saved_target_assignment_run(
        run_path,
        evaluation_config=evaluation_config,
        output_path=destination / "offline_assignment_ablation_evaluation.json",
    )
    stages = _stages(run_path)
    animations = _animations(destination, run_path, stages)
    components = _copy_stage_components(run_path, destination, _stage_dirs(run_path))
    comparison = _comparison(destination, animations, offline)
    _write_report(destination, run_path, offline, animations, components, comparison)
    report_data = {
        "schema_version": 1,
        "run_directory": str(run_path),
        "offline_evaluation": offline,
        "visualizations": animations,
        "stage_components": components,
        "comparison": comparison,
    }
    (destination / "report_data.json").write_text(
        json.dumps(report_data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result = {
        "report_directory": str(destination),
        "presentation_html": str(destination / "presentation_report.html"),
        "markdown": str(destination / "README.md"),
        "comparison": str(destination / comparison),
        "all_expected_results_matched": offline["all_expected_results_matched"],
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Ablation 3 report")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--evaluation-config", type=Path, default=DEFAULT_EVALUATION_CONFIG)
    arguments = parser.parse_args()
    generate_target_assignment_report(
        arguments.run_dir,
        arguments.output_dir,
        evaluation_config=arguments.evaluation_config,
    )


if __name__ == "__main__":
    main()
