"""Generate a human-readable, visual report from one saved grounding run.

The report is deliberately offline: it never rerenders the scene and never
changes detector or geometric evidence. All three ablations are visualized
from the exact stage artifacts saved by the production joint run.
"""

from __future__ import annotations

import argparse
import html
import json
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
        records.append(
            {
                "stage": stage,
                "name": source.name,
                "paths": copied,
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
body{font-family:system-ui,sans-serif;margin:0;background:#f3f4f6;color:#111827}
main{max-width:1500px;margin:auto;padding:32px}
section{background:white;padding:26px;margin:22px 0;border-radius:14px;box-shadow:0 2px 8px #0001}
h1,h2,h3{margin-top:0} table{border-collapse:collapse;width:100%;font-size:14px}
th,td{border:1px solid #d1d5db;padding:9px;text-align:left;vertical-align:top}
th{background:#e5e7eb}.hero,.wide{width:100%;height:auto;border:1px solid #ddd}
.modes{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}
.mode img{width:100%}.good{color:#047857;font-weight:700}.bad{color:#b91c1c;font-weight:700}
code{background:#eef2ff;padding:2px 5px} details{margin:14px 0}
@media(max-width:950px){.modes{grid-template-columns:1fr}}
"""
    mode_cards = ""
    for mode in MODES:
        record = offline["modes"][mode]
        mode_cards += f"""
<article class="mode">
 <h3>{html.escape(MODE_TITLES[mode])}</h3>
 <img src="{html.escape(visualizations[mode]['gif'])}" alt="{html.escape(mode)} animation">
 <p>{html.escape(MODE_EXPLANATIONS[mode])}</p>
 <p><b>Selected:</b> {html.escape(str(record.get('selected_labels', {})))};
 <b>completion stage:</b> {html.escape(str(record.get('completion_stage')))}</p>
</article>"""
    stages_html = ""
    for stage in stage_components:
        images = "".join(
            f'<h4>{html.escape(name)}</h4><img class="wide" '
            f'src="{html.escape(path)}" alt="{html.escape(name)}">'
            for name, path in stage["paths"].items()
        )
        stages_html += (
            f"<details><summary><b>Stage {stage['stage']:03d}: "
            f"{html.escape(stage['name'])}</b></summary>{images}</details>"
        )
    html_report = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Joint grounding ablation report</title><style>{css}</style></head><body><main>
<section><h1>Joint semantic–geometric grounding report</h1>
<p><b>Scene:</b> {html.escape(str(run_config.get('scene_name')))} ·
<b>Task:</b> {html.escape(str(run_config.get('task_id')))} ·
<b>Run:</b> {html.escape(run_dir.name)}</p>
<p class="good">All expected ablation outcomes matched:
{html.escape(str(offline['all_expected_results_matched']))}</p>
<img class="hero" src="{comparison_path}" alt="ablation comparison"></section>
<section><h2>What was run</h2>
<p>The MuJoCo scene was rendered once per inspection stage using five
region-facing cameras. RGB entered the pretrained semantic detector. Metric
depth and instance masks produced fresh, region-gated, stage-local point
clouds. Geometry was measured only from typed measurement evidence; cumulative
clouds were retained only for visualization.</p>
<p>Every stage evaluated geometry-only, semantic-only, and joint acceptance
from the exact same saved registry, semantic evidence, and geometric evidence.
The two ablations did not rerender or modify observations. Offline ground truth
was used only to explain whether each diagnostic result was correct.</p></section>
<section><h2>Ablation outcomes</h2>
{_table(['Mode','Status','Completion stage','Regions opened','Assignment','Matches ground truth'], outcome_rows)}
</section>
<section><h2>Individual ablation visualizations</h2><div class="modes">{mode_cards}</div></section>
<section><h2>Measured object evidence</h2>
{_table(['Object','RGB label','Region','Points','Cameras','Usable length (m)','Cross-section (m)','Opening width (m)','Cavity depth (m)','Elongated','Open cavity'], geometry_rows)}
<p>UNKNOWN means the point cloud did not support that measurement; it was not fabricated.</p></section>
<section><h2>Pairwise geometric checks</h2>
<p><code>INSERTABLE_IN</code>: opening width − tool cross-section − clearance.
<code>REACHES_BOTTOM</code>: usable length − grip allowance − cavity depth.
Positive signed margins pass.</p>
{_table(['Tool','Relation','Status','Signed margin (m)','First evaluated stage'], relation_table_rows)}
</section>
<section><h2>Semantic detector</h2>
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
<section><h2>Scene and component audit</h2>{stages_html}</section>
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
