"""Generate the self-contained Ablation 2 presentation report.

The generator is offline and consumes one fully observed primary run. It does
not rerender MuJoCo or rerun semantic/geometric inference. All three policy
modes are visualized from the same saved stage comparisons.
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

from PIL import Image, ImageDraw

from mujoco_scenes.evaluate_usage_policy_run import (
    DEFAULT_EVALUATION_CONFIG,
    evaluate_saved_usage_policy_run,
)
from mujoco_scenes.generate_grounding_report import (
    _copy_stage_components,
    _fit,
    _font,
    _stage_dirs,
    _table,
    _wrap,
)


MODES = ("always-reusable", "always-distinct", "function-aware")
MODE_TITLES = {
    "always-reusable": "Always reusable diagnostic",
    "always-distinct": "Always distinct diagnostic",
    "function-aware": "Function-aware production policy",
}
MODE_COLORS = {
    "always-reusable": "#b45309",
    "always-distinct": "#7c3aed",
    "function-aware": "#047857",
}
MODE_EXPLANATIONS = {
    "always-reusable": (
        "Incorrectly lets one valid utensil fill both coffee slots and both "
        "dedicated soup slots. It is a false positive because it discards "
        "the soup function's dedicated-per-target constraint."
    ),
    "always-distinct": (
        "Incorrectly requires a different physical utensil for all four target "
        "slots. It is a false negative because it discards valid sequential "
        "reuse for coffee and allowed reuse across operation groups."
    ),
    "function-aware": (
        "Reuses one valid utensil across both coffee cups, requires two "
        "different utensil IDs across the soup bowls, and allows the coffee "
        "utensil to be one of those two soup utensils. The final witness uses "
        "the initial spoon plus the D2 fork, so the requirement is derived as "
        "two physical objects rather than hard-coded by category."
    ),
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _data_uri(output_dir: Path, relative: str) -> str:
    path = output_dir / relative
    media_type = mimetypes.guess_type(path.name)[0] or (
        "application/octet-stream"
    )
    return (
        f"data:{media_type};base64,"
        + base64.b64encode(path.read_bytes()).decode("ascii")
    )


def _stage_records(run_dir: Path) -> list[dict[str, Any]]:
    summary = _load(run_dir / "policy_ablation_summary.json")
    return sorted(summary["stages"], key=lambda item: int(item["stage"]))


def _first_completion(
    stages: list[dict[str, Any]], mode: str
) -> int | None:
    return next(
        (
            int(record["stage"])
            for record in stages
            if record["modes"][mode]["status"] == "COMPLETE"
        ),
        None,
    )


def _assignment_lines(result: dict[str, Any]) -> list[str]:
    assignments = result.get("operation_assignments", [])
    if not assignments:
        return ["No complete global assignment"]
    lines = []
    for group in result.get("function_group_evaluations", []):
        lines.append(
            f"{group['function_group_id']}: "
            f"{group['counts']['satisfied_target_slots']}/"
            f"{group['counts']['required_target_slots']} target slots; "
            f"{group['counts']['distinct_assigned_physical_objects']}/"
            f"{group['counts']['required_distinct_physical_objects']} "
            "distinct tools"
        )
        for assignment in group.get("selected_assignments", []):
            suffix = (
                " · REUSED"
                if assignment.get("reused_assignment")
                else " · DEDICATED"
                if assignment.get("dedicated_assignment")
                else ""
            )
            lines.append(
                f"  {assignment['utensil_object_id']} → "
                f"{assignment['target_object_id']}{suffix}"
            )
    return lines


def _draw_policy_frame(
    *,
    mode: str,
    stage_record: dict[str, Any],
    stage_dir: Path,
    destination: Path,
    terminal_exhausted: bool = False,
) -> None:
    result = stage_record["modes"][mode]
    canvas = Image.new("RGB", (1920, 1080), "#f1f5f9")
    draw = ImageDraw.Draw(canvas)
    color = MODE_COLORS[mode]
    status = "EXHAUSTED" if terminal_exhausted else result["status"]
    status_color = "#15803d" if status == "COMPLETE" else "#b45309"
    draw.rectangle((0, 0, 1920, 132), fill=color)
    draw.text(
        (44, 20),
        MODE_TITLES[mode],
        font=_font(41, bold=True),
        fill="white",
    )
    draw.text(
        (44, 79),
        f"Stage {int(stage_record['stage']):03d} · "
        f"region {stage_record['region_id']} · identical saved perception",
        font=_font(25),
        fill="white",
    )
    draw.rounded_rectangle(
        (1570, 28, 1875, 102), radius=18, fill=status_color
    )
    draw.text(
        (1722, 65),
        status,
        anchor="mm",
        font=_font(29, bold=True),
        fill="white",
    )

    semantic = _fit(
        Image.open(stage_dir / "semantic_overview.png").convert("RGB"),
        (1040, 900),
    )
    overview = _fit(
        Image.open(stage_dir / "overview.png").convert("RGB"),
        (780, 410),
    )
    canvas.paste(semantic, (25, 150))
    canvas.paste(overview, (1115, 150))
    draw.rounded_rectangle(
        (1115, 585, 1895, 1050),
        radius=18,
        fill="white",
        outline="#cbd5e1",
        width=2,
    )
    draw.text(
        (1142, 610),
        "Assignment and count decision",
        font=_font(27, bold=True),
        fill="#111827",
    )
    y = 652
    for raw_line in _assignment_lines(result):
        for line in _wrap(
            draw, raw_line, font=_font(19, bold=True), width=710
        ):
            draw.text(
                (1142, y),
                line,
                font=_font(19, bold=True),
                fill=status_color,
            )
            y += 27
    y += 9
    count_text = (
        f"Policy requirement: "
        f"{result['policy_required_distinct_physical_tool_count']} "
        f"distinct physical tools · currently assigned: "
        f"{result['distinct_physical_tool_count']}"
    )
    for line in _wrap(
        draw, count_text, font=_font(18), width=710
    ):
        draw.text((1142, y), line, font=_font(18), fill="#334155")
        y += 25
    for line in _wrap(
        draw, MODE_EXPLANATIONS[mode], font=_font(17), width=710
    ):
        draw.text((1142, y + 7), line, font=_font(17), fill="#475569")
        y += 24
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def _make_visualizations(
    output_dir: Path,
    run_dir: Path,
    stages: list[dict[str, Any]],
) -> dict[str, Any]:
    stage_dirs = _stage_dirs(run_dir)
    result: dict[str, Any] = {}
    for mode in MODES:
        completion = _first_completion(stages, mode)
        terminal = completion if completion is not None else max(stage_dirs)
        mode_dir = output_dir / "ablations" / mode.replace("-", "_")
        frames = []
        for record in stages:
            stage = int(record["stage"])
            if stage > terminal:
                continue
            destination = mode_dir / f"stage_{stage:03d}.png"
            _draw_policy_frame(
                mode=mode,
                stage_record=record,
                stage_dir=stage_dirs[stage],
                destination=destination,
                terminal_exhausted=(
                    completion is None and stage == terminal
                ),
            )
            frames.append(destination)
        gif_path = mode_dir / f"{mode.replace('-', '_')}.gif"
        images = [Image.open(path).convert("P") for path in frames]
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=[
                1600 if index < len(images) - 1 else 3600
                for index in range(len(images))
            ],
            loop=0,
        )
        mp4_path = mode_dir / f"{mode.replace('-', '_')}.mp4"
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            concat = mode_dir / "frames.txt"
            lines = []
            for frame in frames:
                lines.extend((f"file '{frame.name}'", "duration 2.0"))
            lines.append(f"file '{frames[-1].name}'")
            concat.write_text("\n".join(lines) + "\n", encoding="utf-8")
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    concat.name,
                    "-vf",
                    "fps=30,format=yuv420p",
                    mp4_path.name,
                ],
                cwd=mode_dir,
                check=False,
            )
            concat.unlink(missing_ok=True)
        result[mode] = {
            "terminal_stage": terminal,
            "frames": [
                path.relative_to(output_dir).as_posix() for path in frames
            ],
            "gif": gif_path.relative_to(output_dir).as_posix(),
            "mp4": (
                mp4_path.relative_to(output_dir).as_posix()
                if mp4_path.exists()
                else None
            ),
        }
    return result


def _comparison_image(
    output_dir: Path,
    visualizations: dict[str, Any],
) -> str:
    canvas = Image.new("RGB", (1920, 770), "#f8fafc")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (45, 25),
        "Ablation 2 · same-evidence usage-policy comparison",
        font=_font(40, bold=True),
        fill="#111827",
    )
    summaries = {
        "always-reusable": "False positive · one utensil at INITIAL",
        "always-distinct": "False negative · assumes four utensils",
        "function-aware": "Correct · spoon + fork after D2",
    }
    for index, mode in enumerate(MODES):
        x = 35 + index * 630
        frame = _fit(
            Image.open(
                output_dir / visualizations[mode]["frames"][-1]
            ).convert("RGB"),
            (590, 470),
        )
        canvas.paste(frame, (x, 100))
        draw.rectangle((x, 590, x + 590, 735), fill="white")
        draw.text(
            (x + 18, 607),
            MODE_TITLES[mode],
            font=_font(22, bold=True),
            fill=MODE_COLORS[mode],
        )
        draw.text(
            (x + 18, 655),
            summaries[mode],
            font=_font(19, bold=True),
            fill="#111827",
        )
    destination = output_dir / "policy_ablation_comparison.png"
    canvas.save(destination)
    return destination.relative_to(output_dir).as_posix()


def _object_rows(registry: dict[str, Any]) -> list[list[str]]:
    rows = []
    for object_id, record in sorted(registry["objects"].items()):
        semantics = record.get("semantics", {}).get("validated") or {}
        geometry = record.get("geometric_properties", {})
        predicates = record.get("geometric_predicates", {})
        quality = record.get("measurement_quality", {})
        value = lambda name: geometry.get(name, {}).get("value")
        rows.append(
            [
                object_id,
                str(semantics.get("canonical_label") or "UNKNOWN"),
                f"{float(semantics.get('mean_confidence') or 0):.3f}",
                str(semantics.get("supporting_view_count", 0)),
                str(record.get("last_property_source_region")),
                str(quality.get("point_count")),
                str(quality.get("contributing_camera_count")),
                (
                    f"{float(value('usable_length_m')):.4f}"
                    if value("usable_length_m") is not None
                    else "—"
                ),
                (
                    f"{float(value('maximum_cross_section_m')):.4f}"
                    if value("maximum_cross_section_m") is not None
                    else "—"
                ),
                (
                    f"{float(value('opening_width_m')):.4f}"
                    if value("opening_width_m") is not None
                    else "—"
                ),
                (
                    f"{float(value('cavity_depth_m')):.4f}"
                    if value("cavity_depth_m") is not None
                    else "—"
                ),
                str(
                    predicates.get("ELONGATED_OBJECT", {}).get(
                        "status", "UNKNOWN"
                    )
                ),
                str(
                    predicates.get("OPEN_CAVITY", {}).get(
                        "status", "UNKNOWN"
                    )
                ),
                str(record.get("measurement_cloud_path")),
            ]
        )
    return rows


def _relation_rows(
    run_dir: Path, stages: list[dict[str, Any]]
) -> list[list[str]]:
    rows = []
    seen: set[tuple[str, str, str]] = set()
    for stage_record in stages:
        stage = int(stage_record["stage"])
        stage_dir = next(
            path
            for number, path in _stage_dirs(run_dir).items()
            if number == stage
        )
        comparison = _load(stage_dir / "policy_mode_comparison.json")
        result = comparison["modes"]["function-aware"]
        for group in result["function_group_evaluations"]:
            for edge in group["candidate_target_evaluations"]:
                key_base = (
                    edge["utensil_object_id"],
                    edge["target_object_id"],
                )
                for relation in edge["relation_checks"]:
                    key = (*key_base, relation["relation"])
                    if key in seen:
                        continue
                    seen.add(key)
                    evidence = relation.get("evidence", {})
                    rows.append(
                        [
                            str(stage),
                            group["function_group_id"],
                            edge["utensil_object_id"],
                            edge["target_object_id"],
                            relation["relation"],
                            relation["status"],
                            (
                                f"{float(evidence['pass_margin_m']):.4f}"
                                if evidence.get("pass_margin_m") is not None
                                else "—"
                            ),
                            (
                                f"{float(evidence['maximum_cross_section_m']):.4f}"
                                if evidence.get("maximum_cross_section_m")
                                is not None
                                else "—"
                            ),
                            (
                                f"{float(evidence['opening_width_m']):.4f}"
                                if evidence.get("opening_width_m") is not None
                                else "—"
                            ),
                            (
                                f"{float(evidence['usable_length_m']):.4f}"
                                if evidence.get("usable_length_m") is not None
                                else "—"
                            ),
                            (
                                f"{float(evidence['cavity_depth_m']):.4f}"
                                if evidence.get("cavity_depth_m") is not None
                                else "—"
                            ),
                        ]
                    )
    return rows


def _write_report(
    *,
    output_dir: Path,
    run_dir: Path,
    offline: dict[str, Any],
    visualizations: dict[str, Any],
    stage_components: list[dict[str, Any]],
    comparison: str,
) -> None:
    run_config = _load(run_dir / "run_config.json")
    registry = _load(run_dir / "object_registry.json")
    stages = _stage_records(run_dir)
    object_rows = _object_rows(registry)
    relation_rows = _relation_rows(run_dir, stages)
    outcome_rows = []
    for mode in MODES:
        item = offline["modes"][mode]
        outcome_rows.append(
            [
                MODE_TITLES[mode],
                item["actual_status"],
                str(item["completion_stage"]),
                str(item["policy_required_distinct_physical_tool_count"]),
                str(item["distinct_physical_tool_count"]),
                "YES" if item["matches_expected_result"] else "NO",
            ]
        )

    css = """
*{box-sizing:border-box}body{margin:0;background:#eef2f7;color:#111827;font-family:Inter,system-ui,sans-serif;line-height:1.55}
nav{position:sticky;top:0;z-index:10;background:#0f172aee;padding:12px 25px;display:flex;gap:20px}nav a{color:#e2e8f0;text-decoration:none;font-weight:800}
main{max-width:1500px;margin:auto;padding:28px}section{background:white;border-radius:18px;padding:30px;margin:23px 0;box-shadow:0 5px 18px #0f172a13}
h1{font-size:44px;line-height:1.1;margin:5px 0 15px}h2{font-size:30px;margin-top:0}.lede{font-size:20px;color:#334155;max-width:1100px}
.pill{display:inline-block;background:#e2e8f0;padding:7px 12px;border-radius:999px;margin:4px;font-weight:800}.ok{background:#dcfce7;color:#166534}
.hero,.wide{width:100%;height:auto;border:1px solid #dbe3ef;border-radius:12px;display:block}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:10px;border:1px solid #dbe3ef;text-align:left;vertical-align:top}th{background:#e9eef5}
.claim{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.card{border:1px solid #dbe3ef;border-radius:13px;padding:18px;background:#f8fafc}
.mode{border-left:8px solid var(--accent);padding:22px;border-radius:14px;background:#fbfdff;border-top:1px solid #dbe3ef;border-right:1px solid #dbe3ef;border-bottom:1px solid #dbe3ef;margin:22px 0}
.mode-grid{display:grid;grid-template-columns:1.4fr .8fr;gap:20px}video{width:100%;border-radius:12px;background:#111827}.decision{border:1px solid #dbe3ef;border-radius:12px;padding:18px;background:white}
details{border:1px solid #dbe3ef;border-radius:12px;margin:14px 0;overflow:hidden}summary{padding:14px 17px;background:#f8fafc;font-weight:800;cursor:pointer}.inside{padding:17px}
.stage{display:grid;grid-template-columns:2fr 1fr;gap:16px}.side{display:grid;gap:14px}.cams{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}.cams img{width:100%}
code{background:#eef2ff;padding:2px 5px;border-radius:4px}.small{font-size:13px;color:#64748b}
@media(max-width:950px){.claim,.mode-grid,.stage{grid-template-columns:1fr}.cams{grid-template-columns:1fr}main{padding:14px}h1{font-size:34px}}
"""
    mode_html = ""
    for mode in MODES:
        item = offline["modes"][mode]
        viz = visualizations[mode]
        poster = _data_uri(output_dir, viz["frames"][-1])
        if viz["mp4"]:
            media = (
                f'<video controls muted loop playsinline poster="{poster}">'
                f'<source type="video/mp4" '
                f'src="{_data_uri(output_dir, viz["mp4"])}"></video>'
            )
        else:
            media = (
                f'<img class="wide" '
                f'src="{_data_uri(output_dir, viz["gif"])}">'
            )
        mode_html += f"""
<article class="mode" style="--accent:{MODE_COLORS[mode]}">
<h3>{html.escape(MODE_TITLES[mode])}</h3>
<div class="mode-grid"><div>{media}</div><div class="decision">
<b>Status: {html.escape(item['actual_status'])}</b>
<p>Completion stage: {html.escape(str(item['completion_stage']))}</p>
<p>Policy distinct-tool requirement:
{item['policy_required_distinct_physical_tool_count']}</p>
<p>{html.escape(MODE_EXPLANATIONS[mode])}</p>
<p><b>Expected result matched:
{html.escape(str(item['matches_expected_result']))}</b></p></div></div>
<details><summary>Animated GIF</summary><div class="inside">
<img class="wide" src="{_data_uri(output_dir, viz['gif'])}"></div></details>
</article>"""
    stage_html = ""
    for record in stage_components:
        paths = record["paths"]
        side = "".join(
            f'<div><b>{html.escape(label)}</b>'
            f'<img class="wide" src="{_data_uri(output_dir, paths[name])}"></div>'
            for name, label in (
                ("overview.png", "Point cloud + graph"),
                ("pointcloud.png", "Stage-local point cloud"),
                ("graph.png", "Observed graph"),
            )
            if name in paths
        )
        cameras = "".join(
            f'<div><b>{html.escape(camera["camera_id"])}</b>'
            f'<img src="{_data_uri(output_dir, camera["path"])}"></div>'
            for camera in record["camera_overlays"]
        )
        stage_html += f"""
<details {'open' if record['stage'] == 0 else ''}>
<summary>Stage {record['stage']:03d} · {html.escape(record['name'])}</summary>
<div class="inside"><div class="stage"><div>
<img class="wide" src="{_data_uri(output_dir, paths['semantic_overview.png'])}">
</div><div class="side">{side}</div></div>
<details><summary>All five detector overlays</summary>
<div class="inside cams">{cameras}</div></details></div></details>"""

    detector = run_config["semantic_detector"]
    document = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ablation 2 · function-aware utensil reuse</title>
<style>{css}</style></head><body><nav>
<a href="#summary">Summary</a><a href="#modes">Policy modes</a>
<a href="#objects">Objects</a><a href="#relations">Relations</a>
<a href="#stages">Scene audit</a><a href="#provenance">Provenance</a>
</nav><main>
<section id="summary"><div class="small">PRESENTATION REPORT · ABLATION 2</div>
<h1>Function-aware utensil cardinality and reuse</h1>
<p class="lede">Raw object count is not functional satisfaction. One valid
utensil can be reused sequentially across two coffee cups, while two soup
bowls need two dedicated utensil IDs. Spoon and fork are both acceptable
after semantic and geometric validation. Reuse belongs to the operation
group—not permanently to an object category.</p>
<span class="pill">Scene: {html.escape(run_config['scene_name'])}</span>
<span class="pill">Run: {html.escape(run_dir.name)}</span>
<span class="pill ok">Expected results matched:
{offline['all_expected_results_matched']}</span>
<img class="hero" src="{_data_uri(output_dir, comparison)}">
<div class="claim"><div class="card"><b>Always reusable</b><br>
False positive at INITIAL: one utensil is illegally duplicated across dedicated
soup slots.</div><div class="card"><b>Always distinct</b><br>
False negative: four different utensils are demanded even though coffee permits
sequential reuse.</div><div class="card"><b>Function-aware</b><br>
Correct at D2: four target slots are covered using one spoon and one fork.</div>
</div></section>
<section><h2>Outcome matrix</h2>
{_table(['Mode','Outcome','Completion stage','Policy requires distinct','Assigned distinct','Expected matched'], outcome_rows)}
</section>
<section><h2>What was inferred</h2>
<p>Candidate eligibility always required RGB semantic compatibility,
stage-local unary geometry, and target-specific pairwise geometry. The policy
ablation changed only assignment distinctness. Every mode consumed the same
RGB, detections, associations, semantic fusion, point clouds, geometric
properties, and relation results.</p>
<p>No global <code>REUSABLE_OBJECT</code> property exists. Requirements were
manually declared; no FM generated them. No robot, navigation, IK, grasp,
placement, TAMP, cleaning, or task execution occurred.</p></section>
<section id="modes"><h2>Individual policy visualizations</h2>{mode_html}</section>
<section id="objects"><h2>Observed-object evidence</h2>
<p>Repeated labels remain separate generic IDs. Counts become functional only
after semantic, unary, and pairwise validation.</p>
{_table(['Object','RGB label','Confidence','Views','Region','Points','Cameras','Usable L','Cross-section','Opening','Cavity depth','Elongated','Open cavity','Measurement evidence'], object_rows)}
</section>
<section id="relations"><h2>Numeric candidate–target relations</h2>
{_table(['Stage','Group','Utensil','Target','Relation','Status','Margin m','Cross-section m','Opening m','Usable length m','Cavity depth m'], relation_rows)}
</section>
<section id="stages"><h2>Scene and component audit</h2>{stage_html}</section>
<section id="provenance"><h2>Detector and evidence provenance</h2>
<p>Backend: {html.escape(str(detector['name']))}; checkpoint:
{html.escape(str(detector['checkpoint']))}; version:
{html.escape(str(detector['version']))}; device:
{html.escape(str(detector['device']))}; inference size:
{html.escape(str(detector['inference_size']))}; isolated worker:
{html.escape(str(detector['process_isolation']))}.</p>
<p>Geometry consumed typed, stage-local, region-gated MeasurementEvidence.
Cumulative and combined clouds were visualization-only. Offline evaluation
annotations were not imported by runtime inference.</p></section>
</main></body></html>"""
    (output_dir / "presentation_report.html").write_text(
        document, encoding="utf-8"
    )
    (output_dir / "ablation_report.html").write_text(
        document, encoding="utf-8"
    )

    markdown_rows = "\n".join(
        "| " + " | ".join(row) + " |" for row in outcome_rows
    )
    readme = f"""# Ablation 2: function-aware utensil reuse

This report demonstrates that raw utensil count is not sufficient. Reuse is
declared per function group:

- coffee: one valid spoon or fork may be reused across two target cups;
- soup: each bowl requires a different valid spoon-or-fork object ID;
- cross-group reuse: the coffee utensil may also serve one soup bowl.

All policy modes use identical saved perception evidence.

![Policy comparison]({comparison})

| Mode | Outcome | Completion stage | Policy distinct requirement | Assigned distinct | Expected matched |
|---|---:|---:|---:|---:|---:|
{markdown_rows}

## Animations

- [Always reusable GIF]({visualizations['always-reusable']['gif']}) · [MP4]({visualizations['always-reusable']['mp4']})
- [Always distinct GIF]({visualizations['always-distinct']['gif']}) · [MP4]({visualizations['always-distinct']['mp4']})
- [Function-aware GIF]({visualizations['function-aware']['gif']}) · [MP4]({visualizations['function-aware']['mp4']})

Open `presentation_report.html` for the self-contained presentation with scene
views, overlays, point clouds, graphs, assignments, and numeric margins.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    (output_dir / "ablation_report.md").write_text(
        readme, encoding="utf-8"
    )


def generate_usage_policy_report(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    evaluation_config: str | Path = DEFAULT_EVALUATION_CONFIG,
) -> dict[str, Any]:
    run_path = Path(run_dir).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    offline_path = destination / "offline_policy_ablation_evaluation.json"
    offline = evaluate_saved_usage_policy_run(
        run_path,
        evaluation_config=evaluation_config,
        output_path=offline_path,
    )
    stages = _stage_records(run_path)
    visualizations = _make_visualizations(destination, run_path, stages)
    components = _copy_stage_components(
        run_path, destination, _stage_dirs(run_path)
    )
    comparison = _comparison_image(destination, visualizations)
    report_data = {
        "schema_version": 1,
        "run_directory": str(run_path),
        "offline_evaluation": offline,
        "visualizations": visualizations,
        "stage_components": components,
        "comparison": comparison,
    }
    (destination / "report_data.json").write_text(
        json.dumps(report_data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(
        output_dir=destination,
        run_dir=run_path,
        offline=offline,
        visualizations=visualizations,
        stage_components=components,
        comparison=comparison,
    )
    result = {
        "report_directory": str(destination),
        "html": str(destination / "ablation_report.html"),
        "presentation_html": str(destination / "presentation_report.html"),
        "markdown": str(destination / "README.md"),
        "comparison": str(destination / comparison),
        "all_expected_results_matched": offline[
            "all_expected_results_matched"
        ],
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the Ablation 2 usage-policy report"
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--evaluation-config",
        type=Path,
        default=DEFAULT_EVALUATION_CONFIG,
    )
    arguments = parser.parse_args()
    generate_usage_policy_report(
        arguments.run_dir,
        arguments.output_dir,
        evaluation_config=arguments.evaluation_config,
    )


if __name__ == "__main__":
    main()
