"""Self-contained report for target-specific Region Ablation 3."""

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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mujoco_scenes.generate_region_ablation2_report import (
    _render_measurements,
)


POLICIES = (
    "target_agnostic_count",
    "greedy_target_specific",
    "global_target_specific",
)
TITLES = {
    "target_agnostic_count": "Target-agnostic count",
    "greedy_target_specific": "Greedy target-specific",
    "global_target_specific": "Global target-specific",
}


def _load(path: Path):
    return json.loads(path.read_text())


def _font(size: int, bold: bool = False):
    try:
        return ImageFont.truetype(
            "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size
        )
    except OSError:
        return ImageFont.load_default()


def _data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def _render_matrix(run: Path, output: Path) -> None:
    rows = _load(run / "region_target_compatibility.json")["rows"]
    regions = sorted({row["region_id"] for row in rows})
    targets = sorted({row["seating_target_id"] for row in rows})
    fig, axis = plt.subplots(figsize=(8.5, 4.4))
    for y, region in enumerate(reversed(regions)):
        for x, target in enumerate(targets):
            row = next(
                item
                for item in rows
                if item["region_id"] == region
                and item["seating_target_id"] == target
            )
            status = row["compatibility_status"]
            color = (
                "#2a9d61"
                if status == "TRUE"
                else "#d64a4a" if status == "FALSE" else "#87909d"
            )
            axis.add_patch(plt.Rectangle((x, y), 1, 1, color=color, ec="white"))
            axis.text(
                x + 0.5,
                y + 0.5,
                f"{status}\nd={row['measured_distance_m']:.3f} m\n"
                f"margin={row['near_seat_margin_m']:+.3f} m",
                ha="center",
                va="center",
                color="white",
                weight="bold",
            )
    axis.set_xticks(np.arange(len(targets)) + 0.5, targets)
    axis.set_yticks(np.arange(len(regions)) + 0.5, reversed(regions))
    axis.set_xlim(0, len(targets))
    axis.set_ylim(0, len(regions))
    axis.tick_params(length=0)
    axis.set_title("Measured region × seating-target compatibility", weight="bold")
    for spine in axis.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def _render_seats(run: Path, output: Path) -> None:
    seats = _load(run / "seating_target_registry.json")["seating_targets"]
    fig, axis = plt.subplots(figsize=(8, 4.3))
    for target_id, record in seats.items():
        centroid = record["centroid_world_m"]
        semantic = record["semantics"]
        axis.scatter(centroid[0], centroid[1], s=1200, color="#3678b8")
        axis.text(
            centroid[0],
            centroid[1],
            f"{target_id}\n{semantic['canonical_label']}\n"
            f"{semantic['confidence']:.2f} · "
            f"{semantic['supporting_view_count']} views",
            ha="center",
            va="center",
            color="white",
            fontsize=8,
            weight="bold",
        )
    axis.set_xlabel("world x (m)")
    axis.set_ylabel("world y (m)")
    axis.set_title("Observed seating-target centroids", weight="bold")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def _render_policy_graph(run: Path, policy: str, output: Path) -> None:
    matrix = _load(run / "region_target_compatibility.json")
    result = _load(run / "policy_evaluations.json")["policies"][policy]
    regions = sorted(matrix["adjacency_by_region"])
    targets = sorted(matrix["adjacency_by_target"])
    selected = {
        (item["region_id"], item["seating_target_id"])
        for item in result["region_target_assignments"]
    }
    fig, axis = plt.subplots(figsize=(9, 5))
    region_y = {
        region: 0.75 - index * 0.5 / max(1, len(regions) - 1)
        for index, region in enumerate(regions)
    }
    target_y = {
        target: 0.75 - index * 0.5 / max(1, len(targets) - 1)
        for index, target in enumerate(targets)
    }
    for region, y in region_y.items():
        axis.scatter(0.22, y, s=3200, color="#238b76", zorder=3)
        axis.text(0.22, y, region, ha="center", va="center", color="white", weight="bold", fontsize=7)
    for target, y in target_y.items():
        axis.scatter(0.78, y, s=3200, color="#3678b8", zorder=3)
        axis.text(0.78, y, target, ha="center", va="center", color="white", weight="bold", fontsize=7)
    for row in matrix["rows"]:
        pair = (row["region_id"], row["seating_target_id"])
        compatible = row["compatibility_status"] == "TRUE"
        axis.plot(
            [0.25, 0.75],
            [region_y[row["region_id"]], target_y[row["seating_target_id"]]],
            color="#16a34a" if compatible else "#cbd5e1",
            linewidth=5 if pair in selected else 1.4,
            linestyle="-" if compatible else "--",
            alpha=1 if pair in selected else 0.75,
            zorder=1,
        )
    for target in result["uncovered_target_ids"]:
        axis.text(
            0.78,
            target_y[target] - 0.10,
            "UNCOVERED",
            ha="center",
            color="#c43d3d",
            weight="bold",
        )
    axis.set_title(
        f"{TITLES[policy]} · {result['status']} · {result['classification']}\n"
        f"matching cardinality {result['maximum_matching_cardinality']} / "
        f"{result['target_count']}",
        weight="bold",
    )
    axis.set_xlim(0, 1)
    axis.set_ylim(0.05, 0.95)
    axis.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def _animation(report: Path, paths: list[Path]) -> dict[str, Any]:
    images = [Image.open(path).convert("RGB") for path in paths]
    width, height = max(i.width for i in images), max(i.height for i in images)
    frames = []
    for index, image in enumerate(images):
        frame = Image.new("RGB", (width, height + 58), "#0f172a")
        frame.paste(image, ((width - image.width) // 2, 58))
        ImageDraw.Draw(frame).text(
            (20, 14),
            f"Policy comparison frame {index + 1}/{len(images)}",
            fill="white",
            font=_font(24, True),
        )
        frames.append(frame)
    gif = report / "policy_ablation_comparison.gif"
    frames[0].save(
        gif,
        save_all=True,
        append_images=frames[1:],
        duration=1700,
        loop=0,
        optimize=True,
    )
    frame_dir = report / ".frames"
    frame_dir.mkdir()
    for index, frame in enumerate(frames):
        frame.save(frame_dir / f"frame_{index:03d}.png")
    mp4 = report / "policy_ablation_comparison.mp4"
    completed = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            "1/2",
            "-i",
            str(frame_dir / "frame_%03d.png"),
            "-vf",
            "format=yuv420p",
            "-r",
            "30",
            str(mp4),
        ],
        capture_output=True,
        text=True,
    )
    shutil.rmtree(frame_dir)
    return {"gif": gif.name, "mp4": mp4.name if completed.returncode == 0 else None}


def generate_report(
    primary: str | Path,
    matching: str | Path,
    valid: str | Path,
    permuted: str | Path,
    report_dir: str | Path,
) -> dict[str, Any]:
    runs = {
        "primary": Path(primary).resolve(),
        "matching_trap": Path(matching).resolve(),
        "valid": Path(valid).resolve(),
        "permuted": Path(permuted).resolve(),
    }
    report = Path(report_dir).resolve()
    if report.exists():
        raise RuntimeError(f"Report directory already exists: {report}")
    report.mkdir(parents=True)
    primary_run = runs["primary"]
    copy_names = (
        "run_config.json",
        "task_requirements.json",
        "payload_registry.json",
        "seating_target_registry.json",
        "region_registry.json",
        "observed_graph.json",
        "events.jsonl",
        "evidence_manifest.json",
        "region_target_compatibility.json",
        "region_target_compatibility.csv",
        "policy_evaluations.json",
        "target_agnostic_count_result.json",
        "greedy_target_specific_result.json",
        "global_target_specific_result.json",
        "matching_diagnostics.json",
        "region_assignments.json",
        "offline_region_target_ablation_evaluation.json",
        "region_ablation3_summary.json",
    )
    for name in copy_names:
        shutil.copy2(primary_run / name, report / name)
    for source, name in (
        ("observation/initial_scene_overview.png", "initial_scene_overview.png"),
        ("observation/semantic_overview.png", "semantic_overview.png"),
        ("observation/region_masks_overview.png", "region_masks_overview.png"),
    ):
        shutil.copy2(primary_run / source, report / name)
    camera_root = report / "cameras"
    for camera in ("inspection_left", "inspection_right", "inspection_top", "inspection_front", "inspection_close"):
        camera_root.joinpath(camera).mkdir(parents=True)
        shutil.copy2(
            primary_run / "observation/semantics/cameras" / camera / "association_overlay.png",
            camera_root / camera / "association_overlay.png",
        )
    _render_measurements(
        primary_run,
        _load(primary_run / "payload_registry.json"),
        kind="payload",
        output=report / "payload_measurements.png",
    )
    _render_measurements(
        primary_run,
        _load(primary_run / "region_registry.json"),
        kind="region",
        output=report / "region_measurements.png",
    )
    _render_seats(primary_run, report / "seating_target_measurements.png")
    _render_matrix(primary_run, report / "region_target_compatibility_matrix.png")
    graph_paths = []
    filenames = {
        "target_agnostic_count": "target_agnostic_count_graph.png",
        "greedy_target_specific": "greedy_target_specific_graph.png",
        "global_target_specific": "global_target_specific_graph.png",
    }
    for policy in POLICIES:
        path = report / filenames[policy]
        _render_policy_graph(primary_run, policy, path)
        graph_paths.append(path)
    variants = report / "variants"
    variants.mkdir()
    for variant, run in runs.items():
        summary = _load(run / "region_ablation3_summary.json")
        (variants / f"{variant}_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        for policy in ("greedy_target_specific", "global_target_specific"):
            path = variants / f"{variant}_{policy}.png"
            _render_policy_graph(run, policy, path)
            if variant == "matching_trap":
                graph_paths.append(path)
        handoff = run / "verified_region_allocation_handoff.json"
        if handoff.exists():
            shutil.copy2(handoff, variants / f"{variant}_verified_handoff.json")
    animation = _animation(report, graph_paths)
    summaries = {
        name: _load(run / "policy_evaluations.json")["policies"]
        for name, run in runs.items()
    }
    run_config = _load(primary_run / "run_config.json")
    rows = []
    for scene, policy in (
        ("primary", "target_agnostic_count"),
        ("primary", "global_target_specific"),
        ("matching_trap", "greedy_target_specific"),
        ("matching_trap", "global_target_specific"),
        ("valid", "global_target_specific"),
        ("permuted", "global_target_specific"),
    ):
        value = summaries[scene][policy]
        rows.append(
            f"<tr><td>{scene}</td><td>{TITLES[policy]}</td>"
            f"<td>{value['status']}</td><td>{value['classification']}</td>"
            f"<td>{value['maximum_matching_cardinality']} / {value['target_count']}</td></tr>"
        )
    images = [
        "policy_ablation_comparison.gif",
        "initial_scene_overview.png",
        "semantic_overview.png",
        "payload_measurements.png",
        "seating_target_measurements.png",
        "region_measurements.png",
        "region_target_compatibility_matrix.png",
        *filenames.values(),
        "variants/matching_trap_greedy_target_specific.png",
        "variants/matching_trap_global_target_specific.png",
    ]
    image_html = "".join(
        f"<figure><img src='{_data_uri(report / name)}'><figcaption>{html.escape(name)}</figcaption></figure>"
        for name in images
    )
    camera_html = "".join(
        f"<figure><img src='{_data_uri(report / 'cameras' / camera / 'association_overlay.png')}'><figcaption>{camera}</figcaption></figure>"
        for camera in ("inspection_left", "inspection_right", "inspection_top", "inspection_front", "inspection_close")
    )
    css = "*{box-sizing:border-box}body{margin:0;background:#eef2f7;color:#111827;font-family:system-ui;line-height:1.5}main{max-width:1500px;margin:auto;padding:22px}section{background:white;border-radius:16px;padding:26px;margin:18px 0;box-shadow:0 4px 18px #0001}.hero{background:#13203a;color:white}h1{font-size:40px}img{width:100%;height:auto;border:1px solid #d7dfeb;border-radius:9px}figure{margin:15px 0}figcaption{font-weight:800}table{width:100%;border-collapse:collapse}th,td{border:1px solid #d7dfeb;padding:9px;text-align:left}th{background:#e8eef6}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}code{background:#e8eef6;padding:2px 5px}@media(max-width:900px){.grid{grid-template-columns:1fr}main{padding:8px}}"
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Region Ablation 3</title><style>{css}</style></head><body><main>
<section class="hero"><h1>Target-specific region allocation</h1><p><b>Goal:</b> {html.escape(run_config['natural_language_goal'])}</p><p>One initial five-view RGB-D observation · actual YOLO-World · identical evidence for every policy · no region discovery, FM, navigation, placement, or TAMP.</p></section>
<section><h2>Scientific result</h2><p>Raw suitable-region count does not prove target coverage. The primary scene has two valid side tables but both serve one seat, so count-only is a false positive. The matching trap proves that target relations plus greedy allocation are still insufficient: global matching must reassign the flexible table to cover both targets.</p><table><thead><tr><th>Scene</th><th>Policy</th><th>Status</th><th>Classification</th><th>Matching</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>
<section><h2>Actual evidence and policy visualizations</h2>{image_html}<p><a href="policy_ablation_comparison.gif">GIF</a> · <a href="policy_ablation_comparison.mp4">MP4</a> · <a href="region_target_compatibility.json">matrix JSON</a> · <a href="region_target_compatibility.csv">matrix CSV</a></p></section>
<section><h2>Five actual semantic views</h2><div class="grid">{camera_html}</div></section>
<section><h2>Research boundary</h2><p>All side-table dimensions, payload footprints, seating centroids, distances, and margins come from fresh typed point-cloud evidence. Configured volumes select pixels only. Runtime allocation does not consume body names, geom names, intended dimensions, evaluation mappings, or expected persistent IDs. Requirements remain manual; the output is allocation evidence for future TAMP, not execution.</p></section>
</main></body></html>"""
    (report / "presentation_report.html").write_text(page, encoding="utf-8")
    (report / "README.md").write_text(
        "# Living-room Region Ablation 3\n\nOpen `presentation_report.html`."
    )
    data = {
        "schema_version": 1,
        "runs": {key: str(value) for key, value in runs.items()},
        "animation": animation,
        "self_contained_html": True,
        "policy_results": summaries,
    }
    (report / "report_data.json").write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n"
    )
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("primary")
    parser.add_argument("--matching-trap", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument("--permuted", required=True)
    parser.add_argument("--report-dir", required=True)
    args = parser.parse_args()
    data = generate_report(
        args.primary,
        args.matching_trap,
        args.valid,
        args.permuted,
        args.report_dir,
    )
    print(json.dumps({"report": data["runs"], "output": args.report_dir}, indent=2))


if __name__ == "__main__":
    main()
