"""Render one labelled five-camera closed-start snapshot per Workshop variant."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .workshop_ground_truth_planner import load_variant_specs, solve_gt_assignment
from .workshop_scene import WORKSHOP_CAMERAS, WORKSHOP_REGIONS, WorkshopScene


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "workshop_variant_visualizations"


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def render_variant(
    variant_id: str,
    output_root: Path,
    width: int,
    height: int,
    *,
    open_storage: bool = False,
) -> dict:
    spec = load_variant_specs()[variant_id]
    assignment = solve_gt_assignment(variant_id)
    scene = WorkshopScene(robot="none", variant=variant_id)
    # A catalogue image is also used as the scene-start reference.  Storage
    # must therefore remain closed unless an explicitly requested inspection
    # preview is being rendered.
    if open_storage:
        for region in WORKSHOP_REGIONS:
            scene.open_container(region, steps=700)

    labels = ("LEFT", "RIGHT", "TOP", "FRONT", "CLOSE")
    frames = []
    for camera, label in zip(WORKSHOP_CAMERAS, labels, strict=True):
        frame = Image.fromarray(scene.render_frame(camera, width=width, height=height))
        draw = ImageDraw.Draw(frame)
        draw.rounded_rectangle((8, 8, 110, 40), radius=5, fill=(8, 18, 28, 205))
        draw.text((18, 13), label, font=_font(18, True), fill=(245, 248, 252))
        frames.append(frame)

    panel = Image.new("RGB", (width, height), (13, 22, 32))
    draw = ImageDraw.Draw(panel)
    outcome = spec["intended_outcome"]
    accent = (50, 205, 125) if outcome == "FEASIBLE" else (245, 92, 92)
    y = 18
    draw.text((18, y), variant_id, font=_font(19, True), fill=(245, 248, 252)); y += 32
    draw.text((18, y), outcome, font=_font(17, True), fill=accent); y += 30
    draw.text((18, y), f"Inspect: {len(spec['expected_inspection_regions'])} region(s)",
              font=_font(15), fill=(188, 205, 220)); y += 28
    if assignment.driver:
        mode = "POWER" if assignment.driver == "workshop_power_driver" else "MANUAL"
        draw.text((18, y), f"Selected: {mode} driver", font=_font(15), fill=(255, 211, 94)); y += 28
    else:
        draw.text((18, y), f"Reason: {assignment.rejection_reason}",
                  font=_font(14), fill=accent); y += 28
    for region in WORKSHOP_REGIONS:
        short = {
            "workshop_long_phillips_driver": "manual driver",
            "workshop_power_driver": "power driver",
            "workshop_medium_phillips_screw": "Phillips screw",
            "workshop_wooden_hammer": "hammer",
        }
        contents = ", ".join(short[name] for name in spec["storage_contents"][region]) or "empty"
        draw.text((18, y), region.replace("_", " "), font=_font(13, True), fill=(132, 174, 210)); y += 21
        draw.text((28, y), contents, font=_font(13), fill=(225, 231, 237)); y += 25
    frames.append(panel)

    mosaic = Image.new("RGB", (width * 3, height * 2), (0, 0, 0))
    for index, frame in enumerate(frames):
        mosaic.paste(frame, ((index % 3) * width, (index // 3) * height))
    variant_dir = output_root / variant_id
    variant_dir.mkdir(parents=True, exist_ok=True)
    image_path = variant_dir / "five_camera_open_storage.png"
    mosaic.save(image_path, optimize=True)
    return {
        "variant_id": variant_id,
        "intended_outcome": outcome,
        "expected_inspection_regions": spec["expected_inspection_regions"],
        "storage_contents": spec["storage_contents"],
        "storage_state": "OPEN" if open_storage else "CLOSED",
        "selected_driver": assignment.driver,
        "rejection_reason": assignment.rejection_reason,
        "image": str(image_path.relative_to(ROOT)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=270)
    parser.add_argument(
        "--open-storage",
        action="store_true",
        help="Render an explicit inspection preview with storage opened.",
    )
    args = parser.parse_args()
    records = []
    for variant_id in load_variant_specs():
        print(f"Rendering {variant_id}", flush=True)
        records.append(
            render_variant(
                variant_id,
                args.output_root,
                args.width,
                args.height,
                open_storage=args.open_storage,
            )
        )
    manifest = {
        "schema_version": 1,
        "camera_count": 5,
        "snapshot_storage_state": "OPEN" if args.open_storage else "CLOSED",
        "variants": records,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
