#!/usr/bin/env python3
"""Generate deterministic, furniture-scale movie-night visual assets.

The meshes are project-authored visual geometry.  Scene MJCFs deliberately use
simple analytic collision proxies instead of these multi-part visual meshes.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "movie_night"


class Obj:
    def __init__(self) -> None:
        self.vertices: list[tuple[float, float, float]] = []
        self.uvs: list[tuple[float, float]] = []
        self.faces: list[tuple[int, int, int, int]] = []

    def rounded_box(
        self,
        center: tuple[float, float, float],
        half: tuple[float, float, float],
        exponent: float = 8.0,
        nu: int = 28,
        nv: int = 14,
    ) -> None:
        start = len(self.vertices)
        power = 2.0 / exponent

        def signed_power(value: float) -> float:
            return math.copysign(abs(value) ** power, value)

        for iv in range(nv + 1):
            latitude = -math.pi / 2 + math.pi * iv / nv
            for iu in range(nu):
                longitude = -math.pi + 2 * math.pi * iu / nu
                clat = signed_power(math.cos(latitude))
                self.vertices.append(
                    (
                        center[0] + half[0] * clat * signed_power(math.cos(longitude)),
                        center[1] + half[1] * clat * signed_power(math.sin(longitude)),
                        center[2] + half[2] * signed_power(math.sin(latitude)),
                    )
                )
                self.uvs.append((iu / nu, iv / nv))
        for iv in range(nv):
            for iu in range(nu):
                a = start + iv * nu + iu
                b = start + iv * nu + (iu + 1) % nu
                c = start + (iv + 1) * nu + (iu + 1) % nu
                d = start + (iv + 1) * nu + iu
                self.faces.append((a, b, c, d))

    def write(self, path: Path) -> None:
        lines = ["# Deterministic project-authored movie-night furniture"]
        lines += [f"v {x:.7f} {y:.7f} {z:.7f}" for x, y, z in self.vertices]
        lines += [f"vt {u:.7f} {v:.7f}" for u, v in self.uvs]
        for face in self.faces:
            lines.append(
                "f " + " ".join(f"{index + 1}/{index + 1}" for index in face)
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def furniture() -> dict[str, Obj]:
    assets: dict[str, Obj] = {}

    sofa = Obj()
    sofa.rounded_box((0, 0, 0.22), (1.02, 0.38, 0.20))
    for x in (-0.66, 0.0, 0.66):
        # A residential-depth cushion provides a robust measured support patch
        # for Ablation 1 without relying on configured region dimensions.
        sofa.rounded_box((x, -0.05, 0.50), (0.31, 0.40, 0.10), 5.0)
        sofa.rounded_box((x, 0.27, 0.91), (0.31, 0.11, 0.39), 5.0)
    sofa.rounded_box((-1.08, 0.02, 0.62), (0.10, 0.36, 0.34), 5.0)
    sofa.rounded_box((1.08, 0.02, 0.62), (0.10, 0.36, 0.34), 5.0)
    assets["sofa"] = sofa

    chair = Obj()
    chair.rounded_box((0, 0, 0.23), (0.43, 0.40, 0.21))
    chair.rounded_box((0, -0.05, 0.52), (0.34, 0.30, 0.11), 5.0)
    chair.rounded_box((0, 0.30, 0.91), (0.36, 0.12, 0.39), 5.0)
    chair.rounded_box((-0.45, 0.02, 0.63), (0.09, 0.36, 0.33), 5.0)
    chair.rounded_box((0.45, 0.02, 0.63), (0.09, 0.36, 0.33), 5.0)
    assets["armchair"] = chair

    coffee = Obj()
    coffee.rounded_box((0, 0, 0.44), (0.78, 0.46, 0.055), 10.0)
    for x in (-0.65, 0.65):
        for y in (-0.34, 0.34):
            coffee.rounded_box((x, y, 0.22), (0.055, 0.055, 0.22), 8.0)
    coffee.rounded_box((0, 0, 0.16), (0.64, 0.34, 0.025), 10.0)
    assets["coffee_table"] = coffee

    end = Obj()
    end.rounded_box((0, 0, 0.59), (0.28, 0.24, 0.04), 10.0)
    for x in (-0.21, 0.21):
        for y in (-0.17, 0.17):
            end.rounded_box((x, y, 0.30), (0.035, 0.035, 0.27), 8.0)
    end.rounded_box((0, 0, 0.10), (0.22, 0.18, 0.025), 10.0)
    assets["end_table"] = end

    ctable = Obj()
    ctable.rounded_box((0, 0, 0.64), (0.23, 0.15, 0.035), 10.0)
    ctable.rounded_box((-0.18, 0, 0.33), (0.025, 0.12, 0.30), 8.0)
    ctable.rounded_box((0, 0, 0.035), (0.22, 0.15, 0.025), 10.0)
    assets["c_table"] = ctable

    console = Obj()
    console.rounded_box((0, 0, 0.34), (0.92, 0.25, 0.32), 14.0)
    console.rounded_box((0, -0.255, 0.34), (0.36, 0.012, 0.25), 10.0)
    for x in (-0.70, 0.70):
        console.rounded_box((x, -0.255, 0.34), (0.16, 0.012, 0.25), 10.0)
    assets["media_console"] = console

    shelf = Obj()
    for x in (-0.48, 0.48):
        shelf.rounded_box((x, 0, 0.90), (0.035, 0.18, 0.90), 10.0)
    for z in (0.04, 0.62, 1.20, 1.76):
        shelf.rounded_box((0, 0, z), (0.50, 0.20, 0.035), 10.0)
    assets["bookshelf"] = shelf

    # The upstream GSO catalogue contains no TV-remote scan.  This
    # project-authored, rounded visual prop therefore replaces the old stack
    # of raw MJCF boxes while retaining a separate analytic collision proxy.
    remote = Obj()
    remote.rounded_box((0, 0, 0), (0.105, 0.038, 0.016), 7.0, 32, 16)
    remote.rounded_box((-0.073, 0, 0.018), (0.010, 0.010, 0.004), 4.0, 16, 8)
    for x in (-0.038, -0.005, 0.028, 0.061):
        for y in (-0.016, 0.016):
            remote.rounded_box((x, y, 0.018), (0.008, 0.008, 0.003), 4.0, 12, 6)
    assets["remote_control"] = remote

    return assets


def texture(path: Path, base: tuple[int, int, int], accent: tuple[int, int, int]) -> None:
    image = Image.new("RGB", (512, 512), base)
    draw = ImageDraw.Draw(image)
    for offset in range(-512, 1024, 48):
        draw.line((offset, 0, offset + 512, 512), fill=accent, width=3)
    image.save(path, optimize=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, mesh in furniture().items():
        mesh.write(OUT / f"{name}.obj")
    texture(OUT / "fabric_warm.png", (112, 105, 96), (133, 124, 112))
    texture(OUT / "fabric_chair.png", (93, 111, 116), (111, 129, 133))
    texture(OUT / "wood_walnut.png", (105, 67, 39), (134, 91, 52))
    texture(OUT / "wood_oak.png", (151, 113, 72), (177, 140, 94))
    texture(OUT / "remote_charcoal.png", (35, 38, 43), (67, 72, 78))
    print(f"Generated {len(furniture())} visual meshes in {OUT}")


if __name__ == "__main__":
    main()
