# L2 living-room Region Ablation 1

Goal: **Place the loaded refreshment tray on a stable serving surface within easy reach of both people.**

Kitchen grounding searches for functional objects. This report searches for a
functional destination region. A large planar patch is not automatically a
valid serving region, and a semantically suitable table is not automatically
large enough for the measured payload.

| Mode | Primary selected region | Stage | Correct? |
|---|---|---:|---|
| Geometry-only | `region_0001` (rug) | 0 | No |
| Semantic-only | `region_0002` (small side table) | 1 | No |
| Joint | `region_0003` (coffee table) | 2 | Yes |

| Region | YOLO parent | support L (m) | support W (m) | fit margin (m) | semantic role | FITS_ON | joint |
|---|---|---:|---:|---:|---|---|---|
| region_0001 | sofa | 1.340 | 0.442 | +0.032 | FALSE | TRUE | FALSE |
| region_0002 | side_table | 0.317 | 0.242 | -0.168 | TRUE | FALSE | FALSE |
| region_0003 | side_table | 0.701 | 0.535 | +0.125 | TRUE | TRUE | TRUE |

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
