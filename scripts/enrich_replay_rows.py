#!/usr/bin/env python3
"""Recompute the deterministic front half for every row of a replay artifact.

Two jobs, both offline and both with zero FM calls:

1. verification -- re-derive wire validity, canonical graph emission, operation
   count and role cardinality from the same archived raw response and report any
   row that no longer reproduces, and
2. enrichment -- add the fields this reporting round needs and the original
   harness did not record: ``canonical_graph_emitted`` and
   ``executable_contract_complete`` as separate quantities, plus the count of
   non-scene-resolvable compile blockers.

No ground truth is read here; the offline GT columns already in the rows are
carried through untouched.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_frozen_trial import derive  # noqa: E402


def load_raw(path: Path):
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return json.loads(data["content"]) if isinstance(data.get("content"), str) else data
    except Exception:
        return None


COMPARE = ("strict_v3_valid", "task_valid", "graph_compiled", "n_operation_groups",
           "canonical_roles", "role_counts")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--frozen-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = json.loads(Path(args.rows).read_text())
    root = Path(args.frozen_root)
    mismatches = []
    for row in rows:
        raw_path = root / row["domain"] / row["variant"] / row["trial"] / "raw_v3.json"
        fresh = derive(load_raw(raw_path), row["domain"])
        for key in COMPARE:
            if key in row and row[key] != fresh.get(key):
                mismatches.append((row["variant"], row["trial"], key,
                                   str(row[key])[:90], str(fresh.get(key))[:90]))
        row["canonical_graph_emitted"] = bool(fresh.get("graph_compiled"))
        row["executable_contract_complete"] = bool(fresh.get("executable_contract_complete"))
        row["required_contract_complete"] = bool(fresh.get("contract_complete"))
        row["non_scene_resolvable_blockers"] = list(fresh.get("non_scene_resolvable_blockers") or ())
        row["false_completion"] = bool(row.get("success")) and not bool(row.get("gt_full_task_satisfied"))
    Path(args.out).write_text(json.dumps(rows, indent=2, default=str) + "\n")
    print(f"rows={len(rows)} mismatches={len(mismatches)}")
    for item in mismatches[:60]:
        print("   MISMATCH", item)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
