#!/usr/bin/env python3
"""Compare the deterministic front half of every frozen trial against a baseline.

Offline, zero FM calls, no ground truth: re-derives wire validity, the canonical
graph, its operation groups and role cardinalities from each archived raw
response and prints every trial whose front half moved.  This is the cheap
regression check between full pipeline replays.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_frozen_trial import derive  # noqa: E402

FIELDS = ("strict_v3_valid", "task_valid", "canonical_graph_emitted",
          "executable_contract_complete", "n_operation_groups")


def load_raw(path: Path):
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return json.loads(data["content"]) if isinstance(data.get("content"), str) else data
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--frozen-root", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    base = json.loads(Path(args.baseline).read_text())
    root = Path(args.frozen_root)
    fresh_rows = []
    changed = []
    for row in base:
        fresh = derive(load_raw(root / row["domain"] / row["variant"] / row["trial"] / "raw_v3.json"),
                       row["domain"])
        fresh.update({k: row[k] for k in ("domain", "variant", "trial", "feasible")})
        fresh_rows.append(fresh)
        moved = {f: (row.get(f), fresh.get(f)) for f in FIELDS if row.get(f) != fresh.get(f)}
        roles_before, roles_after = row.get("canonical_roles") or [], fresh.get("canonical_roles") or []
        if roles_before != roles_after:
            moved["roles"] = (roles_before, roles_after)
        counts_before, counts_after = row.get("role_counts") or {}, fresh.get("role_counts") or {}
        if counts_before != counts_after:
            moved["counts"] = ({k: v for k, v in counts_before.items() if counts_after.get(k) != v},
                               {k: v for k, v in counts_after.items() if counts_before.get(k) != v})
        if moved:
            changed.append((row["domain"], row["variant"], row["trial"], moved))
    tally = {f: (sum(1 for r in base if r.get(f)), sum(1 for r in fresh_rows if r.get(f)))
             for f in FIELDS[:4]}
    print("front-half totals (before -> after):")
    for f, (b, a) in tally.items():
        print(f"   {f:34s} {b:3d} -> {a:3d}")
    print(f"   operation groups (sum)             "
          f"{sum(r.get('n_operation_groups', 0) or 0 for r in base):3d} -> "
          f"{sum(r.get('n_operation_groups', 0) or 0 for r in fresh_rows):3d}")
    print(f"\n{len(changed)} trials moved:")
    for dom, var, trial, moved in changed:
        print(f"  {dom[:4]:5s} {var:4s} {trial}")
        for key, value in moved.items():
            print(f"      {key}: {str(value)[:230]}")
    if args.out:
        Path(args.out).write_text(json.dumps(fresh_rows, indent=2, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
