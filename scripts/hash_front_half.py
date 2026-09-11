#!/usr/bin/env python3
"""One hash over the deterministic front half of every archived contract.

Used by the reproducibility audit: run under different PYTHONHASHSEED values,
the hash must not move.  Zero FM calls, no ground truth.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from replay_frozen_trial import derive  # noqa: E402
from probe_frozen_contract import load_raw  # noqa: E402

DOMAINS = {"kitchen": [f"K{i}" for i in range(1, 13)],
           "living_room": [f"L{i}" for i in range(1, 11)],
           "workshop": [f"W{i}" for i in range(1, 11)]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen-root", required=True)
    args = ap.parse_args()
    root = Path(args.frozen_root)
    blobs = []
    for domain, variants in DOMAINS.items():
        for variant in variants:
            for trial in (1, 2, 3):
                path = root / domain / variant / f"trial_{trial:02d}" / "raw_v3.json"
                if not path.exists():
                    continue
                blobs.append(json.dumps(derive(load_raw(path), domain),
                                        sort_keys=True, default=str))
    print(hashlib.sha256("\n".join(blobs).encode()).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
