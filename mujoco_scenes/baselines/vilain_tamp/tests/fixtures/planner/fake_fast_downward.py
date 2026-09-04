#!/usr/bin/env python3
"""Deterministic fake used only by Stage-6 subprocess tests."""

import os
from pathlib import Path
import shutil
import sys
import time


arguments = sys.argv[1:]
mode = os.environ.get("FAKE_FD_MODE", "success")
if arguments == ["--version"]:
    print("Fast Downward 24.06")
    raise SystemExit(0)
if "--translate" in arguments:
    if mode == "translator_error":
        print("translate: undeclared predicate", file=sys.stderr)
        raise SystemExit(2)
    destination = Path(arguments[arguments.index("--sas-file") + 1])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("begin_version\n3\nend_version\n", encoding="utf-8")
    print("translate: success")
    raise SystemExit(0)
if "--alias" in arguments:
    if mode == "timeout":
        time.sleep(2)
        raise SystemExit(0)
    if mode == "search_error":
        print("search: failure", file=sys.stderr)
        raise SystemExit(3)
    if mode == "no_plan":
        print("search: completed without a plan")
        raise SystemExit(0)
    else:
        source = Path(os.environ["FAKE_PLAN_FIXTURE"])
        prefix = Path(arguments[arguments.index("--plan-file") + 1])
        shutil.copyfile(source / "sas_plan", prefix)
        shutil.copyfile(source / "sas_plan.1", Path(str(prefix) + ".1"))
    print("search: completed")
    raise SystemExit(0)
raise SystemExit(64)
