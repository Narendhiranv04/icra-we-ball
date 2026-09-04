#!/usr/bin/env python3
"""Deterministic fake used only by Stage-6 subprocess tests."""

import os
import sys


if sys.argv[1:] == ["--version"]:
    print("VAL 4.2.09")
    raise SystemExit(0)
if os.environ.get("FAKE_VAL_MODE") == "invalid":
    print("Plan invalid: unmet precondition")
else:
    print("Plan valid")
raise SystemExit(0)
