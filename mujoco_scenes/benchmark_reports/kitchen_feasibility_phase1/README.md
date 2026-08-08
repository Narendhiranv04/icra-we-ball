# Phase 1 kitchen feasibility freeze

This tracked package summarizes the deterministic no-FM/no-robot RGB-D feasibility benchmark. The exact goal text is shared by all 16 controlled variants. The oracle loads the instantiated MuJoCo model and derives geometry from its actual mesh/primitive geometry; it does not read the observed registry. Production uses fresh region-gated RGB-D measurement evidence and YOLO-World semantics.

The authoritative run is `runs/feasibility_benchmarks/kitchen_feasibility_phase1_freeze2_20260808`. At 1280x960, 14/16 classifications matched. F0_REUSE_ONE and P0_LAYOUT_BASE are false-infeasible because two hidden bowls have low-confidence/conflicting multi-view detector evidence and therefore correctly remain UNKNOWN under the two-view semantic gate. No geometry, oracle, or hidden-location shortcut was used to turn them into matches.

Use `reproduction_command.sh` for a fresh run. Raw PLYs and model weights are intentionally not tracked; each variant directory contains compact JSON/CSV evidence plus compressed initial/terminal overview and semantic images.
