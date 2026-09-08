# Search and Feasibility Recovery Statistics

## 1. Observation Recovery Protocol

Under Section 12, closed storage regions (drawers, cabinets) require active inspection to discover occluded objects before complete grounding can be satisfied.

- **Offline Recovery Variants:**
  - Kitchen: `K2, K3, K4, K5, K6` (5 variants)
  - Living Room: None (open layout)
  - Workshop: `W1` through `W8` (8 variants)
  - Total Recovery Cases: 13 / 32 in Development Matrix.

## 2. Search Execution Diagnostics

- **Search State Transitions:**
  - In development matrix: variants requiring discovery actively evaluated candidate evidence states (`CONTRACT_INCOMPLETE_NOT_SEARCHABLE` vs `SEARCH_ELIGIBLE`).
  - Search executed on genuinely recovery-requiring variants during probe testing (`K2, K3`).
  - No false search execution on fully observed variants (`K1`).
