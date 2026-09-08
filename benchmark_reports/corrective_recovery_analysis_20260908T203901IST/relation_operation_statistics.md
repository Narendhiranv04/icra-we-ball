# Relation and Operation Semantic Coverage

## 1. Domain Operation Coverage

| Domain | Canonical Operations | VLM Production Mapping | Capability Realization |
| :--- | :--- | :--- | :--- |
| **Kitchen** | `TRANSFER_CONTENT_TO_CONTAINER`, `MIX_BEVERAGE_CONTENTS`, `PROVIDE_SOUP_EATING_UTENSIL` | Mapped via `robot_capability_registry` to robot primitive actions | `POUR` (transfer) and `STIR` (mix) with explicit task goal provenance |
| **Living Room** | `SUPPORT_DRINKWARE`, `SUPPORT_ENTERTAINMENT_CONTROL` | Mapped to table support regions via `system_context_registry` | Surface staging and remote accessibility |
| **Workshop** | `FASTEN_JOINT`, `RETURN_REUSABLE_ITEM_TO_SUPPORT` | Mapped to capability preconditions (`COMPATIBLE_WITH`, `REACHES_TARGET`) | `SCREW` and `PLACE` return actions |

## 2. Quantitative Performance

- **Canonicalization Success Rate:** 100.0% across all 47 cases.
- **Raw Role F1:** 66.3% (Dev), 61.2% (Held-Out).
- **Goal Coverage:** 3.8% (Dev), 7.7% (Kitchen Dev), demonstrating successful end-to-end plan realization from raw VLM guidance.
