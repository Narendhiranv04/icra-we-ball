# Expected GT action catalogue

These are the authoritative high-level GT task actions expected before physical execution.
Every final-paper run saves the physically executed sequence beside this plan and reports an exact comparison.

| Variant | Environment | Outcome | What it does | Actions | File |
| --- | --- | --- | --- | ---: | --- |
| `K1` | kitchen | FEASIBLE | All required objects are initially visible. | 24 | [kitchen/K1/expected_gt_actions.txt](kitchen/K1/expected_gt_actions.txt) |
| `K2` | kitchen | FEASIBLE | The second coffee vessel is in cupboard C2. | 27 | [kitchen/K2/expected_gt_actions.txt](kitchen/K2/expected_gt_actions.txt) |
| `K3` | kitchen | FEASIBLE | The second soup bowl is in box B1. | 25 | [kitchen/K3/expected_gt_actions.txt](kitchen/K3/expected_gt_actions.txt) |
| `K4` | kitchen | FEASIBLE | The second coffee vessel is in C2 and the second soup bowl is in B1. | 28 | [kitchen/K4/expected_gt_actions.txt](kitchen/K4/expected_gt_actions.txt) |
| `K5` | kitchen | FEASIBLE | The coffee stirrer and one soup utensil are in drawers D1 and D2. | 26 | [kitchen/K5/expected_gt_actions.txt](kitchen/K5/expected_gt_actions.txt) |
| `K6` | kitchen | FEASIBLE | One required object is distributed through each storage region. | 31 | [kitchen/K6/expected_gt_actions.txt](kitchen/K6/expected_gt_actions.txt) |
| `K7` | kitchen | INFEASIBLE | The second coffee vessel is absent; otherwise identical to F3. | 6 | [kitchen/K7/expected_gt_actions.txt](kitchen/K7/expected_gt_actions.txt) |
| `K8` | kitchen | INFEASIBLE | The second soup bowl is absent; otherwise identical to F3. | 6 | [kitchen/K8/expected_gt_actions.txt](kitchen/K8/expected_gt_actions.txt) |
| `K9` | kitchen | INFEASIBLE | The reusable coffee stirrer is absent; otherwise identical to F5. | 6 | [kitchen/K9/expected_gt_actions.txt](kitchen/K9/expected_gt_actions.txt) |
| `K10` | kitchen | INFEASIBLE | One of the two required soup utensils is absent; otherwise identical to F5. | 6 | [kitchen/K10/expected_gt_actions.txt](kitchen/K10/expected_gt_actions.txt) |
| `K11` | kitchen | INFEASIBLE | The kettle is absent; otherwise identical to F5. | 6 | [kitchen/K11/expected_gt_actions.txt](kitchen/K11/expected_gt_actions.txt) |
| `K12` | kitchen | INFEASIBLE | The coffee jar is absent; otherwise identical to F5. | 6 | [kitchen/K12/expected_gt_actions.txt](kitchen/K12/expected_gt_actions.txt) |
| `L1` | living_room | FEASIBLE | All five objects begin in the staging area. | 10 | [living_room/L1/expected_gt_actions.txt](living_room/L1/expected_gt_actions.txt) |
| `L2` | living_room | FEASIBLE | The left saucer is already correctly placed on the left personal table. | 8 | [living_room/L2/expected_gt_actions.txt](living_room/L2/expected_gt_actions.txt) |
| `L3` | living_room | FEASIBLE | The left saucer begins on the shared table and must be moved to a personal table. | 10 | [living_room/L3/expected_gt_actions.txt](living_room/L3/expected_gt_actions.txt) |
| `L4` | living_room | FEASIBLE | The left cup begins on the shared table and must be moved to a personal table. | 10 | [living_room/L4/expected_gt_actions.txt](living_room/L4/expected_gt_actions.txt) |
| `L5` | living_room | FEASIBLE | The left saucer is correctly preplaced while the left cup begins on the shared table. | 8 | [living_room/L5/expected_gt_actions.txt](living_room/L5/expected_gt_actions.txt) |
| `L6` | living_room | FEASIBLE | The left cup and left saucer both begin on the shared table and must be moved. | 10 | [living_room/L6/expected_gt_actions.txt](living_room/L6/expected_gt_actions.txt) |
| `L7` | living_room | INFEASIBLE | The required shared table is absent. Grounding termination: FUNCTIONAL_WITNESS_NOT_COMPLETE. | 1 | [living_room/L7/expected_gt_actions.txt](living_room/L7/expected_gt_actions.txt) |
| `L8` | living_room | INFEASIBLE | One of the two required personal tables is absent. Grounding termination: FUNCTIONAL_WITNESS_NOT_COMPLETE. | 1 | [living_room/L8/expected_gt_actions.txt](living_room/L8/expected_gt_actions.txt) |
| `L9` | living_room | INFEASIBLE | Both required personal tables are absent. Grounding termination: FUNCTIONAL_WITNESS_NOT_COMPLETE. | 1 | [living_room/L9/expected_gt_actions.txt](living_room/L9/expected_gt_actions.txt) |
| `L10` | living_room | INFEASIBLE | All three placement tables are absent. Grounding termination: FUNCTIONAL_WITNESS_NOT_COMPLETE. | 1 | [living_room/L10/expected_gt_actions.txt](living_room/L10/expected_gt_actions.txt) |
| `W1` | workshop | FEASIBLE | Manual driver and screw are both in the first region; search stops after one inspection. | 6 | [workshop/W1/expected_gt_actions.txt](workshop/W1/expected_gt_actions.txt) |
| `W2` | workshop | FEASIBLE | Power driver and screw are both in the first region; search stops after one inspection. | 6 | [workshop/W2/expected_gt_actions.txt](workshop/W2/expected_gt_actions.txt) |
| `W3` | workshop | FEASIBLE | Screw in the left drawer; the first driver found is the manual driver in the right drawer. | 7 | [workshop/W3/expected_gt_actions.txt](workshop/W3/expected_gt_actions.txt) |
| `W4` | workshop | FEASIBLE | Screw in the left drawer; the first driver found is the power driver in the right drawer. | 7 | [workshop/W4/expected_gt_actions.txt](workshop/W4/expected_gt_actions.txt) |
| `W5` | workshop | FEASIBLE | Manual driver is found second; the screw is only found in the third region. | 8 | [workshop/W5/expected_gt_actions.txt](workshop/W5/expected_gt_actions.txt) |
| `W6` | workshop | FEASIBLE | Power driver is found first; the screw is only found in the third region. | 8 | [workshop/W6/expected_gt_actions.txt](workshop/W6/expected_gt_actions.txt) |
| `W7` | workshop | FEASIBLE | Power driver absent; manual driver and screw remain sufficient. | 8 | [workshop/W7/expected_gt_actions.txt](workshop/W7/expected_gt_actions.txt) |
| `W8` | workshop | FEASIBLE | Manual driver absent; power driver and screw remain sufficient. | 8 | [workshop/W8/expected_gt_actions.txt](workshop/W8/expected_gt_actions.txt) |
| `W9` | workshop | INFEASIBLE | Screw and hammer present; both compatible drivers absent. | 3 | [workshop/W9/expected_gt_actions.txt](workshop/W9/expected_gt_actions.txt) |
| `W10` | workshop | INFEASIBLE | Both compatible drivers and hammer present; required screw absent. | 3 | [workshop/W10/expected_gt_actions.txt](workshop/W10/expected_gt_actions.txt) |
