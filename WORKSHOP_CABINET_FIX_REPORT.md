# Workshop cabinet retrieval fix

All W1–W10 direct Workshop executions pass: **64/64 actions**, with terminal validation. Cabinet hammer retrieval also passes independently. No packaging, Kitchen, or Living Room executions were run.

## W5 root cause

Reproduced the original miss at **0.427145 m** (reported previously as 0.4272 m). The final target was `[0.328, 0.504, 0.786392]`; the actual gripper was `[0.742815, 0.508227, 0.888195]`. IK position error was **0.000073 m**, but maximum joint tracking error was **0.337618 rad**.

The trajectory entered laterally along X while already inside the cabinet's Y bounds. The palm hit `cabinet_col_right`, with additional robot/door contact. Both cabinet bodies were admitted by the candidate collision checker. `_animate_configuration` then accepted low-speed contact stalls, and subsequent waypoints continued from planned rather than achieved configurations. This was physical obstruction plus accepted control stall, not a 43 cm IK or coordinate-transform error. Intermediate measured reach errors were 0.140173 m and 0.297948 m. The screw remained at its storage pose.

Full target/joint/contact evidence: [original failure trace](runs/workshop_cabinet_fix/w5_original_failure.jsonl).

## Storage articulation — preserved

Following the user's updated instruction, all experimental opening changes were removed. `_articulate_storage`, `_animate_configuration`, handle closure, drawer OPEN/PICK, and the existing hinge calibration remain unchanged. AST comparison confirmed the opening/control functions match HEAD.

The direct W5 audit measured LEFT_DRAWER **0.24096 m**, RIGHT_DRAWER **0.24100 m**, and TOOL_CABINET **0.96764 rad**, and the mechanisms remained open. Successful OPEN updates `container_open_state` and `opened_containers`; the handle weld is released afterward and the existing servo holds the reached position.

**The retained cabinet OPEN does not attain its nominal 1.45 rad target.** It passes its existing opening contract. This report does not claim full nominal stroke; the user's later instruction explicitly froze opening behavior. Direct and Phase-4 entry points retain their existing hinge configuration behavior.

All audited cabinet objects are independent free bodies with storage welds to **`tool_cabinet`**, never the moving door. No hinge or handle-weld implementation changes remain.

## Cabinet geometry and shared retrieval

Interior X bounds: **[0.233, 0.647] m**; front/back: approximately **[0.466, 0.644] m**; floor top **0.696 m**, shelf top **0.766 m**, roof underside **1.322 m**.

| Variant | Cabinet contents and authored body positions, metres |
|---|---|
| W5 | screw `(0.340,0.535,0.767)`; side-resting power driver `(0.560,0.474,0.721)` |
| W6 | screw `(0.340,0.535,0.767)`; upright long driver `(0.560,0.535,0.767)` |
| W7 | upright long driver `(0.340,0.535,0.767)` |
| W8 | upright power driver `(0.320,0.535,0.767)` |
| W9 | empty |
| W10 | upright hammer `(0.340,0.535,0.767)` |

The old side-resting power-driver pose extended through the back wall. The corrected lower slot fits its 165 mm depth and uses the actual floor top. For selected retrieval, an upright slot exposes the handle below the motor housing. The hammer uses the accessible left lane. Objects retain shelf/floor support and the existing storage fixtures. W1–W4 keep their original storage poses, covered by explicit regression tests.

The W6 long driver needs **no relocation**: the corrected front approach avoids it. Neither its identity nor its collisions are excluded.

All four object families share cabinet navigation, aperture crossing, physical-contact gating, and extraction. Screw grasps use the shaft; drivers and hammer use their handle geometry. Jaw preshape accounts for handle width. The site offset is 32 mm behind the physical pad centre. Entry crosses in front of the door, then moves through the left aperture. The loaded exit accounts for casing width before turning in front of the cabinet. Cabinet extraction uses arm-only payload gravity compensation; no effort or pose writes are applied to the free payload.

W6's safe base stance and open-right-drawer envelope rejection remain intact. The power driver's final return uses the clear left workbench lane when the cabinet is open; normal W1–W4 placement behavior is unchanged.

Detailed per-geom collision/visual bounds, orientations, parents, and corridor coordinates: [geometry audit](runs/workshop_cabinet_fix/geometry.json).

## Validation

| Variant | Result | Actions | First remaining failure |
|---|---|---:|---|
| W1 | PASS | 6/6 | None |
| W2 | PASS | 6/6 | None |
| W3 | PASS | 7/7 | None |
| W4 | PASS | 7/7 | None |
| W5 | PASS | 8/8 | None |
| W6 | PASS | 8/8 | None |
| W7 | PASS | 8/8 | None |
| W8 | PASS | 8/8 | None |
| W9 | NO_COMPATIBLE_DRIVER confirmed | 3/3 | None |
| W10 | NO_COMPATIBLE_SCREW confirmed | 3/3 | None |

Final cabinet preclose errors: W5 **2.911 mm**, W6 **2.911 mm**, W7 **2.903 mm**, W8 **2.414 mm**. Each established bilateral physical contact before attachment and verified extraction clearance. The independent hammer pick passed with **3.116 mm** preclose error and physical contact.

**14 focused tests passed**, including cabinet mesh/collision containment, original W1–W4 cabinet poses, and existing Workshop contract tests. Full per-variant direct execution summaries, action traces, and terminal checks are in [validation](runs/workshop_cabinet_fix/validation). The hammer check is [separate](runs/workshop_cabinet_fix/hammer_retrieval.jsonl); its success is the PICK result, not the infeasibility terminal result.

Execution used `run_workshop_ground_truth_execution.run_variant`, with exceptions visible, no recording or packaging. These are direct Workshop validations, not a new Phase-4 video certification.

## Physical integrity

- No collision suppression added; cabinet walls/door are removed from PICK allowances. Unrelated stored objects remain collision checked.
- No global IK, collision, or preclose thresholds loosened. The legacy 180 mm manual-driver preclose exception is removed.
- Loaded cabinet motion retains the existing effective 0.020 rad joint-tracking tolerance, but cannot pass by contact stall. Preclose uses its stricter tracking path.
- No direct payload pose/qpos writes or teleportation added. Gravity compensation acts on arm DOFs only and is restored after each motion.
- All cabinet pick welds require bilateral contact and snap validation. The benchmark grasp-recovery fallback is disabled for cabinet picks.
- No planner, assignment, inspection-order, robot-profile, or generic collision-policy changes.
- Existing drawer/OPEN benchmark behavior is preserved; these cabinet guarantees do not retroactively certify its legacy assistance paths.

## Files and diff

- `mujoco_scenes/workshop_ground_truth_execution.py`: `execute` cabinet PICK/exit, `_reach` stall control, load effort and diagnostics, `_activate_grasp` bilateral snap gate, `_navigate_robot` cabinet carry transition, `_base_stance` and `_destination_position` open-cabinet return clearance. Preserves the pre-existing safe stance diff.
- `mujoco_scenes/workshop_scene.py`: deterministic cabinet storage corrections through `_get_object_storage_pose`/`build_workshop_xml`, preserving reference layouts.
- `mujoco_scenes/tests/test_workshop_cabinet_storage.py`: geometry containment and frozen reference pose regressions.
- `WORKSHOP_CABINET_FIX_REPORT.md`: this report.
- `runs/workshop_cabinet_fix/`: local, git-ignored diagnostic/geometry/validation artifacts and diagnostic scripts.

No changes remain in `workshop_tool_cabinet_hinge.py`. Other pre-existing or concurrently edited working-tree files were left intact. No commit was created. `git diff --check` passes.
