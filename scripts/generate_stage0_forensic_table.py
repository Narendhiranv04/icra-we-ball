import json
import os

target_variants = ["K1", "K2", "K3", "L1", "L6", "W1", "W2", "W8"]
base = "benchmark_reports/final_corrected_32x1_20260908T192500IST_rerun"

entries = []

for v in target_variants:
    domain = "kitchen" if v.startswith("K") else ("living_room" if v.startswith("L") else "workshop")
    v_dir = os.path.join(base, domain, v, "vlm")
    if not os.path.exists(v_dir):
        print(f"Missing {v_dir}")
        continue

    with open(os.path.join(v_dir, "evaluation_record.json")) as f:
        rec = json.load(f)
    with open(os.path.join(v_dir, "fm_diagnostics", "fm_call_001.json")) as f:
        diag = json.load(f)
    with open(os.path.join(v_dir, "raw_semantic_evaluation.json")) as f:
        raw_eval = json.load(f)
    with open(os.path.join(v_dir, "result.json")) as f:
        res = json.load(f)

    try:
        raw_content = json.loads(diag.get("content", "{}"))
    except Exception:
        raw_content = {}

    tc = raw_content.get("task_contract", {})
    raw_roles = tc.get("functional_roles", [])
    raw_ops = tc.get("operation_pairings", [])
    raw_rels = tc.get("functional_relations", [])

    # Forensic interpretation
    manual_interp = ""
    if v == "K1":
        manual_interp = "Inference token runout / tab loop caused JSON truncation ('finish_reason: length'). Structural parsing failure upstream of compiler."
    elif v == "K2":
        manual_interp = "FM expressed complete kitchen task (sources, cups, bowls, stirrers, spoons, transfer ops, stir op, pair op). Compiler failed on soup_serving_vessel ('UNRESOLVED_SEMANTIC') and stirring_utensil ('AMBIGUOUS_ROLE_MAPPING collision with coffee_serving_vessel'). Missing transfer capability marked operations UNSUPPORTED_OPERATOR. Search blocked by contract_incomplete gate."
    elif v == "K3":
        manual_interp = "FM expressed valid roles/relations/ops. Compiler suffered role collision/unresolved mapping. Missing material-transfer capability. Search starved by contract gate even though physical objects were unobserved."
    elif v == "L1":
        manual_interp = "FM expressed refreshment and media roles and placement operations. Compiler rejected refreshment_component ('UNRESOLVED_SEMANTIC') and entertainment_control ('UNRESOLVED_SEMANTIC'). Relations treated as SOFT_SEMANTIC_ONLY. Search never considered."
    elif v == "L6":
        manual_interp = "FM expressed valid refreshment item & remote roles. Compiler rejected refreshment_item ('UNRESOLVED_SEMANTIC') and rejected move_remote_to_accessible_surface ('UNSUPPORTED_OPERATOR'). Planner never engaged."
    elif v == "W1":
        manual_interp = "FM expressed fastening tool, component, marked location, and surface. Compiler rejected fastening_component ('UNRESOLVED_SEMANTIC') and fastening_tool ('UNRESOLVED_SEMANTIC'). Workshop V2 preconditions skipped. Search starved by contract gate."
    elif v == "W2":
        manual_interp = "FM expressed valid fastening tool, component, fastener, and location. Compiler rejected fastening_tool ('UNRESOLVED_SEMANTIC') and collided marked_location with fastener. Search starved."
    elif v == "W8":
        manual_interp = "FM expressed fastening_tool, fastener, target_assembly, workbench_surface. Roles mapped, but perform_fastening marked UNSUPPORTED_OPERATOR because V2 capability preconditions were skipped / operator not bridged. Search starved."

    entry = {
        "variant": v,
        "domain": domain,
        "feasible": rec.get("gt_feasible"),
        "raw_roles": [f"{r.get('id')}: {r.get('function')}" for r in raw_roles],
        "raw_ops": [f"{o.get('id')}: {o.get('operation')} ({o.get('source_role')}->{o.get('target_role')})" for o in raw_ops],
        "raw_rels": [f"{rel.get('subject_role')} -> {rel.get('relation')} -> {rel.get('object_role')}" for rel in raw_rels],
        "unresolved_roles": rec.get("unresolved_roles", []),
        "req_statuses": rec.get("candidate_requirement_statuses", []),
        "dropped_relations": rec.get("dropped_relations", []),
        "contract_complete": rec.get("runtime_contract_complete"),
        "exec_contract_complete": rec.get("executable_contract_complete"),
        "search_state": rec.get("search_state_trace", []),
        "search_eligible": rec.get("search_eligible"),
        "grounding_state": rec.get("grounded_roles", {}),
        "astar_invoked": rec.get("astar_invocations", 0) > 0,
        "first_cause": rec.get("first_cause_category"),
        "failure_reason": rec.get("failure_reason"),
        "manual_interp": manual_interp
    }
    entries.append(entry)

print("Generated entries for", len(entries), "variants.")

# Write markdown table
md_lines = [
    "# Forensic Baseline Table (af2dde73)",
    "",
    "| Variant | Domain | Feasible (GT) | Raw Semantics Expressed (Roles / Ops / Rels) | Production Mappings (Roles / Ops / Rels) | Contract Complete | Search State | Grounding State | A* Invoked? | First Cause (Reported) | Forensic Root Cause |",
    "|---|---|---|---|---|---|---|---|---|---|---|"
]

for e in entries:
    raw_sem = f"Roles: {len(e['raw_roles'])}, Ops: {len(e['raw_ops'])}, Rels: {len(e['raw_rels'])}"
    prod_role = "OK" if not e["unresolved_roles"] else f"{len(e['unresolved_roles'])} unresolved/collided"
    unsupp_ops = [s.get('id') for s in e['req_statuses'] if s.get('status') == 'UNSUPPORTED_OPERATOR']
    prod_ops = f"{len(unsupp_ops)} unsupported" if unsupp_ops else "Mapped/instantiated"
    prod_map = f"Roles: {prod_role}; Ops: {prod_ops}"
    contract = f"Exec: {e['exec_contract_complete']}, Runtime: {e['contract_complete']}"
    search = f"Eligible: {e['search_eligible']}, Trace: {e['search_state'][:1]}"
    ground = f"Grounded: {len(e['grounding_state'])} roles"
    astar = "Yes" if e['astar_invoked'] else "No"

    line = f"| **{e['variant']}** | {e['domain']} | {e['feasible']} | {raw_sem} | {prod_map} | {contract} | {search} | {ground} | {astar} | `{e['first_cause']}` | {e['manual_interp']} |"
    md_lines.append(line)

with open("scripts/forensic_table.md", "w") as f:
    f.write("\n".join(md_lines) + "\n")
print("Saved scripts/forensic_table.md")
