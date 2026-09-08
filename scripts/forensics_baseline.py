import json

for v in ["W1", "W2", "W8", "L1", "L6"]:
    domain = "workshop" if v.startswith("W") else "living_room"
    base = f"benchmark_reports/final_corrected_32x1_20260908T192500IST_rerun/{domain}/{v}/vlm"
    with open(f"{base}/evaluation_record.json") as f:
        rec = json.load(f)
    with open(f"{base}/fm_diagnostics/fm_call_001.json") as f:
        diag = json.load(f)
    with open(f"{base}/raw_semantic_evaluation.json") as f:
        raw_eval = json.load(f)

    c = json.loads(diag.get("content", "{}"))
    tc = c.get("task_contract", {})

    print("=" * 80)
    print(f"VARIANT: {v}")
    print("ROLES in FM:")
    for r in tc.get("functional_roles", []):
        print(f"  [{r.get('id')}] {r.get('function')}")
    print("RELATIONS in FM:")
    for rel in tc.get("functional_relations", []):
        print(f"  {rel.get('subject_role')} -- {rel.get('relation')} --> {rel.get('object_role')}")
    print("OPERATIONS in FM:")
    for op in tc.get("operation_pairings", []):
        print(f"  [{op.get('id')}] {op.get('operation')} (source: {op.get('source_role')} -> target: {op.get('target_role')})")
    print("UNRESOLVED ROLES in rec:", rec.get("unresolved_roles"))
    print("DROPPED RELATIONS in rec:", rec.get("dropped_relations"))
    print("CANDIDATE REQUIREMENT STATUSES:", rec.get("candidate_requirement_statuses"))
    print("SEARCH STATE TRACE:", rec.get("search_state_trace"))
    print("RAW EVAL mapping:", raw_eval.get("raw_id_to_semantic_role"))
    print("RAW EVAL rel:", raw_eval.get("relation"))
    print("RAW EVAL op:", raw_eval.get("operation"))
