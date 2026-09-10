#!/usr/bin/env python3
"""Inspect what the deterministic front half makes of one frozen FM contract.

Offline development instrument: zero FM calls, and it never reads ground truth.
It prints, for one archived raw response, how every role was typed, how every
relation and operation was read, and what the compiled graph and its
canonicalization trace contain.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

INSTRUCTIONS = {
    "kitchen": "Prepare and serve one coffee and one soup for each of two people. Make each coffee using coffee and water and stir it before serving. Serve each soup bowl with its own suitable eating utensil.",
    "living_room": "Prepare the living room for two people to enjoy refreshments while watching television. Provide each person with their own refreshment setting nearby, and place the entertainment control where it is accessible to both people.",
    "workshop": "Identify the compatible components required to complete the fastening at the marked workbench location, complete the fastening, and leave any reusable equipment used for the task safely on the workbench.",
}


def load_raw(path: Path):
    data = json.loads(path.read_text())
    return json.loads(data["content"]) if isinstance(data.get("content"), str) else data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--stage", default="all",
                    help="roles | relations | operations | compile | all")
    args = ap.parse_args()
    domain = args.domain
    instruction = INSTRUCTIONS[domain]
    raw = load_raw(Path(args.raw))

    from mujoco_scenes.functional_tamp_pipeline.fm_schema_v3 import (
        normalize_v3_live_document, validate_v3_live_contract,
        convert_v3_to_canonical_document)
    from mujoco_scenes.functional_tamp_pipeline.semantic_typing import (
        build_role_type_hypotheses, detect_role_families, role_text_scopes)

    normalized, ntrace = normalize_v3_live_document(raw, domain=domain, task_instruction=instruction)
    print("### NORMALIZED WIRE")
    notes = ntrace if isinstance(ntrace, list) else (ntrace or {}).get("repairs", []) or []
    for note in notes:
        print("   repair:", json.dumps(note, default=str)[:200])
    try:
        validate_v3_live_contract(normalized, domain=None)
        print("   strict wire: VALID")
    except Exception as exc:
        print(f"   strict wire: INVALID  {type(exc).__name__}: {exc}")
        return 1
    contract = normalized["task_contract"]

    if args.stage in {"roles", "all"}:
        print("\n### ROLE TYPING")
        hyp = build_role_type_hypotheses(domain, contract)
        for rid, h in sorted(hyp.items()):
            role = next((r for r in contract["functional_roles"] if r["id"] == rid), {})
            fams, _ = detect_role_families(domain, dict(role))
            print(f"  {rid:26s} -> {list(h.canonical_role_candidates)}  status={h.status}")
            print(f"      count={role.get('required_count')} {role.get('binding_policy')} kind={role.get('entity_kind')} families={sorted(fams)}")
            print(f"      scopes={json.dumps(role_text_scopes(dict(role)))[:220]}")

    if args.stage in {"relations", "all"}:
        print("\n### RELATION READING")
        from mujoco_scenes.functional_tamp_pipeline.relation_interpreter import (
            extract_relation_semantic_candidates)
        for rel in contract.get("functional_relations", ()) or ():
            cands = extract_relation_semantic_candidates(domain, rel["relation"])
            print(f"  {rel['id']:30s} {rel['relation']!r} {rel['participant_roles']} req={rel.get('required')}")
            print(f"      candidates={[getattr(c,'predicate_name',c) for c in cands]}")

    if args.stage in {"operations", "all"}:
        print("\n### OPERATION READING")
        from mujoco_scenes.functional_tamp_pipeline.robot_capability_registry import (
            extract_operation_semantic_candidates, operation_action_phrase)
        from mujoco_scenes.functional_tamp_pipeline.fm_schema_v3 import resolve_v3_operation_slots
        from mujoco_scenes.functional_tamp_pipeline.operation_slot_completion import (
            collect_expressed_semantics, complete_operation_slots)
        hyp = build_role_type_hypotheses(domain, contract)
        roles_by_id = {r["id"]: r for r in contract["functional_roles"]}
        evidence = collect_expressed_semantics(domain, contract, hyp)
        for op in contract.get("operation_pairings", ()) or ():
            parts = list(op.get("participant_roles", ()))
            phrase = str(op["operation"])
            caps = extract_operation_semantic_candidates(domain, phrase, parts)
            print(f"  {op['id']:26s} {phrase!r} x{op.get('operation_count')} {parts}")
            print(f"      action_phrase={operation_action_phrase(phrase, parts)!r}")
            print(f"      capabilities={[c.capability_id for c in caps]}")
            opts = resolve_v3_operation_slots(domain, op, roles_by_id, hyp)
            print(f"      direct_slots={[{k:v for k,v in o.items() if k in ('capability_id','source_role','target_role','anchor_role','source_type','target_type','anchor_type','usage_policy')} for o in opts]}")
            if not opts:
                done = complete_operation_slots(domain, op, roles_by_id, hyp, evidence)
                if done is None:
                    print("      slot_completion=REFUSED")
                else:
                    print(f"      slot_completion={done.capability_id} options={done.options}")
                    for r in done.synthesized_roles:
                        print(f"         synthesized {r['id']} role={r['canonical_role']} count={r['required_count']} {r['binding_policy']} slot={r['induced_slot']}")

    if args.stage in {"compile", "all"}:
        print("\n### CANONICAL + COMPILE")
        canonical = convert_v3_to_canonical_document(normalized, domain=domain, task_instruction=instruction)
        from mujoco_scenes.functional_tamp_pipeline.semantic_compiler import compile_candidate_graph
        graph = compile_candidate_graph(domain, instruction, canonical)
        meta = graph.metadata or {}
        trace = meta.get("canonicalization_trace", {}) or {}
        print("  nodes:")
        for n, role in sorted(graph.nodes.items()):
            print(f"    {n:28s} kind={role.entity_kind:12s} count={role.count} min={role.minimum_count} max={role.maximum_count} {role.binding_policy} raw={role.raw_role_id}")
        print("  operation_groups:")
        for g in graph.operation_groups or ():
            print(f"    {g.id:26s} {g.function:28s} src={g.tool_role} tgt={g.target_role} ctx={g.context_role} n={g.required_target_count} {g.usage_policy}")
            print(f"        preconditions={list(g.physical_preconditions or ())}")
        print("  relations:", [(r.subject_role, r.predicate, r.object_role) for r in graph.relations or ()])
        print(f"  required_contract_complete={meta.get('required_contract_complete')} online_executable={meta.get('online_executable_contract_complete')}")
        for key in ("unresolved_roles", "unresolved_required_relations", "unresolved_required_operations",
                    "disabled_groups", "surplus_unrepresentable_constraints"):
            items = trace.get(key) or []
            if items:
                print(f"  {key}:")
                for item in items:
                    print("    -", json.dumps(item, default=str)[:400])
        for reason in meta.get("executable_contract_missing_reasons") or []:
            print("  EXEC_MISSING:", str(reason)[:300])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
