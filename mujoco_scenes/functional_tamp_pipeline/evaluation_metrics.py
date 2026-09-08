"""Offline metrics derived from saved evidence and independently replayed atoms."""
import json
from pathlib import Path


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {} if default is None else default


def rate(successes, eligible):
    return successes / eligible if eligible else None


def format_rate(value):
    return 'N/A' if value is None else f'{100 * value:.1f}%'


def compute_primary_metrics(records):
    """Compute the seven paper metrics with one shared denominator policy."""
    records = list(records)
    feasible = [record for record in records if record.get('gt_feasible')]
    infeasible = [record for record in records if not record.get('gt_feasible')]
    recovery = [record for record in records if record.get('requires_observation_recovery')]
    return {
        'outcome_correct': 100.0 * rate(sum(bool(r.get('outcome_correct')) for r in records), len(records)) if records else 0.0,
        'feasible_success': 100.0 * rate(sum(bool(r.get('full_task_satisfied')) for r in feasible), len(feasible)) if feasible else 0.0,
        'feasibility_recovery': 100.0 * rate(sum(bool(r.get('full_task_satisfied')) for r in recovery), len(recovery)) if recovery else 0.0,
        # Infeasible scenes have no achievable user-level goal denominator.
        'goal_coverage': 100.0 * rate(sum(float(r.get('full_task_goal_coverage', 0.0)) for r in feasible), len(feasible)) if feasible else 0.0,
        'false_completion': 100.0 * rate(sum(bool(r.get('false_completion')) for r in infeasible), len(infeasible)) if infeasible else 0.0,
        'vlm_requests': rate(sum(int(r.get('semantic_vlm_requests', 0)) for r in records), len(records)) or 0.0,
        'high_level_replans': rate(sum(int(r.get('high_level_replans', 0)) for r in records), len(records)) or 0.0,
    }


def validation_artifacts(run_dir):
    run_dir = Path(run_dir)
    for name in ('action_sequence/action_plan.json', 'action_plan.json', 'action_sequence/plan.json',
                 'observed_grounding/action_sequence/plan.json'):
        data = read_json(run_dir / name)
        val = data.get('validation', {})
        if val:
            return val
    for name in ('action_sequence/replay_validation.json', 'observed_grounding/action_sequence/replay_validation.json'):
        val = read_json(run_dir / name)
        if val:
            return val
    return {}


def candidate_plan_valid(run_dir):
    validation = replay_saved_plan(run_dir) if (Path(run_dir)/"symbolic_problem.json").exists() else validation_artifacts(run_dir)
    audit = read_json(Path(run_dir) / 'plan_grounding_audit.json')
    return bool(validation.get('status') == 'VALID' and
        audit.get('all_assignment_nodes_observed') and
        audit.get('all_required_relations_true') and
        audit.get('plan_uses_only_grounded_task_objects') and
        audit.get('preparation_accessibility_valid', True) and
        not audit.get('violations'))


def full_task_coverage(domain, run_dir):
    """Score concrete final-state goals; candidate-goal completion is insufficient.

    The benchmark's per-person goal cardinality is evaluation-only. Candidate
    role bindings select which observed entities are assessed, but never change
    the required number of settings/servings. Missing evidence scores unsatisfied.
    """
    run_dir = Path(run_dir)
    total = {'kitchen': 4, 'living_room': 3, 'workshop': 3}[domain]
    val = replay_saved_plan(run_dir) if (run_dir/'symbolic_problem.json').exists() else validation_artifacts(run_dir)
    if val.get('status') != 'VALID':
        return total, 0, 0.0, False
    atoms = {tuple(a) for a in val.get('final_atoms', [])}
    grounding = read_json(run_dir / 'graph_grounding_result.json')
    assignment = grounding.get('assignment') or {}
    def ids(role):
        value = assignment.get(role, [])
        return [value] if isinstance(value, str) else list(value)
    sat = 0
    if domain == 'kitchen':
        coffee = ids('coffee_container')
        soups = ids('soup_container')
        utensils = ids('soup_eating_utensil')
        sat += min(2, sum(all(a in atoms for a in [('at', c, 'dining_table'), ('contains', c, 'coffee'), ('contains', c, 'water'), ('stirred', c)]) for c in coffee))
        used = set()
        for bowl in soups[:2]:
            if ('at', bowl, 'dining_table') not in atoms or ('contains', bowl, 'soup') not in atoms:
                continue
            available = [u for u in utensils if u not in used and ('at', u, bowl) in atoms]
            if available:
                used.add(available[0]); sat += 1
    elif domain == 'workshop':
        target = 'workshop_frame_joint'
        sat += int(any(('inserted', f, target) in atoms for f in ids('fastener')))
        sat += int(('repaired', target) in atoms)
        sat += int(bool(ids('driver')) and all(('at', d, 'MAIN_WORKBENCH_ZONE') in atoms for d in ids('driver')) and ('hand_empty',) in atoms)
    else:
        observed = read_json(run_dir / 'observed_scene_graph.json')
        nodes = observed.get('nodes', {})
        if isinstance(nodes, list):
            nodes = {n.get('instance_id'): n for n in nodes}
        observed_relations = observed.get('relations', [])
        def verified(predicate, subject, target):
            return any(r.get('predicate') == predicate and r.get('subject_id', r.get('subject')) == subject
                       and r.get('object_id', r.get('object')) == target and r.get('status') == 'TRUE'
                       for r in observed_relations)
        # Operation IDs are FM-authored bookkeeping, not task semantics.  Score
        # any verified binding whose endpoints have the required semantic roles.
        operation_bindings = grounding.get('operation_bindings', {})
        personal_bindings = []
        personal_regions = set(ids('PERSONAL_CUP_SAUCER_REGION'))
        personal_sets = set(ids('CUP_SAUCER_SET'))
        seats = set(ids('SEATING_POSITION'))
        for bindings in operation_bindings.values() if isinstance(operation_bindings, dict) else ():
            for binding in bindings if isinstance(bindings, list) else ():
                context = binding.get('context', {}) if isinstance(binding, dict) else {}
                seat = context.get('SEATING_POSITION')
                if (binding.get('tool_id') in personal_regions
                        and binding.get('target_id') in personal_sets
                        and seat in seats):
                    personal_bindings.append(binding)
        for binding in personal_bindings[:2]:
            slot = binding.get('target_id')
            support = binding.get('tool_id')
            seat = binding.get('context', {}).get('SEATING_POSITION')
            node = nodes.get(slot, {})
            parts = node.get('unary_properties', {}).get('payload_ids', [])
            correct_support = (
                verified('FITS_SET_ON', support, slot)
                and verified('NEAR_SEAT', support, seat)
            )
            sat += int(
                correct_support
                and len(parts) >= 2
                and all(('on', part, support) in atoms for part in parts[:2])
            )
        sat += int(any(('on', remote, dest) in atoms and verified('FITS_ON', dest, remote) and verified('ACCESSIBLE_FROM_BOTH_SEATS', dest, 'SEATING_PAIR') for remote in ids('REMOTE') for dest in ids('SHARED_REMOTE_REGION')))
    sat = min(total, sat)
    is_fully_satisfied = bool(sat == total and candidate_plan_valid(run_dir))
    return total, sat, sat / total, is_fully_satisfied


def enrich_record(row, run_dir, task):
    """Add stage diagnostics from raw archive even when no canonical graph exists."""
    from .raw_semantic_evaluation import evaluate_raw_semantics
    from .structural_sanitizer import sanitize_functional_graph
    from .models import FunctionalRequirementGraph
    from .executability import analyze_executability
    from .audit import compute_prompt_and_schema_hash
    run_dir = Path(run_dir)
    manifest = read_json(run_dir / 'run_manifest.json')
    graph_dict = read_json(run_dir / 'functional_specification.json')
    metadata = graph_dict.get('metadata', {})
    raw = metadata.get('raw_vlm_response')
    diagnostics = list((run_dir / 'fm_diagnostics').glob('fm_call_*.json'))
    diagnostic = read_json(diagnostics[0]) if diagnostics else {}
    if raw is None:
        content = diagnostic.get('content')
        try:
            raw = json.loads(content) if isinstance(content, str) else content
        except ValueError:
            raw = None
    sanitizer = sanitize_functional_graph(raw).to_dict()
    trace = metadata.get('canonicalization_trace', {})
    raw_metrics = evaluate_raw_semantics(row['domain'], task, raw)
    for kind in ('role', 'relation', 'operation'):
        for metric, value in raw_metrics[kind].items():
            row[f'raw_{kind}_{metric}'] = value
    # Historical name retained only as a report compatibility alias.
    for metric, value in raw_metrics['operation'].items():
        row[f'raw_group_{metric}'] = value
    grounding = read_json(run_dir / 'graph_grounding_result.json')
    assignments = grounding.get('assignment') or {}
    grounding_status = grounding.get('status', '')
    is_complete_grounding = bool(grounding.get('complete', False) or grounding_status == 'COMPLETE')
    nodes_in_graph = graph_dict.get('nodes', {}) if graph_dict else {}
    expressed_role_count = len(nodes_in_graph)
    grounded_expressed_role_cov = (len(assignments) / expressed_role_count) if expressed_role_count > 0 else 0.0

    statuses = []
    if graph_dict:
        try:
            statuses = analyze_executability(FunctionalRequirementGraph.from_dict(graph_dict), assignments)
        except Exception:
            statuses = []
    compiled_graph = None
    runtime_interface_valid = False
    if graph_dict:
        try:
            compiled_graph = FunctionalRequirementGraph.from_dict(graph_dict)
            from .task_interface_validator import validate_runtime_gf
            validate_runtime_gf(compiled_graph)
            runtime_interface_valid = True
        except Exception:
            compiled_graph = None
    row.update(task_instruction=task, git_commit=manifest.get('git_commit'),git_dirty=manifest.get('git_dirty'),
        model=diagnostic.get('model') or manifest.get('provider_model'), prompt_hash=compute_prompt_and_schema_hash(),
        inference_config=manifest.get('inference_config', {}), transport_retries=manifest.get('transport_retries', 0),
        vlm_json_valid=isinstance(raw,dict), sanitization_succeeded=sanitizer['succeeded'],
        sanitizer_repairs=sanitizer['repairs'], canonicalization_status=metadata.get('canonicalization_status','FAILED'),
        soft_semantic_evidence=metadata.get('soft_semantic_evidence', []), context_only_roles=trace.get('context_only_roles', []),
        unresolved_roles=trace.get('unresolved_roles', []),
        dropped_relations=[r for r in sanitizer['repairs'] if r['code'] == 'DANGLING_RELATION_REFERENCE'],
        merged_roles=trace.get('merged_roles', []), disambiguated_roles=trace.get('disambiguated_roles', []),
        grounded_roles=assignments, ungrounded_expressed_roles=grounding.get('missing_roles', []),
        candidate_requirement_statuses=statuses,
        candidate_grounding_succeeded=bool(assignments),
        complete_candidate_grounding=is_complete_grounding,
        grounded_expressed_role_coverage=grounded_expressed_role_cov,
        candidate_plan_eligible=row.get('astar_invocations',0)>0,
        candidate_plan_found=bool(row.get('candidate_plan_length')),
        nonempty_candidate_plan_generated=bool(row.get('candidate_plan_length', 0) > 0),
        raw_vlm_spec_complete=bool(raw_metrics['complete_task_contract']),
        offline_reference_task_complete=bool(raw_metrics['complete_task_contract']),
        fm_count_correct=raw_metrics['count_correct'],
        fm_binding_correct=raw_metrics['binding_correct'],
        full_task_semantic_goal_count=row['full_task_goal_count'],
        full_task_semantic_goal_satisfied_count=row['full_task_goal_satisfied_count'],
        canonical_role_coverage=raw_metrics['role']['recall'],
        execution_state=manifest.get('execution_state'),
        raw_semantic_matching_method=raw_metrics['matching_method'])
    row['inspection_order_source'] = 'FM' if graph_dict.get('region_ranking') else 'SYSTEM_FALLBACK'

    # Online executable completeness is authored by the compiler and then
    # independently gated by runtime interface validation. GT coverage is not
    # consulted here.
    compiler_contract_complete = bool(
        metadata.get(
            'online_executable_contract_complete',
            metadata.get('required_contract_complete', row.get('runtime_contract_complete', False)),
        )
    )
    canon_status = metadata.get('canonicalization_status', 'FAILED')
    unresolved_semantics = metadata.get('unresolved_semantics', [])
    unresolved_roles = trace.get('unresolved_roles', [])
    disabled_groups = trace.get('disabled_groups', [])
    unresolved_relations = trace.get('unresolved_required_relations', [])
    unresolved_operations = trace.get('unresolved_required_operations', [])
    unverified_props = metadata.get('unverified_required_properties', [])
    executable_contract_complete = bool(
        compiler_contract_complete and runtime_interface_valid
        and canon_status == 'FULL' and not unresolved_semantics
        and not unresolved_roles and not disabled_groups
        and not unresolved_relations and not unresolved_operations
        and not unverified_props
    )
    row['online_executable_contract_complete'] = executable_contract_complete
    row['offline_reference_task_complete'] = bool(raw_metrics.get('complete_task_contract', False))
    row['required_contract_complete'] = executable_contract_complete
    row['executable_contract_complete'] = executable_contract_complete
    row['runtime_contract_complete'] = executable_contract_complete
    row['required_compiler_interpretation'] = bool(
        runtime_interface_valid and canon_status == 'FULL'
        and not unresolved_semantics and not unresolved_roles
        and not disabled_groups and not unresolved_relations and not unresolved_operations
    )
    row['candidate_grounding_eligible'] = executable_contract_complete

    # Causal search recovery: initial grounding before search was incomplete,
    # at least one region was inspected, and final grounding became complete.
    initial_grounding_complete = None
    search_states = []
    snapshots = read_json(run_dir / 'grounding_snapshots.json', default=[])
    if isinstance(snapshots, list) and snapshots:
        initial = snapshots[0]
        initial_payload = initial.get('grounding', {})
        initial_grounding_complete = bool(initial_payload.get('complete', False) or initial_payload.get('satisfied', False) or initial_payload.get('status') == 'COMPLETE')
        search_states = [snap.get('search_state') for snap in snapshots]
    else:
        # Historical archives before grounding_snapshots.json recorded the same
        # pre-search state in their first grounding_updated trajectory event.
        events_file = run_dir / 'trajectory_events.jsonl'
        if events_file.exists():
            try:
                for line in events_file.read_text().splitlines():
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    if event.get('event') == 'grounding_updated':
                        payload = event.get('payload', {})
                        initial_payload = payload.get('grounding', {})
                        initial_grounding_complete = bool(
                            initial_payload.get('complete', False)
                            or initial_payload.get('satisfied', False)
                            or initial_payload.get('status') == 'COMPLETE'
                        )
                        if payload.get('search_state'):
                            search_states.append(payload['search_state'])
                        break
            except (OSError, ValueError, TypeError):
                pass
    if initial_grounding_complete is None:
        if not row.get('regions_inspected'):
            initial_grounding_complete = is_complete_grounding
        else:
            initial_grounding_complete = False

    causal_search_recovery = bool(
        initial_grounding_complete is False
        and row.get('regions_inspected')
        and is_complete_grounding
    )
    row['search_recovery_succeeded'] = causal_search_recovery
    row['initial_grounding_complete'] = initial_grounding_complete
    row['search_state_trace'] = search_states
    row['search_eligible'] = bool(executable_contract_complete and 'SEARCH_RECOVERABLE' in search_states)

    # Incomplete semantics cannot prove full-task scene infeasibility.
    if not row['gt_feasible'] and not row['runtime_contract_complete']:
        row['outcome_correct'] = False

    # Corrected first-cause failure precedence (Section 17.3)
    first_cause = None
    category = None
    stage = None
    if row.get('terminal_status') == 'PIPELINE_EXCEPTION' or row.get('failure_category') == 'EVAL_INFRA_ERROR':
        first_cause = 'EVAL_INFRA_ERROR'
        category, stage = 'EVAL_INFRA_ERROR', 'INFRASTRUCTURE'
    elif not row['gt_feasible']:
        if row.get('false_completion'):
            first_cause = 'TASK_SPECIFICATION_FAILURE'
            category, stage = 'FALSE_COMPLETION', 'EVALUATION'
        else:
            first_cause = None
    elif row.get('full_task_satisfied'):
        first_cause = None
        category, stage = 'NONE', 'SUCCESS'
    elif not row['vlm_json_valid']:
        first_cause = 'TASK_SPECIFICATION_FAILURE'
        category, stage = 'FM_STRUCTURAL_ERROR', 'RAW_FM'
    elif not sanitizer['succeeded']:
        first_cause = 'GRAPH_COMPILATION_FAILURE'
        category, stage = 'SANITIZER_UNRECOVERABLE', 'SANITIZER'
    elif not row['canonicalization_succeeded']:
        first_cause = 'GRAPH_COMPILATION_FAILURE'
        category, stage = 'CANONICALIZATION_AMBIGUITY', 'CANONICALIZER'
    elif trace.get('unresolved_roles'):
        first_cause = 'GRAPH_COMPILATION_FAILURE'
        category, stage = 'CANONICALIZATION_AMBIGUITY', 'CANONICALIZER'
    elif metadata.get('unresolved_semantics') or trace.get('disabled_groups') or trace.get('unresolved_required_operations'):
        first_cause = 'GRAPH_COMPILATION_FAILURE'
        category, stage = 'CANONICALIZATION_UNRESOLVED_REQUIRED_SEMANTIC', 'EXECUTABILITY'
    elif not executable_contract_complete:
        first_cause = 'GRAPH_COMPILATION_FAILURE'
        category, stage = 'CONTRACT_INCOMPLETE', 'EXECUTABILITY'
    elif not row.get('offline_reference_task_complete', row.get('raw_vlm_spec_complete', False)):
        first_cause = 'TASK_SPECIFICATION_FAILURE'
        category, stage = 'FM_SEMANTIC_OMISSION', 'RAW_FM'
    elif not assignments:
        if row.get('search_exhausted'):
            first_cause = 'OBJECT_DISCOVERY_FAILURE'
            category, stage = 'SEARCH_EXHAUSTED', 'SEARCH'
        else:
            first_cause = 'FUNCTIONAL_ASSIGNMENT_FAILURE'
            category, stage = 'NO_VALID_ASSIGNMENT', 'GROUNDING'
    elif not is_complete_grounding:
        if row.get('search_exhausted'):
            first_cause = 'OBJECT_DISCOVERY_FAILURE'
            category, stage = 'SEARCH_EXHAUSTED', 'SEARCH'
        else:
            first_cause = 'FUNCTIONAL_ASSIGNMENT_FAILURE'
            category, stage = 'PARTIAL_VERIFIED_GROUNDING', 'GROUNDING'
    else:
        audit_data = read_json(Path(run_dir) / 'plan_grounding_audit.json')
        grounding_invalid = bool(
            audit_data and (
                audit_data.get('all_assignment_nodes_observed') is False
                or audit_data.get('all_required_relations_true') is False
            )
        )
        if grounding_invalid:
            first_cause = 'FUNCTIONAL_ASSIGNMENT_FAILURE'
            category, stage = 'INVALID_GROUNDING_EVIDENCE', 'GROUNDING'
        elif row.get('candidate_plan_eligible') and not row.get('candidate_plan_valid'):
            first_cause = 'PLANNING_FAILURE'
            category, stage = 'PLAN_VALIDATION_FAILURE', 'VALIDATION'
        elif row.get('candidate_plan_eligible') and not row.get('full_task_satisfied'):
            first_cause = 'PLANNING_FAILURE'
            category, stage = 'PLANNING_FAILURE', 'PLANNING'
        else:
            first_cause = 'FUNCTIONAL_ASSIGNMENT_FAILURE'
            category, stage = 'PLANNING_NOT_REACHED', 'GROUNDING'

    row.update(first_cause_category=first_cause, failure_category=category, failure_stage=stage)
    (run_dir/'raw_semantic_evaluation.json').write_text(json.dumps(raw_metrics,indent=2)+'\n')
    (run_dir/'structural_sanitization.json').write_text(json.dumps(sanitizer,indent=2)+'\n')
    (run_dir/'executability_analysis.json').write_text(json.dumps(statuses,indent=2)+'\n')
    (run_dir/'evaluation_record.json').write_text(json.dumps(row,indent=2)+'\n')
    return row


def write_detailed_report(output_root, records, *, live=False):
    from collections import Counter
    output_root = Path(output_root)
    diagnostics = {
        'Raw VLM role precision': lambda r: r.get('raw_role_precision'),
        'Raw VLM role recall': lambda r: r.get('raw_role_recall'),
        'Raw VLM role F1': lambda r: r.get('raw_role_f1'),
        'Interpreter-matched raw relation F1': lambda r: r.get('interpreter_matched_raw_relation_f1', r.get('raw_relation_f1')),
        'Raw VLM group F1': lambda r: r.get('raw_group_f1'),
        'Raw complete spec rate': lambda r: r.get('raw_vlm_spec_complete'),
        'Sanitization success': lambda r: r.get('sanitization_succeeded', False),
        'Full canonicalization': lambda r: r.get('canonicalization_status') == 'FULL',
        'Partial canonicalization': lambda r: r.get('canonicalization_status') == 'PARTIAL',
        'Any canonicalization success': lambda r: r['canonicalization_succeeded'],
        'Executable contract complete rate': lambda r: r.get('executable_contract_complete', r.get('runtime_contract_complete')),
        'Search eligible': lambda r: r.get('search_eligible', False),
        'Search recovery success': lambda r: r.get('search_recovery_succeeded', False) if r.get('search_eligible') else None,
        'Any verified grounding / eligible': lambda r: r['candidate_grounding_succeeded'] if r['candidate_grounding_eligible'] else None,
        'Complete candidate grounding / eligible': lambda r: r.get('complete_candidate_grounding', False) if r['candidate_grounding_eligible'] else None,
        'Grounded expressed role coverage': lambda r: r.get('grounded_expressed_role_coverage') if r['candidate_grounding_eligible'] else None,
        'A* invoked': lambda r: r.get('candidate_plan_eligible', False),
        'Non-empty plan generated': lambda r: r.get('candidate_plan_length', 0) > 0,
        'Candidate planning success / generated': lambda r: r['candidate_plan_valid'] if r.get('candidate_plan_length', 0) > 0 else None,
        'Partial-plan rate': lambda r: 'PARTIAL' in r['candidate_plan_status'],
        'Candidate goal coverage': lambda r: r['candidate_goal_coverage'] if r.get('candidate_plan_eligible') else None,
        'Full-task success': lambda r: r['full_task_satisfied'] if r.get('gt_feasible') else None,
    }
    lines=['| Metric | Kitchen | Living | Workshop | Overall |','|---|---:|---:|---:|---:|']
    for name, fn in diagnostics.items():
        cells=[]
        for domain in ['kitchen','living_room','workshop',None]:
            values=[fn(r) for r in records if domain is None or r['domain']==domain]
            values=[v for v in values if v is not None]
            cells.append(format_rate(sum(values)/len(values) if values else None))
        lines.append('| '+name+' | '+' | '.join(cells)+' |')
    cells=[]
    for domain in ['kitchen','living_room','workshop',None]:
        rows=[r for r in records if domain is None or r['domain']==domain]
        cells.append(f'{sum(len(r["regions_inspected"]) for r in rows)/len(rows):.2f}' if rows else 'N/A')
    lines.append('| Mean regions inspected | '+' | '.join(cells)+' |')
    (output_root/'pipeline_diagnostic_table.md').write_text('\n'.join(lines)+'\n')

    feasible_records = [r for r in records if r.get('gt_feasible')]
    first_cause_counts = Counter(r.get('first_cause_category') or 'NONE' for r in feasible_records)
    detailed_counts = Counter(r.get('failure_category') or 'NONE' for r in records)
    failure_payload = {
        'first_cause_categories_feasible': dict(first_cause_counts),
        'detailed_failure_categories_all': dict(detailed_counts),
    }
    (output_root/'failure_analysis.json').write_text(json.dumps(failure_payload, indent=2)+'\n')

    failures = [
        '# First-Cause Failure Analysis (Feasible Tasks)',
        '',
        '| First-Cause Category | Count |',
        '|---|---:|',
    ] + [f'| {k} | {v} |' for k, v in sorted(first_cause_counts.items())] + [
        '',
        '# Detailed Pipeline Diagnostics (All Variants)',
        '',
        '| Detailed Cause | Count |',
        '|---|---:|',
    ] + [f'| {k} | {v} |' for k, v in sorted(detailed_counts.items())]
    (output_root/'failure_analysis.md').write_text('\n'.join(failures)+'\n')
    errors=[]
    if live:
        if len(records)!=32:errors.append('Expected 32 variants')
        if sum(r['gt_feasible'] for r in records)!=20:errors.append('Expected 20 feasible variants')
        if sum(r['requires_observation_recovery'] for r in records)!=13:errors.append('Expected 13 recovery variants')
        if len({r.get('git_commit') for r in records})!=1:errors.append('Mixed commits')
        for r in records:
            if (r['spec_acquisition']!='live_provider' or r.get('specification_input') is not None
                    or r['semantic_vlm_requests']!=1 or r['high_level_replans']!=0
                    or r.get('git_dirty') or r.get('execution_state')!='planning_only'
                    or r.get('inspection_order_source') not in {'FM', 'SYSTEM_FALLBACK'}):
                errors.append(f'{r["variant"]}: live invariant failed')
        if len({r.get('prompt_hash') for r in records}) != 1:
            errors.append('Mixed prompts')
        if len({r.get('model') for r in records}) != 1:
            errors.append('Mixed models')
    (output_root/'invariants.json').write_text(json.dumps({'status':'INVALID' if errors else 'VALID' if live else 'REPLAY', 'errors':errors},indent=2)+'\n')
    write_representative_trajectories(output_root, records)
    return errors


def write_representative_trajectories(output_root, records):
    """Persist reviewable end-to-end evidence for the requested case classes."""
    output_root = Path(output_root)
    cases = (
        ('kitchen_full_success', lambda r: r['domain'] == 'kitchen' and r['full_task_satisfied']),
        ('kitchen_recovery_success', lambda r: r['domain'] == 'kitchen' and r['requires_observation_recovery'] and r['full_task_satisfied']),
        ('kitchen_partial_candidate', lambda r: r['domain'] == 'kitchen' and 'PARTIAL' in r['candidate_plan_status'] and r['candidate_plan_found']),
        ('living_full_success', lambda r: r['domain'] == 'living_room' and r['full_task_satisfied']),
        ('living_infeasible', lambda r: r['domain'] == 'living_room' and not r['gt_feasible']),
        ('workshop_recovery_success', lambda r: r['domain'] == 'workshop' and r['requires_observation_recovery'] and r['full_task_satisfied']),
        ('workshop_partial_candidate', lambda r: r['domain'] == 'workshop' and 'PARTIAL' in r['candidate_plan_status'] and r['candidate_plan_found']),
        ('workshop_infeasible', lambda r: r['domain'] == 'workshop' and not r['gt_feasible']),
    )
    report = {}
    for label, predicate in cases:
        row = next((candidate for candidate in records if predicate(candidate)), None)
        if row is None:
            report[label] = {'available': False, 'reason': 'No run matched this case class'}
            continue
        run_dir = output_root / row['domain'] / row['variant'] / 'vlm'
        canonical = read_json(run_dir / 'functional_specification.json')
        metadata = canonical.get('metadata', {})
        plan = {}
        for name in ('action_sequence/action_plan.json', 'action_plan.json', 'action_sequence/plan.json'):
            plan = read_json(run_dir / name)
            if plan:
                break
        events = []
        try:
            events = [json.loads(line) for line in (run_dir / 'trajectory_events.jsonl').read_text().splitlines() if line.strip()]
        except (OSError, ValueError):
            pass
        report[label] = {
            'available': True,
            'domain': row['domain'],
            'variant': row['variant'],
            'raw_G_F': metadata.get('raw_vlm_response'),
            'sanitizer_repairs': row.get('sanitizer_repairs', []),
            'canonical_G_F': canonical,
            'unresolved_semantics': metadata.get('unresolved_semantics', []),
            'G_O_before_search': read_json(run_dir / 'observed_graph_before_search.json'),
            'inspection_sequence': events,
            'G_O_after_search': read_json(run_dir / 'observed_graph_after_search.json', read_json(run_dir / 'observed_scene_graph.json')),
            'phi': read_json(run_dir / 'graph_grounding_result.json').get('assignment', {}),
            'candidate_requirements': row.get('candidate_requirement_statuses', []),
            'astar_action_sequence': plan.get('actions', []),
            'candidate_replay_result': read_json(run_dir / 'candidate_replay_validation.json', validation_artifacts(run_dir)),
            'candidate_goal_coverage': row['candidate_goal_coverage'],
            'full_task_goal_coverage': row['full_task_goal_coverage'],
            'final_status': row['terminal_status'],
        }
    (output_root / 'representative_trajectories.json').write_text(json.dumps(report, indent=2) + '\n')


def replay_saved_plan(run_dir):
    """Rebuild the frozen symbolic problem and replay the actual saved actions."""
    from mujoco_scenes.symbolic_planning_core import SymbolicAction, SymbolicProblem, independent_replay
    run_dir = Path(run_dir)
    data = read_json(run_dir/'symbolic_problem.json')
    if not data:
        return {'status':'INVALID','reason':'No serialized symbolic problem'}
    atoms=lambda rows:frozenset(tuple(r) for r in rows)
    operators=tuple(SymbolicAction(name=a['name'],arguments=tuple(a['arguments']),
        positive_preconditions=atoms(a['positive_preconditions']),negative_preconditions=atoms(a['negative_preconditions']),
        add_effects=atoms(a['add_effects']),delete_effects=atoms(a['delete_effects']),cost=a['cost']) for a in data['actions'])
    problem=SymbolicProblem(initial_atoms=atoms(data['initial_atoms']),goal_atoms=atoms(data['goal_atoms']),actions=operators)
    plan_data={}
    for name in ('action_sequence/action_plan.json','action_plan.json','action_sequence/plan.json'):
        plan_data=read_json(run_dir/name)
        if 'actions' in plan_data:break
    if 'actions' not in plan_data:
        return {'status':'INVALID','reason':'No saved plan'}
    state=set(problem.initial_atoms); sequence=[]
    for row in plan_data['actions']:
        args=row.get('arguments',[])
        args=tuple(args.values()) if isinstance(args,dict) else tuple(args)
        matches=[a for a in operators if a.name.upper()==row.get('operator','').upper() and a.arguments==args
                 and a.positive_preconditions<=state and not a.negative_preconditions.intersection(state)]
        if not matches:
            return {'status':'INVALID','reason':'Action is unknown or has false preconditions','action':row}
        action=matches[0];sequence.append(action)
        state.difference_update(action.delete_effects);state.update(action.add_effects)
    result=independent_replay(problem,sequence,allow_partial=True)
    (run_dir/'candidate_replay_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
