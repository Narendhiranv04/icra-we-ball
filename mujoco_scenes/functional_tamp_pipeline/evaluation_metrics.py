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
    if not candidate_plan_valid(run_dir):
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
        personal_bindings = grounding.get('operation_bindings', {}).get(
            'personal_support_group', []
        )
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
    return total, sat, sat / total, sat == total


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
    for kind in ('role', 'relation', 'group'):
        for metric, value in raw_metrics[kind].items():
            row[f'raw_{kind}_{metric}'] = value
    grounding = read_json(run_dir / 'graph_grounding_result.json')
    assignments = grounding.get('assignment') or {}
    statuses = []
    if graph_dict:
        statuses = analyze_executability(FunctionalRequirementGraph.from_dict(graph_dict), assignments)
    required_raw_recalls = [raw_metrics[kind]['recall'] for kind in ('role', 'relation', 'group')
                            if raw_metrics[kind]['recall'] is not None]
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
        candidate_requirement_statuses=statuses, candidate_grounding_eligible=bool(graph_dict),
        candidate_grounding_succeeded=bool(assignments),candidate_plan_eligible=row.get('astar_invocations',0)>0,
        candidate_plan_found=bool(row.get('candidate_plan_length')),
        raw_vlm_spec_complete=bool(isinstance(raw, dict) and required_raw_recalls
                                   and all(value == 1.0 for value in required_raw_recalls)),
        full_task_semantic_goal_count=row['full_task_goal_count'],
        full_task_semantic_goal_satisfied_count=row['full_task_goal_satisfied_count'],
        search_eligible=bool(graph_dict and row['regions_available']),
        canonical_role_coverage=raw_metrics['role']['recall'],
        search_recovery_succeeded=bool(row['regions_inspected'] and grounding.get('complete')),
        execution_state=manifest.get('execution_state'),
        raw_semantic_matching_method=raw_metrics['matching_method'])
    row['inspection_order_source'] = 'FM' if graph_dict.get('region_ranking') else 'SYSTEM_FALLBACK'
    # Incomplete semantics cannot prove full-task scene infeasibility.
    if not row['gt_feasible'] and not row['runtime_contract_complete']:
        row['outcome_correct'] = False
    category = None
    stage = None
    if not row['vlm_json_valid']:
        category, stage = 'FM_STRUCTURAL_ERROR', 'RAW_FM'
    elif not sanitizer['succeeded']:
        category, stage = 'SANITIZER_UNRECOVERABLE', 'SANITIZER'
    elif not row['canonicalization_succeeded']:
        category, stage = 'CANONICALIZATION_AMBIGUITY', 'CANONICALIZER'
    elif trace.get('unresolved_roles'):
        category, stage = 'CANONICALIZATION_AMBIGUITY', 'CANONICALIZER'
    elif metadata.get('unresolved_semantics') or trace.get('disabled_groups'):
        category, stage = 'CANONICALIZATION_UNRESOLVED_REQUIRED_SEMANTIC', 'EXECUTABILITY'
    elif not row['raw_vlm_spec_complete']:
        category, stage = 'FM_SEMANTIC_OMISSION', 'RAW_FM'
    elif not assignments:
        category, stage = ('SEARCH_EXHAUSTED' if row['search_exhausted'] else 'NO_VALID_ASSIGNMENT'), 'GROUNDING'
    elif row['candidate_plan_eligible'] and not row['candidate_plan_valid']:
        category, stage = 'PLAN_VALIDATION_FAILURE', 'VALIDATION'
    elif not row['full_task_satisfied']:
        category, stage = 'PLANNING_FAILURE', 'PLANNING'
    row.update(failure_category=category,failure_stage=stage)
    (run_dir/'raw_semantic_evaluation.json').write_text(json.dumps(raw_metrics,indent=2)+'\n')
    (run_dir/'structural_sanitization.json').write_text(json.dumps(sanitizer,indent=2)+'\n')
    (run_dir/'executability_analysis.json').write_text(json.dumps(statuses,indent=2)+'\n')
    (run_dir/'evaluation_record.json').write_text(json.dumps(row,indent=2)+'\n')
    return row


def write_detailed_report(output_root, records, *, live=False):
    from collections import Counter
    output_root = Path(output_root)
    diagnostics = {
        'Raw VLM role F1': lambda r: r.get('raw_role_f1'),
        'Raw VLM relation F1': lambda r: r.get('raw_relation_f1'),
        'Raw VLM group F1': lambda r: r.get('raw_group_f1'),
        'Sanitization success': lambda r:r.get('sanitization_succeeded',False),
        'Full canonicalization':lambda r:r.get('canonicalization_status')=='FULL',
        'Partial canonicalization':lambda r:r.get('canonicalization_status')=='PARTIAL',
        'Any canonicalization success':lambda r:r['canonicalization_succeeded'],
        'Runtime contract coverage':lambda r:r.get('canonical_role_coverage'),
        'Search eligible':lambda r:r.get('search_eligible',False),
        'Search recovery success':lambda r:r.get('search_recovery_succeeded',False) if r.get('search_eligible') else None,
        'Grounding success / eligible':lambda r:r['candidate_grounding_succeeded'] if r['candidate_grounding_eligible'] else None,
        'Candidate planning success / eligible':lambda r:r['candidate_plan_valid'] if r.get('candidate_plan_eligible') else None,
        'Partial-plan rate':lambda r:'PARTIAL' in r['candidate_plan_status'],
        'Candidate goal coverage':lambda r:r['candidate_goal_coverage'] if r.get('candidate_plan_eligible') else None,
        'Full-task success':lambda r:r['full_task_satisfied'],
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
    counts=Counter(r.get('failure_category') or 'NONE' for r in records)
    (output_root/'failure_analysis.json').write_text(json.dumps(dict(counts),indent=2)+'\n')
    failures=['| Primary cause | Count |','|---|---:|']+[f'| {k} | {v} |' for k,v in sorted(counts.items())]
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
