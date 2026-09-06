from copy import deepcopy
import json
from pathlib import Path

import pytest

from mujoco_scenes.functional_tamp_pipeline.structural_sanitizer import sanitize_functional_graph
from mujoco_scenes.functional_tamp_pipeline.semantic_compiler import compile_candidate_graph, causal_position, can_merge_roles


def role(rid='vessel', **updates):
    return dict(id=rid, entity_kind='OBJECT', function='hold coffee', description='',
                required_count=1, binding_policy='DISTINCT', required_properties=[],
                candidate_categories=['cup'], visible_candidates=[], **updates)


def doc(*roles, **updates):
    return dict(functional_roles=list(roles), functional_relations=[], interaction_groups=[], **updates)


def test_default_count_dangling_relation_and_input_immutability():
    r=role(); del r['required_count']
    raw=doc(r); raw['functional_relations']=[dict(subject_role='vessel', relation='near', object_role='absent')]
    before=deepcopy(raw)
    result=sanitize_functional_graph(raw)
    assert raw == before
    assert result.succeeded and result.semantically_incomplete
    assert result.document['functional_roles'][0]['required_count'] == 1
    assert not result.document['functional_relations']
    assert {r['code'] for r in result.repairs} == {'DEFAULTED_REQUIRED_COUNT', 'DANGLING_RELATION_REFERENCE'}


@pytest.mark.parametrize('text', ['two vessels', 'one for each person', 'both vessels', '3 vessels'])
def test_plural_missing_count_is_not_defaulted(text):
    r=role(); del r['required_count'];r['description']=text
    result=sanitize_functional_graph(doc(r, role('other')))
    assert [r['id'] for r in result.document['functional_roles']] == ['other']
    assert result.semantically_incomplete


def test_duplicate_candidates_and_missing_group():
    r=role();r['visible_candidates']=[dict(id='c1', label='cup')]*2
    raw=doc(r);raw['interaction_groups']=[dict(id='g',tool_role='vessel',target_role='absent')]
    result=sanitize_functional_graph(raw)
    assert len(result.document['functional_roles'][0]['visible_candidates']) == 1
    assert not result.document['interaction_groups']
    assert {r['code'] for r in result.repairs} == {'DUPLICATE_VISIBLE_CANDIDATE_REMOVED', 'INVALID_GROUP_REFERENCE'}


def test_multiple_required_relations_are_legal():
    raw=doc(role('a'),role('b'))
    raw['interaction_groups']=[dict(id='g', tool_role='a', target_role='b',required_relations=['aligned_with','inserted_into'])]
    result=sanitize_functional_graph(raw)
    assert result.document['interaction_groups'][0]['required_relations'] == ['aligned_with','inserted_into']
    assert not result.semantically_incomplete


def test_unknown_properties_survive_without_losing_other_roles():
    r=role();r['required_properties']=['sufficient surface area','unusual decorative profile']
    g=compile_candidate_graph('kitchen','task',doc(r))
    assert 'coffee_container' in g.nodes
    assert len(g.metadata['soft_semantic_evidence']) == 2


def test_causal_source_destination_core_is_material_independent():
    source=role('a');source['function']='supplies granular material'
    dest=role('b');dest['function']='receives prepared payload'
    raw=doc(source,dest)
    assert causal_position(source,raw) == {'source'}
    assert causal_position(dest,raw) == {'destination'}
    assert not can_merge_roles(source,dest,raw)


def test_source_disambiguation_domain_adapter():
    source=role('source');source['function']='contain coffee powder'
    dest=role('target');dest['function']='contain prepared coffee'
    g=compile_candidate_graph('kitchen','task',doc(source,dest))
    assert set(g.nodes) == {'coffee_source','coffee_container'}
    assert g.metadata['canonicalization_trace']['disambiguated_roles']


def test_true_duplicate_merge_preserves_provenance():
    g=compile_candidate_graph('kitchen','task',doc(role('a'),role('b')))
    assert len(g.nodes) == 1
    assert g.metadata['canonicalization_trace']['merged_roles'][0]['raw_ids'] == ['a','b']


def test_context_and_unresolved_roles_are_separate():
    raw=json.loads((Path(__file__).parent/'fixtures/ideal_raw_vlm/living_room_L1.json').read_text())
    context=role('screen');context.update(entity_kind='FIXED_TARGET',function='display television',candidate_categories=['screen'])
    unknown=role('unknown');unknown['function']='emit unspecified output';unknown['candidate_categories']=[]
    raw['functional_roles'] += [context,unknown]
    g=compile_candidate_graph('living_room','task',raw)
    trace=g.metadata['canonicalization_trace']
    assert trace['context_only_roles'][0]['raw_role']['id'] == 'screen'
    assert trace['unresolved_roles'][0]['raw_role']['id'] == 'unknown'
    assert g.metadata['canonicalization_status'] == 'PARTIAL'
    assert 'REMOTE' in g.nodes


def test_missing_participant_stays_missing():
    g=compile_candidate_graph('kitchen','task',doc(role()))
    assert set(g.nodes) == {'coffee_container'}
    assert not g.operation_groups


@pytest.mark.parametrize('domain', ['kitchen','living_room','workshop'])
def test_ideal_graphs_remain_full(domain):
    paths=list((Path(__file__).parent/'fixtures/ideal_raw_vlm').glob(domain+'*.json'))
    raw=json.loads(paths[0].read_text())
    g=compile_candidate_graph(domain,'task',raw)
    assert g.metadata['canonicalization_status'] == 'FULL'
    assert len(g.metadata['raw_role_to_canonical']) == len(raw['functional_roles'])


def test_search_controller_updates_observed_graph_and_retains_partial():
    from mujoco_scenes.functional_tamp_pipeline.scene_graph import ObservedNode, ObservedSceneGraph
    from mujoco_scenes.functional_tamp_pipeline.grounding import ground_graph, ground_verified_candidate_subgraph
    from mujoco_scenes.functional_tamp_pipeline.search import search_until_satisfied
    from mujoco_scenes.functional_tamp_pipeline.search_contract import freeze_search_region_contract
    source=role('source');source.update(function='source of water', candidate_categories=['kettle'])
    graph=compile_candidate_graph('kitchen','Prepare a drink',doc(role(),source))
    class Domain:
        def __init__(self):
            self.graph=ObservedSceneGraph();self.opened=[]
        def observe_initial(self):
            pass
        def evaluate_satisfaction(self, search_exhausted=False):
            return ground_graph(graph,self.graph,{'search_exhausted':search_exhausted})
        def open_region(self,region):
            self.opened.append(region);return {'success':True}
        def observe_after_open(self,region):
            self.graph.add_node(ObservedNode(instance_id='found_cup',entity_kind='OBJECT',canonical_category=graph.nodes['coffee_container'].semantic_categories[0],source_region=region))
    domain=Domain()
    contract=freeze_search_region_contract(graph,domain='kitchen',mode='vlm',source='auto',variant='K1')
    result, inspected=search_until_satisfied(domain,graph,search_contract=contract,emit=lambda _:None)
    assert inspected and domain.opened == list(inspected)
    assert 'found_cup' in domain.graph.nodes and not result.complete
    partial=ground_verified_candidate_subgraph(graph,domain.graph)
    assert partial.assignment == {'coffee_container':'found_cup'}
    assert partial.missing_roles == ('water_source',)


def test_raw_scoring_precedes_structural_acceptance():
    from mujoco_scenes.functional_tamp_pipeline.raw_semantic_evaluation import evaluate_raw_semantics
    raw=doc(role());raw['functional_relations']=[dict(subject_role='vessel',relation='fits inside',object_role='missing')]
    metrics=evaluate_raw_semantics('kitchen','task',raw)
    assert metrics['role']['recall'] > 0
    assert metrics['relation']['precision'] == 0


def test_zero_eligible_denominator():
    from mujoco_scenes.functional_tamp_pipeline.evaluation_metrics import rate, format_rate
    assert format_rate(rate(0,0)) == 'N/A'


def test_invalid_nonempty_plan_is_not_valid(tmp_path):
    from mujoco_scenes.functional_tamp_pipeline.evaluation_metrics import candidate_plan_valid
    (tmp_path/'action_plan.json').write_text(json.dumps({'actions':[{'operator':'PICK','arguments':['ghost']}], 'validation':{'status':'INVALID'}}))
    assert not candidate_plan_valid(tmp_path)


def test_single_astar_partial_is_independently_replayed(tmp_path):
    from mujoco_scenes.functional_tamp_pipeline.planning import plan_with_common_astar
    from mujoco_scenes.functional_tamp_pipeline.telemetry import current_run, RunTelemetry
    from mujoco_scenes.symbolic_planning_core import SymbolicAction, SymbolicProblem
    class Compiler:
        def compile_problem(self,assignment,context):
            action=SymbolicAction(name='MOVE',arguments=('observed',),positive_preconditions=frozenset({('ready',)}),negative_preconditions=frozenset(),add_effects=frozenset({('done',)}),delete_effects=frozenset({('ready',)}))
            return SymbolicProblem(initial_atoms=frozenset({('ready',)}),goal_atoms=frozenset({('done',),('unreachable',)}),actions=(action,))
    telemetry=RunTelemetry(tmp_path);token=current_run.set(telemetry)
    try:
        result=plan_with_common_astar(Compiler(),{}, {},allow_partial=True)
    finally:
        current_run.reset(token)
    assert result.actions and result.validation['status'] == 'VALID'
    assert telemetry.astar_invocations == 1
    assert result.search.statistics['is_partial']


def test_gt_evaluation_cannot_change_compiler_result(monkeypatch):
    from mujoco_scenes.functional_tamp_pipeline.gt_spec_provider import GTSpecProvider
    raw=doc(role())
    before=compile_candidate_graph('kitchen','task',raw).to_dict()
    monkeypatch.setattr(GTSpecProvider,'provide',lambda *a,**k: (_ for _ in ()).throw(AssertionError('GT accessed online')))
    after=compile_candidate_graph('kitchen','task',raw).to_dict()
    assert after == before


def test_required_safety_property_blocks_only_its_role():
    from mujoco_scenes.functional_tamp_pipeline.executability import analyze_executability
    r=role();r['required_properties']=['must be electrically insulated']
    other=role('other');other['function']='source of water'
    graph=compile_candidate_graph('kitchen','task',doc(r,other))
    assert graph.metadata['canonicalization_status']=='PARTIAL'
    assert analyze_executability(graph)[0]['status']=='UNVERIFIABLE_REQUIRED_PROPERTY'
    assert 'water_source' in graph.nodes


def test_multisignal_instrument_role_and_all_group_relations_preserved():
    raw=json.loads((Path(__file__).parent/'fixtures/ideal_raw_vlm/workshop_W1.json').read_text())
    raw['functional_roles'][0]['function']='connection implement'
    raw['functional_roles'][0]['description']='reusable operation instrument'
    raw['interaction_groups'][0]['required_relations']=['compatible with','unknown interface condition']
    graph=compile_candidate_graph('workshop','task',raw)
    assert 'driver' in graph.nodes
    assert any(r.predicate=='COMPATIBLE_WITH' for r in graph.relations)
    assert graph.metadata['unresolved_semantics']
    assert graph.metadata['canonicalization_status']=='PARTIAL'


def test_serialized_replay_rejects_tampered_actions(tmp_path):
    from mujoco_scenes.functional_tamp_pipeline.evaluation_metrics import replay_saved_plan
    from mujoco_scenes.functional_tamp_pipeline.telemetry import RunTelemetry, current_run
    from mujoco_scenes.symbolic_planning_core import SymbolicAction, SymbolicProblem, deterministic_astar
    action=SymbolicAction('MOVE',('observed',),frozenset({('ready',)}),frozenset(),frozenset({('done',)}),frozenset())
    problem=SymbolicProblem(frozenset({('ready',)}),frozenset({('done',)}),(action,))
    token=current_run.set(RunTelemetry(tmp_path))
    try:
        deterministic_astar(problem)
    finally:
        current_run.reset(token)
    (tmp_path/'action_plan.json').write_text(json.dumps({'actions':[{'operator':'MOVE','arguments':['ghost']}]}))
    assert replay_saved_plan(tmp_path)['status']=='INVALID'
    (tmp_path/'action_plan.json').write_text(json.dumps({'actions':[{'operator':'MOVE','arguments':['observed']}]}))
    assert replay_saved_plan(tmp_path)['status']=='VALID'


def test_observer_diagnostics_bound_combinatorial_assignment_logs(tmp_path):
    from mujoco_scenes.observed_state import _atomic_json
    evaluations = [{'decision': 'REJECTED', 'index': index} for index in range(750)]
    payload = {'modes': {'joint': {'assignment_evaluations': evaluations}}}
    _atomic_json(tmp_path / 'witness.json', payload)
    saved = json.loads((tmp_path / 'witness.json').read_text())
    joint = saved['modes']['joint']
    assert len(joint['assignment_evaluations']) == 500
    assert joint['assignment_evaluations_total_count'] == 750
    assert joint['assignment_evaluations_truncated'] is True
    assert len(evaluations) == 750


def test_joint_grounding_streams_large_rejection_diagnostics():
    from mujoco_scenes.task_witness import evaluate_joint_task_witness
    objects = [
        {'id': f'object:{index}', 'type': 'object', 'attributes': {
            'object_id': f'object_{index:04d}', 'canonical_label': 'item',
            'semantic_observations': [],
        }} for index in range(6)
    ]
    graph = {'stage': 0, 'nodes': objects, 'edges': []}
    requirements = {
        'task_id': 'bounded-diagnostics', '_task_schema': 'JOINT_ROLE_GROUNDING',
        'specification_source': 'test',
        'roles': {
            f'role_{index}': {
                'count': 1, 'assignment_order': index,
                'semantic_preferences': [{'canonical_label': 'item', 'rank': 1}],
                'unary_geometry': [], 'allow_empty_geometry': True,
            } for index in range(4)
        },
        'constraints': {'distinct_objects': True, 'pairwise': []},
        'selection': {},
    }
    result = evaluate_joint_task_witness(graph, requirements, grounding_mode='semantic-only')
    assert result['assignment_evaluations_total_count'] == 6 ** 4
    assert len(result['assignment_evaluations']) == 500
    assert result['assignment_evaluations_truncated'] is True
