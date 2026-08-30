"""Pass 3.6A.7.2 Final Non-Oracular VLM Interface Freeze Contract Tests.

Tests:
1. RESPONSE_SCHEMA <-> Python validator parity (generic & kitchen)
2. Purely generic runtime G_F validator (zero expected task nouns)
3. Runtime vs offline evaluation separation (incomplete candidate passes runtime, fails offline eval)
4. Static import graph isolation (runtime modules never import gf_reference_evaluator)
5. True raw-VLM -> canonical G_F synthetic path and three-layer provenance
6. Downstream graph compiler non-mutation invariance
7. Living Room production G_O grounding & Workshop open-vocabulary category preservation
8. Version consistency and zero benchmark leakage
"""

from __future__ import annotations

import ast
from copy import deepcopy
import inspect
import json
from pathlib import Path
from typing import Any
import pytest

try:
    import jsonschema
except ImportError:
    jsonschema = None

from mujoco_scenes.environment_vlm_requirements import (
    EnvironmentVLMRequirementProvider,
    VLM_CANONICALIZATION_VERSION as ENV_VLM_VERSION,
)
from mujoco_scenes.functional_tamp_pipeline.errors import (
    MalformedVLMSpecificationError,
    VLMSpecificationError,
)
from mujoco_scenes.functional_tamp_pipeline.gf_reference_evaluator import (
    GFReferenceEvaluationResult,
    evaluate_gf_against_reference,
)
from mujoco_scenes.functional_tamp_pipeline.grounding import ground_graph
from mujoco_scenes.functional_tamp_pipeline.gt_spec_provider import GTSpecProvider
from mujoco_scenes.functional_tamp_pipeline.models import (
    FunctionalRelation,
    FunctionalRequirementGraph,
    FunctionalRole,
    OperationGroup,
)
from mujoco_scenes.functional_tamp_pipeline.scene_graph import (
    ObservedNode,
    ObservedRelation,
    ObservedSceneGraph,
)
from mujoco_scenes.functional_tamp_pipeline.task_interface_validator import (
    validate_canonical_task_interface,
    validate_runtime_gf,
)
from mujoco_scenes.functional_tamp_pipeline.vlm_spec_provider import (
    VLM_CANONICALIZATION_VERSION,
    VLMSpecProvider,
)
from mujoco_scenes.kitchen_vlm_functional_graph import (
    VLM_CANONICALIZATION_VERSION as KITCHEN_VLM_VERSION,
    compile_vlm_functional_graph,
    validate_kitchen_functional_specification,
)
from mujoco_scenes.workshop_phase1.fm_adapter import (
    KITCHEN_FUNCTIONAL_GRAPH_SCHEMA,
    RESPONSE_SCHEMA,
    FMAdapter,
    FMResponseValidationError,
    SYSTEM_PROMPT,
    validate_requirement_response,
)
from mujoco_scenes.workshop_phase1.requirements import FMRequirementProvider


class MockFMAdapter:
    def __init__(self, doc: dict[str, Any], raw_doc: dict[str, Any] | None = None):
        self._doc = doc
        self.last_raw_requirement_response = raw_doc or doc
        self.last_validated_requirement_response = doc
        self.last_raw_kitchen_graph_response = raw_doc or doc
        self.last_validated_kitchen_graph_response = doc
        self.last_raw_inspection_response = None
        self.last_raw_response = raw_doc or doc
        self.last_observation_images = []
        self.metrics = type("Metrics", (), {"total_calls": 1, "requirement_calls": 1, "inspection_calls": 0})()

    def generate_task_requirements(self, task_instruction: str, *, observation_images: list[Any] | None = None) -> dict[str, Any]:
        return validate_requirement_response(deepcopy(self._doc))

    def generate_kitchen_functional_graph(self, task_instruction: str, *, observation_images: list[Any] | None = None) -> dict[str, Any]:
        return validate_kitchen_functional_specification(deepcopy(self._doc))

    def generate_inspection_priors(self, task_instruction: str, search_region_descriptors: Any = None, *, observation_images: list[Any] | None = None) -> dict[str, Any]:
        return {"inspectable_regions": [], "inspection_order": [], "confidence": 1.0}


def _valid_generic_vlm_doc() -> dict[str, Any]:
    return {
        "status": "SUPPORTED",
        "task_summary": "Perform assembly with tool and fastener",
        "functional_roles": [
            {
                "id": "driver_tool",
                "entity_kind": "OBJECT",
                "function": "drive screws into hole",
                "description": "tool for driving screws",
                "required_count": 1,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["screwdriver"],
                "visible_candidates": [{"label": "driver", "visual_description": "black tool"}],
                "required_properties": [],
            },
            {
                "id": "fastener_part",
                "entity_kind": "OBJECT",
                "function": "threaded screw to join parts",
                "description": "fastener part",
                "required_count": 1,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["screw"],
                "visible_candidates": [{"label": "screw", "visual_description": "silver screw"}],
                "required_properties": [],
            },
            {
                "id": "target_joint",
                "entity_kind": "FIXED_TARGET",
                "function": "workbench hole for repair",
                "description": "repair target hole",
                "required_count": 1,
                "binding_policy": "SHARED",
                "candidate_categories": ["hole"],
                "visible_candidates": [{"label": "hole", "visual_description": "recess"}],
                "required_properties": [],
            },
        ],
        "functional_relations": [
            {"subject_role": "driver_tool", "relation": "compatible with", "object_role": "fastener_part"},
            {"subject_role": "driver_tool", "relation": "reaches target", "object_role": "target_joint"},
            {"subject_role": "fastener_part", "relation": "threads into target hole", "object_role": "target_joint"},
        ],
        "interaction_groups": [
            {
                "id": "fastening_op",
                "function": "fasten joint",
                "tool_role": "driver_tool",
                "target_role": "fastener_part",
                "required_target_count": 1,
                "usage_policy": "SEQUENTIAL_REUSE_ALLOWED",
                "required_relations": ["compatible with"],
            }
        ],
        "inspectable_regions": [],
        "inspection_order": [],
        "unsupported_reason": "",
    }


def _valid_kitchen_vlm_doc() -> dict[str, Any]:
    return {
        "status": "SUPPORTED",
        "task_summary": "Prepare two cups of coffee and two bowls of soup",
        "functional_roles": [
            {
                "id": "coffee_mug",
                "entity_kind": "OBJECT",
                "function": "contain individual coffee serving",
                "required_count": 2,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["coffee mug", "cup"],
                "visible_candidates": [
                    {"label": "mug_1", "visual_description": "white mug"},
                    {"label": "mug_2", "visual_description": "blue mug"},
                ],
                "required_properties": ["open cavity"],
            },
            {
                "id": "soup_bowl",
                "entity_kind": "OBJECT",
                "function": "contain individual soup serving",
                "required_count": 2,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["soup bowl", "bowl"],
                "visible_candidates": [
                    {"label": "bowl_1", "visual_description": "ceramic bowl"},
                    {"label": "bowl_2", "visual_description": "glass bowl"},
                ],
                "required_properties": ["open cavity"],
            },
            {
                "id": "spoon_tool",
                "entity_kind": "OBJECT",
                "function": "stir coffee in container",
                "required_count": 1,
                "binding_policy": "REUSABLE",
                "candidate_categories": ["spoon", "stirrer"],
                "visible_candidates": [{"label": "spoon", "visual_description": "metal spoon"}],
                "required_properties": ["elongated object"],
            },
            {
                "id": "soup_spoon_tool",
                "entity_kind": "OBJECT",
                "function": "eat soup from bowl",
                "required_count": 2,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["soup spoon", "spoon"],
                "visible_candidates": [
                    {"label": "soup_spoon_1", "visual_description": "large spoon 1"},
                    {"label": "soup_spoon_2", "visual_description": "large spoon 2"},
                ],
                "required_properties": ["elongated object"],
            },
            {
                "id": "coffee_jar",
                "entity_kind": "OBJECT",
                "function": "provide coffee material",
                "required_count": 1,
                "binding_policy": "REUSABLE",
                "candidate_categories": ["coffee jar", "coffee container"],
                "visible_candidates": [{"label": "jar", "visual_description": "glass jar with beans"}],
                "required_properties": ["open cavity"],
            },
            {
                "id": "water_kettle",
                "entity_kind": "OBJECT",
                "function": "provide hot water",
                "required_count": 1,
                "binding_policy": "REUSABLE",
                "candidate_categories": ["kettle", "water jug"],
                "visible_candidates": [{"label": "kettle", "visual_description": "electric kettle"}],
                "required_properties": ["open cavity"],
            },
        ],
        "functional_relations": [
            {"subject_role": "spoon_tool", "relation": "reaches bottom", "object_role": "coffee_mug"},
            {"subject_role": "soup_spoon_tool", "relation": "enter opening", "object_role": "soup_bowl"},
        ],
        "interaction_groups": [
            {
                "id": "coffee_stirring_op",
                "function": "stir coffee",
                "tool_role": "spoon_tool",
                "target_role": "coffee_mug",
                "required_target_count": 2,
                "usage_policy": "SEQUENTIAL_REUSE_ALLOWED",
                "required_relations": ["reaches bottom"],
            },
            {
                "id": "soup_serving_op",
                "function": "serve soup",
                "tool_role": "soup_spoon_tool",
                "target_role": "soup_bowl",
                "required_target_count": 2,
                "usage_policy": "DEDICATED_PER_TARGET",
                "required_relations": ["enter opening"],
            },
        ],
        "cross_group_reuse_allowed": False,
        "inspectable_regions": [
            {"id": "drawer_top", "label": "top kitchen drawer", "visual_description": "wooden upper drawer", "reason": "utensil storage"}
        ],
        "inspection_order": ["drawer_top"],
        "unsupported_reason": "",
    }


def _valid_living_vlm_doc() -> dict[str, Any]:
    return {
        "status": "SUPPORTED",
        "task_summary": "Two person tea serving in living room",
        "functional_roles": [
            {"id": "cup_set", "entity_kind": "OBJECT", "function": "cup saucer set beverage payload", "required_count": 2, "binding_policy": "DISTINCT", "candidate_categories": ["cup"], "visible_candidates": [], "required_properties": []},
            {"id": "seat", "entity_kind": "FIXED_TARGET", "function": "seating position armchair", "required_count": 2, "binding_policy": "DISTINCT", "candidate_categories": ["armchair"], "visible_candidates": [], "required_properties": []},
            {"id": "personal_surface", "entity_kind": "REGION", "function": "personal beverage support surface", "required_count": 2, "binding_policy": "DISTINCT", "candidate_categories": ["side_table"], "visible_candidates": [], "required_properties": ["planar support surface"]},
            {"id": "remote_obj", "entity_kind": "OBJECT", "function": "television remote control", "required_count": 1, "binding_policy": "DISTINCT", "candidate_categories": ["remote_control"], "visible_candidates": [], "required_properties": []},
            {"id": "seat_pair", "entity_kind": "FIXED_TARGET", "function": "seating pair both seats", "required_count": 1, "binding_policy": "SHARED", "candidate_categories": ["seating_pair"], "visible_candidates": [], "required_properties": []},
            {"id": "shared_surface", "entity_kind": "REGION", "function": "shared central remote placement table", "required_count": 1, "binding_policy": "SHARED", "candidate_categories": ["coffee_table"], "visible_candidates": [], "required_properties": ["planar support surface"]},
        ],
        "functional_relations": [
            {"subject_role": "personal_surface", "relation": "fits cup and saucer set", "object_role": "cup_set"},
            {"subject_role": "personal_surface", "relation": "near seat", "object_role": "seat"},
            {"subject_role": "shared_surface", "relation": "fits television remote control", "object_role": "remote_obj"},
            {"subject_role": "shared_surface", "relation": "accessible from both seats", "object_role": "seat_pair"},
        ],
        "interaction_groups": [
            {
                "id": "personal_support",
                "function": "provide personal beverage surface for seated viewer",
                "tool_role": "personal_surface",
                "target_role": "cup_set",
                "required_target_count": 2,
                "usage_policy": "DEDICATED_PER_TARGET",
                "required_relations": ["fits cup and saucer set"],
                "context_role": "seat",
                "context_relations": ["near seat"],
            }
        ],
        "inspectable_regions": [],
        "inspection_order": [],
        "unsupported_reason": "",
    }


# ===========================================================================
# 1. Schema & Validator Parity Tests
# ===========================================================================
def test_a_generic_response_schema_parity_valid():
    assert "interaction_groups" in RESPONSE_SCHEMA["required"]
    assert "required_relations" in RESPONSE_SCHEMA["properties"]["interaction_groups"]["items"]["required"]
    assert RESPONSE_SCHEMA["properties"]["interaction_groups"]["items"]["properties"]["required_relations"]["minItems"] == 1
    doc = _valid_generic_vlm_doc()
    if jsonschema is not None:
        jsonschema.validate(instance=doc, schema=RESPONSE_SCHEMA)
    validated = validate_requirement_response(doc)
    assert validated["status"] == "SUPPORTED"
    assert len(validated["interaction_groups"]) == 1


def test_b_generic_response_schema_missing_interaction_groups_rejected():
    doc = _valid_generic_vlm_doc()
    del doc["interaction_groups"]
    if jsonschema is not None:
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=doc, schema=RESPONSE_SCHEMA)
    with pytest.raises(FMResponseValidationError) as exc:
        validate_requirement_response(doc)
    assert "interaction_groups" in str(exc.value)


def test_c_generic_response_schema_empty_required_relations_rejected():
    doc = _valid_generic_vlm_doc()
    doc["interaction_groups"][0]["required_relations"] = []
    if jsonschema is not None:
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=doc, schema=RESPONSE_SCHEMA)
    with pytest.raises(FMResponseValidationError) as exc:
        validate_requirement_response(doc)
    assert "required_relations" in str(exc.value)


def test_d_kitchen_response_schema_parity_valid():
    assert "interaction_groups" in KITCHEN_FUNCTIONAL_GRAPH_SCHEMA["required"]
    assert "required_relations" in KITCHEN_FUNCTIONAL_GRAPH_SCHEMA["properties"]["interaction_groups"]["items"]["required"]
    assert KITCHEN_FUNCTIONAL_GRAPH_SCHEMA["properties"]["interaction_groups"]["items"]["properties"]["required_relations"]["minItems"] == 1
    doc = _valid_kitchen_vlm_doc()
    if jsonschema is not None:
        jsonschema.validate(instance=doc, schema=KITCHEN_FUNCTIONAL_GRAPH_SCHEMA)
    validated = validate_kitchen_functional_specification(doc)
    assert validated["status"] == "SUPPORTED"
    assert len(validated["interaction_groups"]) == 2


def test_e_kitchen_response_schema_missing_required_relations_rejected():
    doc = _valid_kitchen_vlm_doc()
    del doc["interaction_groups"][0]["required_relations"]
    if jsonschema is not None:
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=doc, schema=KITCHEN_FUNCTIONAL_GRAPH_SCHEMA)
    with pytest.raises(FMResponseValidationError):
        validate_kitchen_functional_specification(doc)


# ===========================================================================
# 2. Generic Runtime Validator Tests
# ===========================================================================
def test_f_generic_runtime_gf_passes_valid_graphs():
    for domain, instr in [
        ("kitchen", "prepare coffee and soup"),
        ("living_room", "serve tea for two"),
        ("workshop", "fasten joint with driver and screw"),
    ]:
        gt_gf = GTSpecProvider().provide(domain, instr)
        validate_runtime_gf(gt_gf)
        validate_canonical_task_interface(gt_gf)


def test_g_generic_runtime_gf_rejects_missing_endpoint():
    gf = GTSpecProvider().provide("workshop", "fasten joint")
    bad_rel = FunctionalRelation(subject_role="driver", predicate="COMPATIBLE_WITH", object_role="non_existent_node", expected=True)
    bad_gf = FunctionalRequirementGraph(
        domain="workshop", task_instruction=gf.task_instruction,
        nodes=dict(gf.nodes), relations=(bad_rel,),
    )
    with pytest.raises(MalformedVLMSpecificationError) as exc:
        validate_runtime_gf(bad_gf)
    assert "non_existent_node" in str(exc.value)


def test_h_generic_runtime_gf_rejects_empty_required_relations():
    gf = GTSpecProvider().provide("kitchen", "prepare coffee and soup")
    bad_op = OperationGroup(
        id="op1", function="stir", tool_role="coffee_stirrer", target_role="coffee_container",
        required_target_count=2, usage_policy="SEQUENTIAL_REUSE_ALLOWED", required_relations=(),
    )
    bad_gf = FunctionalRequirementGraph(
        domain="kitchen", task_instruction=gf.task_instruction,
        nodes=dict(gf.nodes), relations=gf.relations, operation_groups=(bad_op,),
    )
    with pytest.raises(MalformedVLMSpecificationError) as exc:
        validate_runtime_gf(bad_gf)
    assert "empty required_relations" in str(exc.value)


def test_i_generic_runtime_gf_rejects_context_role_without_relations():
    bad_op = OperationGroup(
        id="op1", function="personal", tool_role="table", target_role="cup",
        required_target_count=1, usage_policy="DEDICATED_PER_TARGET",
        required_relations=("fits",), context_role="seat", context_relations=(),
    )
    nodes = {
        "table": FunctionalRole(name="table", entity_kind="REGION", count=1, binding_policy="DISTINCT"),
        "cup": FunctionalRole(name="cup", entity_kind="OBJECT", count=1, binding_policy="DISTINCT"),
        "seat": FunctionalRole(name="seat", entity_kind="FIXED_TARGET", count=1, binding_policy="DISTINCT"),
    }
    bad_gf = FunctionalRequirementGraph(domain="living_room", task_instruction="task", nodes=nodes, operation_groups=(bad_op,))
    with pytest.raises(MalformedVLMSpecificationError) as exc:
        validate_runtime_gf(bad_gf)
    assert "empty context_relations" in str(exc.value)


def test_j_generic_runtime_gf_rejects_context_relations_without_role():
    bad_op = OperationGroup(
        id="op1", function="personal", tool_role="table", target_role="cup",
        required_target_count=1, usage_policy="DEDICATED_PER_TARGET",
        required_relations=("fits",), context_role=None, context_relations=("near",),
    )
    nodes = {
        "table": FunctionalRole(name="table", entity_kind="REGION", count=1, binding_policy="DISTINCT"),
        "cup": FunctionalRole(name="cup", entity_kind="OBJECT", count=1, binding_policy="DISTINCT"),
    }
    bad_gf = FunctionalRequirementGraph(domain="living_room", task_instruction="task", nodes=nodes, operation_groups=(bad_op,))
    with pytest.raises(MalformedVLMSpecificationError) as exc:
        validate_runtime_gf(bad_gf)
    assert "context_relations without context_role" in str(exc.value)


def test_k_generic_runtime_gf_has_zero_domain_nouns():
    src = Path("mujoco_scenes/functional_tamp_pipeline/task_interface_validator.py").read_text(encoding="utf-8")
    forbidden = [
        "coffee_container", "coffee_stirrer", "soup_container", "soup_eating_utensil",
        "CUP_SAUCER_SET", "PERSONAL_CUP_SAUCER_REGION", "SHARED_REMOTE_REGION",
        "driver", "fastener", "repair_target", "FITS_ON", "NEAR_SEAT",
        "COMPATIBLE_WITH", "REACHES_TARGET", "COMPATIBLE_WITH_TARGET",
    ]
    for noun in forbidden:
        assert noun not in src, f"Forbidden task/benchmark noun {noun!r} found in task_interface_validator.py"


# ===========================================================================
# 3. Runtime vs Offline Evaluation Separation / Non-Oracle Tests
# ===========================================================================
def test_l_incomplete_candidate_passes_runtime_fails_offline_eval():
    # Candidate graph specifies ONLY driver and fastener (omitting repair_target)
    cand_nodes = {
        "driver": FunctionalRole(name="driver", entity_kind="OBJECT", count=1, semantic_categories=("screwdriver",), binding_policy="DISTINCT"),
        "fastener": FunctionalRole(name="fastener", entity_kind="OBJECT", count=1, semantic_categories=("screw",), binding_policy="DISTINCT"),
    }
    cand_rels = (
        FunctionalRelation(subject_role="driver", predicate="COMPATIBLE_WITH", object_role="fastener", expected=True),
    )
    candidate = FunctionalRequirementGraph(
        domain="workshop",
        task_instruction="Fasten joint securely",
        nodes=cand_nodes,
        relations=cand_rels,
    )
    # 1. RUNTIME VALIDATOR MUST PASS (generic structural consistency only)
    validate_runtime_gf(candidate)

    # 2. OFFLINE EVALUATOR MUST DETECT INCOMPLETENESS
    eval_result = evaluate_gf_against_reference(candidate)
    assert isinstance(eval_result, GFReferenceEvaluationResult)
    assert not eval_result.structurally_complete
    assert "repair_target" in eval_result.missing_roles
    assert eval_result.role_recall < 1.0
    assert ("driver", "REACHES_TARGET", "repair_target") in eval_result.missing_relations


def test_m_incomplete_kitchen_passes_runtime_fails_offline_eval():
    # Candidate graph specifies coffee tools but omits soup eating utensil
    ref = GTSpecProvider().provide("kitchen", "prepare coffee and soup")
    nodes = dict(ref.nodes)
    del nodes["soup_eating_utensil"]
    rels = tuple(r for r in ref.relations if r.subject_role in nodes and r.object_role in nodes)
    ops = tuple(g for g in ref.operation_groups if g.tool_role in nodes and g.target_role in nodes)
    candidate = FunctionalRequirementGraph(
        domain="kitchen",
        task_instruction=ref.task_instruction,
        nodes=nodes,
        relations=rels,
        operation_groups=ops,
    )
    # RUNTIME PASSES (candidate graph is structurally self-consistent)
    validate_runtime_gf(candidate)

    # OFFLINE EVAL DETECTS MISSING ROLE
    eval_result = evaluate_gf_against_reference(candidate, ref)
    assert not eval_result.structurally_complete
    assert "soup_eating_utensil" in eval_result.missing_roles
    assert eval_result.role_recall < 1.0


def test_n_static_import_graph_isolation():
    runtime_modules = [
        "mujoco_scenes/functional_tamp_pipeline/grounding.py",
        "mujoco_scenes/functional_tamp_pipeline/vlm_spec_provider.py",
        "mujoco_scenes/functional_tamp_pipeline/gt_spec_provider.py",
        "mujoco_scenes/functional_tamp_pipeline/domains/kitchen.py",
        "mujoco_scenes/functional_tamp_pipeline/domains/living_room.py",
        "mujoco_scenes/functional_tamp_pipeline/domains/workshop.py",
        "mujoco_scenes/workshop_phase1/perception.py",
        "mujoco_scenes/workshop_phase1/tracking.py",
        "mujoco_scenes/workshop_phase1/geometric_grounding.py",
        "mujoco_scenes/workshop_phase1/inspection_controller.py",
        "mujoco_scenes/workshop_phase1/requirements.py",
        "mujoco_scenes/environment_vlm_requirements.py",
        "mujoco_scenes/kitchen_vlm_functional_graph.py",
    ]
    for mod_path in runtime_modules:
        p = Path(mod_path)
        if not p.is_file():
            continue
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "gf_reference_evaluator" not in alias.name, f"{mod_path} imports gf_reference_evaluator"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert "gf_reference_evaluator" not in node.module, f"{mod_path} imports from gf_reference_evaluator"


# ===========================================================================
# 4. True Synthetic Raw-VLM -> Canonical G_F Pipeline & 3-Layer Provenance
# ===========================================================================
def test_o_kitchen_synthetic_raw_to_canonical_provenance():
    raw_doc = _valid_kitchen_vlm_doc()
    adapter = MockFMAdapter(raw_doc)
    spec_provider = VLMSpecProvider()
    gf = spec_provider._kitchen("prepare coffee and soup", [], adapter=adapter)

    assert gf.domain == "kitchen"
    assert "coffee_container" in gf.nodes
    assert "soup_container" in gf.nodes
    assert "coffee_stirrer" in gf.nodes
    assert "soup_eating_utensil" in gf.nodes
    assert len(gf.operation_groups) == 2

    # Three-layer provenance assertions
    assert "raw_vlm_response" in gf.metadata
    assert "validated_vlm_specification" in gf.metadata
    assert "canonicalization_trace" in gf.metadata
    assert gf.metadata["vlm_canonicalization_version"] == "phase3_6a7_2_v1"
    assert gf.metadata["raw_vlm_response"] == raw_doc


def test_p_workshop_synthetic_raw_to_canonical_provenance():
    raw_doc = _valid_generic_vlm_doc()
    adapter = MockFMAdapter(raw_doc)
    provider = FMRequirementProvider(fm_adapter=adapter)
    spec_provider = VLMSpecProvider()
    gf = spec_provider._workshop("fasten joint", [], provider=provider)

    assert gf.domain == "workshop"
    assert "driver" in gf.nodes
    assert "fastener" in gf.nodes
    assert "repair_target" in gf.nodes

    # Three-layer provenance assertions
    assert "raw_vlm_response" in gf.metadata
    assert "validated_vlm_specification" in gf.metadata
    assert "canonicalization_trace" in gf.metadata
    assert gf.metadata["vlm_canonicalization_version"] == "phase3_6a7_2_v1"
    assert gf.metadata["raw_vlm_response"] == raw_doc


def test_q_living_room_synthetic_raw_to_canonical_provenance():
    raw_doc = _valid_living_vlm_doc()
    adapter = MockFMAdapter(raw_doc)
    provider = EnvironmentVLMRequirementProvider("living_room", fm_adapter=adapter)
    spec_provider = VLMSpecProvider()
    gf = spec_provider._living_room("serve tea for two", [], provider=provider)

    assert gf.domain == "living_room"
    assert "PERSONAL_CUP_SAUCER_REGION" in gf.nodes
    assert "SHARED_REMOTE_REGION" in gf.nodes
    assert "CUP_SAUCER_SET" in gf.nodes
    assert "REMOTE" in gf.nodes
    assert "SEATING_POSITION" in gf.nodes
    assert "SEATING_PAIR" in gf.nodes

    # Three-layer provenance assertions
    assert "raw_vlm_response" in gf.metadata
    assert "validated_vlm_specification" in gf.metadata
    assert "canonicalization_trace" in gf.metadata
    assert gf.metadata["vlm_canonicalization_version"] == "phase3_6a7_2_v1"


# ===========================================================================
# 5. Graph Compiler Non-Mutation Invariance
# ===========================================================================
def test_r_graph_compilers_do_not_mutate_gf():
    from mujoco_scenes.functional_tamp_pipeline.domains.kitchen import compile_kitchen_contract_from_graph
    from mujoco_scenes.functional_tamp_pipeline.domains.workshop import compile_workshop_requirements_from_graph

    # Kitchen compiler invariance
    kitchen_gf = GTSpecProvider().provide("kitchen", "prepare coffee and soup")
    kitchen_before = deepcopy(kitchen_gf)
    _ = compile_kitchen_contract_from_graph(kitchen_gf)
    assert kitchen_gf.nodes == kitchen_before.nodes
    assert kitchen_gf.relations == kitchen_before.relations
    assert kitchen_gf.operation_groups == kitchen_before.operation_groups
    assert kitchen_gf.metadata == kitchen_before.metadata

    # Workshop compiler invariance
    workshop_gf = GTSpecProvider().provide("workshop", "fasten joint")
    workshop_before = deepcopy(workshop_gf)
    _ = compile_workshop_requirements_from_graph(workshop_gf)
    assert workshop_gf.nodes == workshop_before.nodes
    assert workshop_gf.relations == workshop_before.relations
    assert workshop_gf.metadata == workshop_before.metadata


# ===========================================================================
# 6. Living Room Production G_O Grounding & Workshop Open-Vocab
# ===========================================================================
def test_s_living_room_production_go_grounding():
    doc = _valid_living_vlm_doc()
    adapter = MockFMAdapter(doc)
    provider = EnvironmentVLMRequirementProvider("living_room", fm_adapter=adapter)
    spec = VLMSpecProvider()._living_room("serve tea for two", [], provider=provider)

    graph = ObservedSceneGraph()
    graph.add_node(ObservedNode(instance_id="table_left", entity_kind="REGION", canonical_category="side_table", unary_predicates={"PLANAR_SUPPORT": "TRUE"}))
    graph.add_node(ObservedNode(instance_id="table_right", entity_kind="REGION", canonical_category="side_table", unary_predicates={"PLANAR_SUPPORT": "TRUE"}))
    graph.add_node(ObservedNode(instance_id="coffee_table", entity_kind="REGION", canonical_category="coffee_table", unary_predicates={"PLANAR_SUPPORT": "TRUE"}))
    graph.add_node(ObservedNode(instance_id="armchair_left", entity_kind="FIXED_TARGET", canonical_category="armchair"))
    graph.add_node(ObservedNode(instance_id="armchair_right", entity_kind="FIXED_TARGET", canonical_category="armchair"))
    graph.add_node(ObservedNode(instance_id="pair_1", entity_kind="FIXED_TARGET", canonical_category="seating_pair"))
    graph.add_node(ObservedNode(instance_id="set_1", entity_kind="OBJECT", canonical_category="cup"))
    graph.add_node(ObservedNode(instance_id="set_2", entity_kind="OBJECT", canonical_category="cup"))
    graph.add_node(ObservedNode(instance_id="remote_1", entity_kind="OBJECT", canonical_category="remote_control"))

    graph.add_relation(ObservedRelation(subject_id="table_left", predicate="NEAR_SEAT", object_id="armchair_left", status="TRUE"))
    graph.add_relation(ObservedRelation(subject_id="table_right", predicate="NEAR_SEAT", object_id="armchair_right", status="TRUE"))
    graph.add_relation(ObservedRelation(subject_id="coffee_table", predicate="ACCESSIBLE_FROM_BOTH_SEATS", object_id="pair_1", status="TRUE"))
    graph.add_relation(ObservedRelation(subject_id="table_left", predicate="FITS_SET_ON", object_id="set_1", status="TRUE"))
    graph.add_relation(ObservedRelation(subject_id="table_right", predicate="FITS_SET_ON", object_id="set_2", status="TRUE"))
    graph.add_relation(ObservedRelation(subject_id="coffee_table", predicate="FITS_ON", object_id="remote_1", status="TRUE"))

    result = ground_graph(spec, graph)
    assert result.satisfied is True
    assert result.status == "COMPLETE"
    assert len(result.assignment["PERSONAL_CUP_SAUCER_REGION"]) == 2
    assert result.assignment["SHARED_REMOTE_REGION"] == "coffee_table"


def test_t_workshop_open_vocab_categories_evaluate_on_go():
    # Workshop G_F with open-vocabulary categories
    gf = FunctionalRequirementGraph(
        domain="workshop",
        task_instruction="Fasten joint",
        nodes={
            "driver": FunctionalRole(name="driver", entity_kind="OBJECT", count=1, semantic_categories=("powered_screw_tool",), binding_policy="DISTINCT"),
            "fastener": FunctionalRole(name="fastener", entity_kind="OBJECT", count=1, semantic_categories=("threaded_metal_screw",), binding_policy="DISTINCT"),
            "repair_target": FunctionalRole(name="repair_target", entity_kind="FIXED_TARGET", count=1, semantic_categories=("pre_drilled_hole",), binding_policy="SHARED"),
        },
        relations=(
            FunctionalRelation(subject_role="driver", predicate="COMPATIBLE_WITH", object_role="fastener", expected=True),
            FunctionalRelation(subject_role="driver", predicate="REACHES_TARGET", object_role="repair_target", expected=True),
            FunctionalRelation(subject_role="fastener", predicate="COMPATIBLE_WITH_TARGET", object_role="repair_target", expected=True),
        ),
    )
    validate_runtime_gf(gf)

    graph = ObservedSceneGraph()
    graph.add_node(ObservedNode(instance_id="d1", entity_kind="OBJECT", canonical_category="powered_screw_tool"))
    graph.add_node(ObservedNode(instance_id="f1", entity_kind="OBJECT", canonical_category="threaded_metal_screw"))
    graph.add_node(ObservedNode(instance_id="h1", entity_kind="FIXED_TARGET", canonical_category="pre_drilled_hole"))

    graph.add_relation(ObservedRelation(subject_id="d1", predicate="COMPATIBLE_WITH", object_id="f1", status="TRUE"))
    graph.add_relation(ObservedRelation(subject_id="d1", predicate="REACHES_TARGET", object_id="h1", status="TRUE"))
    graph.add_relation(ObservedRelation(subject_id="f1", predicate="COMPATIBLE_WITH_TARGET", object_id="h1", status="TRUE"))

    result = ground_graph(gf, graph)
    assert result.satisfied is True
    assert result.status == "COMPLETE"
    assert result.assignment["driver"] == "d1"
    assert result.assignment["fastener"] == "f1"


# ===========================================================================
# 7. Version Constants and Zero Benchmark Leakage
# ===========================================================================
def test_u_canonicalization_version_constants_consistent():
    assert VLM_CANONICALIZATION_VERSION == "phase3_6a7_2_v1"
    assert ENV_VLM_VERSION == "phase3_6a7_2_v1"
    assert KITCHEN_VLM_VERSION == "phase3_6a7_2_v1"


def test_v_zero_prompt_and_schema_leakage():
    prompt_lower = SYSTEM_PROMPT.lower()
    leaks = [
        "has_handle", "has_threaded_body", "fit_into_hole",
        "wrench", "phillips", "allen_key", "mug_with_handle",
    ]
    for leak in leaks:
        assert leak not in prompt_lower, f"Leaked phrase {leak!r} found in SYSTEM_PROMPT"
