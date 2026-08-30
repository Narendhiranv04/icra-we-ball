"""Pass 3.6A.7 Deterministic VLM Interface Closure and Acceptance Contract Tests.

Tests all required acceptance criteria A through N:
A. required_properties cannot map to a binary relation (UnsupportedCheckerCapabilityError)
B. unknown unary required property -> UnsupportedCheckerCapabilityError
C. valid binary relation -> canonical relation
D. valid synthetic VLM Living JSON -> final VLM G_F -> two-person G_O -> ground_graph COMPLETE
E. invalid two-seat pairing -> NOT COMPLETE
F. GT/VLM Living grounding-relevant structural equivalence
G. Workshop novel driver token -> _sync_common_graph() -> REACHES_TARGET evaluated/stored
H. Workshop novel fastener token -> _sync_common_graph() -> COMPATIBLE_WITH_TARGET evaluated/stored
I. Workshop driver+fastener -> _sync_common_graph() -> COMPATIBLE_WITH evaluated/stored
J. no global ontology entry added for synthetic Workshop labels
K. selectable role with empty candidate_categories fails (MalformedVLMSpecificationError)
L. canonicalization version is final and consistent (phase3_6a7_v1)
M. raw / validated / canonical provenance is clearly separated
N. failure_category backward compatibility works
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
import pytest

from mujoco_scenes.environment_vlm_requirements import (
    EnvironmentVLMRequirementProvider,
    VLM_CANONICALIZATION_VERSION as ENV_VLM_VERSION,
    map_living_room_relation,
)
from mujoco_scenes.functional_tamp_pipeline.domains.workshop import (
    WorkshopDomainAdapter,
)
from mujoco_scenes.functional_tamp_pipeline.errors import (
    AmbiguousCanonicalizationError,
    MalformedVLMSpecificationError,
    TransportOrStructuredOutputError,
    UnmappedFunctionalConceptError,
    UnsupportedCheckerCapabilityError,
    VLMSpecificationError,
)
from mujoco_scenes.functional_tamp_pipeline.grounding import ground_graph
from mujoco_scenes.functional_tamp_pipeline.gt_spec_provider import GTSpecProvider
from mujoco_scenes.functional_tamp_pipeline.models import (
    FunctionalRelation,
    FunctionalRequirementGraph,
    FunctionalRole,
    OperationGroup,
    PipelineResult,
)
from mujoco_scenes.functional_tamp_pipeline.scene_graph import (
    ObservedNode,
    ObservedRelation,
    ObservedSceneGraph,
)
from mujoco_scenes.functional_tamp_pipeline.vlm_spec_provider import (
    VLMSpecProvider,
    VLM_CANONICALIZATION_VERSION,
)
from mujoco_scenes.kitchen_vlm_functional_graph import (
    VLM_CANONICALIZATION_VERSION as KITCHEN_VLM_VERSION,
)
from mujoco_scenes.workshop_phase1.fm_adapter import (
    FMAdapter,
    FMResponseValidationError,
    validate_requirement_response,
)
from mujoco_scenes.workshop_phase1.types import ObservedObjectTrack


class MockFMAdapter(FMAdapter):
    def __init__(self, response_doc: dict[str, Any]):
        super().__init__()
        self._doc = response_doc
        self.last_raw_requirement_response = response_doc
        self.last_raw_inspection_response = {}
        self.last_observation_images = []

    def generate_task_requirements(self, *args, **kwargs) -> dict[str, Any]:
        self.last_raw_requirement_response = self._doc
        self.last_observation_images = []
        return self._doc

    def query_requirements(self, *args, **kwargs) -> dict[str, Any]:
        self.last_raw_requirement_response = self._doc
        self.last_observation_images = []
        return self._doc


# ---------------------------------------------------------------------------
# Test A: required_properties cannot map to a binary relation
# ---------------------------------------------------------------------------
def test_a_required_properties_cannot_map_to_binary_relation():
    provider = EnvironmentVLMRequirementProvider("living_room")
    with pytest.raises(UnsupportedCheckerCapabilityError) as exc_info:
        provider._map_properties(["near seat"], fail_closed=True)
    assert "not supported by any available checker" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test B: unknown unary required property -> UNSUPPORTED_CHECKER_CAPABILITY
# ---------------------------------------------------------------------------
def test_b_unknown_unary_required_property_raises_unsupported_checker():
    provider = EnvironmentVLMRequirementProvider("living_room")
    for unk_prop in ["has_handle", "has_threaded_body", "rigid_material", "magnetic"]:
        with pytest.raises(UnsupportedCheckerCapabilityError) as exc_info:
            provider._map_properties([unk_prop], fail_closed=True)
        assert "not supported by any available checker" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test C: valid binary relation -> canonical relation
# ---------------------------------------------------------------------------
def test_c_valid_binary_relation_maps_to_canonical():
    provider = EnvironmentVLMRequirementProvider("living_room")
    binary_aliases = provider.binary_relation_aliases
    
    assert map_living_room_relation("near seat", binary_aliases) == "NEAR_SEAT"
    assert map_living_room_relation("near armchair", binary_aliases) == "NEAR_SEAT"
    assert map_living_room_relation("fits cup and saucer set", binary_aliases) == "FITS_SET_ON"
    assert map_living_room_relation("support drinkware", binary_aliases) == "FITS_SET_ON"
    assert map_living_room_relation("fit remote", binary_aliases) == "FITS_ON"
    assert map_living_room_relation("accessible from both seats", binary_aliases) == "ACCESSIBLE_FROM_BOTH_SEATS"

    with pytest.raises(UnmappedFunctionalConceptError):
        map_living_room_relation("unknown imaginary relation", binary_aliases)


# ---------------------------------------------------------------------------
# Test D: valid synthetic VLM Living JSON -> final VLM G_F -> two-person G_O -> ground_graph COMPLETE
# ---------------------------------------------------------------------------
def test_d_living_room_vlm_spec_to_two_person_grounding_complete():
    vlm_doc = {
        "status": "SUPPORTED",
        "task_summary": "Two person tea serving in living room",
        "functional_roles": [
            {
                "id": "cup_set",
                "entity_kind": "OBJECT",
                "function": "cup saucer set beverage payload",
                "description": "payload tea set",
                "required_count": 2,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["cup", "saucer"],
                "visible_candidates": [],
                "required_properties": [],
            },
            {
                "id": "seat",
                "entity_kind": "FIXED_TARGET",
                "function": "seating position armchair",
                "description": "armchair position",
                "required_count": 2,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["armchair"],
                "visible_candidates": [],
                "required_properties": [],
            },
            {
                "id": "personal_surface",
                "entity_kind": "REGION",
                "function": "personal beverage support surface",
                "description": "side table next to armchair",
                "required_count": 2,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["side_table"],
                "visible_candidates": [],
                "required_properties": ["planar support surface"],
            },
            {
                "id": "remote_obj",
                "entity_kind": "OBJECT",
                "function": "television remote control",
                "description": "remote control payload",
                "required_count": 1,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["remote_control"],
                "visible_candidates": [],
                "required_properties": [],
            },
            {
                "id": "seat_pair",
                "entity_kind": "FIXED_TARGET",
                "function": "seating pair both seats",
                "description": "viewer armchairs",
                "required_count": 1,
                "binding_policy": "SHARED",
                "candidate_categories": ["seating_pair"],
                "visible_candidates": [],
                "required_properties": [],
            },
            {
                "id": "shared_surface",
                "entity_kind": "REGION",
                "function": "shared central remote placement table",
                "description": "coffee table",
                "required_count": 1,
                "binding_policy": "SHARED",
                "candidate_categories": ["coffee_table"],
                "visible_candidates": [],
                "required_properties": ["planar support surface"],
            },
        ],
        "functional_relations": [
            {"subject_role": "personal_surface", "relation": "fits cup and saucer set", "object_role": "cup_set"},
            {"subject_role": "personal_surface", "relation": "near seat", "object_role": "seat"},
            {"subject_role": "shared_surface", "relation": "fit remote", "object_role": "remote_obj"},
            {"subject_role": "shared_surface", "relation": "accessible from both", "object_role": "seat_pair"},
        ],
        "interaction_groups": [
            {
                "id": "personal_support_group",
                "function": "support drinkware",
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
    adapter = MockFMAdapter(vlm_doc)
    provider = EnvironmentVLMRequirementProvider("living_room", fm_adapter=adapter)
    spec_provider = VLMSpecProvider()
    vlm_gf = spec_provider._living_room("serve tea for two", [], provider=provider)

    assert vlm_gf.domain == "living_room"
    assert len(vlm_gf.operation_groups) == 1
    og = vlm_gf.operation_groups[0]
    assert og.tool_role == "PERSONAL_CUP_SAUCER_REGION"
    assert og.target_role == "CUP_SAUCER_SET"
    assert og.context_role == "SEATING_POSITION"
    assert og.usage_policy == "DEDICATED_PER_TARGET"

    # Construct two-person G_O
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

    # Relations
    graph.add_relation(ObservedRelation(subject_id="table_left", predicate="NEAR_SEAT", object_id="armchair_left", status="TRUE"))
    graph.add_relation(ObservedRelation(subject_id="table_right", predicate="NEAR_SEAT", object_id="armchair_right", status="TRUE"))
    graph.add_relation(ObservedRelation(subject_id="table_left", predicate="FITS_SET_ON", object_id="set_1", status="TRUE"))
    graph.add_relation(ObservedRelation(subject_id="table_right", predicate="FITS_SET_ON", object_id="set_2", status="TRUE"))
    graph.add_relation(ObservedRelation(subject_id="coffee_table", predicate="FITS_ON", object_id="remote_1", status="TRUE"))
    graph.add_relation(ObservedRelation(subject_id="coffee_table", predicate="ACCESSIBLE_FROM_BOTH_SEATS", object_id="pair_1", status="TRUE"))

    result = ground_graph(vlm_gf, graph)
    assert result.satisfied is True
    assert result.status == "COMPLETE"
    assert len(result.assignment["PERSONAL_CUP_SAUCER_REGION"]) == 2
    assert result.assignment["SHARED_REMOTE_REGION"] == "coffee_table"
    assert len(result.operation_bindings["personal_support_group"]) == 2


# ---------------------------------------------------------------------------
# Test E: invalid two-seat pairing -> NOT COMPLETE
# ---------------------------------------------------------------------------
def test_e_invalid_two_seat_pairing_rejected():
    vlm_doc = {
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
            {"subject_role": "shared_surface", "relation": "fit remote", "object_role": "remote_obj"},
            {"subject_role": "shared_surface", "relation": "accessible from both", "object_role": "seat_pair"},
        ],
        "interaction_groups": [
            {
                "id": "personal_support_group", "function": "support drinkware", "tool_role": "personal_surface", "target_role": "cup_set",
                "required_target_count": 2, "usage_policy": "DEDICATED_PER_TARGET", "required_relations": ["fits cup and saucer set"],
                "context_role": "seat", "context_relations": ["near seat"],
            }
        ],
        "inspectable_regions": [], "inspection_order": [], "unsupported_reason": "",
    }
    adapter = MockFMAdapter(vlm_doc)
    provider = EnvironmentVLMRequirementProvider("living_room", fm_adapter=adapter)
    spec_provider = VLMSpecProvider()
    vlm_gf = spec_provider._living_room("serve tea for two", [], provider=provider)

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

    # Both tables claim to be near armchair_left only; neither is near armchair_right
    graph.add_relation(ObservedRelation(subject_id="table_left", predicate="NEAR_SEAT", object_id="armchair_left", status="TRUE"))
    graph.add_relation(ObservedRelation(subject_id="table_right", predicate="NEAR_SEAT", object_id="armchair_left", status="TRUE"))
    graph.add_relation(ObservedRelation(subject_id="table_left", predicate="NEAR_SEAT", object_id="armchair_right", status="FALSE"))
    graph.add_relation(ObservedRelation(subject_id="table_right", predicate="NEAR_SEAT", object_id="armchair_right", status="FALSE"))

    graph.add_relation(ObservedRelation(subject_id="table_left", predicate="FITS_SET_ON", object_id="set_1", status="TRUE"))
    graph.add_relation(ObservedRelation(subject_id="table_right", predicate="FITS_SET_ON", object_id="set_2", status="TRUE"))
    graph.add_relation(ObservedRelation(subject_id="coffee_table", predicate="FITS_ON", object_id="remote_1", status="TRUE"))
    graph.add_relation(ObservedRelation(subject_id="coffee_table", predicate="ACCESSIBLE_FROM_BOTH_SEATS", object_id="pair_1", status="TRUE"))

    result = ground_graph(vlm_gf, graph)
    assert result.satisfied is False
    assert result.status != "COMPLETE"


# ---------------------------------------------------------------------------
# Test F: GT/VLM Living grounding-relevant structural equivalence
# ---------------------------------------------------------------------------
def test_f_gt_vlm_living_grounding_equivalence():
    gt_gf = GTSpecProvider().provide("living_room", "two person tea setup")

    vlm_doc = {
        "status": "SUPPORTED",
        "task_summary": "Two person tea serving",
        "functional_roles": [
            {"id": "cup_set", "entity_kind": "OBJECT", "function": "cup saucer set beverage payload", "required_count": 2, "binding_policy": "DISTINCT", "candidate_categories": ["cup", "saucer"], "visible_candidates": [], "required_properties": []},
            {"id": "seat", "entity_kind": "FIXED_TARGET", "function": "seating position armchair", "required_count": 2, "binding_policy": "DISTINCT", "candidate_categories": ["armchair"], "visible_candidates": [], "required_properties": []},
            {"id": "personal_surface", "entity_kind": "REGION", "function": "personal beverage support surface", "required_count": 2, "binding_policy": "DISTINCT", "candidate_categories": ["side_table"], "visible_candidates": [], "required_properties": ["planar support surface"]},
            {"id": "remote_obj", "entity_kind": "OBJECT", "function": "television remote control", "required_count": 1, "binding_policy": "DISTINCT", "candidate_categories": ["remote_control"], "visible_candidates": [], "required_properties": []},
            {"id": "seat_pair", "entity_kind": "FIXED_TARGET", "function": "seating pair both seats", "required_count": 1, "binding_policy": "SHARED", "candidate_categories": ["armchair"], "visible_candidates": [], "required_properties": []},
            {"id": "shared_surface", "entity_kind": "REGION", "function": "shared central remote placement table", "required_count": 1, "binding_policy": "SHARED", "candidate_categories": ["coffee_table"], "visible_candidates": [], "required_properties": ["planar support surface"]},
        ],
        "functional_relations": [
            {"subject_role": "personal_surface", "relation": "fits cup and saucer set", "object_role": "cup_set"},
            {"subject_role": "personal_surface", "relation": "near seat", "object_role": "seat"},
            {"subject_role": "shared_surface", "relation": "fit remote", "object_role": "remote_obj"},
            {"subject_role": "shared_surface", "relation": "accessible from both", "object_role": "seat_pair"},
        ],
        "interaction_groups": [
            {
                "id": "personal_support_group", "function": "support drinkware", "tool_role": "personal_surface", "target_role": "cup_set",
                "required_target_count": 2, "usage_policy": "DEDICATED_PER_TARGET", "required_relations": ["fits cup and saucer set"],
                "context_role": "seat", "context_relations": ["near seat"],
            }
        ],
        "inspectable_regions": [], "inspection_order": [], "unsupported_reason": "",
    }
    adapter = MockFMAdapter(vlm_doc)
    provider = EnvironmentVLMRequirementProvider("living_room", fm_adapter=adapter)
    spec_provider = VLMSpecProvider()
    vlm_gf = spec_provider._living_room("serve tea for two", [], provider=provider)

    # 1. Canonical functional roles
    assert set(gt_gf.nodes.keys()) == set(vlm_gf.nodes.keys())
    for role_name, gt_node in gt_gf.nodes.items():
        vlm_node = vlm_gf.nodes[role_name]
        assert gt_node.entity_kind == vlm_node.entity_kind
        assert gt_node.count == vlm_node.count
        assert gt_node.binding_policy == vlm_node.binding_policy
        assert gt_node.unary_predicates == vlm_node.unary_predicates

    # 2. Operation groups
    assert len(gt_gf.operation_groups) == len(vlm_gf.operation_groups)
    for gt_og, vlm_og in zip(gt_gf.operation_groups, vlm_gf.operation_groups):
        assert gt_og.id == vlm_og.id
        assert gt_og.tool_role == vlm_og.tool_role
        assert gt_og.target_role == vlm_og.target_role
        assert gt_og.context_role == vlm_og.context_role
        assert gt_og.usage_policy == vlm_og.usage_policy
        assert gt_og.required_target_count == vlm_og.required_target_count
        assert gt_og.required_relations == vlm_og.required_relations
        assert gt_og.context_relations == vlm_og.context_relations


# ---------------------------------------------------------------------------
# Test G, H, I: Workshop novel tokens routed through actual _sync_common_graph()
# ---------------------------------------------------------------------------
class DummyTracker:
    def __init__(self, tracks: dict[str, Any]):
        self.tracks = tracks


class DummyGeometricGrounder:
    def evaluate_reaches_target(self, geom: dict[str, Any]) -> dict[str, Any]:
        return {"status": "TRUE", "length_m": 0.15}

    def evaluate_compatible_with_target(self, geom: dict[str, Any]) -> dict[str, Any]:
        return {"status": "TRUE", "diameter_m": 0.005}


class DummyController:
    def __init__(self, tracks: dict[str, Any]):
        self.tracker = DummyTracker(tracks)
        self.geometric_grounder = DummyGeometricGrounder()


def _make_dummy_track(instance_id: str, canonical_label: str) -> ObservedObjectTrack:
    return ObservedObjectTrack(
        instance_id=instance_id,
        source_inspection_region_id="INITIAL_WORKBENCH",
        first_seen_stage=0,
        last_seen_stage=0,
        current_geometric_properties={"length_m": 0.15, "diameter_m": 0.005, "head_type": "hex"},
        current_semantic_belief={
            "status": "SUPPORTED",
            "canonical_label": canonical_label,
            "confidence": 0.95,
        },
    )


def test_g_workshop_novel_driver_token_routes_and_stores_reaches_target():
    novel_driver_cat = "compact_powered_screw_tool"
    nodes = {
        "driver": FunctionalRole(
            name="driver", entity_kind="OBJECT", count=1,
            semantic_categories=(novel_driver_cat,), binding_policy="DISTINCT",
        ),
        "fastener": FunctionalRole(
            name="fastener", entity_kind="OBJECT", count=1,
            semantic_categories=("standard_bolt",), binding_policy="DISTINCT",
        ),
    }
    spec = FunctionalRequirementGraph(
        domain="workshop", task_instruction="repair", nodes=nodes,
        relations=(), detector_vocabulary=(novel_driver_cat,),
        candidate_regions=(), region_ranking=(), source="VLM_CANONICAL_G_F",
    )

    track1 = _make_dummy_track("tool_track_1", novel_driver_cat)
    controller = DummyController({"tool_track_1": track1})

    adapter = object.__new__(WorkshopDomainAdapter)
    adapter.specification = spec
    adapter.graph = ObservedSceneGraph()
    adapter.controller = controller
    adapter._stage = 0

    adapter._sync_common_graph()

    rel = adapter.graph.get_relation("REACHES_TARGET", "tool_track_1", "repair_target")
    assert rel is not None
    assert rel.status == "TRUE"


def test_h_workshop_novel_fastener_token_routes_and_stores_compatible_with_target():
    novel_fastener_cat = "threaded_metal_joining_pin"
    nodes = {
        "driver": FunctionalRole(
            name="driver", entity_kind="OBJECT", count=1,
            semantic_categories=("standard_driver",), binding_policy="DISTINCT",
        ),
        "fastener": FunctionalRole(
            name="fastener", entity_kind="OBJECT", count=1,
            semantic_categories=(novel_fastener_cat,), binding_policy="DISTINCT",
        ),
    }
    spec = FunctionalRequirementGraph(
        domain="workshop", task_instruction="repair", nodes=nodes,
        relations=(), detector_vocabulary=(novel_fastener_cat,),
        candidate_regions=(), region_ranking=(), source="VLM_CANONICAL_G_F",
    )

    track1 = _make_dummy_track("fastener_track_1", novel_fastener_cat)
    controller = DummyController({"fastener_track_1": track1})

    adapter = object.__new__(WorkshopDomainAdapter)
    adapter.specification = spec
    adapter.graph = ObservedSceneGraph()
    adapter.controller = controller
    adapter._stage = 0

    adapter._sync_common_graph()

    rel = adapter.graph.get_relation("COMPATIBLE_WITH_TARGET", "fastener_track_1", "repair_target")
    assert rel is not None
    assert rel.status == "TRUE"


def test_i_workshop_driver_and_fastener_evaluate_compatible_with():
    d_cat = "compact_powered_screw_tool"
    f_cat = "threaded_metal_joining_pin"
    nodes = {
        "driver": FunctionalRole(name="driver", entity_kind="OBJECT", count=1, semantic_categories=(d_cat,), binding_policy="DISTINCT"),
        "fastener": FunctionalRole(name="fastener", entity_kind="OBJECT", count=1, semantic_categories=(f_cat,), binding_policy="DISTINCT"),
    }
    spec = FunctionalRequirementGraph(
        domain="workshop", task_instruction="repair", nodes=nodes, relations=(),
        detector_vocabulary=(d_cat, f_cat), candidate_regions=(), region_ranking=(), source="VLM_CANONICAL_G_F",
    )

    d_track = _make_dummy_track("d1", d_cat)
    f_track = _make_dummy_track("f1", f_cat)
    controller = DummyController({"d1": d_track, "f1": f_track})

    adapter = object.__new__(WorkshopDomainAdapter)
    adapter.specification = spec
    adapter.graph = ObservedSceneGraph()
    adapter.controller = controller
    adapter._stage = 0

    adapter._sync_common_graph()

    rel = adapter.graph.get_relation("COMPATIBLE_WITH", "d1", "f1")
    assert rel is not None
    assert rel.status in ("TRUE", "FALSE", "UNKNOWN")


# ---------------------------------------------------------------------------
# Test J: no global ontology entry added for synthetic Workshop labels
# ---------------------------------------------------------------------------
def test_j_no_global_ontology_modified_for_synthetic_labels():
    from mujoco_scenes.workshop_phase1.requirements import FMRequirementProvider
    provider = FMRequirementProvider()
    cat_map = provider.ontology_contract.get_alias_to_canonical_map()
    assert "compact_powered_screw_tool" not in cat_map
    assert "threaded_metal_joining_pin" not in cat_map


# ---------------------------------------------------------------------------
# Test K: selectable role with empty candidate_categories fails
# ---------------------------------------------------------------------------
def test_k_empty_candidate_categories_fails_closed():
    # Living Room empty categories on selectable REGION
    vlm_doc = {
        "status": "SUPPORTED",
        "task_summary": "Living Room setup",
        "functional_roles": [
            {
                "id": "r1", "entity_kind": "REGION", "function": "personal beverage support surface",
                "description": "side table", "required_count": 2, "binding_policy": "DISTINCT",
                "candidate_categories": [], "visible_candidates": [], "required_properties": ["planar support surface"],
            }
        ],
        "functional_relations": [], "inspectable_regions": [], "inspection_order": [], "unsupported_reason": "",
    }
    adapter = MockFMAdapter(vlm_doc)
    provider = EnvironmentVLMRequirementProvider("living_room", fm_adapter=adapter)
    with pytest.raises(MalformedVLMSpecificationError) as exc_info:
        provider.generate_canonical(instruction="serve tea")
    assert "must specify non-empty candidate_categories" in str(exc_info.value)

    # Workshop empty categories on driver
    nodes = {
        "driver": FunctionalRole(name="driver", entity_kind="OBJECT", count=1, semantic_categories=(), binding_policy="DISTINCT"),
        "fastener": FunctionalRole(name="fastener", entity_kind="OBJECT", count=1, semantic_categories=("screw",), binding_policy="DISTINCT"),
    }
    spec = FunctionalRequirementGraph(
        domain="workshop", task_instruction="repair", nodes=nodes, relations=(),
        detector_vocabulary=(), candidate_regions=(), region_ranking=(), source="VLM_CANONICAL_G_F",
    )
    adapter = object.__new__(WorkshopDomainAdapter)
    adapter.specification = spec
    adapter.controller = DummyController({})
    adapter.graph = ObservedSceneGraph()
    with pytest.raises(MalformedVLMSpecificationError) as exc_info:
        adapter._sync_common_graph()
    assert "must have non-empty candidate_categories" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test L: canonicalization version is final and consistent (phase3_6a7_v1)
# ---------------------------------------------------------------------------
def test_l_canonicalization_version_is_phase3_6a7_v1():
    assert VLM_CANONICALIZATION_VERSION == "phase3_6a7_v1"
    assert ENV_VLM_VERSION == "phase3_6a7_v1"
    assert KITCHEN_VLM_VERSION == "phase3_6a7_v1"


# ---------------------------------------------------------------------------
# Test M: raw / validated / canonical provenance is clearly separated
# ---------------------------------------------------------------------------
def test_m_provenance_separation():
    vlm_doc = {
        "status": "SUPPORTED",
        "task_summary": "Two person tea serving",
        "functional_roles": [
            {"id": "r1", "entity_kind": "REGION", "function": "personal beverage support surface", "required_count": 2, "binding_policy": "DISTINCT", "candidate_categories": ["side_table"], "visible_candidates": [], "required_properties": ["planar support surface"]},
        ],
        "functional_relations": [], "inspectable_regions": [], "inspection_order": [], "unsupported_reason": "",
    }
    adapter = MockFMAdapter(vlm_doc)
    provider = EnvironmentVLMRequirementProvider("living_room", fm_adapter=adapter)
    res = provider.generate_canonical(instruction="serve tea")
    
    assert res["raw_vlm_requirement_response"] == vlm_doc
    assert res["raw_vlm_decomposition"] == vlm_doc
    assert isinstance(res["normalized_requirements"], list)
    assert res["normalized_requirements"][0]["function"] == "PERSONAL_CUP_SAUCER_REGION"


# ---------------------------------------------------------------------------
# Test N: failure_category backward compatibility works
# ---------------------------------------------------------------------------
def test_n_failure_category_backward_compatibility():
    old_data = {
        "domain": "living_room",
        "variant": "L1",
        "mode": "vlm",
        "status": "VLM_SPEC_FAILED",
        "grounding_status": "INFEASIBLE",
        "failure_reason": "Some legacy reason",
    }
    res = PipelineResult.from_dict(old_data)
    assert res.failure_category is None
    assert res.failure_reason == "Some legacy reason"

    d = res.to_dict()
    assert d["failure_category"] is None

