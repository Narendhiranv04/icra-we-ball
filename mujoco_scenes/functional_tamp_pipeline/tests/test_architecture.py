"""Focused structural tests for the canonical two-graph functional TAMP pipeline."""

from pathlib import Path
from unittest.mock import MagicMock
import pytest
import yaml

from mujoco_scenes.final_paper_variant_labels import (
    PREFIXES, VARIANT_LABELS, paper_variant_label, resolve_variant_name,
)
from mujoco_scenes.functional_tamp_pipeline.grounding import ground_graph
from mujoco_scenes.functional_tamp_pipeline.gt_spec_provider import GTSpecProvider
from mujoco_scenes.functional_tamp_pipeline.models import (
    FunctionalRelation,
    FunctionalRequirementGraph,
    FunctionalRole,
    GraphGroundingResult,
    NumericConstraint,
    OperationGroup,
)
from mujoco_scenes.functional_tamp_pipeline.planning import plan_with_common_astar
from mujoco_scenes.functional_tamp_pipeline.scene_graph import (
    ObservedNode,
    ObservedRelation,
    ObservedSceneGraph,
)
from mujoco_scenes.functional_tamp_pipeline.spec_provider import (
    FunctionalSpecProvider,
    provider_for_mode,
)
from mujoco_scenes.functional_tamp_pipeline.vlm_spec_provider import VLMSpecProvider
from mujoco_scenes.functional_tamp_pipeline.domains.workshop import WorkshopPlanningCompiler
from mujoco_scenes.functional_tamp_pipeline.domains.kitchen import (
    KitchenPlanningCompiler, compile_kitchen_contract_from_graph,
)
from mujoco_scenes.functional_tamp_pipeline.domains.living_room import compile_living_room_task_from_graph
from mujoco_scenes.living_room_variants import load_living_room_variants
from mujoco_scenes.workshop_scene import WORKSHOP_VARIANTS_CONFIG


# Suite 1: Functional graph integrity and validation
def test_functional_graph_integrity() -> None:
    # 1. Reject missing relation subject/object
    role_a = FunctionalRole(name="role_a")
    with pytest.raises(ValueError, match="relation object 'missing_b' not in nodes"):
        g_invalid = FunctionalRequirementGraph(
            domain="workshop",
            task_instruction="invalid",
            nodes={"role_a": role_a},
            relations=(FunctionalRelation(subject_role="role_a", predicate="COMPATIBLE_WITH", object_role="missing_b"),),
        )
        g_invalid.validate()

    # 2. Reject missing operation group roles
    with pytest.raises(ValueError, match="operation group 'op1' tool_role 'missing_tool' not in nodes"):
        g_invalid_op = FunctionalRequirementGraph(
            domain="kitchen",
            task_instruction="invalid",
            nodes={"role_a": role_a},
            operation_groups=(OperationGroup(
                id="op1", function="stir", tool_role="missing_tool", target_role="role_a", required_target_count=1,
                usage_policy="SEQUENTIAL_REUSE_ALLOWED",
            ),),
        )
        g_invalid_op.validate()

    # 3. Reject zero/negative count
    role_zero = FunctionalRole(name="role_z", count=0)
    with pytest.raises(ValueError, match="count must be >= 1"):
        g_zero = FunctionalRequirementGraph(
            domain="workshop",
            task_instruction="invalid",
            nodes={"role_z": role_zero},
        )
        g_zero.validate()


# Suite 2: Workshop functional topology
def test_workshop_functional_topology() -> None:
    provider = GTSpecProvider()
    graph = provider.provide("workshop", "task instruction")
    assert "driver" in graph.nodes or "CAN_DRIVE_SCREW" in graph.nodes
    assert "fastener" in graph.nodes or "CAN_FASTEN" in graph.nodes
    assert "repair_target" in graph.nodes
    assert graph.nodes["repair_target"].entity_kind == "FIXED_TARGET"

    driver_key = "driver" if "driver" in graph.nodes else "CAN_DRIVE_SCREW"
    fastener_key = "fastener" if "fastener" in graph.nodes else "CAN_FASTEN"

    driver_rel_targets = [r.object_role for r in graph.relations if r.subject_role == driver_key]
    assert fastener_key in driver_rel_targets
    assert "repair_target" in driver_rel_targets

    # Ensure driver -> REACHES_TARGET -> fastener is NOT present
    for r in graph.relations:
        if r.subject_role == driver_key and r.predicate == "REACHES_TARGET":
            assert r.object_role == "repair_target"


# Suite 3: Synthetic grounding positive, negative, and ternary unknown
def test_synthetic_grounding_positive_negative_and_unknown() -> None:
    role_d = FunctionalRole(name="driver", semantic_categories=("screwdriver",))
    role_f = FunctionalRole(name="fastener", semantic_categories=("screw",))
    role_t = FunctionalRole(name="target", entity_kind="FIXED_TARGET")
    g_f = FunctionalRequirementGraph(
        domain="workshop",
        task_instruction="repair",
        nodes={"driver": role_d, "fastener": role_f, "target": role_t},
        relations=(
            FunctionalRelation(subject_role="driver", predicate="COMPATIBLE_WITH", object_role="fastener"),
            FunctionalRelation(subject_role="driver", predicate="REACHES_TARGET", object_role="target"),
        ),
    )

    # Positive case: all TRUE
    g_o_pos = ObservedSceneGraph()
    g_o_pos.add_node(ObservedNode(instance_id="d1", canonical_category="screwdriver"))
    g_o_pos.add_node(ObservedNode(instance_id="f1", canonical_category="screw"))
    g_o_pos.add_node(ObservedNode(instance_id="target", entity_kind="FIXED_TARGET"))
    g_o_pos.add_relation(ObservedRelation(subject_id="d1", predicate="COMPATIBLE_WITH", object_id="f1", status="TRUE"))
    g_o_pos.add_relation(ObservedRelation(subject_id="d1", predicate="REACHES_TARGET", object_id="target", status="TRUE"))

    res_pos = ground_graph(g_f, g_o_pos)
    assert res_pos.complete is True
    assert res_pos.status == "COMPLETE"
    assert res_pos.assignment == {"driver": "d1", "fastener": "f1", "target": "target"}

    # Negative case: relation is FALSE
    g_o_neg = ObservedSceneGraph()
    g_o_neg.add_node(ObservedNode(instance_id="d1", canonical_category="screwdriver"))
    g_o_neg.add_node(ObservedNode(instance_id="f1", canonical_category="screw"))
    g_o_neg.add_node(ObservedNode(instance_id="target", entity_kind="FIXED_TARGET"))
    g_o_neg.add_relation(ObservedRelation(subject_id="d1", predicate="COMPATIBLE_WITH", object_id="f1", status="FALSE"))
    g_o_neg.add_relation(ObservedRelation(subject_id="d1", predicate="REACHES_TARGET", object_id="target", status="TRUE"))

    res_neg = ground_graph(g_f, g_o_neg)
    assert res_neg.complete is False
    assert res_neg.status == "INFEASIBLE"

    # Ternary Unknown case: relation is UNKNOWN -> INCOMPLETE, not INFEASIBLE
    g_o_unk = ObservedSceneGraph()
    g_o_unk.add_node(ObservedNode(instance_id="d1", canonical_category="screwdriver"))
    g_o_unk.add_node(ObservedNode(instance_id="f1", canonical_category="screw"))
    g_o_unk.add_node(ObservedNode(instance_id="target", entity_kind="FIXED_TARGET"))
    g_o_unk.add_relation(ObservedRelation(subject_id="d1", predicate="COMPATIBLE_WITH", object_id="f1", status="TRUE"))
    g_o_unk.add_relation(ObservedRelation(subject_id="d1", predicate="REACHES_TARGET", object_id="target", status="UNKNOWN"))

    res_unk = ground_graph(g_f, g_o_unk)
    assert res_unk.complete is False
    assert res_unk.status == "INCOMPLETE"


# Suite 4: Operation group dedicated matching
def test_operation_group_dedicated_matching() -> None:
    role_tool = FunctionalRole(name="soup_utensil", count=2, semantic_categories=("spoon",), binding_policy="DISTINCT")
    role_bowl = FunctionalRole(name="soup_bowl", count=2, semantic_categories=("bowl",), binding_policy="DISTINCT")
    op_group = OperationGroup(
        id="soup_serving",
        function="serve_soup",
        tool_role="soup_utensil",
        target_role="soup_bowl",
        required_target_count=2,
        usage_policy="DEDICATED_PER_TARGET",
        required_relations=("INSERTABLE_IN", "REACHES_BOTTOM"),
    )
    g_f = FunctionalRequirementGraph(
        domain="kitchen",
        task_instruction="serve two soups",
        nodes={"soup_utensil": role_tool, "soup_bowl": role_bowl},
        operation_groups=(op_group,),
    )

    g_o = ObservedSceneGraph()
    g_o.add_node(ObservedNode(instance_id="spoon_1", canonical_category="spoon"))
    g_o.add_node(ObservedNode(instance_id="spoon_2", canonical_category="spoon"))
    g_o.add_node(ObservedNode(instance_id="bowl_1", canonical_category="bowl"))
    g_o.add_node(ObservedNode(instance_id="bowl_2", canonical_category="bowl"))

    # spoon_1 fits bowl_1 only, spoon_2 fits bowl_2 only
    g_o.add_relation(ObservedRelation(subject_id="spoon_1", predicate="INSERTABLE_IN", object_id="bowl_1", status="TRUE"))
    g_o.add_relation(ObservedRelation(subject_id="spoon_1", predicate="REACHES_BOTTOM", object_id="bowl_1", status="TRUE"))
    g_o.add_relation(ObservedRelation(subject_id="spoon_1", predicate="INSERTABLE_IN", object_id="bowl_2", status="FALSE"))

    g_o.add_relation(ObservedRelation(subject_id="spoon_2", predicate="INSERTABLE_IN", object_id="bowl_2", status="TRUE"))
    g_o.add_relation(ObservedRelation(subject_id="spoon_2", predicate="REACHES_BOTTOM", object_id="bowl_2", status="TRUE"))
    g_o.add_relation(ObservedRelation(subject_id="spoon_2", predicate="INSERTABLE_IN", object_id="bowl_1", status="FALSE"))

    res = ground_graph(g_f, g_o)
    assert res.complete is True
    assert res.status == "COMPLETE"
    assert set(res.assignment["soup_utensil"]) == {"spoon_1", "spoon_2"}

    # If spoon_2 fails on bowl_2 -> INFEASIBLE
    g_o.add_relation(ObservedRelation(subject_id="spoon_2", predicate="INSERTABLE_IN", object_id="bowl_2", status="FALSE"))
    res_fail = ground_graph(g_f, g_o)
    assert res_fail.complete is False
    assert res_fail.status == "INFEASIBLE"


# Suite 5: Operation group reusable tool
def test_operation_group_reusable_tool() -> None:
    role_tool = FunctionalRole(name="stirrer", count=1, semantic_categories=("spoon",), binding_policy="REUSABLE")
    role_cup = FunctionalRole(name="cup", count=2, semantic_categories=("cup",), binding_policy="DISTINCT")
    op_group = OperationGroup(
        id="coffee_stirring",
        function="stir_coffee",
        tool_role="stirrer",
        target_role="cup",
        required_target_count=2,
        usage_policy="SEQUENTIAL_REUSE_ALLOWED",
        required_relations=("INSERTABLE_IN",),
    )
    g_f = FunctionalRequirementGraph(
        domain="kitchen",
        task_instruction="stir two coffees",
        nodes={"stirrer": role_tool, "cup": role_cup},
        operation_groups=(op_group,),
    )

    g_o = ObservedSceneGraph()
    g_o.add_node(ObservedNode(instance_id="spoon_1", canonical_category="spoon"))
    g_o.add_node(ObservedNode(instance_id="cup_1", canonical_category="cup"))
    g_o.add_node(ObservedNode(instance_id="cup_2", canonical_category="cup"))

    # Single spoon_1 satisfies both cup_1 and cup_2
    g_o.add_relation(ObservedRelation(subject_id="spoon_1", predicate="INSERTABLE_IN", object_id="cup_1", status="TRUE"))
    g_o.add_relation(ObservedRelation(subject_id="spoon_1", predicate="INSERTABLE_IN", object_id="cup_2", status="TRUE"))

    res = ground_graph(g_f, g_o)
    assert res.complete is True
    assert res.status == "COMPLETE"
    assert res.assignment["stirrer"] == "spoon_1"


# Suite 6: Cross-group reuse policy
def test_cross_group_reuse_policy() -> None:
    role_stirrer = FunctionalRole(name="stirrer", count=1, semantic_categories=("spoon",), binding_policy="REUSABLE")
    role_soup_utensil = FunctionalRole(name="soup_utensil", count=1, semantic_categories=("spoon",), binding_policy="DISTINCT")
    role_cup = FunctionalRole(name="cup", count=1, semantic_categories=("cup",), binding_policy="DISTINCT")
    role_bowl = FunctionalRole(name="bowl", count=1, semantic_categories=("bowl",), binding_policy="DISTINCT")

    op_stir = OperationGroup(
        id="stirring", function="stir", tool_role="stirrer", target_role="cup", required_target_count=1,
        usage_policy="SEQUENTIAL_REUSE_ALLOWED", required_relations=("INSERTABLE_IN",),
    )
    op_soup = OperationGroup(
        id="soup", function="soup", tool_role="soup_utensil", target_role="bowl", required_target_count=1,
        usage_policy="DEDICATED_PER_TARGET", required_relations=("INSERTABLE_IN",),
    )

    g_f = FunctionalRequirementGraph(
        domain="kitchen",
        task_instruction="both",
        nodes={"stirrer": role_stirrer, "soup_utensil": role_soup_utensil, "cup": role_cup, "bowl": role_bowl},
        operation_groups=(op_stir, op_soup),
        cross_group_reuse_allowed=False,  # Disallow tool reuse across groups
    )

    # Scene with only 1 spoon
    g_o = ObservedSceneGraph()
    g_o.add_node(ObservedNode(instance_id="spoon_1", canonical_category="spoon"))
    g_o.add_node(ObservedNode(instance_id="cup_1", canonical_category="cup"))
    g_o.add_node(ObservedNode(instance_id="bowl_1", canonical_category="bowl"))
    g_o.add_relation(ObservedRelation(subject_id="spoon_1", predicate="INSERTABLE_IN", object_id="cup_1", status="TRUE"))
    g_o.add_relation(ObservedRelation(subject_id="spoon_1", predicate="INSERTABLE_IN", object_id="bowl_1", status="TRUE"))

    # Must fail because cross_group_reuse is False and only 1 spoon exists
    res = ground_graph(g_f, g_o)
    assert res.complete is False

    # Add second spoon -> succeeds
    g_o.add_node(ObservedNode(instance_id="spoon_2", canonical_category="spoon"))
    g_o.add_relation(ObservedRelation(subject_id="spoon_2", predicate="INSERTABLE_IN", object_id="bowl_1", status="TRUE"))
    res2 = ground_graph(g_f, g_o)
    assert res2.complete is True
    assert res2.assignment["stirrer"] != res2.assignment["soup_utensil"]


# Suite 7: Legacy compilation independence from raw_requirements
def test_legacy_compilation_from_functional_graph() -> None:
    from dataclasses import replace

    provider = GTSpecProvider()
    kitchen_g = provider.provide("kitchen", "task")
    kitchen_g_clean = replace(kitchen_g, raw_requirements=())

    contract = compile_kitchen_contract_from_graph(kitchen_g_clean)
    assert "roles" in contract
    assert "operation_groups" in contract
    assert "relations" in contract
    assert "symbolic_task" in contract
    assert contract["roles"]["coffee_stirrer"]["binding_cardinality"]["preferred"] == "minimize_distinct"

    living_g = provider.provide("living_room", "task")
    living_g_clean = replace(living_g, raw_requirements=())
    lr_contract = compile_living_room_task_from_graph(living_g_clean)
    assert "function_groups" in lr_contract
    assert "semantic_requirements" in lr_contract
    assert "geometric_requirements" in lr_contract


# Suite 8: Workshop VLM anti-oracle
def test_workshop_vlm_anti_oracle(monkeypatch) -> None:
    from mujoco_scenes.workshop_phase1.requirements import FMRequirementProvider

    mock_doc = {
        "status": "SUPPORTED",
        "task_summary": "Repair workpiece with screwdriver and screw",
        "unsupported_reason": "",
        "functional_requirements": [
            {
                "id": "driver_role",
                "entity_kind": "OBJECT",
                "function": "drive fastener",
                "description": "Phillips screwdriver",
                "required_count": 1,
                "candidate_objects": [{"label": "Phillips screwdriver", "visual_description": "tool", "suitability_reason": "ok"}],
                "required_properties": ["reaches repair hole", "compatible with screw head"],
            },
            {
                "id": "fastener_role",
                "entity_kind": "OBJECT",
                "function": "thread into joint",
                "description": "Phillips screw",
                "required_count": 1,
                "candidate_objects": [{"label": "Phillips screw", "visual_description": "screw", "suitability_reason": "ok"}],
                "required_properties": ["thread into hole"],
            },
        ],
    }

    class MockAdapter:
        def generate_task_requirements(self, *args, **kwargs):
            return mock_doc

        def generate_inspection_priors(self, *args, **kwargs):
            return {
                "inspection_order": [
                    {"region_id": "RIGHT_DRAWER", "reason": "first"},
                    {"region_id": "LEFT_DRAWER", "reason": "second"},
                    {"region_id": "TOOL_CABINET", "reason": "third"},
                ],
                "initial_requirements_satisfied": False,
                "decision_reason": "ok",
            }

    provider = FMRequirementProvider(fm_adapter=MockAdapter())
    # Should successfully normalize using generic ontology without error
    reqs = provider.get_requirements("task", observation_images=[Path("/fake/image.png")])
    assert len(reqs) == 2
    policy = provider.generate_inspection_policy("task", observation_images=[Path("/fake/image.png")])
    assert policy == ("RIGHT_DRAWER", "LEFT_DRAWER", "TOOL_CABINET")
    assert provider.region_ranking == ("RIGHT_DRAWER", "LEFT_DRAWER", "TOOL_CABINET")


# Suite 9: Living Room VLM anti-oracle
def test_living_room_vlm_anti_oracle() -> None:
    from mujoco_scenes.environment_vlm_requirements import EnvironmentVLMRequirementProvider

    mock_doc = {
        "status": "SUPPORTED",
        "task_summary": "Living room custom placement",
        "unsupported_reason": "",
        "functional_requirements": [
            {
                "id": "personal_support_role",
                "entity_kind": "REGION",
                "function": "support drink",
                "description": "personal side table",
                "required_count": 2,
                "candidate_objects": [{"label": "side table", "visual_description": "small table", "suitability_reason": "near seat"}],
                "required_properties": ["planar support", "near seating area"],
            },
            {
                "id": "shared_support_role",
                "entity_kind": "REGION",
                "function": "support remote control",
                "description": "shared coffee table",
                "required_count": 1,
                "candidate_objects": [{"label": "coffee table", "visual_description": "central table", "suitability_reason": "accessible"}],
                "required_properties": ["planar support", "accessible from both seats"],
            },
        ],
    }

    class FakeAdapter:
        last_observation_images = []
        last_raw_requirement_response = {}
        last_raw_inspection_response = {}
        metrics = MagicMock(total_calls=1)

        def generate_task_requirements(self, *args, **kwargs):
            return mock_doc

    provider = EnvironmentVLMRequirementProvider("living_room", fm_adapter=FakeAdapter())
    result = provider.generate("custom instruction", observation_images=[Path("/fake/image.png")])
    normalized = result["normalized_requirements"]
    assert len(normalized) == 2


# Suite 10: Both modes share the provider boundary
def test_both_modes_share_the_provider_boundary() -> None:
    assert isinstance(provider_for_mode("gt"), FunctionalSpecProvider)
    assert isinstance(provider_for_mode("vlm"), FunctionalSpecProvider)


# Suite 11: Graph serialization roundtrip
def test_graph_serialization_roundtrip() -> None:
    role_a = FunctionalRole(
        name="driver",
        entity_kind="OBJECT",
        count=1,
        semantic_categories=("screwdriver", "power_driver"),
        unary_predicates=("CAN_DRIVE_SCREW",),
        numeric_constraints=(
            NumericConstraint(property_name="usable_length_m", operator=">=", threshold=0.10, unit="m"),
        ),
        binding_policy="DISTINCT",
    )
    role_b = FunctionalRole(
        name="fastener",
        entity_kind="OBJECT",
        count=1,
        semantic_categories=("screw",),
        binding_policy="DISTINCT",
    )
    rel = FunctionalRelation(
        subject_role="driver",
        predicate="COMPATIBLE_WITH",
        object_role="fastener",
        expected=True,
    )
    g_f = FunctionalRequirementGraph(
        domain="workshop",
        task_instruction="Drive screw with driver",
        nodes={"driver": role_a, "fastener": role_b},
        relations=(rel,),
        detector_vocabulary=("screwdriver", "screw"),
        candidate_regions=("LEFT_DRAWER", "RIGHT_DRAWER"),
        region_ranking=("RIGHT_DRAWER", "LEFT_DRAWER"),
        source="VLM_FUNCTIONAL_SPEC",
    )

    data_f = g_f.to_dict()
    reconstructed_f = FunctionalRequirementGraph.from_dict(data_f)
    assert reconstructed_f.domain == g_f.domain
    assert reconstructed_f.task_instruction == g_f.task_instruction
    assert len(reconstructed_f.nodes) == 2
    assert reconstructed_f.nodes["driver"].name == "driver"
    assert reconstructed_f.nodes["driver"].numeric_constraints[0].threshold == 0.10
    assert len(reconstructed_f.relations) == 1
    assert reconstructed_f.relations[0].predicate == "COMPATIBLE_WITH"
    assert reconstructed_f.region_ranking == ("RIGHT_DRAWER", "LEFT_DRAWER")


# Suite 12: Workshop and Kitchen common A* planning
def test_workshop_plan_uses_common_astar_and_excludes_search_open() -> None:
    assignment = {
        "driver": "workshop_long_phillips_driver",
        "fastener": "workshop_medium_phillips_screw",
        "driver_source": "RIGHT_DRAWER",
        "fastener_source": "LEFT_DRAWER",
        "work_surface": "MAIN_WORKBENCH_ZONE",
        "target_joint": "workshop_frame_joint",
    }
    result = plan_with_common_astar(
        WorkshopPlanningCompiler(), assignment,
        {"opened_regions": ("LEFT_DRAWER", "RIGHT_DRAWER")},
    )
    assert result.search.statistics["algorithm"] == "deterministic_astar_symbolic_state_search"
    assert [row["operator"] for row in result.actions] == [
        "PICK", "PLACE", "PICK", "SCREW", "PLACE",
    ]
    assert all(row["operator"] != "OPEN" for row in result.actions)


def test_kitchen_compiles_observed_witness_into_common_astar() -> None:
    objects = ("cup", "bowl", "coffee_source", "water_source", "soup_source", "spoon", "fork")
    compiled = {
        "role_assignments": {"coffee_targets": ["cup"], "soup_targets": ["bowl"]},
        "capabilities": {
            "source_contains": [
                ["coffee_source", "coffee"], ["water_source", "water"],
                ["soup_source", "soup"],
            ],
            "initial_target_contents": [],
            "can_stir": [["spoon", "cup"]],
            "assigned_soup_utensil": [["fork", "bowl"]],
        },
        "requirements": {
            "home_region": "countertop", "serving_destination": "serving_area",
        },
        "objects": {
            obj: {"location": {"region_id": "countertop"}} for obj in objects
        },
    }
    result = plan_with_common_astar(
        KitchenPlanningCompiler(), compiled["role_assignments"],
        {"compiled_observed_state": compiled},
    )
    assert result.search.statistics["algorithm"] == "deterministic_astar_symbolic_state_search"
    assert result.validation["goal_status"] == "GOAL_SATISFIED"
    assert {row["operator"] for row in result.actions} == {
        "PICK", "PLACE", "POUR", "STIR",
    }
    assert all(row["operator"] != "OPEN" for row in result.actions)


# Suite 13: Solution boundary and variant resolution
def test_new_pipeline_does_not_import_oracle_solution_generators() -> None:
    root = Path(__file__).resolve().parents[1]
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in root.rglob("*.py")
        if "tests" not in path.parts
    )
    for forbidden in (
        "solve_gt_assignment", "generate_gt_plan", "EXPECTED_GT_ACTIONS",
        "expected_inspection_regions", "privileged_validate_variant_feasibility",
    ):
        assert forbidden not in source


def test_every_paper_variant_resolves_to_a_backing_scene_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    kitchen = yaml.safe_load(
        (root / "configs" / "kitchen_feasibility_variants.yaml").read_text(
            encoding="utf-8"
        )
    )["variants"]
    workshop = yaml.safe_load(
        WORKSHOP_VARIANTS_CONFIG.read_text(encoding="utf-8")
    )["variants"]
    living_room = load_living_room_variants()
    backing = {
        "kitchen": kitchen,
        "workshop": workshop,
        "living_room": living_room,
    }
    assert {domain: len(variants) for domain, variants in VARIANT_LABELS.items()} == {
        "kitchen": 12, "workshop": 10, "living_room": 10,
    }
    for domain, variants in VARIANT_LABELS.items():
        for index, internal in enumerate(variants, start=1):
            short = f"{PREFIXES[domain]}{index}"
            assert resolve_variant_name(domain, short) == internal
            assert resolve_variant_name(domain, short.lower()) == internal
            assert paper_variant_label(domain, internal) == short
            assert internal in backing[domain]


# Suite 14: Operation-group relation skipping in Cartesian check
def test_ground_graph_skips_operation_managed_edges_in_cartesian_check() -> None:
    # Tool and target roles with 2 tools and 2 targets
    role_tool = FunctionalRole(name="stirrer", count=2, semantic_categories=("spoon",), binding_policy="DISTINCT")
    role_target = FunctionalRole(name="cup", count=2, semantic_categories=("cup",), binding_policy="DISTINCT")
    
    # Operation group specifies INSERTABLE_IN
    op_group = OperationGroup(
        id="stirring_group",
        function="stir",
        tool_role="stirrer",
        target_role="cup",
        required_target_count=2,
        usage_policy="DEDICATED_PER_TARGET",
        required_relations=("INSERTABLE_IN",),
    )
    # Also explicitly add INSERTABLE_IN in relations (which would fail if all-pairs Cartesian check were run)
    explicit_rel = FunctionalRelation(subject_role="stirrer", predicate="INSERTABLE_IN", object_role="cup")

    g_f = FunctionalRequirementGraph(
        domain="kitchen",
        task_instruction="stir cups",
        nodes={"stirrer": role_tool, "cup": role_target},
        relations=(explicit_rel,),
        operation_groups=(op_group,),
    )

    # In G_O, tool1 fits cup1 only, tool2 fits cup2 only (NOT all pairs)
    g_o = ObservedSceneGraph()
    g_o.add_node(ObservedNode(instance_id="t1", canonical_category="spoon"))
    g_o.add_node(ObservedNode(instance_id="t2", canonical_category="spoon"))
    g_o.add_node(ObservedNode(instance_id="c1", canonical_category="cup"))
    g_o.add_node(ObservedNode(instance_id="c2", canonical_category="cup"))

    g_o.add_relation(ObservedRelation(subject_id="t1", predicate="INSERTABLE_IN", object_id="c1", status="TRUE"))
    g_o.add_relation(ObservedRelation(subject_id="t1", predicate="INSERTABLE_IN", object_id="c2", status="FALSE"))
    g_o.add_relation(ObservedRelation(subject_id="t2", predicate="INSERTABLE_IN", object_id="c2", status="TRUE"))
    g_o.add_relation(ObservedRelation(subject_id="t2", predicate="INSERTABLE_IN", object_id="c1", status="FALSE"))

    # If managed relations were double-checked as Cartesian all-pairs, it would fail.
    # Because it is skipped, DEDICATED_PER_TARGET matches (t1->c1, t2->c2) and succeeds!
    res = ground_graph(g_f, g_o)
    assert res.complete is True
    assert res.status == "COMPLETE"


# Suite 15: Variable cardinality with preferred = "minimize_distinct"
def test_ground_graph_variable_cardinality_prefers_min_count() -> None:
    role_tool = FunctionalRole(
        name="stirrer",
        min_count=1,
        max_count=2,
        preference="minimize_distinct",
        semantic_categories=("spoon",),
        binding_policy="REUSABLE",
    )
    role_target = FunctionalRole(name="cup", count=2, semantic_categories=("cup",), binding_policy="DISTINCT")
    op_group = OperationGroup(
        id="stirring",
        function="stir",
        tool_role="stirrer",
        target_role="cup",
        required_target_count=2,
        usage_policy="SEQUENTIAL_REUSE_ALLOWED",
        required_relations=("INSERTABLE_IN",),
        same_tool_must_cover_all_targets=False,
    )

    g_f = FunctionalRequirementGraph(
        domain="kitchen",
        task_instruction="stir cups",
        nodes={"stirrer": role_tool, "cup": role_target},
        operation_groups=(op_group,),
    )

    # 2 spoons available. spoon_1 fits both cups.
    g_o = ObservedSceneGraph()
    g_o.add_node(ObservedNode(instance_id="s1", canonical_category="spoon"))
    g_o.add_node(ObservedNode(instance_id="s2", canonical_category="spoon"))
    g_o.add_node(ObservedNode(instance_id="c1", canonical_category="cup"))
    g_o.add_node(ObservedNode(instance_id="c2", canonical_category="cup"))

    g_o.add_relation(ObservedRelation(subject_id="s1", predicate="INSERTABLE_IN", object_id="c1", status="TRUE"))
    g_o.add_relation(ObservedRelation(subject_id="s1", predicate="INSERTABLE_IN", object_id="c2", status="TRUE"))
    g_o.add_relation(ObservedRelation(subject_id="s2", predicate="INSERTABLE_IN", object_id="c1", status="TRUE"))
    g_o.add_relation(ObservedRelation(subject_id="s2", predicate="INSERTABLE_IN", object_id="c2", status="TRUE"))

    res = ground_graph(g_f, g_o)
    assert res.complete is True
    # Should assign exactly 1 spoon because preference is "minimize_distinct"
    assert res.assignment["stirrer"] == "s1"


# Suite 16: same_tool_must_cover_all_targets semantics
def test_ground_graph_same_tool_must_cover_all_targets_true_vs_false() -> None:
    role_tool = FunctionalRole(name="stirrer", min_count=1, max_count=2, semantic_categories=("spoon",), binding_policy="REUSABLE")
    role_target = FunctionalRole(name="cup", count=2, semantic_categories=("cup",), binding_policy="DISTINCT")

    # When same_tool_must_cover_all_targets is True
    op_strict = OperationGroup(
        id="stirring", function="stir", tool_role="stirrer", target_role="cup",
        required_target_count=2, usage_policy="SEQUENTIAL_REUSE_ALLOWED",
        required_relations=("INSERTABLE_IN",), same_tool_must_cover_all_targets=True,
    )
    g_f_strict = FunctionalRequirementGraph(
        domain="kitchen", task_instruction="stir",
        nodes={"stirrer": role_tool, "cup": role_target}, operation_groups=(op_strict,),
    )

    # s1 fits c1 only, s2 fits c2 only (no single spoon fits both)
    g_o = ObservedSceneGraph()
    g_o.add_node(ObservedNode(instance_id="s1", canonical_category="spoon"))
    g_o.add_node(ObservedNode(instance_id="s2", canonical_category="spoon"))
    g_o.add_node(ObservedNode(instance_id="c1", canonical_category="cup"))
    g_o.add_node(ObservedNode(instance_id="c2", canonical_category="cup"))
    g_o.add_relation(ObservedRelation(subject_id="s1", predicate="INSERTABLE_IN", object_id="c1", status="TRUE"))
    g_o.add_relation(ObservedRelation(subject_id="s1", predicate="INSERTABLE_IN", object_id="c2", status="FALSE"))
    g_o.add_relation(ObservedRelation(subject_id="s2", predicate="INSERTABLE_IN", object_id="c1", status="FALSE"))
    g_o.add_relation(ObservedRelation(subject_id="s2", predicate="INSERTABLE_IN", object_id="c2", status="TRUE"))

    res_strict = ground_graph(g_f_strict, g_o)
    assert res_strict.complete is False
    assert res_strict.status == "INFEASIBLE"

    # When same_tool_must_cover_all_targets is False
    op_lax = OperationGroup(
        id="stirring", function="stir", tool_role="stirrer", target_role="cup",
        required_target_count=2, usage_policy="SEQUENTIAL_REUSE_ALLOWED",
        required_relations=("INSERTABLE_IN",), same_tool_must_cover_all_targets=False,
    )
    g_f_lax = FunctionalRequirementGraph(
        domain="kitchen", task_instruction="stir",
        nodes={"stirrer": role_tool, "cup": role_target}, operation_groups=(op_lax,),
    )
    res_lax = ground_graph(g_f_lax, g_o)
    assert res_lax.complete is True
    assert set(res_lax.assignment["stirrer"]) == {"s1", "s2"}


# Suite 17: FunctionalRelation.expected = False semantics
def test_ground_graph_expected_false_relation_semantics() -> None:
    role_a = FunctionalRole(name="driver", semantic_categories=("screwdriver",))
    role_b = FunctionalRole(name="fastener", semantic_categories=("screw",))
    # Expect driver NOT to be incompatible
    rel_not_incompatible = FunctionalRelation(
        subject_role="driver", predicate="INCOMPATIBLE_WITH", object_role="fastener", expected=False,
    )
    g_f = FunctionalRequirementGraph(
        domain="workshop", task_instruction="repair",
        nodes={"driver": role_a, "fastener": role_b}, relations=(rel_not_incompatible,),
    )

    # When observed relation is FALSE, expected=False is satisfied!
    g_o_ok = ObservedSceneGraph()
    g_o_ok.add_node(ObservedNode(instance_id="d1", canonical_category="screwdriver"))
    g_o_ok.add_node(ObservedNode(instance_id="f1", canonical_category="screw"))
    g_o_ok.add_relation(ObservedRelation(subject_id="d1", predicate="INCOMPATIBLE_WITH", object_id="f1", status="FALSE"))

    res_ok = ground_graph(g_f, g_o_ok)
    assert res_ok.complete is True
    assert res_ok.status == "COMPLETE"

    # When observed relation is TRUE, expected=False fails
    g_o_bad = ObservedSceneGraph()
    g_o_bad.add_node(ObservedNode(instance_id="d1", canonical_category="screwdriver"))
    g_o_bad.add_node(ObservedNode(instance_id="f1", canonical_category="screw"))
    g_o_bad.add_relation(ObservedRelation(subject_id="d1", predicate="INCOMPATIBLE_WITH", object_id="f1", status="TRUE"))

    res_bad = ground_graph(g_f, g_o_bad)
    assert res_bad.complete is False
    assert res_bad.status == "INFEASIBLE"


# Suite 18: UNKNOWN candidate nodes return INCOMPLETE, not INFEASIBLE
def test_ground_graph_unknown_candidate_returns_incomplete_not_infeasible() -> None:
    role_d = FunctionalRole(
        name="driver",
        semantic_categories=("screwdriver",),
        unary_predicates=("CAN_DRIVE_SCREW",),
    )
    g_f = FunctionalRequirementGraph(
        domain="workshop", task_instruction="repair",
        nodes={"driver": role_d},
    )

    # Node has UNKNOWN predicate CAN_DRIVE_SCREW
    g_o = ObservedSceneGraph()
    g_o.add_node(ObservedNode(
        instance_id="obj_unknown",
        canonical_category="screwdriver",
        unary_predicates={"CAN_DRIVE_SCREW": "UNKNOWN"},
    ))

    res = ground_graph(g_f, g_o)
    assert res.complete is False
    assert res.status == "INCOMPLETE"


# Suite 19: FunctionalRequirementGraph validation strengthening
def test_functional_graph_validate_checks_min_max_and_region_ranking() -> None:
    # 1. min_count < 1
    role_invalid_min = FunctionalRole(name="r1", min_count=0, max_count=2)
    with pytest.raises(ValueError, match="minimum count must be >= 1"):
        FunctionalRequirementGraph(domain="test", task_instruction="t", nodes={"r1": role_invalid_min}).validate()

    # 2. max_count < min_count
    role_invalid_max = FunctionalRole(name="r2", min_count=3, max_count=1)
    with pytest.raises(ValueError, match="max_count .* < min_count"):
        FunctionalRequirementGraph(domain="test", task_instruction="t", nodes={"r2": role_invalid_max}).validate()

    # 3. region_ranking mismatch with candidate_regions
    role_valid = FunctionalRole(name="r3", count=1)
    with pytest.raises(ValueError, match="region_ranking .* must match candidate_regions"):
        FunctionalRequirementGraph(
            domain="workshop", task_instruction="t", nodes={"r3": role_valid},
            candidate_regions=("A", "B"), region_ranking=("A", "C"),
        ).validate()

    # 4. duplicate regions in region_ranking
    with pytest.raises(ValueError, match="duplicate regions in region_ranking"):
        FunctionalRequirementGraph(
            domain="workshop", task_instruction="t", nodes={"r3": role_valid},
            candidate_regions=("A", "B"), region_ranking=("A", "A"),
        ).validate()


# Suite 20: Workshop requirements compilation derivation from graph
def test_compile_workshop_requirements_from_graph_derivation() -> None:
    from mujoco_scenes.functional_tamp_pipeline.domains.workshop import compile_workshop_requirements_from_graph

    role_d = FunctionalRole(name="driver", semantic_categories=("screwdriver", "power_driver"), description="Driver tool")
    role_f = FunctionalRole(name="fastener", semantic_categories=("screw",), description="Fastener screw")
    role_t = FunctionalRole(name="repair_target", entity_kind="FIXED_TARGET")

    g_f = FunctionalRequirementGraph(
        domain="workshop", task_instruction="repair",
        nodes={"driver": role_d, "fastener": role_f, "repair_target": role_t},
    )

    reqs = compile_workshop_requirements_from_graph(g_f)
    assert len(reqs) == 2
    req_names = [r.function_name for r in reqs]
    assert "CAN_DRIVE_SCREW" in req_names
    assert "CAN_FASTEN" in req_names


# Suite 21: Living room G_O builder ignores production_result assignments
def test_living_room_scene_graph_builder_ignores_production_result_assignments() -> None:
    from mujoco_scenes.functional_tamp_pipeline.domains.living_room import build_living_room_observed_scene_graph

    class MockRun:
        def __init__(self):
            self.region_registry = {
                "region_1": {"geometry": {"PLANAR_SUPPORT": True}, "semantics": {"canonical_label": "side_table"}},
                "region_2": {"geometry": {"PLANAR_SUPPORT": True}, "semantics": {"canonical_label": "coffee_table"}},
            }
            self.personal_rows = [
                {"region_id": "region_1", "slot_id": "personal_table_slot_1", "FITS_SET_ON": "TRUE", "NEAR_SEAT": "TRUE", "fit_evidence": {}, "context_evidence": {}},
            ]
            self.shared_rows = [
                {"region_id": "region_2", "slot_id": "shared_remote_slot", "payload_ids": ["tv_remote"], "FITS_ON": "TRUE", "ACCESSIBLE_FROM_BOTH_SEATS": "TRUE", "fit_evidence": {}, "context_evidence": {}},
            ]
            # Deliberately inject bogus production_result
            self.production_result = {
                "assignments": [{"slot_id": "bogus_slot", "region_id": "bogus_region"}],
            }

    run = MockRun()
    graph_o = build_living_room_observed_scene_graph(run)
    assert "region_1" in graph_o.nodes
    assert "region_2" in graph_o.nodes
    assert "bogus_region" not in graph_o.nodes
    rel_p = graph_o.get_relation("FITS_SET_ON", "region_1", "personal_table_slot_1")
    assert rel_p is not None
    assert rel_p.status == "TRUE"
    rel_s = graph_o.get_relation("ACCESSIBLE_FROM_BOTH_SEATS", "region_2", "SEATING_PAIR")
    assert rel_s is not None
    assert rel_s.status == "TRUE"


# Suite 22: Workshop sync common graph calls real GeometricGrounder methods
def test_workshop_sync_common_graph_calls_real_geometric_grounder_methods() -> None:
    from mujoco_scenes.functional_tamp_pipeline.domains.workshop import WorkshopDomainAdapter
    from mujoco_scenes.workshop_phase1.types import ObservedObjectTrack

    internal_variant = resolve_variant_name("workshop", "W1")
    spec = GTSpecProvider().provide("workshop", "task")
    adapter = WorkshopDomainAdapter(internal_variant, spec)
    adapter.graph = ObservedSceneGraph()
    adapter._stage = 0

    # Add mock driver track and fastener track
    driver_track = ObservedObjectTrack(
        instance_id="driver_1",
        source_inspection_region_id="RIGHT_DRAWER",
        first_seen_stage=0,
        last_seen_stage=0,
    )
    driver_track.current_semantic_belief = {"canonical_label": "screwdriver"}
    driver_track.current_geometric_properties = {
        "usable_length_m": 0.15,
        "maximum_cross_section_m": 0.02,
        "shaft_clearance_diameter_m": 0.004,
        "tip_profile": "phillips_ph1",
        "tip_diameter_m": 0.004,
        "working_end_interface": "phillips_ph1",
    }

    fastener_track = ObservedObjectTrack(
        instance_id="fastener_1",
        source_inspection_region_id="LEFT_DRAWER",
        first_seen_stage=0,
        last_seen_stage=0,
    )
    fastener_track.current_semantic_belief = {"canonical_label": "screw"}
    fastener_track.current_geometric_properties = {
        "recess_profile": "phillips_ph1",
        "head_interface": "phillips_ph1",
        "recess_diameter_m": 0.004,
        "shaft_diameter_m": 0.003,
        "thread_major_diameter_m": 0.003,
        "head_diameter_m": 0.006,
        "maximum_cross_section_m": 0.006,
        "total_length_m": 0.015,
    }

    adapter.controller.tracker._tracks = {
        "driver_1": driver_track,
        "fastener_1": fastener_track,
    }

    # Set mock target recess evidence
    from mujoco_scenes.workshop_phase1.types import GroundingStatus
    target_evidence = MagicMock()
    target_evidence.validity = GroundingStatus.PASS
    target_evidence.estimated_opening_diameter_m = 0.008
    target_evidence.estimated_recess_depth_m = 0.012
    adapter.controller.geometric_grounder.target_evidence = target_evidence

    # Run _sync_common_graph
    adapter._sync_common_graph()

    # Check that relations were populated using the real GeometricGrounder methods
    rel_reach = adapter.graph.get_relation("REACHES_TARGET", "driver_1", "repair_target")
    assert rel_reach is not None
    assert rel_reach.status == "TRUE"

    rel_fastener = adapter.graph.get_relation("COMPATIBLE_WITH_TARGET", "fastener_1", "repair_target")
    assert rel_fastener is not None
    assert rel_fastener.status == "TRUE"

    rel_compat = adapter.graph.get_relation("COMPATIBLE_WITH", "driver_1", "fastener_1")
    assert rel_compat is not None
    assert rel_compat.status == "TRUE"


# Suite 23: Living Room planar support parsing
def test_living_planar_support_parsing() -> None:
    from mujoco_scenes.functional_tamp_pipeline.domains.living_room import parse_planar_support

    assert parse_planar_support({"PLANAR_SUPPORT": {"value": False, "status": "DERIVED"}}) == "FALSE"
    assert parse_planar_support({"PLANAR_SUPPORT": {"value": True, "status": "DERIVED"}}) == "TRUE"
    assert parse_planar_support({"PLANAR_SUPPORT": {"status": "UNKNOWN"}}) == "UNKNOWN"
    assert parse_planar_support({"PLANAR_SUPPORT": {"value": None}}) == "UNKNOWN"
    assert parse_planar_support({"PLANAR_SUPPORT": False}) == "FALSE"
    assert parse_planar_support({"PLANAR_SUPPORT": True}) == "TRUE"
    assert parse_planar_support({}) == "UNKNOWN"
    assert parse_planar_support(False) == "FALSE"
    assert parse_planar_support(True) == "TRUE"


# Suite 24: Living Room target-specific edges coexist without overwriting
def test_living_target_specific_seat_edges_coexist() -> None:
    from mujoco_scenes.functional_tamp_pipeline.domains.living_room import build_living_room_observed_scene_graph

    class MockRun:
        def __init__(self):
            self.region_registry = {
                "region_A": {"geometry": {"PLANAR_SUPPORT": True}, "semantics": {"canonical_label": "side_table"}},
            }
            self.seating_registry = {
                "seat_left": {"geometry": {}},
                "seat_right": {"geometry": {}},
            }
            self.personal_rows = [
                {
                    "region_id": "region_A",
                    "slot_id": "bundle_1",
                    "seating_target_id": "seat_left",
                    "payload_ids": ["cup_1", "saucer_1"],
                    "FITS_SET_ON": "TRUE",
                    "NEAR_SEAT": "TRUE",
                    "fit_evidence": {},
                    "context_evidence": {},
                },
                {
                    "region_id": "region_A",
                    "slot_id": "bundle_2",
                    "seating_target_id": "seat_right",
                    "payload_ids": ["cup_2", "saucer_2"],
                    "FITS_SET_ON": "FALSE",
                    "NEAR_SEAT": "FALSE",
                    "fit_evidence": {},
                    "context_evidence": {},
                },
            ]
            self.shared_rows = []

    run = MockRun()
    graph_o = build_living_room_observed_scene_graph(run)

    # Verify both seat relations coexist on region_A
    rel_left = graph_o.get_relation("NEAR_SEAT", "region_A", "seat_left")
    rel_right = graph_o.get_relation("NEAR_SEAT", "region_A", "seat_right")
    assert rel_left is not None
    assert rel_left.status == "TRUE"
    assert rel_right is not None
    assert rel_right.status == "FALSE"

    # Verify both bundle fit relations coexist on region_A
    rel_b1 = graph_o.get_relation("FITS_SET_ON", "region_A", "bundle_1")
    rel_b2 = graph_o.get_relation("FITS_SET_ON", "region_A", "bundle_2")
    assert rel_b1 is not None
    assert rel_b1.status == "TRUE"
    assert rel_b2 is not None
    assert rel_b2.status == "FALSE"


# Suite 25: Living Room VLM anti-oracle respects VLM counts and omitted relations
def test_living_room_vlm_anti_oracle_count_and_relations() -> None:
    from mujoco_scenes.environment_vlm_requirements import EnvironmentVLMRequirementProvider

    # VLM emits 1 personal support role and omits NEAR_SEAT
    mock_doc = {
        "status": "SUPPORTED",
        "task_summary": "Single drink placement",
        "unsupported_reason": "",
        "functional_requirements": [
            {
                "id": "personal_table",
                "entity_kind": "REGION",
                "function": "support a cup and saucer",
                "description": "one side table for coffee",
                "required_count": 1,
                "candidate_objects": [{"label": "side table", "visual_description": "table", "suitability_reason": "fits set"}],
                "required_properties": ["planar support", "fit the complete set"],
            },
        ],
    }

    class FakeAdapter:
        last_observation_images = []
        last_raw_requirement_response = {}
        last_raw_inspection_response = {}
        metrics = MagicMock(total_calls=1)

        def generate_task_requirements(self, *args, **kwargs):
            return mock_doc

    provider = EnvironmentVLMRequirementProvider("living_room", fm_adapter=FakeAdapter())
    result = provider.generate_canonical("place cup and saucer", observation_images=[Path("/fake/image.png")])
    records = result["normalized_requirements"]
    assert len(records) == 1
    # Check count is preserved as 1 (not forced to 2 from GT)
    assert records[0]["vlm_required_count"] == 1
    # Check required_properties does not contain NEAR_SEAT since VLM omitted it
    assert "NEAR_SEAT" not in records[0]["required_properties"]
    assert "FITS_SET_ON" in records[0]["required_properties"]
    assert "PLANAR_SUPPORT" in records[0]["required_properties"]


# Suite 26: Workshop VLM unmapped category fails closed
def test_workshop_vlm_unmapped_category_fails_closed() -> None:
    from mujoco_scenes.workshop_phase1.requirements import FMRequirementProvider

    mock_doc = {
        "status": "SUPPORTED",
        "task_summary": "Workshop unmapped tool",
        "unsupported_reason": "",
        "functional_requirements": [
            {
                "id": "driver_tool",
                "entity_kind": "OBJECT",
                "function": "drive screws into wood",
                "description": "mysterious alien tool",
                "required_count": 1,
                "candidate_objects": [{"label": "alien_blaster_9000", "visual_description": "shiny", "suitability_reason": "drives"}],
                "required_properties": ["reaches target recess"],
            },
        ],
    }

    class FakeAdapter:
        last_observation_images = []
        last_raw_requirement_response = {}
        last_raw_inspection_response = {}
        metrics = MagicMock(total_calls=1)

        def generate_task_requirements(self, *args, **kwargs):
            return mock_doc

    provider = FMRequirementProvider(fm_adapter=FakeAdapter())
    with pytest.raises(ValueError, match="VLM_SPEC_FAILED"):
        provider.get_requirements("repair joint", observation_images=[Path("/fake/image.png")])


# Suite 27: Workshop legacy compiler derives relations strictly from G_F
def test_workshop_compiler_derives_relations_strictly_from_gf() -> None:
    from mujoco_scenes.functional_tamp_pipeline.domains.workshop import compile_workshop_requirements_from_graph

    role_driver = FunctionalRole(
        name="driver",
        entity_kind="OBJECT",
        count=1,
        semantic_categories=("screwdriver",),
        unary_predicates=("CAN_DRIVE_SCREW",),
    )
    # Only declare REACHES_TARGET in G_F (omit COMPATIBLE_WITH)
    g_f = FunctionalRequirementGraph(
        domain="workshop",
        task_instruction="repair",
        nodes={"driver": role_driver},
        relations=(
            FunctionalRelation(subject_role="driver", predicate="REACHES_TARGET", object_role="repair_target", expected=True),
        ),
        detector_vocabulary=("screwdriver",),
    )

    reqs = compile_workshop_requirements_from_graph(g_f)
    assert len(reqs) == 1
    assert reqs[0].required_relations == ["REACHES_TARGET"]
    assert "COMPATIBLE_WITH" not in reqs[0].required_relations


# Suite 28: Kitchen VLM variable cardinality and OperationGroup roundtrip
def test_kitchen_vlm_variable_cardinality_and_policies_roundtrip(monkeypatch) -> None:
    from mujoco_scenes.functional_tamp_pipeline.vlm_spec_provider import VLMSpecProvider
    from mujoco_scenes.workshop_phase1.fm_adapter import FMAdapter
    from mujoco_scenes.tests.test_kitchen_vlm_functional_graph import qwen_graph

    mock_kitchen_raw = qwen_graph()
    mock_kitchen_raw["inspection_order"] = ["D1", "D2", "C2", "B1", "C1"]
    mock_kitchen_raw["candidate_regions"] = [{"region_id": r, "description": r} for r in ["D1", "D2", "C2", "B1", "C1"]]
    # Add variable cardinality to mixing_implement
    for r in mock_kitchen_raw["roles"]:
        if r["id"] == "mixing_implement":
            r["binding_cardinality"] = {
                "mode": "assignment_driven",
                "minimum_distinct_physical_objects": 1,
                "maximum_distinct_physical_objects": 2,
                "preferred": "minimize_distinct",
            }

    monkeypatch.setattr(
        FMAdapter, "generate_kitchen_functional_graph",
        lambda *args, **kwargs: mock_kitchen_raw
    )

    spec = VLMSpecProvider().provide("kitchen", "prepare coffee and soup", observation_images=[])
    assert spec.domain == "kitchen"
    stirrer = spec.nodes.get("mixing_implement")
    assert stirrer is not None
    assert stirrer.min_count == 1
    assert stirrer.max_count == 2
    assert stirrer.preference == "minimize_distinct"

    # Check operation group policies
    assert len(spec.operation_groups) >= 1
    coffee_grp = next(g for g in spec.operation_groups if "mix" in g.id)
    assert coffee_grp.usage_policy == "SEQUENTIAL_REUSE_ALLOWED"
    assert coffee_grp.selection_preference == "minimize_distinct_tools"


# Suite 29: Search-stage missing roles: INCOMPLETE before exhaustion, INFEASIBLE after
def test_search_missing_role_incomplete_before_exhaustion_infeasible_after() -> None:
    from mujoco_scenes.functional_tamp_pipeline.grounding import ground_graph

    role_missing = FunctionalRole(
        name="rare_tool",
        entity_kind="OBJECT",
        count=1,
        semantic_categories=("rare_tool",),
        unary_predicates=(),
    )
    g_f = FunctionalRequirementGraph(
        domain="test",
        task_instruction="find tool",
        nodes={"rare_tool": role_missing},
        relations=(),
        detector_vocabulary=("rare_tool",),
    )
    g_o = ObservedSceneGraph()  # empty

    # Before search exhaustion
    res_incomplete = ground_graph(g_f, g_o, {"search_exhausted": False})
    assert res_incomplete.status == "INCOMPLETE"
    assert not res_incomplete.complete
    assert "rare_tool" in res_incomplete.missing_roles

    # After search exhaustion
    res_infeasible = ground_graph(g_f, g_o, {"search_exhausted": True})
    assert res_infeasible.status == "INFEASIBLE"
    assert not res_infeasible.complete
    assert "rare_tool" in res_infeasible.missing_roles



