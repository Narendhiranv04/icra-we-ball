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
        "measurement_provenance": {"camera_ids": ["cam1", "cam2"]},
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


# Suite 30: OperationGroup context matching and operation_bindings
def test_operation_group_context_matching_and_operation_bindings() -> None:
    role_tool = FunctionalRole(name="personal_support", count=2, semantic_categories=("table_region",), binding_policy="DISTINCT")
    role_target = FunctionalRole(name="cup_saucer_set", count=2, semantic_categories=("cup_saucer_set",), binding_policy="DISTINCT")
    role_context = FunctionalRole(name="seating_position", count=2, entity_kind="FIXED_TARGET", semantic_categories=("chair",), binding_policy="DISTINCT")

    op_group = OperationGroup(
        id="personal_support_group",
        function="SUPPORT_DRINKWARE",
        tool_role="personal_support",
        target_role="cup_saucer_set",
        usage_policy="DEDICATED_PER_TARGET",
        required_relations=("FITS_SET_ON",),
        context_role="seating_position",
        context_relations=("NEAR_SEAT",),
        required_target_count=2,
    )
    g_f = FunctionalRequirementGraph(
        domain="living_room",
        task_instruction="set up personal drinks",
        nodes={"personal_support": role_tool, "cup_saucer_set": role_target, "seating_position": role_context},
        operation_groups=(op_group,),
    )

    g_o = ObservedSceneGraph()
    g_o.add_node(ObservedNode(instance_id="region_0001", canonical_category="table_region"))
    g_o.add_node(ObservedNode(instance_id="region_0003", canonical_category="table_region"))
    g_o.add_node(ObservedNode(instance_id="slot_1", canonical_category="cup_saucer_set"))
    g_o.add_node(ObservedNode(instance_id="slot_2", canonical_category="cup_saucer_set"))
    g_o.add_node(ObservedNode(instance_id="seat_left", canonical_category="chair", entity_kind="FIXED_TARGET"))
    g_o.add_node(ObservedNode(instance_id="seat_right", canonical_category="chair", entity_kind="FIXED_TARGET"))

    # region_0001 satisfies slot_1 and seat_left; region_0003 satisfies slot_2 and seat_right
    g_o.add_relation(ObservedRelation(subject_id="region_0001", predicate="FITS_SET_ON", object_id="slot_1", status="TRUE"))
    g_o.add_relation(ObservedRelation(subject_id="region_0001", predicate="NEAR_SEAT", object_id="seat_left", status="TRUE"))
    g_o.add_relation(ObservedRelation(subject_id="region_0001", predicate="NEAR_SEAT", object_id="seat_right", status="FALSE"))

    g_o.add_relation(ObservedRelation(subject_id="region_0003", predicate="FITS_SET_ON", object_id="slot_2", status="TRUE"))
    g_o.add_relation(ObservedRelation(subject_id="region_0003", predicate="NEAR_SEAT", object_id="seat_right", status="TRUE"))
    g_o.add_relation(ObservedRelation(subject_id="region_0003", predicate="NEAR_SEAT", object_id="seat_left", status="FALSE"))

    res = ground_graph(g_f, g_o)
    assert res.complete is True
    assert res.status == "COMPLETE"
    assert "personal_support_group" in res.operation_bindings
    bindings = res.operation_bindings["personal_support_group"]
    assert len(bindings) == 2
    assert bindings[0] == {"tool_id": "region_0001", "target_id": "slot_1", "context": {"seating_position": "seat_left"}}
    assert bindings[1] == {"tool_id": "region_0003", "target_id": "slot_2", "context": {"seating_position": "seat_right"}}


# Suite 31: GraphGroundingResult serialization roundtrip
def test_graph_grounding_result_serialization_roundtrip() -> None:
    res = GraphGroundingResult(
        status="COMPLETE",
        complete=True,
        assignment={"tool": "t1", "target": "s1"},
        operation_bindings={"op1": [{"tool_id": "t1", "target_id": "s1", "context": {"seat": "c1"}}]},
        missing_roles=(),
        unsatisfied_relations=(),
        unresolved_constraints=(),
        evidence={"confidence": 0.99},
    )
    d = res.to_dict()
    assert d["operation_bindings"]["op1"][0]["tool_id"] == "t1"
    reconstructed = GraphGroundingResult.from_dict(d)
    assert reconstructed.status == "COMPLETE"
    assert reconstructed.complete is True
    assert reconstructed.assignment == {"tool": "t1", "target": "s1"}
    assert reconstructed.operation_bindings == res.operation_bindings
    assert reconstructed.evidence == res.evidence


# Suite 32: Living room observed scene graph no duplicate seating pair or fake relations
def test_living_room_observed_graph_no_duplicate_seating_pair_or_fake_near_seat() -> None:
    from mujoco_scenes.functional_tamp_pipeline.domains.living_room import build_living_room_observed_scene_graph

    mock_run = MagicMock()
    mock_run.region_registry = {
        "region_0001": {
            "semantics": {"canonical_label": "coffee_table"},
            "geometry": {"unary_region": {"PLANAR_SUPPORT": "TRUE"}},
        }
    }
    mock_run.personal_rows = [
        {
            "region_id": "region_0001",
            "slot_id": "personal_table_slot_1",
            "seating_target_id": "seat_left",
            "payload_ids": ["cup_1", "saucer_1"],
            "FITS_SET_ON": "TRUE",
            "NEAR_SEAT": "TRUE",
        }
    ]
    mock_run.shared_rows = [
        {
            "region_id": "region_0001",
            "payload_ids": ["tv_remote"],
            "FITS_ON": "TRUE",
            "ACCESSIBLE_FROM_BOTH_SEATS": "TRUE",
        }
    ]
    mock_run.seating_registry = {"seat_left": {"geometry": {}}}

    graph_o = build_living_room_observed_scene_graph(mock_run)
    # Check single uppercase SEATING_PAIR
    assert "SEATING_PAIR" in graph_o.nodes
    assert "seating_pair" not in graph_o.nodes

    # Check no NEAR_SEAT relation to slot_id
    for rel in graph_o.relations.values():
        if rel.predicate == "NEAR_SEAT":
            assert rel.object_id == "seat_left"
            assert not rel.object_id.startswith("personal_table_slot")


# Suite 33: Workshop VLM omitted target relation omits repair_target node
def test_workshop_vlm_omitted_target_relation_omits_repair_target_node() -> None:
    from mujoco_scenes.functional_tamp_pipeline.vlm_spec_provider import VLMSpecProvider
    from mujoco_scenes.workshop_phase1.fm_adapter import FMAdapter

    mock_workshop_doc = {
        "status": "SUPPORTED",
        "task_summary": "Workshop driver fastener pair without frame target",
        "unsupported_reason": "",
        "functional_requirements": [
            {
                "id": "driver_req",
                "entity_kind": "OBJECT",
                "function": "drive screws",
                "description": "screwdriver",
                "required_count": 1,
                "candidate_objects": [{"label": "screwdriver", "visual_description": "screwdriver", "suitability_reason": "drives"}],
                "required_properties": ["compatible with phillips screw"],
            },
            {
                "id": "fastener_req",
                "entity_kind": "OBJECT",
                "function": "fasten wood",
                "description": "phillips screw",
                "required_count": 1,
                "candidate_objects": [{"label": "screw", "visual_description": "screw", "suitability_reason": "threads"}],
                "required_properties": [],
            },
        ],
    }

    class FakeAdapter:
        last_observation_images = []
        last_raw_requirement_response = {}
        last_raw_inspection_response = {}
        metrics = MagicMock(total_calls=1)

        def generate_task_requirements(self, *args, **kwargs):
            return mock_workshop_doc

        def generate_inspection_priors(self, *args, **kwargs):
            return {"inspection_order": [{"region_id": "LEFT_DRAWER"}, {"region_id": "RIGHT_DRAWER"}, {"region_id": "TOOL_CABINET"}]}

    from mujoco_scenes.workshop_phase1.requirements import FMRequirementProvider
    provider = FMRequirementProvider(fm_adapter=FakeAdapter())
    graph = VLMSpecProvider._workshop("repair joint", [Path("/fake/img.png")], provider=provider)

    assert "repair_target" not in graph.nodes
    for rel in graph.relations:
        assert rel.object_role != "repair_target"
        assert rel.subject_role != "repair_target"


# Suite 34: Kitchen UNKNOWN semantic remains UNKNOWN in G_O (no fallback promotion)
def test_kitchen_unknown_semantic_remains_unknown_in_go() -> None:
    from mujoco_scenes.functional_tamp_pipeline.domains.kitchen import build_kitchen_observed_scene_graph

    class FakeSession:
        registry = {
            "objects": {
                "object_0001": {
                    "semantics": {
                        "status": "UNKNOWN",
                        "canonical_label": None,
                        "latest_observation": {
                            "canonical_label": None,
                            "alternatives": [
                                {"label": "cup", "supporting_view_count": 1, "confidence": 0.15},
                            ],
                        },
                    },
                    "geometric_predicates": {"OPEN_CAVITY": {"status": "TRUE"}},
                    "geometric_properties": {"cavity_depth_m": 0.08},
                }
            }
        }
        fused_clouds = {}
        events_path = Path("/fake/events.jsonl")

    graph_o = build_kitchen_observed_scene_graph(FakeSession())
    node = graph_o.get_node("object_0001")
    assert node is not None
    assert node.canonical_category is None
    assert node.semantic_labels.get("status") == "UNKNOWN"


# Suite 35: Kitchen supported semantic passes through
def test_kitchen_supported_semantic_passes_through() -> None:
    from mujoco_scenes.functional_tamp_pipeline.domains.kitchen import build_kitchen_observed_scene_graph

    class FakeSession:
        registry = {
            "objects": {
                "object_0001": {
                    "semantics": {
                        "status": "SUPPORTED",
                        "canonical_label": "cup",
                        "validated": {"canonical_label": "cup"},
                        "latest_observation": {
                            "canonical_label": "cup",
                            "alternatives": [
                                {"label": "cup", "supporting_view_count": 3, "confidence": 0.85},
                            ],
                        },
                    },
                    "geometric_predicates": {"OPEN_CAVITY": {"status": "TRUE"}},
                    "geometric_properties": {"cavity_depth_m": 0.08},
                }
            }
        }
        fused_clouds = {}
        events_path = Path("/fake/events.jsonl")

    graph_o = build_kitchen_observed_scene_graph(FakeSession())
    node = graph_o.get_node("object_0001")
    assert node is not None
    assert node.canonical_category == "cup"


# Suite 36: OperationGroup policy serialization in kitchen contract compilation
def test_kitchen_contract_operation_group_policy_serialization() -> None:
    from mujoco_scenes.functional_tamp_pipeline.domains.kitchen import compile_kitchen_contract_from_graph

    graph_f = FunctionalRequirementGraph(
        domain="kitchen",
        task_instruction="Prepare meal",
        nodes={
            "stirrer": FunctionalRole(name="stirrer", semantic_categories=("spoon",)),
            "cup": FunctionalRole(name="cup", semantic_categories=("cup",), count=2),
        },
        operation_groups=(
            OperationGroup(
                id="stirring",
                function="stir",
                tool_role="stirrer",
                target_role="cup",
                required_target_count=2,
                usage_policy="SHARED_ACROSS_ALL_TARGETS",
                distinct_within_group=False,
                same_tool_must_cover_all_targets=True,
                selection_preference="minimize_distinct_tools",
                required_relations=("INSERTABLE_IN", "REACHES_BOTTOM"),
            ),
        ),
    )
    contract = compile_kitchen_contract_from_graph(graph_f)
    assert "stirring" in contract["operation_groups"]
    op = contract["operation_groups"]["stirring"]
    assert op["usage_policy"]["mode"] == "shared_across_all_targets"
    assert op["usage_policy"]["distinct_within_group"] is False
    assert op["usage_policy"]["same_tool_must_cover_all_targets"] is True
    assert op["usage_policy"]["selection_preference"] == "minimize_distinct_tools"
    assert op["relations"] == ["INSERTABLE_IN", "REACHES_BOTTOM"]


# Suite 37: Exact operation bindings preserved without zip reconstruction
def test_exact_operation_bindings_preserved() -> None:
    from mujoco_scenes.functional_tamp_pipeline.models import PipelineResult

    graph_f = FunctionalRequirementGraph(
        domain="kitchen",
        task_instruction="Serve soup",
        nodes={
            "soup_container": FunctionalRole(name="soup_container", count=2),
            "soup_eating_utensil": FunctionalRole(name="soup_eating_utensil", count=2),
        },
        operation_groups=(
            OperationGroup(
                id="soup_serving",
                function="serve",
                tool_role="soup_eating_utensil",
                target_role="soup_container",
                required_target_count=2,
                usage_policy="DEDICATED_PER_TARGET",
                required_relations=("INSERTABLE_IN", "REACHES_BOTTOM"),
            ),
        ),
    )
    graph_o = ObservedSceneGraph()
    for obj_id, cat in [
        ("bowl_A", "bowl"), ("bowl_B", "bowl"),
        ("spoon_X", "spoon"), ("spoon_Y", "spoon"),
    ]:
        graph_o.add_node(ObservedNode(instance_id=obj_id, entity_kind="OBJECT", canonical_category=cat))

    # Add relations with explicit pairing: spoon_X -> bowl_B, spoon_Y -> bowl_A
    for tool, tgt in [("spoon_X", "bowl_B"), ("spoon_Y", "bowl_A")]:
        graph_o.add_relation(ObservedRelation(
            subject_id=tool, predicate="INSERTABLE_IN", object_id=tgt, status="TRUE",
            evidence={"margin": 0.05},
        ))
        graph_o.add_relation(ObservedRelation(
            subject_id=tool, predicate="REACHES_BOTTOM", object_id=tgt, status="TRUE",
            evidence={"depth": 0.08},
        ))

    res = ground_graph(graph_f, graph_o)
    assert res.complete
    bindings = res.operation_bindings["soup_serving"]
    # Check that tool-target pairings are preserved
    paired_dict = {b["tool_id"]: b["target_id"] for b in bindings}
    assert paired_dict["spoon_X"] == "bowl_B"
    assert paired_dict["spoon_Y"] == "bowl_A"


# Suite 38: Workshop geometry cannot determine semantic category
def test_workshop_geometry_cannot_determine_semantic_category() -> None:
    from mujoco_scenes.functional_tamp_pipeline.grounding import _check_semantic_category

    # Object has hammer semantic category and long length (0.15m)
    node = ObservedNode(
        instance_id="object_0001",
        entity_kind="OBJECT",
        canonical_category="hammer",
        semantic_labels={"canonical_label": "hammer", "status": "SUPPORTED"},
        unary_properties={"total_length_m": 0.15},
    )

    # Fastener role accepts screw, driver role accepts screwdriver
    status_driver, matched_driver = _check_semantic_category(node, ["screwdriver", "power_driver"])
    assert status_driver == "FALSE"
    assert matched_driver is None

    status_fastener, matched_fastener = _check_semantic_category(node, ["screw", "phillips_screw"])
    assert status_fastener == "FALSE"
    assert matched_fastener is None


# Suite 39: Mixed ambiguity produces UNKNOWN for both driver and fastener roles
def test_mixed_driver_fastener_ambiguity_produces_unknown() -> None:
    from mujoco_scenes.functional_tamp_pipeline.grounding import check_semantic_role_compatibility

    node = ObservedNode(
        instance_id="object_0002",
        entity_kind="OBJECT",
        canonical_category=None,
        semantic_labels={
            "status": "UNKNOWN",
            "reason_codes": ["CONFLICTING_MULTI_VIEW_LABELS"],
            "plausible_labels": ["screwdriver", "screw"],
            "ambiguity_hypotheses": ["screwdriver", "screw"],
        },
    )

    status_driver, matched_driver = check_semantic_role_compatibility(node, ["screwdriver", "power_driver"])
    assert status_driver == "UNKNOWN"
    assert matched_driver is None

    status_fastener, matched_fastener = check_semantic_role_compatibility(node, ["screw", "phillips_screw"])
    assert status_fastener == "UNKNOWN"
    assert matched_fastener is None


# Suite 40: Semantic fusion conflict and noise handling
def test_semantic_fusion_conflict_and_noise() -> None:
    from mujoco_scenes.semantic_grounding import fuse_semantic_observations, load_semantic_config

    config = load_semantic_config()

    def make_obs(label: str, conf: float, cam: str, score: float = 1.0, pixels: int = 1000):
        return {
            "detection": {
                "canonical_label": label,
                "confidence": conf,
                "source_camera": cam,
                "input_kind": "FULL_FRAME",
            },
            "association_score": score,
            "metrics": {"visible_mask_pixels": pixels},
        }

    # Case 1: Clear winner (3 views bowl) vs weak noise runner (2 views cup, low confidence and association)
    obs_clear = [
        make_obs("bowl", 0.85, "c1", score=1.0, pixels=1000),
        make_obs("bowl", 0.80, "c2", score=1.0, pixels=1000),
        make_obs("bowl", 0.75, "c3", score=1.0, pixels=1000),
        make_obs("cup", 0.12, "c4", score=0.25, pixels=1000),
        make_obs("cup", 0.11, "c5", score=0.25, pixels=1000),
    ]
    res_clear = fuse_semantic_observations(
        obs_clear, config=config, stage=0, region_id="countertop",
        detector_metadata={},
    )
    assert res_clear["status"] == "SUPPORTED"
    assert res_clear["canonical_label"] == "bowl"

    # Case 2: Genuine conflict (3 views fork vs 2 views spoon with high confidence and high score)
    obs_conflict = [
        make_obs("fork", 0.70, "c1", score=1.0, pixels=1000),
        make_obs("fork", 0.65, "c2", score=1.0, pixels=1000),
        make_obs("fork", 0.60, "c3", score=1.0, pixels=1000),
        make_obs("spoon", 0.72, "c4", score=1.0, pixels=1000),
        make_obs("spoon", 0.68, "c5", score=1.0, pixels=1000),
    ]
    res_conflict = fuse_semantic_observations(
        obs_conflict, config=config, stage=0, region_id="countertop",
        detector_metadata={},
    )
    assert res_conflict["status"] == "UNKNOWN"
    assert res_conflict["canonical_label"] is None
    assert "CONFLICTING_MULTI_VIEW_LABELS" in res_conflict["reason_codes"]


# Suite 41: Workshop failure result retains diagnostics
def test_workshop_failure_result_retains_diagnostics() -> None:
    from mujoco_scenes.functional_tamp_pipeline.models import PipelineResult

    res = PipelineResult(
        domain="workshop",
        variant="W9",
        mode="gt",
        status="INFEASIBLE",
        inspected_regions=["LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET"],
        failure_reason="NO_GLOBAL_ASSIGNMENT",
    )
    d = res.to_dict()
    assert d["domain"] == "workshop"
    assert d["variant"] == "W9"
    assert d["status"] == "INFEASIBLE"
    assert d["inspected_regions"] == ["LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET"]
    assert d["failure_reason"] == "NO_GLOBAL_ASSIGNMENT"


# Suite 42: Semantic role compatibility formal cases A through H (Section 30)
def test_semantic_role_compatibility_cases_a_through_h() -> None:
    from mujoco_scenes.functional_tamp_pipeline.grounding import check_semantic_role_compatibility

    # Case A: SUPPORTED cup, C(r) = {cup, mug} -> TRUE
    node_a = ObservedNode(instance_id="a", entity_kind="OBJECT", canonical_category="cup")
    st, matched = check_semantic_role_compatibility(node_a, ["cup", "mug"])
    assert st == "TRUE"
    assert matched == "cup"

    # Case B: UNKNOWN genuine ambiguity H(o) = {cup, mug}, C(r) = {cup, mug} -> TRUE
    node_b = ObservedNode(
        instance_id="b", entity_kind="OBJECT", canonical_category=None,
        semantic_labels={"status": "UNKNOWN", "plausible_labels": ["cup", "mug"], "reason_codes": ["CONFLICTING_MULTI_VIEW_LABELS"]},
    )
    st, matched = check_semantic_role_compatibility(node_b, ["cup", "mug"])
    assert st == "TRUE"
    assert matched in ["cup", "mug"]

    # Case C: UNKNOWN genuine ambiguity H(o) = {spoon, fork}, C(r) = {spoon} -> UNKNOWN
    node_c = ObservedNode(
        instance_id="c", entity_kind="OBJECT", canonical_category=None,
        semantic_labels={"status": "UNKNOWN", "plausible_labels": ["spoon", "fork"], "reason_codes": ["CONFLICTING_MULTI_VIEW_LABELS"]},
    )
    st, matched = check_semantic_role_compatibility(node_c, ["spoon"])
    assert st == "UNKNOWN"
    assert matched is None

    # Case D: UNKNOWN genuine ambiguity H(o) = {screwdriver, screw}, driver C(r) = {screwdriver} -> UNKNOWN
    node_d = ObservedNode(
        instance_id="d", entity_kind="OBJECT", canonical_category=None,
        semantic_labels={"status": "UNKNOWN", "plausible_labels": ["screwdriver", "screw"], "reason_codes": ["CONFLICTING_MULTI_VIEW_LABELS"]},
    )
    st, matched = check_semantic_role_compatibility(node_d, ["screwdriver"])
    assert st == "UNKNOWN"
    assert matched is None

    # Case E: UNKNOWN genuine ambiguity H(o) = {screwdriver, screw}, fastener C(r) = {screw} -> UNKNOWN
    node_e = ObservedNode(
        instance_id="e", entity_kind="OBJECT", canonical_category=None,
        semantic_labels={"status": "UNKNOWN", "plausible_labels": ["screwdriver", "screw"], "reason_codes": ["CONFLICTING_MULTI_VIEW_LABELS"]},
    )
    st, matched = check_semantic_role_compatibility(node_e, ["screw"])
    assert st == "UNKNOWN"
    assert matched is None

    # Case F: SUPPORTED hammer, driver C(r) = {screwdriver} -> FALSE
    node_f = ObservedNode(instance_id="f", entity_kind="OBJECT", canonical_category="hammer")
    st, matched = check_semantic_role_compatibility(node_f, ["screwdriver"])
    assert st == "FALSE"
    assert matched is None

    # Case G: UNKNOWN due to INSUFFICIENT_SEMANTIC_CAMERA_SUPPORT -> UNKNOWN
    node_g = ObservedNode(
        instance_id="g", entity_kind="OBJECT", canonical_category=None,
        semantic_labels={"status": "UNKNOWN", "plausible_labels": [], "reason_codes": ["INSUFFICIENT_SEMANTIC_CAMERA_SUPPORT"]},
    )
    st, matched = check_semantic_role_compatibility(node_g, ["cup", "mug"])
    assert st == "UNKNOWN"
    assert matched is None

    # Case H: UNKNOWN due to NO_ASSOCIATED_DETECTION -> UNKNOWN
    node_h = ObservedNode(
        instance_id="h", entity_kind="OBJECT", canonical_category=None,
        semantic_labels={"status": "UNKNOWN", "plausible_labels": [], "reason_codes": ["NO_ASSOCIATED_DETECTION"]},
    )
    st, matched = check_semantic_role_compatibility(node_h, ["cup", "mug"])
    assert st == "UNKNOWN"
    assert matched is None


# Suite 43: Object ID instance name cannot promote OBJECT node to TRUE
def test_object_id_does_not_promote_semantics() -> None:
    from mujoco_scenes.functional_tamp_pipeline.grounding import check_semantic_role_compatibility

    node = ObservedNode(
        instance_id="red_screw_01",
        entity_kind="OBJECT",
        canonical_category=None,
        semantic_labels={"status": "UNKNOWN", "plausible_labels": [], "reason_codes": ["INSUFFICIENT_SEMANTIC_CAMERA_SUPPORT"]},
    )
    st, matched = check_semantic_role_compatibility(node, ["screw", "phillips_screw"])
    assert st == "UNKNOWN"
    assert matched is None


# Suite 44: Pure build_canonical_kitchen_witness validation
def test_build_canonical_kitchen_witness() -> None:
    from mujoco_scenes.functional_tamp_pipeline.domains.kitchen import build_canonical_kitchen_witness
    from mujoco_scenes.functional_tamp_pipeline.models import GraphGroundingResult
    from mujoco_scenes.functional_tamp_pipeline.gt_spec_provider import GTSpecProvider

    spec = GTSpecProvider().provide("kitchen", "K1")
    graph_o = ObservedSceneGraph()
    for obj in ["c1", "c2", "s1", "s2", "src", "kettle", "sp1", "sp2", "sp3"]:
        graph_o.add_node(ObservedNode(instance_id=obj, entity_kind="OBJECT"))

    # Add valid pairwise relations
    graph_o.add_relation(ObservedRelation(subject_id="sp1", predicate="INSERTABLE_IN", object_id="c1", status="TRUE", evidence={"margin": 0.01}))
    graph_o.add_relation(ObservedRelation(subject_id="sp1", predicate="REACHES_BOTTOM", object_id="c1", status="TRUE", evidence={"margin": 0.01}))
    graph_o.add_relation(ObservedRelation(subject_id="sp1", predicate="INSERTABLE_IN", object_id="c2", status="TRUE", evidence={"margin": 0.01}))
    graph_o.add_relation(ObservedRelation(subject_id="sp1", predicate="REACHES_BOTTOM", object_id="c2", status="TRUE", evidence={"margin": 0.01}))
    graph_o.add_relation(ObservedRelation(subject_id="sp2", predicate="INSERTABLE_IN", object_id="s1", status="TRUE", evidence={"margin": 0.01}))
    graph_o.add_relation(ObservedRelation(subject_id="sp2", predicate="REACHES_BOTTOM", object_id="s1", status="TRUE", evidence={"margin": 0.01}))
    graph_o.add_relation(ObservedRelation(subject_id="sp3", predicate="INSERTABLE_IN", object_id="s2", status="TRUE", evidence={"margin": 0.01}))
    graph_o.add_relation(ObservedRelation(subject_id="sp3", predicate="REACHES_BOTTOM", object_id="s2", status="TRUE", evidence={"margin": 0.01}))

    ground_res = GraphGroundingResult(
        complete=True,
        status="COMPLETE",
        assignment={
            "coffee_container": ("c1", "c2"),
            "soup_container": ("s1", "s2"),
            "coffee_stirrer": ("sp1",),
            "soup_eating_utensil": ("sp2", "sp3"),
            "coffee_source": ("src",),
            "water_source": ("kettle",),
        },
        operation_bindings={
            "coffee_stirring": (
                {"tool_id": "sp1", "target_id": "c1"},
                {"tool_id": "sp1", "target_id": "c2"},
            ),
            "soup_serving": (
                {"tool_id": "sp2", "target_id": "s1"},
                {"tool_id": "sp3", "target_id": "s2"},
            ),
        },
    )

    witness = build_canonical_kitchen_witness(spec, ground_res, graph_o)
    assert witness["status"] == "COMPLETE"
    assert "coffee_stirring" in [op["function_group_id"] for op in witness["operation_assignments"]]
    assert "soup_serving" in [op["function_group_id"] for op in witness["operation_assignments"]]


# Suite 45: Missing or non-TRUE relation in build_canonical_kitchen_witness raises RuntimeError
def test_kitchen_witness_raises_on_missing_or_false_relation() -> None:
    import pytest
    from mujoco_scenes.functional_tamp_pipeline.domains.kitchen import build_canonical_kitchen_witness
    from mujoco_scenes.functional_tamp_pipeline.models import GraphGroundingResult
    from mujoco_scenes.functional_tamp_pipeline.gt_spec_provider import GTSpecProvider

    spec = GTSpecProvider().provide("kitchen", "K1")
    graph_o = ObservedSceneGraph()
    for obj in ["c1", "c2", "s1", "s2", "src", "kettle", "sp1", "sp2", "sp3"]:
        graph_o.add_node(ObservedNode(instance_id=obj, entity_kind="OBJECT"))

    # Missing relations for c2
    graph_o.add_relation(ObservedRelation(subject_id="sp1", predicate="INSERTABLE_IN", object_id="c1", status="TRUE"))
    graph_o.add_relation(ObservedRelation(subject_id="sp1", predicate="REACHES_BOTTOM", object_id="c1", status="TRUE"))

    ground_res = GraphGroundingResult(
        complete=True,
        status="COMPLETE",
        assignment={
            "coffee_container": ("c1", "c2"),
            "soup_container": ("s1", "s2"),
            "coffee_stirrer": ("sp1",),
            "soup_eating_utensil": ("sp2", "sp3"),
            "coffee_source": ("src",),
            "water_source": ("kettle",),
        },
        operation_bindings={
            "coffee_stirring": (
                {"tool_id": "sp1", "target_id": "c1"},
                {"tool_id": "sp1", "target_id": "c2"},
            ),
        },
    )
    with pytest.raises(RuntimeError, match="was not found in G_O"):
        build_canonical_kitchen_witness(spec, ground_res, graph_o)

    # Non-TRUE relation (status="FALSE")
    graph_o.add_relation(ObservedRelation(subject_id="sp1", predicate="INSERTABLE_IN", object_id="c2", status="FALSE"))
    graph_o.add_relation(ObservedRelation(subject_id="sp1", predicate="REACHES_BOTTOM", object_id="c2", status="TRUE"))
    with pytest.raises(RuntimeError, match="with status 'FALSE'"):
        build_canonical_kitchen_witness(spec, ground_res, graph_o)


# Suite 46: Kitchen causal planning order verification (stirred before served)
def test_kitchen_causal_planning_order() -> None:
    from mujoco_scenes.functional_tamp_pipeline.domains.kitchen import KitchenPlanningCompiler
    from mujoco_scenes.functional_tamp_pipeline.planning import plan_with_common_astar

    compiled_state = {
        "requirements": {
            "home_region": "countertop",
            "serving_destination": "serving_area",
        },
        "role_assignments": {
            "coffee_targets": ["cup_1", "cup_2"],
            "soup_targets": ["bowl_1", "bowl_2"],
        },
        "capabilities": {
            "source_contains": [["kettle_1", "water"], ["jar_1", "coffee"], ["pot_1", "soup"]],
            "can_stir": [["spoon_1", "cup_1"], ["spoon_1", "cup_2"]],
            "assigned_soup_utensil": [["spoon_2", "bowl_1"], ["spoon_3", "bowl_2"]],
            "initial_target_contents": [],
        },
        "objects": {
            "cup_1": {"location": {"region_id": "countertop"}},
            "cup_2": {"location": {"region_id": "countertop"}},
            "bowl_1": {"location": {"region_id": "countertop"}},
            "bowl_2": {"location": {"region_id": "countertop"}},
            "kettle_1": {"location": {"region_id": "countertop"}},
            "jar_1": {"location": {"region_id": "countertop"}},
            "pot_1": {"location": {"region_id": "countertop"}},
            "spoon_1": {"location": {"region_id": "countertop"}},
            "spoon_2": {"location": {"region_id": "countertop"}},
            "spoon_3": {"location": {"region_id": "countertop"}},
        },
    }

    planned = plan_with_common_astar(
        KitchenPlanningCompiler(),
        {},
        {"compiled_observed_state": compiled_state},
    )
    plan_ops = [(a["operator"], tuple(a["arguments"])) for a in planned.actions]

    # Verify stir happens before serve for all coffee targets
    for coffee_tgt in ["cup_1", "cup_2"]:
        stir_idx = [i for i, (op, args) in enumerate(plan_ops) if op == "STIR" and args[1] == coffee_tgt][0]
        serve_idx = [i for i, (op, args) in enumerate(plan_ops) if op == "PLACE" and args == (coffee_tgt, "serving_area")][0]
        assert stir_idx < serve_idx, f"Causal violation: {coffee_tgt} served at {serve_idx} before stirred at {stir_idx}"

    # Verify soup utensil placed before bowl served
    for bowl_tgt, assigned_spoon in [("bowl_1", "spoon_2"), ("bowl_2", "spoon_3")]:
        utensil_idx = [i for i, (op, args) in enumerate(plan_ops) if op == "PLACE" and args == (assigned_spoon, bowl_tgt)][0]
        serve_idx = [i for i, (op, args) in enumerate(plan_ops) if op == "PLACE" and args == (bowl_tgt, "serving_area")][0]
        assert utensil_idx < serve_idx, f"Causal violation: {bowl_tgt} served before {assigned_spoon} placed"


# Suite 47: Direct inspection of Kitchen compiled PLACE action preconditions
def test_kitchen_place_action_preconditions() -> None:
    from mujoco_scenes.functional_tamp_pipeline.domains.kitchen import KitchenPlanningCompiler

    compiled_state = {
        "requirements": {
            "home_region": "countertop",
            "serving_destination": "serving_area",
        },
        "role_assignments": {
            "coffee_targets": ["cup_1"],
            "soup_targets": ["bowl_1"],
        },
        "capabilities": {
            "source_contains": [["kettle_1", "water"], ["jar_1", "coffee"], ["pot_1", "soup"]],
            "can_stir": [["spoon_1", "cup_1"]],
            "assigned_soup_utensil": [["spoon_2", "bowl_1"]],
            "initial_target_contents": [],
        },
        "objects": {
            "cup_1": {"location": {"region_id": "countertop"}},
            "bowl_1": {"location": {"region_id": "countertop"}},
            "kettle_1": {"location": {"region_id": "countertop"}},
            "jar_1": {"location": {"region_id": "countertop"}},
            "pot_1": {"location": {"region_id": "countertop"}},
            "spoon_1": {"location": {"region_id": "countertop"}},
            "spoon_2": {"location": {"region_id": "countertop"}},
        },
    }

    problem = KitchenPlanningCompiler().compile_problem({}, {"compiled_observed_state": compiled_state})

    # 1. Coffee serve action
    coffee_place = next(
        a for a in problem.actions
        if a.name == "PLACE" and a.arguments == ("cup_1", "serving_area")
    )
    assert ("holding", "cup_1") in coffee_place.positive_preconditions
    assert ("contains", "cup_1", "coffee") in coffee_place.positive_preconditions
    assert ("contains", "cup_1", "water") in coffee_place.positive_preconditions
    assert ("stirred", "cup_1") in coffee_place.positive_preconditions

    # 2. Soup serve action
    soup_place = next(
        a for a in problem.actions
        if a.name == "PLACE" and a.arguments == ("bowl_1", "serving_area")
    )
    assert ("holding", "bowl_1") in soup_place.positive_preconditions
    assert ("contains", "bowl_1", "soup") in soup_place.positive_preconditions
    assert ("at", "spoon_2", "bowl_1") in soup_place.positive_preconditions


# Suite 48: Direct applicability testing (is_applicable) on unstirred vs stirred coffee and unassigned vs assigned soup bowl
def test_kitchen_place_action_applicability() -> None:
    from mujoco_scenes.functional_tamp_pipeline.domains.kitchen import KitchenPlanningCompiler
    from mujoco_scenes.symbolic_planning_core import is_applicable

    compiled_state = {
        "requirements": {
            "home_region": "countertop",
            "serving_destination": "serving_area",
        },
        "role_assignments": {
            "coffee_targets": ["cup_1"],
            "soup_targets": ["bowl_1"],
        },
        "capabilities": {
            "source_contains": [["kettle_1", "water"], ["jar_1", "coffee"], ["pot_1", "soup"]],
            "can_stir": [["spoon_1", "cup_1"]],
            "assigned_soup_utensil": [["spoon_2", "bowl_1"]],
            "initial_target_contents": [],
        },
        "objects": {
            "cup_1": {"location": {"region_id": "countertop"}},
            "bowl_1": {"location": {"region_id": "countertop"}},
            "kettle_1": {"location": {"region_id": "countertop"}},
            "jar_1": {"location": {"region_id": "countertop"}},
            "pot_1": {"location": {"region_id": "countertop"}},
            "spoon_1": {"location": {"region_id": "countertop"}},
            "spoon_2": {"location": {"region_id": "countertop"}},
        },
    }

    problem = KitchenPlanningCompiler().compile_problem({}, {"compiled_observed_state": compiled_state})

    coffee_place = next(
        a for a in problem.actions
        if a.name == "PLACE" and a.arguments == ("cup_1", "serving_area")
    )

    # State with holding, coffee, water, but NOT stirred
    state_unstirred = frozenset({
        ("holding", "cup_1"),
        ("contains", "cup_1", "coffee"),
        ("contains", "cup_1", "water"),
    })
    assert not is_applicable(state_unstirred, coffee_place), "Unstirred coffee must not be applicable for serving"

    # Add stirred
    state_stirred = state_unstirred | {("stirred", "cup_1")}
    assert is_applicable(state_stirred, coffee_place), "Stirred coffee must be applicable for serving"

    soup_place = next(
        a for a in problem.actions
        if a.name == "PLACE" and a.arguments == ("bowl_1", "serving_area")
    )

    # State with holding, soup, but utensil NOT at bowl
    state_no_utensil = frozenset({
        ("holding", "bowl_1"),
        ("contains", "bowl_1", "soup"),
        ("at", "spoon_2", "countertop"),
    })
    assert not is_applicable(state_no_utensil, soup_place), "Soup bowl without placed utensil must not be applicable for serving"

    # Add utensil placed
    state_with_utensil = state_no_utensil | {("at", "spoon_2", "bowl_1")}
    assert is_applicable(state_with_utensil, soup_place), "Soup bowl with placed utensil must be applicable for serving"


# Suite 49: Workshop SemanticGrounder role compatibility and alternative promotion ban
def test_workshop_semantic_grounder_role_compatibility() -> None:
    from mujoco_scenes.workshop_phase1.semantic_grounding import SemanticGrounder
    from mujoco_scenes.workshop_phase1.types import (
        ObservedObjectTrack,
        FunctionalRequirement,
        GroundingStatus,
        EntityType,
    )

    grounder = SemanticGrounder()

    driver_req = FunctionalRequirement(
        requirement_id="req_driver",
        entity_type=EntityType.OBJECT,
        function_name="CAN_DRIVE_SCREW",
        description="driver requirement",
        accepted_categories=["screwdriver", "power_driver"],
    )
    fastener_req = FunctionalRequirement(
        requirement_id="req_fastener",
        entity_type=EntityType.OBJECT,
        function_name="CAN_FASTEN",
        description="fastener requirement",
        accepted_categories=["screw"],
    )

    # Case 1: Supported screwdriver with 2-view alternative screw
    # Canonical supported screwdriver must PASS for driver and FAIL for fastener
    track_screwdriver = ObservedObjectTrack(
        instance_id="track_1",
        current_semantic_belief={
            "status": "SUPPORTED",
            "canonical_label": "screwdriver",
            "plausible_labels": ["screwdriver"],
            "ambiguity_hypotheses": ["screwdriver"],
            "reason_codes": [],
            "confidence": 0.95,
            "label_supporting_view_count": {"screwdriver": 3, "screw": 2},
        },
    )
    res_driver = grounder.ground_object_for_requirement(track_screwdriver, driver_req)
    assert res_driver.semantic_status == GroundingStatus.PASS

    res_fastener = grounder.ground_object_for_requirement(track_screwdriver, fastener_req)
    assert res_fastener.semantic_status == GroundingStatus.FAIL, (
        "Alternative screw with 2 views must NOT override supported canonical screwdriver"
    )

    # Case 2: Genuine ambiguity between screwdriver and screw
    track_ambiguous = ObservedObjectTrack(
        instance_id="track_2",
        current_semantic_belief={
            "status": "UNKNOWN",
            "canonical_label": None,
            "plausible_labels": ["screwdriver", "screw"],
            "ambiguity_hypotheses": ["screwdriver", "screw"],
            "reason_codes": ["CONFLICTING_MULTI_VIEW_LABELS"],
            "confidence": 0.50,
        },
    )
    assert grounder.ground_object_for_requirement(track_ambiguous, driver_req).semantic_status == GroundingStatus.UNKNOWN
    assert grounder.ground_object_for_requirement(track_ambiguous, fastener_req).semantic_status == GroundingStatus.UNKNOWN

    # Case 3: Insufficient camera support
    track_insufficient = ObservedObjectTrack(
        instance_id="track_3",
        current_semantic_belief={
            "status": "UNKNOWN",
            "canonical_label": None,
            "plausible_labels": [],
            "ambiguity_hypotheses": [],
            "reason_codes": ["INSUFFICIENT_SEMANTIC_CAMERA_SUPPORT"],
            "confidence": 0.50,
        },
    )
    assert grounder.ground_object_for_requirement(track_insufficient, driver_req).semantic_status == GroundingStatus.UNKNOWN


# Suite 50: Workshop Tracker consensus semantic belief fusion contract
def test_workshop_tracker_consensus_fusion() -> None:
    from mujoco_scenes.workshop_phase1.tracking import PersistentInstanceTracker

    # Case A: 3 high-confidence screwdriver views, 1 weak screw
    obs_supported = [
        {"stage_index": 0, "camera_id": "CAM1", "canonical_label": "screwdriver", "confidence": 0.9, "physical_support_quality": 1.0},
        {"stage_index": 0, "camera_id": "CAM2", "canonical_label": "screwdriver", "confidence": 0.85, "physical_support_quality": 1.0},
        {"stage_index": 0, "camera_id": "CAM3", "canonical_label": "screwdriver", "confidence": 0.88, "physical_support_quality": 1.0},
        {"stage_index": 0, "camera_id": "CAM4", "canonical_label": "screw", "confidence": 0.1, "physical_support_quality": 0.5},
    ]
    belief_supported = PersistentInstanceTracker._compute_consensus_semantic_belief(obs_supported)
    assert belief_supported["status"] == "SUPPORTED"
    assert belief_supported["canonical_label"] == "screwdriver"
    assert belief_supported["plausible_labels"] == ["screwdriver"]
    assert belief_supported["reason_codes"] == []

    # Case B: 3 screwdriver vs 2 strong screw views (meeting conflict criteria)
    obs_conflicting = [
        {"stage_index": 0, "camera_id": "CAM1", "canonical_label": "screwdriver", "confidence": 0.7, "physical_support_quality": 1.0},
        {"stage_index": 0, "camera_id": "CAM2", "canonical_label": "screwdriver", "confidence": 0.65, "physical_support_quality": 1.0},
        {"stage_index": 0, "camera_id": "CAM3", "canonical_label": "screwdriver", "confidence": 0.68, "physical_support_quality": 1.0},
        {"stage_index": 0, "camera_id": "CAM4", "canonical_label": "screw", "confidence": 0.7, "physical_support_quality": 1.0},
        {"stage_index": 0, "camera_id": "CAM5", "canonical_label": "screw", "confidence": 0.65, "physical_support_quality": 1.0},
    ]
    belief_conflicting = PersistentInstanceTracker._compute_consensus_semantic_belief(obs_conflicting)
    assert belief_conflicting["status"] == "UNKNOWN"
    assert belief_conflicting["canonical_label"] is None
    assert set(belief_conflicting["plausible_labels"]) == {"screwdriver", "screw"}
    assert "CONFLICTING_MULTI_VIEW_LABELS" in belief_conflicting["reason_codes"]

    # Case C: 1 single view (insufficient camera support)
    obs_single = [
        {"stage_index": 0, "camera_id": "CAM1", "canonical_label": "screwdriver", "confidence": 0.9, "physical_support_quality": 1.0},
    ]
    belief_single = PersistentInstanceTracker._compute_consensus_semantic_belief(obs_single)
    assert belief_single["status"] == "UNKNOWN"
    assert belief_single["canonical_label"] is None
    assert belief_single["plausible_labels"] == []
    assert "INSUFFICIENT_SEMANTIC_CAMERA_SUPPORT" in belief_single["reason_codes"]


# Suite 51: Explicit UNKNOWN in semantic belief overrides stale canonical_category on OBJECT nodes
def test_explicit_unknown_overrides_stale_canonical_category_on_object() -> None:
    from mujoco_scenes.functional_tamp_pipeline.grounding import check_semantic_role_compatibility
    from mujoco_scenes.functional_tamp_pipeline.scene_graph import ObservedNode

    node = ObservedNode(
        instance_id="obj_1",
        entity_kind="OBJECT",
        canonical_category="screwdriver",  # stale or speculative label
        semantic_labels={
            "status": "UNKNOWN",
            "canonical_label": None,
            "plausible_labels": ["screwdriver", "screw"],
            "ambiguity_hypotheses": ["screwdriver", "screw"],
            "reason_codes": ["CONFLICTING_MULTI_VIEW_LABELS"],
        },
    )

    # For driver role:
    status, _ = check_semantic_role_compatibility(node, ["screwdriver", "power_driver"])
    assert status == "UNKNOWN", f"Expected UNKNOWN for ambiguous node, got {status}"

    # For fastener role:
    status, _ = check_semantic_role_compatibility(node, ["screw"])
    assert status == "UNKNOWN", f"Expected UNKNOWN for ambiguous node, got {status}"








