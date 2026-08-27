"""Focused structural tests for the canonical two-graph functional TAMP pipeline."""

from pathlib import Path
from unittest.mock import MagicMock

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
from mujoco_scenes.functional_tamp_pipeline.domains.kitchen import KitchenPlanningCompiler
from mujoco_scenes.living_room_variants import load_living_room_variants
from mujoco_scenes.workshop_scene import WORKSHOP_VARIANTS_CONFIG


def test_both_modes_share_the_provider_boundary() -> None:
    assert isinstance(provider_for_mode("gt"), FunctionalSpecProvider)
    assert isinstance(provider_for_mode("vlm"), FunctionalSpecProvider)


def test_gt_specs_have_no_variant_solution_input() -> None:
    provider = GTSpecProvider()
    for domain in ("workshop", "kitchen", "living_room"):
        specification = provider.provide(domain, "task")
        assert isinstance(specification, FunctionalRequirementGraph)
        assert specification.source == "GT_FUNCTIONAL_SPEC_ONLY"
        assert specification.roles
        serialized = str(specification.to_dict()).lower()
        assert "expected_solution" not in serialized
        assert "hidden" not in serialized


def test_gt_graph_creation_succeeds_across_all_domains() -> None:
    provider = GTSpecProvider()
    for domain in ("kitchen", "workshop", "living_room"):
        graph = provider.provide(domain, "Test instruction")
        assert isinstance(graph, FunctionalRequirementGraph)
        assert len(graph.nodes) > 0
        assert graph.domain == domain
        # Verify relations are explicit edges
        if domain == "workshop":
            assert any(r.predicate == "COMPATIBLE_WITH" for r in graph.relations)
            assert any(r.predicate == "REACHES_TARGET" for r in graph.relations)
        elif domain == "kitchen":
            assert any(r.predicate == "INSERTABLE_IN" for r in graph.relations)
            assert any(r.predicate == "REACHES_BOTTOM" for r in graph.relations)
        elif domain == "living_room":
            assert any("SUPPORT" in r.predicate or "SEAT" in r.predicate or "FITS" in r.predicate for r in graph.relations)


def test_vlm_provider_returns_same_graph_type_as_gt(monkeypatch) -> None:
    vlm_provider = VLMSpecProvider()
    gt_provider = GTSpecProvider()

    # Mock Workshop VLM
    mock_fm_doc = {
        "status": "SUPPORTED",
        "task_summary": "Repair workpiece with screwdriver and screw",
        "unsupported_reason": "",
        "functional_requirements": [
            {
                "id": "tool_1",
                "entity_kind": "OBJECT",
                "function": "drive screw",
                "description": "screwdriver tool",
                "required_count": 1,
                "candidate_objects": [{"label": "screwdriver", "visual_description": "black handle", "suitability_reason": "fits screw"}],
                "required_properties": ["reach target hole", "fits screw head"],
            },
            {
                "id": "fastener_1",
                "entity_kind": "OBJECT",
                "function": "fasten joint",
                "description": "threaded screw",
                "required_count": 1,
                "candidate_objects": [{"label": "screw", "visual_description": "metal screw", "suitability_reason": "fits hole"}],
                "required_properties": ["thread into hole"],
            },
        ],
    }
    mock_priors = {
        "initial_requirements_satisfied": False,
        "decision_reason": "Need to search containers",
        "inspection_order": [
            {"region_id": "RIGHT_DRAWER", "reason": "likely has tools"},
            {"region_id": "LEFT_DRAWER", "reason": "might have fasteners"},
            {"region_id": "TOOL_CABINET", "reason": "backup"},
        ],
    }

    class FakeAdapter:
        def generate_task_requirements(self, *args, **kwargs):
            return mock_fm_doc

        def generate_inspection_priors(self, *args, **kwargs):
            return mock_priors

    from mujoco_scenes.workshop_phase1 import requirements as w_reqs
    monkeypatch.setattr(w_reqs, "FMAdapter", FakeAdapter)

    vlm_graph = vlm_provider.provide("workshop", "task instruction", [Path("/fake/image.png")])
    gt_graph = gt_provider.provide("workshop", "task instruction")

    assert type(vlm_graph) is type(gt_graph)
    assert isinstance(vlm_graph, FunctionalRequirementGraph)
    assert vlm_graph.region_ranking == ("RIGHT_DRAWER", "LEFT_DRAWER", "TOOL_CABINET")
    assert len(vlm_graph.relations) > 0


def test_graph_serialization_roundtrip() -> None:
    # 1. FunctionalRequirementGraph roundtrip
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
    op_group = OperationGroup(
        id="op_1",
        function="drive",
        tool_role="driver",
        target_role="fastener",
        required_target_count=1,
        usage_policy="DEDICATED_PER_TARGET",
        required_relations=("COMPATIBLE_WITH",),
    )
    g_f = FunctionalRequirementGraph(
        domain="workshop",
        task_instruction="Drive screw with driver",
        nodes={"driver": role_a, "fastener": role_b},
        relations=(rel,),
        operation_groups=(op_group,),
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

    # 2. ObservedSceneGraph roundtrip
    node_1 = ObservedNode(
        instance_id="obj_01",
        entity_kind="OBJECT",
        canonical_category="screwdriver",
        semantic_labels={"canonical_label": "screwdriver", "confidence": 0.95},
        source_region="RIGHT_DRAWER",
        geometry={"usable_length_m": 0.15},
        unary_properties={"usable_length_m": 0.15},
        unary_predicates={"CAN_DRIVE_SCREW": "TRUE"},
    )
    node_2 = ObservedNode(
        instance_id="obj_02",
        entity_kind="OBJECT",
        canonical_category="screw",
        semantic_labels={"canonical_label": "screw", "confidence": 0.90},
        source_region="LEFT_DRAWER",
        geometry={"total_length_m": 0.03},
        unary_properties={"total_length_m": 0.03},
        unary_predicates={"CAN_FASTEN": "TRUE"},
    )
    obs_rel = ObservedRelation(
        subject_id="obj_01",
        predicate="COMPATIBLE_WITH",
        object_id="obj_02",
        status="TRUE",
        evidence={"score": 1.0},
    )
    g_o = ObservedSceneGraph(
        nodes={"obj_01": node_1, "obj_02": node_2},
        relations={("COMPATIBLE_WITH", "obj_01", "obj_02"): obs_rel},
        inspected_regions=["RIGHT_DRAWER", "LEFT_DRAWER"],
        stage_index=2,
    )
    data_o = g_o.to_dict()
    reconstructed_o = ObservedSceneGraph.from_dict(data_o)
    assert reconstructed_o.stage_index == 2
    assert len(reconstructed_o.nodes) == 2
    assert reconstructed_o.nodes["obj_01"].canonical_category == "screwdriver"
    assert reconstructed_o.get_relation("COMPATIBLE_WITH", "obj_01", "obj_02").status == "TRUE"


def test_synthetic_graph_grounding_positive() -> None:
    # G_F: driver --COMPATIBLE_WITH--> fastener
    role_driver = FunctionalRole(name="driver", semantic_categories=("screwdriver",))
    role_fastener = FunctionalRole(name="fastener", semantic_categories=("screw",))
    g_f = FunctionalRequirementGraph(
        domain="workshop",
        task_instruction="test",
        nodes={"driver": role_driver, "fastener": role_fastener},
        relations=(FunctionalRelation(subject_role="driver", predicate="COMPATIBLE_WITH", object_role="fastener"),),
    )

    # G_O: object_A (screwdriver), object_B (screw), object_A --COMPATIBLE_WITH(TRUE)--> object_B
    g_o = ObservedSceneGraph()
    g_o.add_node(ObservedNode(instance_id="object_A", canonical_category="screwdriver"))
    g_o.add_node(ObservedNode(instance_id="object_B", canonical_category="screw"))
    g_o.add_relation(ObservedRelation(subject_id="object_A", predicate="COMPATIBLE_WITH", object_id="object_B", status="TRUE"))

    result = ground_graph(g_f, g_o)
    assert result.complete is True
    assert result.status == "COMPLETE"
    assert result.assignment == {"driver": "object_A", "fastener": "object_B"}


def test_synthetic_graph_grounding_negative() -> None:
    # G_F: driver --COMPATIBLE_WITH--> fastener
    role_driver = FunctionalRole(name="driver", semantic_categories=("screwdriver",))
    role_fastener = FunctionalRole(name="fastener", semantic_categories=("screw",))
    g_f = FunctionalRequirementGraph(
        domain="workshop",
        task_instruction="test",
        nodes={"driver": role_driver, "fastener": role_fastener},
        relations=(FunctionalRelation(subject_role="driver", predicate="COMPATIBLE_WITH", object_role="fastener"),),
    )

    # G_O: same nodes but COMPATIBLE_WITH status is FALSE
    g_o = ObservedSceneGraph()
    g_o.add_node(ObservedNode(instance_id="object_A", canonical_category="screwdriver"))
    g_o.add_node(ObservedNode(instance_id="object_B", canonical_category="screw"))
    g_o.add_relation(ObservedRelation(subject_id="object_A", predicate="COMPATIBLE_WITH", object_id="object_B", status="FALSE"))

    result = ground_graph(g_f, g_o)
    assert result.complete is False
    assert result.status in {"INFEASIBLE", "INCOMPLETE"}
    assert result.assignment is None


def test_grounding_distinct_reuse_and_cardinality() -> None:
    # Test 1: Cardinality 2 distinct requirement fails with only 1 object
    role_cups = FunctionalRole(name="cups", count=2, semantic_categories=("cup",), binding_policy="DISTINCT")
    g_f = FunctionalRequirementGraph(
        domain="kitchen",
        task_instruction="need two distinct cups",
        nodes={"cups": role_cups},
    )
    g_o = ObservedSceneGraph()
    g_o.add_node(ObservedNode(instance_id="cup_1", canonical_category="cup"))

    res = ground_graph(g_f, g_o)
    assert res.complete is False
    assert "cups" in res.missing_roles

    # Add second cup -> succeeds
    g_o.add_node(ObservedNode(instance_id="cup_2", canonical_category="cup"))
    res = ground_graph(g_f, g_o)
    assert res.complete is True
    assert set(res.assignment["cups"]) == {"cup_1", "cup_2"}

    # Test 2: Reusable tool policy allows sharing same tool across roles/targets
    role_tool = FunctionalRole(name="stirrer", count=1, semantic_categories=("spoon",), binding_policy="REUSABLE")
    role_cup1 = FunctionalRole(name="cup_target", count=1, semantic_categories=("cup",), binding_policy="DISTINCT")
    g_f_reusable = FunctionalRequirementGraph(
        domain="kitchen",
        task_instruction="stir cup with reusable spoon",
        nodes={"stirrer": role_tool, "cup_target": role_cup1},
        relations=(FunctionalRelation(subject_role="stirrer", predicate="INSERTABLE_IN", object_role="cup_target"),),
    )
    g_o_reusable = ObservedSceneGraph()
    g_o_reusable.add_node(ObservedNode(instance_id="spoon_1", canonical_category="spoon"))
    g_o_reusable.add_node(ObservedNode(instance_id="cup_1", canonical_category="cup"))
    g_o_reusable.add_relation(ObservedRelation(subject_id="spoon_1", predicate="INSERTABLE_IN", object_id="cup_1", status="TRUE"))

    res_reuse = ground_graph(g_f_reusable, g_o_reusable)
    assert res_reuse.complete is True
    assert res_reuse.assignment["stirrer"] == "spoon_1"


def test_workshop_vlm_provider_does_not_fallback_to_hardcoded_ranking(monkeypatch) -> None:
    from mujoco_scenes.workshop_phase1.requirements import FMRequirementProvider

    # Mock FMAdapter returning custom ranking
    custom_ranking = [
        {"region_id": "TOOL_CABINET", "reason": "first"},
        {"region_id": "RIGHT_DRAWER", "reason": "second"},
        {"region_id": "LEFT_DRAWER", "reason": "third"},
    ]
    mock_doc = {
        "status": "SUPPORTED",
        "task_summary": "test",
        "unsupported_reason": "",
        "functional_requirements": [
            {
                "id": "driver_role",
                "entity_kind": "OBJECT",
                "function": "drive screw",
                "description": "screwdriver",
                "required_count": 1,
                "candidate_objects": [{"label": "screwdriver", "visual_description": "tool", "suitability_reason": "ok"}],
                "required_properties": ["reach", "compatible"],
            },
            {
                "id": "screw_role",
                "entity_kind": "OBJECT",
                "function": "screw fastener",
                "description": "screw",
                "required_count": 1,
                "candidate_objects": [{"label": "screw", "visual_description": "fastener", "suitability_reason": "ok"}],
                "required_properties": ["thread"],
            },
        ],
    }

    class CustomRankingAdapter:
        def generate_task_requirements(self, *args, **kwargs):
            return mock_doc

        def generate_inspection_priors(self, *args, **kwargs):
            return {"inspection_order": custom_ranking, "initial_requirements_satisfied": False, "decision_reason": "ok"}

    provider = FMRequirementProvider(fm_adapter=CustomRankingAdapter())
    reqs = provider.get_requirements("task", observation_images=[Path("/fake/image.png")])
    assert provider.region_ranking == ("TOOL_CABINET", "RIGHT_DRAWER", "LEFT_DRAWER")
    assert provider.region_ranking != ("LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET")


def test_living_room_vlm_result_is_not_manual_contract_deepcopy(monkeypatch) -> None:
    from mujoco_scenes.environment_vlm_requirements import EnvironmentVLMRequirementProvider

    mock_doc = {
        "status": "SUPPORTED",
        "task_summary": "Living room custom placement",
        "unsupported_reason": "",
        "functional_requirements": [
            {
                "id": "custom_personal_support",
                "entity_kind": "REGION",
                "function": "support drink",
                "description": "personal side table",
                "required_count": 2,
                "candidate_objects": [{"label": "side table", "visual_description": "small table", "suitability_reason": "near seat"}],
                "required_properties": ["planar support", "near seating area"],
            },
            {
                "id": "custom_shared_support",
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
    normalized_task = result["normalized_task_contract"]

    # Confirm it was dynamically built from the VLM outputs, not identical to the manual YAML file
    assert normalized_task["task_id"] == "vlm_living_room_functional_graph"
    assert normalized_task["natural_language_goal"] == "custom instruction"


def test_downstream_grounding_callable_without_provider_mode() -> None:
    # Grounding directly takes G_F and G_O with zero knowledge of whether mode is "gt" or "vlm"
    role = FunctionalRole(name="target_role", semantic_categories=("container",))
    g_f = FunctionalRequirementGraph(
        domain="generic",
        task_instruction="find container",
        nodes={"target_role": role},
        source="ANY_SOURCE",
    )
    g_o = ObservedSceneGraph()
    g_o.add_node(ObservedNode(instance_id="box_1", canonical_category="container"))

    result = ground_graph(g_f, g_o)
    assert result.complete is True
    assert result.assignment == {"target_role": "box_1"}


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


def test_runner_stops_at_action_sequence_and_has_no_plan_execution_call() -> None:
    root = Path(__file__).resolve().parents[1]
    runner = (root / "run.py").read_text(encoding="utf-8")
    models = (root / "models.py").read_text(encoding="utf-8")
    assert "ACTION_SEQUENCE_READY" in runner
    assert "execute_plan(" not in runner
    assert "execute_sequence(" not in runner
    assert "execution_result" not in models

