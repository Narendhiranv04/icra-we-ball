from __future__ import annotations

import json

import pytest

from baseline_common.inference import (
    InvalidCompletionError,
    ModelTransportError,
    PlanningError,
    TruncatedCompletionError,
)
from baseline_common.models import (
    Action,
    ActionResult,
    Entity,
    Observation,
    Region,
    ValidationError,
)

from vlm_tamp_baseline.catalog import load_catalog, scene_subgoals
from vlm_tamp_baseline.executive import ObservationFrame, VLMTAMPExecutive
from vlm_tamp_baseline.models import (
    ObjectReference,
    ObjectUniverse,
    EnglishPlan,
    RefinementFailure,
    Subgoal,
    SubgoalPlan,
    parse_subgoal_plan,
)
from vlm_tamp_baseline.planner import (
    SubgoalPlanResult,
    VLMTAMPPlanner,
    VLMTAMPPlannerConfig,
)
from vlm_tamp_baseline.prompt import (
    english_response_schema,
    grounding_response_schema,
)
from vlm_tamp_baseline.refiner import CatalogSubgoalRefiner, subgoal_satisfied
from vlm_tamp_baseline.pddlstream_refiner import (
    KitchenGeometryOracle,
    PDDLStreamProtocol,
    PDDLStreamSubgoalRefiner,
)
from vlm_tamp_baseline.run_kitchen import (
    _print_refinement,
    _print_subgoal_plan,
)


IMAGE = {"camera": "front", "data_url": "data:image/png;base64,AA=="}


def observation(*, entities=("mug_1", "spoon_1"), holding=None, goal=False, revision=0):
    return Observation(
        "kitchen",
        revision,
        tuple(Entity(item, "object", item) for item in entities),
        (Region("D1", "drawer", "closed", False),),
        {"holding": holding},
        goal,
    )


class FakeTransport:
    def __init__(self, content):
        self.content = iter(content if isinstance(content, list) else [content])
        self.payloads = []

    def complete(self, payload):
        self.payloads.append(payload)
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": json.dumps(next(self.content))},
                }
            ]
        }


def test_catalog_covers_all_benchmark_scenes():
    catalog = load_catalog()
    assert set(catalog["scenes"]) == {"kitchen", "living_room", "workshop"}
    assert "STIRRED" in catalog["scenes"]["kitchen"]
    assert "CLEANED" in catalog["scenes"]["living_room"]
    assert "FASTENED" in catalog["scenes"]["workshop"]


def test_structured_schemas_make_status_and_array_cardinality_exclusive():
    english = english_response_schema(20)["oneOf"]
    grounding = grounding_response_schema(["STIRRED"], 20)["oneOf"]

    english_cardinality = [
        (
            row["properties"]["status"]["const"],
            row["properties"]["steps"]["minItems"],
            row["properties"]["steps"]["maxItems"],
        )
        for row in english
    ]
    grounding_cardinality = [
        (
            row["properties"]["status"]["const"],
            row["properties"]["subgoals"]["minItems"],
            row["properties"]["subgoals"]["maxItems"],
        )
        for row in grounding
    ]
    assert english_cardinality == [
        ("STEPS", 1, 20),
        ("NO_VALID_STEPS", 0, 0),
    ]
    assert grounding_cardinality == [
        ("SUBGOALS", 1, 20),
        ("NO_VALID_SUBGOALS", 0, 0),
    ]
    assert english[0]["properties"]["steps"]["items"]["maxLength"] == 240


def test_grounding_schema_restricts_arguments_to_observed_ids():
    schema = grounding_response_schema(
        scene_subgoals(load_catalog(), "kitchen"),
        20,
        object_ids=("object_0001", "object_0002"),
        region_ids=("D1", "countertop"),
    )
    variants = schema["oneOf"][0]["properties"]["subgoals"]["items"]["oneOf"]
    by_predicate = {
        row["properties"]["predicate"]["const"]: row for row in variants
    }
    holding = by_predicate["HOLDING"]["properties"]["arguments"]
    assert holding["properties"]["object_id"]["enum"] == [
        "object_0001",
        "object_0002",
    ]
    placed = by_predicate["PLACED"]["properties"]["arguments"]
    assert placed["properties"]["region_id"]["enum"] == [
        "D1",
        "countertop",
        "object_0001",
        "object_0002",
    ]


def test_planner_uses_independent_goal_verifier_without_calling_model():
    transport = FakeTransport([])
    planner = VLMTAMPPlanner(
        VLMTAMPPlannerConfig(model="test-model"), transport=transport
    )

    result = planner.plan(
        "Stir the mug",
        observation(goal=True),
        (IMAGE,),
    )

    assert result.plan.status == "GOAL_COMPLETE"
    assert result.english_plan.status == "GOAL_COMPLETE"
    assert result.latency_ms == 0.0
    assert transport.payloads == []


def test_false_model_completion_is_rejected_even_without_structured_output():
    transport = FakeTransport({"status": "GOAL_COMPLETE", "steps": []})
    planner = VLMTAMPPlanner(
        VLMTAMPPlannerConfig(model="test-model", structured_output=False),
        transport=transport,
    )

    with pytest.raises(PlanningError, match="independent goal verifier is false"):
        planner.plan("Stir the mug", observation(), (IMAGE,))


def test_terminal_shows_subgoals_and_refined_actions(capsys):
    subgoal = Subgoal(
        "STIRRED", {"tool_id": "spoon_1", "target_id": "mug_1"}
    )
    _print_subgoal_plan(3, SubgoalPlan("SUBGOALS", (subgoal,)))
    _print_refinement(
        subgoal,
        (
            Action("PICK", {"object_id": "spoon_1"}),
            Action(
                "STIR", {"tool_id": "spoon_1", "target_id": "mug_1"}
            ),
        ),
    )

    assert capsys.readouterr().out.splitlines() == [
        "[vlm-tamp subgoals 3] SUBGOALS",
        "  1. STIRRED tool_id=spoon_1, target_id=mug_1",
        "[vlm-tamp actions] STIRRED",
        "  1. PICK object_id=spoon_1",
        "  2. STIR tool_id=spoon_1, target_id=mug_1",
    ]


def test_terminal_appends_observed_object_aliases(capsys):
    subgoal = Subgoal(
        "STIRRED", {"tool_id": "object_0004", "target_id": "object_0002"}
    )
    aliases = {"object_0004": "spoon", "object_0002": "mug"}

    _print_subgoal_plan(
        1, SubgoalPlan("SUBGOALS", (subgoal,)), aliases
    )
    _print_refinement(
        subgoal,
        (Action("STIR", subgoal.arguments),),
        aliases,
    )

    assert capsys.readouterr().out.splitlines() == [
        "[vlm-tamp subgoals 1] SUBGOALS",
        "  1. STIRRED tool_id=object_0004 (spoon), "
        "target_id=object_0002 (mug)",
        "[vlm-tamp actions] STIRRED",
        "  1. STIR tool_id=object_0004 (spoon), "
        "target_id=object_0002 (mug)",
    ]


def test_unknown_object_is_rejected_without_privileged_universe():
    with pytest.raises(ValidationError, match="unknown object"):
        parse_subgoal_plan(
            {
                "status": "SUBGOALS",
                "subgoals": [
                    {"predicate": "HOLDING", "arguments": {"object_id": "hidden"}}
                ],
            },
            scene_subgoals(load_catalog(), "kitchen"),
            observation(),
            ObjectUniverse.observed(observation()),
            max_subgoals=10,
        )


def test_paper_protocol_allows_named_but_hidden_object():
    universe = ObjectUniverse(
        (
            ObjectReference("mug_1"),
            ObjectReference("spoon_1"),
            ObjectReference("hidden"),
        ),
        True,
    )
    plan = parse_subgoal_plan(
        {
            "status": "SUBGOALS",
            "subgoals": [
                {"predicate": "HOLDING", "arguments": {"object_id": "hidden"}}
            ],
        },
        scene_subgoals(load_catalog(), "kitchen"),
        observation(),
        universe,
        max_subgoals=10,
    )
    assert plan.subgoals[0].arguments["object_id"] == "hidden"


def test_prompt_contains_subgoals_but_not_primitive_action_catalog():
    transport = FakeTransport(
        [
            {"status": "STEPS", "steps": ["Stir the mug with the spoon."]},
            {
                "status": "SUBGOALS",
                "subgoals": [
                    {
                        "predicate": "STIRRED",
                        "arguments": {"tool_id": "spoon_1", "target_id": "mug_1"},
                    }
                ],
            },
        ]
    )
    planner = VLMTAMPPlanner(
        VLMTAMPPlannerConfig(model="test-model"), transport=transport
    )
    result = planner.plan("Stir the mug", observation(), (IMAGE,))
    assert result.plan.subgoals[0].predicate == "STIRRED"
    first = json.loads(transport.payloads[0]["messages"][1]["content"][0]["text"])
    second = json.loads(transport.payloads[1]["messages"][1]["content"][0]["text"])
    assert "formal_subgoal_catalog" not in first
    assert "formal_subgoal_catalog" in second
    assert "action_catalog" not in first
    assert "action_catalog" not in second
    assert "observation" not in first
    assert "textualized_state" in first
    assert first["object_universe"]["objects"] == [
        {"id": "mug_1"},
        {"id": "spoon_1"},
    ]
    assert "label" not in json.dumps(first["textualized_state"])
    assert [
        item["type"]
        for item in transport.payloads[1]["messages"][1]["content"]
    ] == ["text", "text", "image_url"]
    assert transport.payloads[0]["response_format"]["json_schema"]["name"] == "vlm_tamp_english_goals"
    assert transport.payloads[1]["response_format"]["json_schema"]["name"] == "vlm_tamp_grounded_subgoals"
    assert "semantic instance aliases" in transport.payloads[0]["messages"][0]["content"]
    assert first["textualized_state"]["semantic_annotations"]["objects"] == [
        {"id": "mug_1", "alias": "mug_1"},
        {"id": "spoon_1", "alias": "spoon_1"},
    ]
    assert "The five RGB views" not in transport.payloads[0]["messages"][0]["content"]
    assert "Do not\n  enumerate every visible object" in transport.payloads[0]["messages"][0]["content"]
    assert len(planner.request_trace) == 2
    assert len(planner.response_trace) == 2
    assert "data:image/" not in json.dumps(planner.request_trace)
    assert "<embedded image omitted; see saved camera PNG>" in json.dumps(
        planner.request_trace
    )


def test_vlm_tamp_config_strips_boolean_environment_value():
    config = VLMTAMPPlannerConfig.from_env(
        {
            "VLM_TAMP_PROFILE": "qwen35-9b",
            "VLM_TAMP_ENABLE_THINKING": " false ",
        }
    )
    assert not config.enable_thinking
    assert config.sampling["temperature"] == 0.7


def test_vlm_tamp_config_rejects_boolean_numeric_limits():
    with pytest.raises(ValueError, match="positive"):
        VLMTAMPPlannerConfig(model="test", max_subgoals=True)


def test_vlm_tamp_config_rejects_non_integer_subgoal_limit():
    with pytest.raises(ValueError, match="max_subgoals"):
        VLMTAMPPlannerConfig(model="test", max_subgoals=2.5)


def test_refiner_maps_subgoal_to_shared_skills():
    result = CatalogSubgoalRefiner().refine(
        Subgoal("STIRRED", {"tool_id": "spoon_1", "target_id": "mug_1"}),
        observation(),
    )
    assert result.success
    assert [action.skill for action in result.actions] == ["PICK", "STIR"]


def test_refiner_returns_ungrounded_failure_for_hidden_object():
    result = CatalogSubgoalRefiner().refine(
        Subgoal("HOLDING", {"object_id": "hidden"}), observation()
    )
    assert not result.success
    assert result.failure.code == "ungrounded_object"


def test_visible_object_can_be_a_placement_destination():
    plan = parse_subgoal_plan(
        {
            "status": "SUBGOALS",
            "subgoals": [
                {
                    "predicate": "PLACED",
                    "arguments": {"object_id": "spoon_1", "region_id": "mug_1"},
                }
            ],
        },
        scene_subgoals(load_catalog(), "kitchen"),
        observation(),
        ObjectUniverse.observed(observation()),
        max_subgoals=10,
    )
    refined = CatalogSubgoalRefiner().refine(plan.subgoals[0], observation())
    assert refined.success
    assert [action.skill for action in refined.actions] == ["PICK", "PLACE"]


def test_pddl_world_keeps_hidden_location_private_from_vlm_but_available_to_tamp():
    inventory = {
        "objects": [
            {
                "generic_object_id": "hidden_spoon",
                "semantic_label": "spoon",
                "observed_dimensions_m": {
                    "length": 0.2,
                    "width": 0.02,
                    "height": 0.01,
                },
                "selected_functions": [],
                "source_context": {
                    "source_container": "D1",
                    "observed_source_region": "D1",
                    "required_workspace": "home",
                },
            }
        ]
    }
    current = observation(entities=())
    refiner = PDDLStreamSubgoalRefiner(
        inventory, KitchenGeometryOracle(inventory)
    )
    facts = refiner._initial_facts(("hidden_spoon",), current)
    assert ("At", "hidden_spoon", "D1") in facts
    assert ("Closed", "D1") in facts
    assert ("Accessible", "D1") not in facts


def test_pddl_world_maps_phase1_initial_table_provenance_to_countertop():
    inventory = {
        "objects": [
            {
                "generic_object_id": "coffee_source",
                "semantic_label": "coffee_source",
                "observed_dimensions_m": {
                    "length": 0.1,
                    "width": 0.1,
                    "height": 0.1,
                },
                "selected_functions": ["coffee_source"],
                "source_context": {
                    "source_container": None,
                    "observed_source_region": "INITIAL",
                    "source_kind": "TABLE",
                    "required_workspace": "home",
                },
            }
        ]
    }
    current = Observation(
        "kitchen",
        0,
        (
            Entity(
                "coffee_source",
                "object",
                "coffee_source",
                {"source_region": "INITIAL"},
            ),
        ),
        (),
        {"workspace": "home"},
        False,
    )
    refiner = PDDLStreamSubgoalRefiner(
        inventory, KitchenGeometryOracle(inventory)
    )

    facts = refiner._initial_facts(("coffee_source",), current)

    assert ("At", "coffee_source", "countertop") in facts
    assert ("Accessible", "countertop") in facts
    assert not any(
        fact[:2] == ("At", "coffee_source") and fact[2] == "INITIAL"
        for fact in facts
    )

    result = refiner.refine(
        Subgoal("HOLDING", {"object_id": "coffee_source"}), current
    )

    assert result.success
    assert [action.skill for action in result.actions] == ["PICK"]


def _inventory_row(object_id, label, region="countertop"):
    return {
        "generic_object_id": object_id,
        "semantic_label": label,
        "observed_dimensions_m": {
            "length": 0.2,
            "width": 0.03,
            "height": 0.02,
        },
        "selected_functions": [],
        "source_context": {
            "source_container": region,
            "observed_source_region": region,
        },
    }


def test_pddlstream_places_an_object_into_a_visible_receptacle():
    inventory = {
        "objects": [
            _inventory_row("spoon", "spoon"),
            _inventory_row("mug", "mug"),
        ]
    }
    current = Observation(
        "kitchen",
        0,
        (
            Entity("spoon", "object", "spoon", {"region_id": "countertop"}),
            Entity("mug", "object", "mug", {"region_id": "countertop"}),
        ),
        (),
        {"workspace": "home"},
        False,
    )
    refiner = PDDLStreamSubgoalRefiner(
        inventory,
        KitchenGeometryOracle(inventory),
        protocol=PDDLStreamProtocol(timeout_seconds=10),
    )

    result = refiner.refine(
        Subgoal("PLACED", {"object_id": "spoon", "region_id": "mug"}),
        current,
    )

    assert result.success
    assert [action.skill for action in result.actions] == ["PICK", "PLACE"]
    assert any(
        row["operator"] == "place-object"
        for row in refiner.last_trace["attempts"][0]["pddl_plan"]
    )


def test_initial_source_region_satisfies_redundant_placed_subgoal():
    current = Observation(
        "kitchen",
        0,
        (
            Entity(
                "object_1", "object", "object_1",
                {"source_region": "countertop"},
            ),
        ),
        (),
        {"workspace": "home"},
        False,
    )
    assert subgoal_satisfied(
        Subgoal(
            "PLACED", {"object_id": "object_1", "region_id": "countertop"}
        ),
        current,
    )


def test_pddlstream_does_not_park_held_vessel_in_stir_target():
    inventory = {
        "objects": [
            _inventory_row("coffee_source", "coffee source"),
            _inventory_row("spoon", "spoon"),
            _inventory_row("mug", "mug"),
        ]
    }
    current = Observation(
        "kitchen",
        0,
        (
            Entity("coffee_source", "object", "coffee source", {}),
            Entity("spoon", "object", "spoon", {"region_id": "countertop"}),
            Entity("mug", "object", "mug", {"region_id": "countertop"}),
        ),
        (),
        {"workspace": "home", "held_object_id": "coffee_source"},
        False,
    )
    refiner = PDDLStreamSubgoalRefiner(
        inventory,
        KitchenGeometryOracle(inventory),
        protocol=PDDLStreamProtocol(timeout_seconds=10),
    )

    result = refiner.refine(
        Subgoal("STIRRED", {"tool_id": "spoon", "target_id": "mug"}),
        current,
    )

    assert result.success
    assert [action.skill for action in result.actions] == ["PLACE", "PICK", "STIR"]
    assert result.actions[0].arguments == {
        "object_id": "coffee_source",
        "region_id": "countertop",
    }
    assert not any(
        row["operator"] == "place-object"
        for row in refiner.last_trace["attempts"][0]["pddl_plan"]
    )


def test_pddlstream_inspects_before_using_a_target_in_closed_storage():
    inventory = {
        "objects": [
            _inventory_row("kettle", "kettle"),
            _inventory_row("cup", "cup", "C2"),
        ]
    }
    current = Observation(
        "kitchen",
        0,
        (Entity("kettle", "object", "kettle", {"region_id": "countertop"}),),
        (Region("C2", "cabinet", "closed", False),),
        {"workspace": "home"},
        False,
    )
    refiner = PDDLStreamSubgoalRefiner(
        inventory,
        KitchenGeometryOracle(inventory),
        protocol=PDDLStreamProtocol(timeout_seconds=10),
    )

    result = refiner.refine(
        Subgoal("POURED", {"source_id": "kettle", "target_id": "cup"}),
        current,
    )

    assert result.success
    assert [action.skill for action in result.actions] == [
        "INSPECT",
        "PICK",
        "POUR",
    ]


def test_pddlstream_dependency_is_activated_once_per_refiner(monkeypatch):
    activations = 0

    def activate():
        nonlocal activations
        activations += 1

    monkeypatch.setattr(
        "vlm_tamp_baseline.pddlstream_refiner.activate_pddlstream", activate
    )
    refiner = PDDLStreamSubgoalRefiner({}, KitchenGeometryOracle({}))
    monkeypatch.setattr(
        refiner,
        "_solve",
        lambda _trial, _objects, _subgoal, _observation: {
            "actions": (),
            "trace": {},
        },
    )
    subgoal = Subgoal("INSPECTED", {"region_id": "D1"})
    current = observation()

    assert refiner.refine(subgoal, current).success
    assert refiner.refine(subgoal, current).success
    assert activations == 1


def test_pddlstream_protocol_rejects_unsupported_trial_count():
    with pytest.raises(ValueError, match="between 1 and 3"):
        PDDLStreamProtocol(max_tamp_trials=4)


def test_refiner_reads_runtime_held_object_key_and_rejects_second_pick():
    current = observation()
    current = Observation(
        current.scene,
        current.revision,
        current.entities,
        current.regions,
        {"held_object": "mug_1"},
        current.goal_satisfied,
    )
    subgoal = Subgoal(
        "STIRRED", {"tool_id": "spoon_1", "target_id": "mug_1"}
    )
    refined = CatalogSubgoalRefiner().refine(subgoal, current)

    assert not refined.success
    assert refined.failure.code == "hand_not_empty"
    assert "mug_1" in refined.failure.message


class FakePlanner:
    def __init__(self, plans):
        self.plans = iter(plans)
        self.failures = []

    def plan(self, *args, failure=None, **kwargs):
        self.failures.append(failure)
        return SubgoalPlanResult(
            next(self.plans),
            EnglishPlan("STEPS", ("test",)),
            "fake",
            0.0,
            False,
        )


class InvalidThenValidPlanner(FakePlanner):
    def __init__(self, plan):
        super().__init__([plan])
        self.attempts = 0

    def plan(self, *args, failure=None, **kwargs):
        self.failures.append(failure)
        self.attempts += 1
        if self.attempts == 1:
            raise PlanningError(
                "Invalid VLM-TAMP English plan: GOAL_COMPLETE must not contain steps"
            )
        return SubgoalPlanResult(
            next(self.plans),
            EnglishPlan("STEPS", ("test",)),
            "fake",
            0.0,
            False,
        )


class TransportFailureThenValidPlanner(FakePlanner):
    def __init__(self, plan):
        super().__init__([plan])
        self.attempts = 0

    def plan(self, *args, failure=None, **kwargs):
        self.failures.append(failure)
        self.attempts += 1
        if self.attempts == 1:
            raise ModelTransportError("Model server returned HTTP 503")
        return SubgoalPlanResult(
            next(self.plans),
            EnglishPlan("STEPS", ("test",)),
            "fake",
            0.0,
            False,
        )


class FakeRefiner:
    def __init__(self):
        self.calls = 0

    def refine(self, subgoal, current):
        self.calls += 1
        if self.calls == 1:
            from vlm_tamp_baseline.refiner import RefinementResult

            return RefinementResult(
                failure=RefinementFailure("ik_failed", "No collision-free IK.", subgoal)
            )
        return CatalogSubgoalRefiner().refine(subgoal, current)


class FakeWorld:
    def __init__(self):
        self.state = observation()

    def observe(self):
        return ObservationFrame(self.state, (IMAGE,))

    def execute(self, action):
        self.state = observation(holding="mug_1", goal=True, revision=1)
        return ActionResult.succeeded()


def test_executive_uses_state_only_observer_between_model_calls():
    subgoal = Subgoal("HOLDING", {"object_id": "mug_1"})
    planner = FakePlanner([SubgoalPlan("SUBGOALS", (subgoal,))])
    world = FakeWorld()
    frame_calls = 0
    state_calls = 0

    def observe_frame():
        nonlocal frame_calls
        frame_calls += 1
        return world.observe()

    def observe_state():
        nonlocal state_calls
        state_calls += 1
        return world.state

    result = VLMTAMPExecutive(
        planner,
        observe_frame,
        world,
        refiner=CatalogSubgoalRefiner(),
        state_observer=observe_state,
    ).run("Pick the mug")

    assert result.success
    assert frame_calls == 1
    assert state_calls == 2


def test_executive_reprompts_after_plan_preparation_failure():
    class PreparationWorld(FakeWorld):
        def __init__(self):
            super().__init__()
            self.preparations = 0

        def prepare(self, _actions):
            self.preparations += 1
            if self.preparations == 1:
                return ActionResult.failed("collision", "Plan intersects cabinet")
            return ActionResult.succeeded()

    subgoal = Subgoal("HOLDING", {"object_id": "mug_1"})
    planner = FakePlanner(
        [SubgoalPlan("SUBGOALS", (subgoal,)), SubgoalPlan("SUBGOALS", (subgoal,))]
    )
    world = PreparationWorld()

    result = VLMTAMPExecutive(
        planner,
        world.observe,
        world,
        refiner=CatalogSubgoalRefiner(),
        max_model_calls=2,
    ).run("Pick the mug")

    assert result.success
    assert result.executed_actions == 1
    assert planner.failures[1].code == "collision"


def test_executive_structures_plan_preparation_exception():
    class BrokenPreparationWorld(FakeWorld):
        def prepare(self, _actions):
            raise RuntimeError("authorization backend failed")

    subgoal = Subgoal("HOLDING", {"object_id": "mug_1"})
    planner = FakePlanner([SubgoalPlan("SUBGOALS", (subgoal,))])
    world = BrokenPreparationWorld()

    result = VLMTAMPExecutive(
        planner,
        world.observe,
        world,
        refiner=CatalogSubgoalRefiner(),
    ).run("Pick the mug")

    assert not result.success
    assert result.status == "NON_RECOVERABLE_FAILURE"
    assert result.terminal_failure.code == "internal_error"


def test_executive_reprompts_after_refinement_failure():
    subgoal = Subgoal("HOLDING", {"object_id": "mug_1"})
    planner = FakePlanner(
        [SubgoalPlan("SUBGOALS", (subgoal,)), SubgoalPlan("SUBGOALS", (subgoal,))]
    )
    world = FakeWorld()
    refinements = []
    result = VLMTAMPExecutive(
        planner,
        world.observe,
        world,
        refiner=FakeRefiner(),
        refinement_sink=lambda goal, actions: refinements.append(
            (goal, actions)
        ),
        max_model_calls=3,
    ).run("Pick the mug")
    assert result.success
    assert result.model_calls == 2
    assert result.reprompts == 1
    assert planner.failures[1].code == "ik_failed"
    assert refinements == [
        (subgoal, (Action("PICK", {"object_id": "mug_1"}),))
    ]


def test_executive_reprompts_after_no_valid_subgoals():
    subgoal = Subgoal("HOLDING", {"object_id": "mug_1"})
    planner = FakePlanner(
        [
            SubgoalPlan("NO_VALID_SUBGOALS", ()),
            SubgoalPlan("SUBGOALS", (subgoal,)),
        ]
    )
    world = FakeWorld()

    result = VLMTAMPExecutive(
        planner,
        world.observe,
        world,
        refiner=CatalogSubgoalRefiner(),
        max_model_calls=3,
    ).run("Pick the mug")

    assert result.success
    assert result.model_calls == 2
    assert result.reprompts == 1
    assert planner.failures[1].code == "no_valid_subgoals"


def test_executive_reprompts_instead_of_crashing_on_invalid_model_output():
    subgoal = Subgoal("HOLDING", {"object_id": "mug_1"})
    planner = InvalidThenValidPlanner(SubgoalPlan("SUBGOALS", (subgoal,)))
    world = FakeWorld()

    result = VLMTAMPExecutive(
        planner,
        world.observe,
        world,
        refiner=CatalogSubgoalRefiner(),
        max_model_calls=3,
    ).run("Pick the mug")

    assert result.success
    assert result.model_calls == 2
    assert result.reprompts == 1
    assert planner.failures[1].code == "invalid_vlm_output"


def test_executive_distinguishes_transport_failure_from_invalid_output():
    subgoal = Subgoal("HOLDING", {"object_id": "mug_1"})
    planner = TransportFailureThenValidPlanner(
        SubgoalPlan("SUBGOALS", (subgoal,))
    )
    world = FakeWorld()

    result = VLMTAMPExecutive(
        planner,
        world.observe,
        world,
        refiner=CatalogSubgoalRefiner(),
        max_model_calls=3,
    ).run("Pick the mug")

    assert result.success
    # A transport fault produced no completion, so it must not be charged to
    # the model-call budget: the retried round is the episode's first call.
    assert result.model_calls == 1
    assert result.reprompts == 0
    assert planner.failures[1].code == "inference_failed"


class AlwaysTransportFailingPlanner(FakePlanner):
    def __init__(self):
        super().__init__([])
        self.attempts = 0

    def plan(self, *args, failure=None, **kwargs):
        self.failures.append(failure)
        self.attempts += 1
        raise ModelTransportError("Model server returned HTTP 503")


def test_executive_reports_inference_failure_without_spending_model_calls():
    planner = AlwaysTransportFailingPlanner()
    world = FakeWorld()

    result = VLMTAMPExecutive(
        planner,
        world.observe,
        world,
        refiner=CatalogSubgoalRefiner(),
        max_model_calls=3,
        max_transport_retries=2,
    ).run("Pick the mug")

    assert not result.success
    assert result.status == "INFERENCE_FAILED"
    assert result.model_calls == 0
    assert planner.attempts == 3
    assert result.terminal_failure is not None
    assert result.terminal_failure.code == "inference_failed"

class AlwaysTruncatingPlanner(FakePlanner):
    def __init__(self):
        super().__init__([])
        self.attempts = 0

    def plan(self, *args, failure=None, **kwargs):
        self.failures.append(failure)
        self.attempts += 1
        raise TruncatedCompletionError("token ceiling reached")


def test_truncated_generation_does_not_spend_model_calls():
    """A generation cut off by the token ceiling produced no plan.

    Charging the episode's model-call budget for it would report the method as
    having planned badly when it was never allowed to finish planning, and it
    would burn the whole budget re-running the same runaway generation.
    """
    planner = AlwaysTruncatingPlanner()
    world = FakeWorld()

    result = VLMTAMPExecutive(
        planner,
        world.observe,
        world,
        refiner=CatalogSubgoalRefiner(),
        max_model_calls=10,
        max_truncation_retries=2,
    ).run("Pick the mug")

    assert not result.success
    assert result.status == "MODEL_OUTPUT_TRUNCATED"
    assert result.model_calls == 0
    # Bounded by the truncation budget, not by max_model_calls.
    assert planner.attempts == 3
    assert result.terminal_failure is not None
    assert result.terminal_failure.code == "model_output_truncated"


class AlwaysInvalidPlanner(FakePlanner):
    def __init__(self):
        super().__init__([])
        self.attempts = 0

    def plan(self, *args, failure=None, **kwargs):
        self.failures.append(failure)
        self.attempts += 1
        raise InvalidCompletionError("Completion content is not valid JSON")


def test_invalid_output_still_spends_the_model_call_budget():
    """Only truncation is exempt; unparseable output is a real model failure."""
    planner = AlwaysInvalidPlanner()
    world = FakeWorld()

    result = VLMTAMPExecutive(
        planner,
        world.observe,
        world,
        refiner=CatalogSubgoalRefiner(),
        max_model_calls=3,
    ).run("Pick the mug")

    assert result.status == "MODEL_CALL_BUDGET_EXHAUSTED"
    assert result.model_calls == 3
    assert planner.attempts == 3

