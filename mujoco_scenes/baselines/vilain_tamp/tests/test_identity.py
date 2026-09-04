from __future__ import annotations

from dataclasses import replace

import pytest

from mujoco_scenes.baselines.vilain_tamp.contracts import (
    ObjectEstimate,
    ObjectEstimateStatus,
    SymbolicAction,
)
from mujoco_scenes.baselines.vilain_tamp.execution import (
    ProjectionError,
    project_action,
)
from mujoco_scenes.baselines.vilain_tamp.identity import (
    BaselineIdentityResolver,
    EntityCandidate,
    IdentityInputError,
    IdentityResolutionError,
    fixed_entity_binding,
)


def estimate(
    object_id: str,
    pddl_type: str,
    centroid: tuple[float, float, float] | None,
) -> ObjectEstimate:
    return ObjectEstimate(
        object_id=object_id,
        label=object_id,
        pddl_type=pddl_type,
        description="visible synthetic object",
        detections=(
            {
                "stage_id": "000_initial",
                "camera_id": "front",
                "xyxy": (0.0, 0.0, 2.0, 2.0),
                "confidence": 0.9,
            },
        ),
        estimated_centroid_m=centroid,
        centroid_covariance=None,
        observation_stage_ids=("000_initial",),
        status=ObjectEstimateStatus.OBSERVED,
    )


def candidate(
    name: str,
    pddl_type: str,
    centroid: tuple[float, float, float],
    *,
    broad_class: str | None = None,
) -> EntityCandidate:
    return EntityCandidate(
        entity_name=name,
        broad_class=broad_class or pddl_type,
        compatible_pddl_types=(pddl_type,),
        centroid_m=centroid,
        aabb_min_m=tuple(value - 0.02 for value in centroid),
        aabb_max_m=tuple(value + 0.02 for value in centroid),
        visible_stage_ids=("000_initial",),
        movable=True,
        evidence_artifacts=(f"evidence/{name}.json",),
    )


def resolver() -> BaselineIdentityResolver:
    return BaselineIdentityResolver(
        maximum_distance_m=0.30,
        ambiguity_margin_m=0.02,
    )


def test_deterministic_correct_one_to_one_binding() -> None:
    estimates = (
        estimate("mug_1", "vessel", (0.02, 0.0, 0.0)),
        estimate("spoon_1", "utensil", (1.01, 0.0, 0.0)),
    )
    candidates = (
        candidate("scene_spoon", "utensil", (1.0, 0.0, 0.0), broad_class="tool"),
        candidate("scene_mug", "vessel", (0.0, 0.0, 0.0)),
    )
    result = resolver().resolve(estimates, candidates)

    assert {
        key: value.entity_name for key, value in result.by_object_id().items()
    } == {"mug_1": "scene_mug", "spoon_1": "scene_spoon"}
    mug = result.by_object_id()["mug_1"]
    assert mug.binding_method == "ONE_TO_ONE_CLASS_CENTROID_AABB"
    assert mug.confidence == pytest.approx(1.0 - 0.02 / 0.30)
    assert mug.entity_aabb_min_m == (-0.02, -0.02, -0.02)
    assert mug.evidence_artifacts == ("evidence/scene_mug.json",)


def test_ambiguous_binding_fails_without_guessing() -> None:
    with pytest.raises(IdentityResolutionError) as captured:
        resolver().resolve(
            (estimate("mug_1", "vessel", (0.0, 0.0, 0.0)),),
            (
                candidate("mug_left", "vessel", (-0.05, 0.0, 0.0)),
                candidate("mug_right", "vessel", (0.05, 0.0, 0.0)),
            ),
        )
    assert captured.value.reason_code == "AMBIGUOUS_ENTITY"
    assert captured.value.candidate_entities == ("mug_left", "mug_right")


def test_unresolved_binding_fails_for_distance_class_and_visibility() -> None:
    hidden = replace(
        candidate("hidden_mug", "vessel", (0.0, 0.0, 0.0)),
        visible_stage_ids=("001_drawer",),
    )
    wrong_class = candidate("near_spoon", "utensil", (0.0, 0.0, 0.0))
    far = candidate("far_mug", "vessel", (1.0, 0.0, 0.0))
    with pytest.raises(IdentityResolutionError) as captured:
        resolver().resolve(
            (estimate("mug_1", "vessel", (0.0, 0.0, 0.0)),),
            (hidden, wrong_class, far),
        )
    assert captured.value.reason_code == "UNRESOLVED_ENTITY"


def test_one_to_one_conflict_is_not_silently_reused() -> None:
    with pytest.raises(IdentityResolutionError) as captured:
        resolver().resolve(
            (
                estimate("mug_1", "vessel", (0.0, 0.0, 0.0)),
                estimate("mug_2", "vessel", (0.20, 0.0, 0.0)),
            ),
            (candidate("only_mug", "vessel", (0.10, 0.0, 0.0)),),
        )
    assert captured.value.reason_code == "UNRESOLVED_ONE_TO_ONE"


def test_non_observed_or_centroidless_estimate_is_unresolved() -> None:
    lost = replace(
        estimate("mug_1", "vessel", None),
        status=ObjectEstimateStatus.LOST,
    )
    with pytest.raises(IdentityResolutionError, match="no usable observed centroid"):
        resolver().resolve((lost,), (candidate("mug", "vessel", (0.0, 0.0, 0.0)),))


def test_non_contract_and_external_method_inputs_are_rejected() -> None:
    unsafe = {
        "object_id": "mug_1",
        "pddl_type": "vessel",
        "functional_role": "best vessel",
    }
    with pytest.raises(IdentityInputError, match="functional-role fields"):
        resolver().resolve((unsafe,), ())  # type: ignore[arg-type]
    with pytest.raises(IdentityInputError, match="external method artifacts"):
        resolver().resolve((), (), external_method_artifacts={"artifact": "foreign"})


def binding(object_id: str, entity_name: str, pddl_type: str):
    result = resolver().resolve(
        (estimate(object_id, pddl_type, (0.0, 0.0, 0.0)),),
        (candidate(entity_name, pddl_type, (0.01, 0.0, 0.0)),),
    )
    return result.bindings[0]


def static_binding(symbolic_id: str, entity_name: str, broad_class: str):
    return fixed_entity_binding(
        symbolic_id,
        entity_name,
        broad_class=broad_class,
        evidence_artifacts=(f"evidence/{entity_name}.json",),
    )


@pytest.mark.parametrize(
    "domain, operator, arguments, movable, fixed, expected",
    [
        (
            "kitchen",
            "open-storage",
            ("drawer",),
            {},
            {"drawer": static_binding("drawer", "drawer_body", "storage")},
            ("OPEN", ("drawer_body",)),
        ),
        (
            "kitchen",
            "place-on",
            ("mug_1", "counter"),
            {"mug_1": binding("mug_1", "mug_body", "vessel")},
            {"counter": static_binding("counter", "counter_body", "surface")},
            ("PLACE", ("mug_body", "counter_body")),
        ),
        (
            "kitchen",
            "pick-from",
            ("mug_1", "counter"),
            {"mug_1": binding("mug_1", "mug_body", "vessel")},
            {},
            ("PICK", ("mug_body",)),
        ),
        (
            "kitchen",
            "pour",
            ("source_1", "mug_1", "coffee"),
            {
                "source_1": binding("source_1", "source_body", "source"),
                "mug_1": binding("mug_1", "mug_body", "vessel"),
            },
            {},
            ("POUR", ("source_body", "mug_body")),
        ),
        (
            "kitchen",
            "stir",
            ("spoon_1", "mug_1"),
            {
                "spoon_1": binding("spoon_1", "spoon_body", "utensil"),
                "mug_1": binding("mug_1", "mug_body", "vessel"),
            },
            {},
            ("STIR", ("spoon_body", "mug_body")),
        ),
        (
            "kitchen",
            "place-in",
            ("spoon_1", "bowl_1"),
            {
                "spoon_1": binding("spoon_1", "spoon_body", "utensil"),
                "bowl_1": binding("bowl_1", "bowl_body", "vessel"),
            },
            {},
            ("PLACE", ("spoon_body", "bowl_body")),
        ),
        (
            "living-room",
            "pick-from",
            ("cup_1", "staging"),
            {"cup_1": binding("cup_1", "cup_body", "cup")},
            {},
            ("PICK", ("cup_body",)),
        ),
        (
            "living-room",
            "place-on",
            ("cup_1", "table_1"),
            {
                "cup_1": binding("cup_1", "cup_body", "cup"),
                "table_1": binding("table_1", "table_body", "support"),
            },
            {},
            ("PLACE", ("cup_body", "table_body")),
        ),
        (
            "workshop",
            "open-storage",
            ("drawer",),
            {},
            {"drawer": static_binding("drawer", "drawer_body", "storage")},
            ("OPEN", ("drawer_body",)),
        ),
        (
            "workshop",
            "pick-from",
            ("screw_1", "drawer"),
            {"screw_1": binding("screw_1", "screw_body", "fastener")},
            {"drawer": static_binding("drawer", "drawer_body", "storage")},
            ("PICK", ("screw_body", "drawer_body")),
        ),
        (
            "workshop",
            "insert",
            ("screw_1", "target"),
            {"screw_1": binding("screw_1", "screw_body", "fastener")},
            {"target": static_binding("target", "target_body", "target")},
            ("PLACE", ("screw_body", "target_body")),
        ),
        (
            "workshop",
            "drive",
            ("driver_1", "screw_1", "target"),
            {
                "driver_1": binding("driver_1", "driver_body", "driver"),
                "screw_1": binding("screw_1", "screw_body", "fastener"),
            },
            {"target": static_binding("target", "target_body", "target")},
            ("SCREW", ("driver_body", "screw_body", "target_body")),
        ),
        (
            "workshop",
            "place-on",
            ("driver_1", "bench"),
            {"driver_1": binding("driver_1", "driver_body", "driver")},
            {"bench": static_binding("bench", "bench_body", "surface")},
            ("PLACE", ("driver_body", "bench_body")),
        ),
    ],
)
def test_section_thirteen_syntactic_projections(
    domain, operator, arguments, movable, fixed, expected
) -> None:
    action = SymbolicAction(0, f"vilain_00_001_{operator}", operator, arguments)
    projected = project_action(domain, action, movable, fixed_bindings=fixed)
    assert (projected.controller_operator, projected.controller_arguments) == expected
    assert projected.pddl_arguments == arguments
    assert projected.resolved_entities == expected[1]
    assert projected.binding_evidence_artifacts


def test_projection_rejects_unknown_unresolved_and_external_inputs() -> None:
    known = {"mug_1": binding("mug_1", "mug_body", "vessel")}
    with pytest.raises(ProjectionError, match="UNSUPPORTED_CONTROLLER_ACTION"):
        project_action(
            "kitchen", SymbolicAction(0, "bad", "teleport", ("mug_1",)), known
        )
    with pytest.raises(ProjectionError, match="UNRESOLVED_ENTITY"):
        project_action(
            "kitchen",
            SymbolicAction(0, "place", "place-on", ("mug_1", "counter")),
            known,
        )
    with pytest.raises(ProjectionError, match="external method artifacts"):
        project_action(
            "kitchen",
            SymbolicAction(0, "pick", "pick-from", ("mug_1", "counter")),
            known,
            external_method_artifacts={"artifact": "foreign"},
        )
