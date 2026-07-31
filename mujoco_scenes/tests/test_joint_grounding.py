from copy import deepcopy
from pathlib import Path

import yaml

from mujoco_scenes.scene_loader import load_all_configs
from mujoco_scenes.task_witness import (
    evaluate_joint_task_witness,
    load_task_requirements,
)


TASK_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "stir_contents_joint.yaml"
)


def _object_node(object_id, label, confidence=0.8):
    validated = None
    if label is not None:
        validated = {
            "status": "SUPPORTED",
            "canonical_label": label,
            "mean_confidence": confidence,
            "quality": {
                "supporting_view_count": 2,
                "mean_confidence": confidence,
                "winning_label_margin": 0.5,
            },
            "observation_source": "RGB_DETECTOR",
        }
    return {
        "id": f"object:{object_id}",
        "type": "object",
        "attributes": {
            "object_id": object_id,
            "semantics": {"validated": validated},
        },
    }


def _candidate_edge(object_id, role, status):
    return {
        "source": f"object:{object_id}",
        "target": f"role:{role}",
        "relation": "SATISFIES_GEOMETRY",
        "status": status,
        "evidence": {
            "checks": [
                {
                    "name": (
                        "OPEN_CAVITY"
                        if role == "mixing_container"
                        else "ELONGATED_OBJECT"
                    ),
                    "status": status,
                }
            ]
        },
    }


def _relation_edge(source, target, relation, status):
    return {
        "source": f"object:{source}",
        "target": f"object:{target}",
        "relation": relation,
        "status": status,
        "evidence": {"source": "stage_local_measurement"},
    }


def _primary_graph(stage=2, include_fork=True):
    nodes = [
        _object_node("object_0001", "bowl"),
        _object_node("object_0002", "marker"),
        _object_node("object_0003", "spoon"),
    ]
    if include_fork:
        nodes.append(_object_node("object_0004", "fork"))
    edges = [
        _candidate_edge("object_0001", "mixing_container", "TRUE"),
        _candidate_edge("object_0001", "mixing_tool", "FALSE"),
        _candidate_edge("object_0002", "mixing_container", "FALSE"),
        _candidate_edge("object_0002", "mixing_tool", "TRUE"),
        _candidate_edge("object_0003", "mixing_container", "FALSE"),
        _candidate_edge("object_0003", "mixing_tool", "TRUE"),
        _relation_edge(
            "object_0002", "object_0001", "INSERTABLE_IN", "TRUE"
        ),
        _relation_edge(
            "object_0002", "object_0001", "REACHES_BOTTOM", "TRUE"
        ),
        _relation_edge(
            "object_0003", "object_0001", "INSERTABLE_IN", "FALSE"
        ),
        _relation_edge(
            "object_0003", "object_0001", "REACHES_BOTTOM", "TRUE"
        ),
    ]
    if include_fork:
        edges.extend(
            [
                _candidate_edge(
                    "object_0004", "mixing_container", "FALSE"
                ),
                _candidate_edge("object_0004", "mixing_tool", "TRUE"),
                _relation_edge(
                    "object_0004",
                    "object_0001",
                    "INSERTABLE_IN",
                    "TRUE",
                ),
                _relation_edge(
                    "object_0004",
                    "object_0001",
                    "REACHES_BOTTOM",
                    "TRUE",
                ),
            ]
        )
    return {"stage": stage, "nodes": nodes, "edges": edges}


def _evaluate(graph, mode="joint"):
    return evaluate_joint_task_witness(
        graph,
        load_task_requirements(TASK_PATH),
        grounding_mode=mode,
    )


def _selected_tool(result):
    witness = result["selected_witness"]
    return witness["mixing_tool"][0] if witness else None


def test_primary_scene_has_intended_visible_hidden_distribution():
    config = load_all_configs()["S1_joint_stir_counterexamples"]
    assert set(config.countertop_objects.values()) == {
        "mixing_bowl",
        "marker",
    }
    assert config.container_contents["D1"] == ["oversized_spoon"]
    assert config.container_contents["D2"] == ["fork"]


def test_marker_is_absent_from_mixing_tool_semantic_preferences():
    task = load_task_requirements(TASK_PATH)
    labels = {
        preference["canonical_label"]
        for preference in task["roles"]["mixing_tool"][
            "semantic_preferences"
        ]
    }
    assert "marker" not in labels


def test_spoon_is_ranked_above_fork():
    preferences = load_task_requirements(TASK_PATH)["roles"][
        "mixing_tool"
    ]["semantic_preferences"]
    ranks = {
        preference["canonical_label"]: preference["rank"]
        for preference in preferences
    }
    assert ranks["spoon"] < ranks["fork"]


def test_semantic_and_unary_geometry_are_evaluated_independently():
    result = _evaluate(_primary_graph())
    marker = next(
        candidate
        for candidate in result["candidate_evaluations"]
        if candidate["object_id"] == "object_0002"
        and candidate["role"] == "mixing_tool"
    )
    assert marker["semantic"]["status"] == "FALSE"
    assert marker["unary_geometry"]["status"] == "TRUE"


def test_marker_is_rejected_semantically_even_when_geometry_passes():
    result = _evaluate(_primary_graph(stage=0, include_fork=False))
    marker = next(
        candidate
        for candidate in result["candidate_evaluations"]
        if candidate["object_id"] == "object_0002"
        and candidate["role"] == "mixing_tool"
    )
    assert marker["decision"] == "REJECTED_SEMANTIC"


def test_oversized_spoon_is_rejected_by_relation_not_semantics():
    result = _evaluate(_primary_graph(include_fork=False))
    spoon_assignments = [
        assignment
        for assignment in result["assignment_evaluations"]
        if assignment["selected_objects"]["mixing_container"]
        == ["object_0001"]
        and assignment["selected_objects"]["mixing_tool"]
        == ["object_0003"]
    ]
    assert spoon_assignments[0]["decision"] == "REJECTED_GEOMETRY"
    assert next(
        check
        for check in spoon_assignments[0]["relation_checks"]
        if check["relation"] == "INSERTABLE_IN"
    )["status"] == "FALSE"


def test_fork_is_selected_only_when_both_evidence_paths_pass():
    result = _evaluate(_primary_graph())
    assert result["status"] == "COMPLETE"
    assert _selected_tool(result) == "object_0004"


def test_required_unknown_prevents_completion():
    graph = _primary_graph()
    edge = next(
        edge
        for edge in graph["edges"]
        if edge["relation"] == "REACHES_BOTTOM"
        and edge["source"] == "object:object_0004"
    )
    edge["status"] = "UNKNOWN"
    result = _evaluate(graph)
    assert result["status"] == "INDETERMINATE"


def test_relation_direction_is_subject_tool_to_object_container():
    graph = _primary_graph()
    graph["edges"] = [
        edge
        for edge in graph["edges"]
        if not (
            edge["relation"] == "INSERTABLE_IN"
            and edge["source"] == "object:object_0004"
        )
    ]
    graph["edges"].append(
        _relation_edge(
            "object_0001",
            "object_0004",
            "INSERTABLE_IN",
            "TRUE",
        )
    )
    assert _evaluate(graph)["status"] == "INDETERMINATE"


def test_role_assignments_are_distinct():
    graph = {
        "stage": 0,
        "nodes": [_object_node("object_0001", "bowl")],
        "edges": [
            _candidate_edge(
                "object_0001", "mixing_container", "TRUE"
            ),
            _candidate_edge("object_0001", "mixing_tool", "TRUE"),
        ],
    }
    result = _evaluate(graph, mode="geometry-only")
    assert result["status"] == "INCOMPLETE"
    assert {
        assignment["decision"]
        for assignment in result["assignment_evaluations"]
    } == {"REJECTED_DISTINCTNESS"}


def test_ranking_is_applied_after_invalid_spoon_is_filtered():
    result = _evaluate(_primary_graph())
    assert _selected_tool(result) == "object_0004"
    assert result["selected_candidate_edges"][1]["semantic_rank"] == 2


def test_spoon_outranks_fork_when_both_are_valid():
    graph = _primary_graph()
    insertable = next(
        edge
        for edge in graph["edges"]
        if edge["relation"] == "INSERTABLE_IN"
        and edge["source"] == "object:object_0003"
    )
    insertable["status"] = "TRUE"
    assert _selected_tool(_evaluate(graph)) == "object_0003"


def test_confidence_cannot_make_fork_outrank_valid_spoon():
    graph = _primary_graph()
    graph["nodes"][2]["attributes"]["semantics"]["validated"][
        "mean_confidence"
    ] = 0.2
    graph["nodes"][3]["attributes"]["semantics"]["validated"][
        "mean_confidence"
    ] = 0.99
    next(
        edge
        for edge in graph["edges"]
        if edge["relation"] == "INSERTABLE_IN"
        and edge["source"] == "object:object_0003"
    )["status"] = "TRUE"
    assert _selected_tool(_evaluate(graph)) == "object_0003"


def test_persistent_id_is_final_tie_breaker():
    graph = _primary_graph()
    graph["nodes"].append(_object_node("object_0005", "fork", 0.8))
    graph["edges"].extend(
        [
            _candidate_edge(
                "object_0005", "mixing_container", "FALSE"
            ),
            _candidate_edge("object_0005", "mixing_tool", "TRUE"),
            _relation_edge(
                "object_0005",
                "object_0001",
                "INSERTABLE_IN",
                "TRUE",
            ),
            _relation_edge(
                "object_0005",
                "object_0001",
                "REACHES_BOTTOM",
                "TRUE",
            ),
        ]
    )
    assert _selected_tool(_evaluate(graph)) == "object_0004"


def test_resolver_considers_cached_global_objects_not_only_newest():
    graph = _primary_graph()
    graph["nodes"][0]["attributes"]["first_seen_stage"] = 0
    graph["nodes"][3]["attributes"]["first_seen_stage"] = 2
    result = _evaluate(graph)
    assert result["selected_witness"]["mixing_container"] == [
        "object_0001"
    ]
    assert _selected_tool(result) == "object_0004"


def test_valid_stage_zero_assignment_can_complete_immediately():
    graph = _primary_graph(stage=0)
    next(
        edge
        for edge in graph["edges"]
        if edge["relation"] == "INSERTABLE_IN"
        and edge["source"] == "object:object_0003"
    )["status"] = "TRUE"
    result = _evaluate(graph)
    assert result["status"] == "COMPLETE"
    assert result["stage"] == 0
    assert _selected_tool(result) == "object_0003"


def test_lower_ranked_observed_alternative_may_complete_task():
    result = _evaluate(_primary_graph())
    assert _selected_tool(result) == "object_0004"
    assert result["selected_candidate_edges"][1]["semantic_rank"] == 2


def test_geometry_only_ablation_selects_marker():
    result = _evaluate(_primary_graph(stage=0, include_fork=False), "geometry-only")
    assert result["status"] == "COMPLETE"
    assert _selected_tool(result) == "object_0002"


def test_semantic_only_ablation_selects_oversized_spoon():
    result = _evaluate(
        _primary_graph(stage=1, include_fork=False), "semantic-only"
    )
    assert result["status"] == "COMPLETE"
    assert _selected_tool(result) == "object_0003"


def test_joint_rejects_both_counterexamples_and_selects_fork():
    result = _evaluate(_primary_graph(), "joint")
    assert _selected_tool(result) == "object_0004"


def test_all_ablation_modes_consume_same_graph_evidence():
    graph = _primary_graph()
    before = deepcopy(graph)
    for mode in ("joint", "geometry-only", "semantic-only"):
        _evaluate(graph, mode)
    assert graph == before


def test_exhaustion_does_not_fabricate_assignment():
    result = _evaluate(_primary_graph(include_fork=False), "joint")
    assert result["status"] == "INCOMPLETE"
    assert result["selected_witness"] is None


def test_runtime_modules_do_not_reference_evaluation_ground_truth():
    root = Path(__file__).resolve().parents[1]
    runtime = "\n".join(
        (root / filename).read_text()
        for filename in (
            "semantic_grounding.py",
            "task_witness.py",
            "observed_state.py",
        )
    )
    assert "joint_grounding_evaluation.yaml" not in runtime


def test_manual_task_spec_has_no_instance_or_region_ids():
    raw = yaml.safe_load(TASK_PATH.read_text())
    serialized = str(raw)
    assert "object_000" not in serialized
    for region in ("D1", "D2", "C1", "C2", "B1"):
        assert region not in serialized
