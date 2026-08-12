from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from mujoco_scenes.scene_loader import (
    COUNTER_SPOTS,
    INTEGRATED_TARGET_VESSELS,
    KitchenScene,
    SCENE_OBJECT_VARIANTS,
    configure_integrated_target_layout,
    validate_integrated_countertop_clearance,
    load_all_configs,
)
from mujoco_scenes.task_witness import (
    evaluate_usage_policy_task_witness,
    load_task_requirements,
)


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
TASK_PATH = CONFIG_DIR / "s1_integrated_kitchen_object_function.yaml"
OBJECT_LIBRARY = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "objects"
    / "object_library.xml"
)
COFFEE = ("coffee_1", "coffee_2", "coffee_3")
SOUP = ("soup_1", "soup_2", "soup_3")
TARGETS = COFFEE + SOUP


def _node(object_id, label):
    return {
        "id": f"object:{object_id}",
        "type": "object",
        "attributes": {
            "object_id": object_id,
            "measurement_cloud_path": (
                f"stages/000_initial/evidence/{object_id}/fused.ply"
            ),
            "semantics": {
                "validated": {
                    "status": "SUPPORTED",
                    "canonical_label": label,
                    "mean_confidence": 0.8,
                    "semantic_record_path": (
                        f"stages/000_initial/semantics/{object_id}/"
                        "semantic_evidence.json"
                    ),
                }
            },
        },
    }


def _unary(object_id, role, status):
    predicate = (
        "OPEN_CAVITY" if role.endswith("container")
        else "ELONGATED_OBJECT"
    )
    return {
        "source": f"object:{object_id}",
        "target": f"role:{role}",
        "relation": "SATISFIES_GEOMETRY",
        "status": status,
        "evidence": {"checks": [{"name": predicate, "status": status}]},
    }


def _relation(tool, target, predicate, status):
    evidence = {
        "status": status,
        "pass_margin_m": 0.02 if status == "TRUE" else -0.02,
        "source_measurement_cloud_path": (
            f"stages/000_initial/evidence/{tool}/fused.ply"
        ),
        "target_measurement_cloud_path": (
            f"stages/000_initial/evidence/{target}/fused.ply"
        ),
    }
    if predicate == "INSERTABLE_IN":
        evidence.update(
            maximum_cross_section_m=0.02,
            clearance_margin_m=0.005,
            opening_width_m=0.045,
        )
    else:
        evidence.update(
            usable_length_m=0.18,
            grip_allowance_m=0.03,
            cavity_depth_m=0.10,
        )
    return {
        "source": f"object:{tool}",
        "target": f"object:{target}",
        "relation": predicate,
        "status": status,
        "evidence": evidence,
    }


COMPATIBILITY = {
    "short": {"coffee_3", "soup_1"},
    "medium": {"coffee_2", "coffee_3", "soup_1", "soup_2"},
    "wide": {"soup_1", "soup_2"},
    "fork": set(TARGETS),
    "marker": set(TARGETS),
    "oversized": set(),
    "partial": {"coffee_2", "coffee_3", "soup_1", "soup_2"},
    "soup_long": {"coffee_3", "soup_1", "soup_2", "soup_3"},
    "near_miss": {"coffee_2", "coffee_3", *SOUP},
    "final": set(TARGETS),
}


def _graph(tools):
    labels = {
        **{target: ("mug" if target == "coffee_2" else "cup") for target in COFFEE},
        **{target: "bowl" for target in SOUP},
        **{tool: ("fork" if tool == "fork" else "marker" if tool == "marker" else "spoon") for tool in tools},
    }
    nodes = [_node(object_id, label) for object_id, label in labels.items()]
    edges = []
    roles = (
        "coffee_container", "soup_container",
        "coffee_stirrer", "soup_eating_utensil",
    )
    for object_id in labels:
        for role in roles:
            is_target_role = role.endswith("container")
            status = "TRUE" if (object_id in TARGETS) == is_target_role else "FALSE"
            edges.append(_unary(object_id, role, status))
    for tool in tools:
        for target in TARGETS:
            status = "TRUE" if target in COMPATIBILITY[tool] else "FALSE"
            edges.extend(
                _relation(tool, target, predicate, status)
                for predicate in ("INSERTABLE_IN", "REACHES_BOTTOM")
            )
    return {
        "stage": 0,
        "pairing": {"strategy": "exhaustive_all_pairs"},
        "nodes": nodes,
        "edges": edges,
    }


def _evaluate(tools, mode="joint-target-specific"):
    return evaluate_usage_policy_task_witness(
        _graph(tools), TASK_PATH, target_assignment_mode=mode
    )


def _groups(result):
    return {
        item["function_group_id"]: item
        for item in result["function_group_evaluations"]
    }


def test_integrated_scene_family_has_three_visible_and_three_stored_targets():
    configs = load_all_configs()
    names = {
        "S1_integrated_kitchen_object_function_primary",
        "S1_integrated_kitchen_object_function_initial_complete",
        "S1_integrated_kitchen_object_function_exhaustion",
    }
    assert names <= configs.keys()
    primary = configs["S1_integrated_kitchen_object_function_primary"]
    assert len(primary.countertop_objects) == 10
    assert {
        "s1i_compact_kettle",
        "s1i_compact_coffee_jar",
    } <= set(
        primary.countertop_objects.values()
    )
    assert "pot_with_soup" not in primary.countertop_objects.values()
    assert sum(map(len, primary.container_contents.values())) == 10
    assert "s1i_final_long_narrow_spoon" not in (
        primary.countertop_objects.values()
    )
    assert "s1i_final_long_narrow_spoon" in primary.container_contents["C1"]
    assert "marker" in primary.countertop_objects.values()
    assert len(
        set(INTEGRATED_TARGET_VESSELS)
        & set(primary.countertop_objects.values())
    ) == 3
    assert all(len(items) <= 2 for items in primary.container_contents.values())
    assert all(
        sum(
            item in INTEGRATED_TARGET_VESSELS
            for item in primary.container_contents[region]
        ) == 1
        for region in ("C2", "B1", "C1")
    )
    positions = [COUNTER_SPOTS[spot] for spot in primary.countertop_objects]
    assert all(-0.70 <= x <= 0.60 for x, _y, _z in positions)
    assert all(-0.40 <= y <= -0.05 for _x, y, _z in positions)


def test_seeded_integrated_layout_is_deterministic_capacity_safe_and_varied():
    configs = load_all_configs()
    first = load_all_configs()[
        "S1_integrated_kitchen_object_function_primary"
    ]
    second = load_all_configs()[
        "S1_integrated_kitchen_object_function_primary"
    ]
    other = load_all_configs()[
        "S1_integrated_kitchen_object_function_primary"
    ]
    manifest_a = configure_integrated_target_layout(first, 17)
    manifest_b = configure_integrated_target_layout(second, 17)
    manifest_c = configure_integrated_target_layout(other, 19)
    assert manifest_a == manifest_b
    assert manifest_a["target_locations"] != manifest_c["target_locations"]
    assert all(len(items) <= 2 for items in first.container_contents.values())
    assert all(
        sum(
            item in INTEGRATED_TARGET_VESSELS
            for item in first.container_contents[region]
        ) == 1
        for region in ("C2", "B1", "C1")
    )

    # Assignment identity changes with the seed, but all visible vessel/tool
    # combinations retain the scene's conservative 15 mm footprint buffer.
    for seed in range(100):
        seeded = load_all_configs()[
            "S1_integrated_kitchen_object_function_primary"
        ]
        configure_integrated_target_layout(seeded, seed)
        validate_integrated_countertop_clearance(seeded)

    initial_complete = configs[
        "S1_integrated_kitchen_object_function_initial_complete"
    ]
    assert list(initial_complete.countertop_objects.values()).count(
        "s1i_final_long_narrow_spoon"
    ) == 3

    exhaustion = configs[
        "S1_integrated_kitchen_object_function_exhaustion"
    ]
    assert "s1i_final_long_narrow_spoon" not in {
        *exhaustion.countertop_objects.values(),
        *(
            item
            for contents in exhaustion.container_contents.values()
            for item in contents
        ),
    }


def test_variant_visual_and_proxy_scales_remain_identical():
    root = ET.parse(OBJECT_LIBRARY).getroot()
    mesh_scales = {
        mesh.get("name"): tuple(
            float(value)
            for value in mesh.get("scale", "1 1 1").split()
        )
        for mesh in root.findall("./asset/mesh")
    }
    for name, variant in SCENE_OBJECT_VARIANTS.items():
        assert mesh_scales[variant["mesh"]] == tuple(variant["scale"]), name


def test_primary_vessels_use_clean_nonstretched_materials():
    expected = {
        "ab3_narrow_deep_cup": "mat_s1i_cup_cream",
        "ab3_medium_deep_mug": "mat_s1i_mug_blue",
        "s1i_wide_shallow_cup": "mat_s1i_cup_sage",
        "ab3_shallow_bowl": "mat_s1i_bowl_ivory",
        "ab3_deep_bowl": "mat_s1i_bowl_blue",
        "s1i_narrow_deep_bowl": "mat_s1i_bowl_sage",
    }
    assert {
        name: SCENE_OBJECT_VARIANTS[name].get("material")
        for name in expected
    } == expected


def test_primary_sources_use_scanned_compact_visuals_without_redundant_pot():
    assert SCENE_OBJECT_VARIANTS["s1i_compact_kettle"] == {
        "base": "kettle",
        "scale": (0.78, 0.78, 0.78),
        "mesh": "mesh_s1i_compact_kettle",
    }
    assert SCENE_OBJECT_VARIANTS["s1i_compact_coffee_jar"] == {
        "base": "coffee_jar",
        "scale": (0.78, 0.78, 0.78),
        "mesh": "mesh_s1i_compact_coffee_jar",
    }
    primary = load_all_configs()[
        "S1_integrated_kitchen_object_function_primary"
    ]
    assert "pot_with_soup" not in primary.countertop_objects.values()


def test_bowls_sources_visibly_expose_soup_powder_and_hot_water():
    root = ET.parse(OBJECT_LIBRARY).getroot()
    assert root.find(
        "./body[@name='bowl']/geom[@name='bowl_soup_surface']"
    ).get("material") == "mat_tomato_soup"
    assert root.find(
        "./body[@name='coffee_jar']/geom[@name='coffee_powder_surface']"
    ).get("material") == "mat_coffee_powder"
    assert root.find(
        "./body[@name='kettle']/geom[@name='kettle_hot_water']"
    ).get("material") == "mat_hot_water"


def test_d2_drawer_and_all_contents_remain_open_after_fixture_release():
    scene = KitchenScene(
        "S1_integrated_kitchen_object_function_primary",
        include_robot=False,
        robot="none",
    )
    scene.open_container("D2", steps=200)
    assert scene.release_storage_fixture("D2")
    for _ in range(80):
        mujoco.mj_step(scene.model, scene.data)
    state = scene.get_region_observation_states()["D2"]
    assert state["open"]
    assert state["open_fraction"] > 0.95
    for body_name in ("ab3_partial_spoon", "tongs"):
        body_id = mujoco.mj_name2id(
            scene.model, mujoco.mjtObj.mjOBJ_BODY, body_name
        )
        assert scene.data.xpos[body_id][1] < -0.48


def test_c2_spoon_stands_above_shelf_with_bowl_up_and_wall_clearance():
    scene = KitchenScene(
        "S1_integrated_kitchen_object_function_primary",
        include_robot=False,
        robot="none",
    )
    scene.open_container("C2", steps=200)
    body_id = mujoco.mj_name2id(
        scene.model, mujoco.mjtObj.mjOBJ_BODY, "s1i_c2_soup_spoon"
    )
    handle_id = mujoco.mj_name2id(
        scene.model, mujoco.mjtObj.mjOBJ_GEOM,
        "s1i_c2_soup_spoon_handle_collision",
    )
    bowl_id = mujoco.mj_name2id(
        scene.model, mujoco.mjtObj.mjOBJ_GEOM,
        "s1i_c2_soup_spoon_bowl_collision",
    )
    shelf_id = mujoco.mj_name2id(
        scene.model, mujoco.mjtObj.mjOBJ_GEOM, "C2_shelf"
    )
    right_id = mujoco.mj_name2id(
        scene.model, mujoco.mjtObj.mjOBJ_GEOM, "C2_right"
    )
    shelf_top = (
        scene.data.geom_xpos[shelf_id, 2] + scene.model.geom_size[shelf_id, 2]
    )
    minimum_z = float("inf")
    maximum_x = float("-inf")
    for geom_id in (handle_id, bowl_id):
        centre = scene.data.geom_xpos[geom_id]
        rotation = scene.data.geom_xmat[geom_id].reshape(3, 3)
        size = scene.model.geom_size[geom_id]
        if scene.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_CAPSULE:
            first = centre - rotation[:, 2] * size[1]
            second = centre + rotation[:, 2] * size[1]
            lower = np.minimum(first, second) - size[0]
            upper = np.maximum(first, second) + size[0]
        else:
            half_extent = np.sqrt(np.sum((rotation * size[None, :]) ** 2, axis=1))
            lower, upper = centre - half_extent, centre + half_extent
        minimum_z = min(minimum_z, float(lower[2]))
        maximum_x = max(maximum_x, float(upper[0]))
    right_inner_face = (
        scene.data.geom_xpos[right_id, 0] - scene.model.geom_size[right_id, 0]
    )
    assert scene.data.geom_xpos[bowl_id, 2] > scene.data.geom_xpos[handle_id, 2]
    assert minimum_z >= shelf_top
    assert maximum_x < right_inner_face
    assert right_inner_face - maximum_x <= 0.001
    assert scene.data.eq_active[
        mujoco.mj_name2id(
            scene.model, mujoco.mjtObj.mjOBJ_EQUALITY,
            "storage_fixture_C2_upright_spoon",
        )
    ]


def test_integrated_manual_specification_has_function_scoped_usage():
    task = load_task_requirements(TASK_PATH)
    assert task["goal_instruction"] == (
        "Prepare and serve coffee and soup for three people using the "
        "available kitchenware. Stir all three coffees and provide each "
        "soup bowl with a suitable utensil. Search the closed kitchen "
        "storage for anything still required."
    )
    assert task["roles"]["coffee_container"]["count"] == 3
    assert task["roles"]["soup_container"]["count"] == 3
    coffee = task["operation_groups"]["coffee_stirring"]
    soup = task["operation_groups"]["soup_serving"]
    assert not coffee["usage_policy"]["same_tool_must_cover_all_targets"]
    assert coffee["usage_policy"]["selection_preference"] == (
        "minimize_distinct_tools"
    )
    assert soup["usage_policy"]["distinct_within_group"]
    assert task["cross_group_reuse"]["allowed"]
    soup_labels = {
        preference["canonical_label"]
        for preference in task["roles"]["soup_eating_utensil"][
            "semantic_preferences"
        ]
    }
    assert soup_labels == {"spoon"}


def test_primary_progression_requires_c1_all_target_spoon():
    initial = ["short", "medium", "wide", "fork", "marker"]
    checkpoints = [
        (initial, "INCOMPLETE", "INCOMPLETE"),
        (initial + ["oversized"], "INCOMPLETE", "INCOMPLETE"),
        (initial + ["oversized", "partial"], "INCOMPLETE", "INCOMPLETE"),
        (initial + ["oversized", "partial", "soup_long"], "INCOMPLETE", "COMPLETE"),
        (initial + ["oversized", "partial", "soup_long", "near_miss"], "INCOMPLETE", "COMPLETE"),
    ]
    for tools, expected_global, expected_soup in checkpoints:
        result = _evaluate(tools)
        assert result["status"] == expected_global
        assert _groups(result)["soup_serving"]["status"] == expected_soup
    complete = _evaluate(checkpoints[-1][0] + ["final"])
    assert complete["status"] == "COMPLETE"
    assert _groups(complete)["coffee_stirring"]["status"] == "COMPLETE"
    assert complete["distinct_physical_tool_count"] == 3
    coffee_tools = {
        item["utensil_object_id"]
        for item in complete["operation_assignments"]
        if item["function_group_id"] == "coffee_stirring"
    }
    soup_tools = {
        item["utensil_object_id"]
        for item in complete["operation_assignments"]
        if item["function_group_id"] == "soup_serving"
    }
    assert coffee_tools == {"final"}
    assert len(soup_tools) == 3
    assert "final" in soup_tools


def test_partial_count_and_semantics_cannot_control_production():
    tools = ["short", "medium", "wide", "fork", "marker"]
    assert _evaluate(tools, "joint-target-specific")["status"] == "INCOMPLETE"
    assert _evaluate(tools, "semantic-only")["status"] == "COMPLETE"
    assert _evaluate(tools, "geometry-only")["status"] == "COMPLETE"
    assert _evaluate(tools, "joint-target-agnostic-count")["status"] == "COMPLETE"


def test_exhaustion_roster_never_fabricates_reusable_coverage():
    result = _evaluate(
        ["short", "medium", "wide", "fork", "marker", "oversized", "partial", "soup_long", "near_miss"]
    )
    assert result["status"] == "INCOMPLETE"
    assert _groups(result)["coffee_stirring"]["status"] == "INCOMPLETE"
