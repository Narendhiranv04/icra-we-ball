import unittest

from mujoco_scenes.task_witness import (
    evaluate_geometric_requirements,
    evaluate_task_witness,
    load_task_requirements,
)


def task(role_counts, pairwise=(), distinct=True):
    return {
        "task_id": "synthetic_geometry_task",
        "roles": {
            role: {
                "count": count,
                "geometric_requirements": [
                    {
                        "predicate": f"GEOMETRY_FOR_{role.upper()}",
                        "required_status": "TRUE",
                    }
                ],
            }
            for role, count in role_counts.items()
        },
        "constraints": {
            "distinct_objects": distinct,
            "pairwise": list(pairwise),
        },
    }


def graph(role_candidates, relations=()):
    object_ids = sorted(
        {
            object_id
            for candidates in role_candidates.values()
            for object_id, _status in candidates
        }
        | {
            object_id
            for _relation, source, target, _status in relations
            for object_id in (source, target)
        }
    )
    nodes = [
        {
            "id": f"object:{object_id}",
            "type": "object",
            "attributes": {"object_id": object_id},
        }
        for object_id in object_ids
    ] + [
        {
            "id": f"role:{role}",
            "type": "geometric_role",
            "attributes": {"role_id": role},
        }
        for role in sorted(role_candidates)
    ]
    edges = [
        {
            "source": f"object:{object_id}",
            "target": f"role:{role}",
            "relation": "SATISFIES_GEOMETRY",
            "status": status,
            "evidence": {"checks": []},
        }
        for role, candidates in sorted(role_candidates.items())
        for object_id, status in candidates
    ] + [
        {
            "source": f"object:{source}",
            "target": f"object:{target}",
            "relation": relation,
            "status": status,
        }
        for relation, source, target, status in relations
    ]
    return {
        "schema_version": 2,
        "inference_basis": "GEOMETRY_ONLY",
        "stage": 0,
        "nodes": nodes,
        "edges": edges,
    }


PAIRWISE = (
    {
        "relation": "INSERTABLE_IN",
        "from_role": "tool",
        "to_role": "receptacle",
        "apply_to": "all_selected_targets",
        "required_status": "TRUE",
    },
)


class TaskWitnessTests(unittest.TestCase):
    def test_roles_must_be_a_mapping(self):
        with self.assertRaisesRegex(ValueError, "roles as a mapping"):
            load_task_requirements({"task_id": "bad", "roles": []})

    def test_property_bounds_must_be_finite(self):
        requirements = task({"tool": 1})
        requirements["roles"]["tool"]["geometric_requirements"] = [
            {"property": "length", "minimum": float("nan")}
        ]
        with self.assertRaisesRegex(ValueError, "finite"):
            load_task_requirements(requirements)

    def test_complete_assignment_uses_distinct_geometric_candidates(self):
        requirements = task({"receptacle": 2, "tool": 1}, PAIRWISE)
        observed = graph(
            {
                "receptacle": [
                    ("object_0002", "TRUE"),
                    ("object_0001", "TRUE"),
                ],
                "tool": [("object_0003", "TRUE")],
            },
            (
                ("INSERTABLE_IN", "object_0003", "object_0001", "TRUE"),
                ("INSERTABLE_IN", "object_0003", "object_0002", "TRUE"),
            ),
        )
        result = evaluate_task_witness(observed, requirements)
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["inference_basis"], "GEOMETRY_ONLY")
        self.assertEqual(
            result["selected_witness"],
            {
                "receptacle": ["object_0001", "object_0002"],
                "tool": ["object_0003"],
            },
        )

    def test_one_object_cannot_fill_two_roles_when_distinct(self):
        result = evaluate_task_witness(
            graph(
                {
                    "receptacle": [("object_0001", "TRUE")],
                    "tool": [("object_0001", "TRUE")],
                }
            ),
            task({"receptacle": 1, "tool": 1}),
        )
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertIn(
            "GLOBAL_DISTINCTNESS_UNSATISFIABLE",
            result["reason_codes"],
        )

    def test_insufficient_geometric_candidates_are_incomplete(self):
        result = evaluate_task_witness(
            graph({"receptacle": [("object_0001", "TRUE")]}),
            task({"receptacle": 2}),
        )
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(result["missing_counts"], {"receptacle": 1})

    def test_unknown_role_geometry_is_indeterminate(self):
        result = evaluate_task_witness(
            graph({"receptacle": [("object_0001", "UNKNOWN")]}),
            task({"receptacle": 1}),
        )
        self.assertEqual(result["status"], "INDETERMINATE")
        self.assertIsNone(result["selected_witness"])
        self.assertEqual(len(result["unknown_requirements"]), 1)

    def test_unknown_relation_never_completes(self):
        result = evaluate_task_witness(
            graph(
                {
                    "receptacle": [("object_0001", "TRUE")],
                    "tool": [("object_0002", "TRUE")],
                },
                (
                    (
                        "INSERTABLE_IN",
                        "object_0002",
                        "object_0001",
                        "UNKNOWN",
                    ),
                ),
            ),
            task({"receptacle": 1, "tool": 1}, PAIRWISE),
        )
        self.assertEqual(result["status"], "INDETERMINATE")
        self.assertEqual(len(result["unknown_relations"]), 1)

    def test_all_false_relations_are_incomplete(self):
        result = evaluate_task_witness(
            graph(
                {
                    "receptacle": [("object_0001", "TRUE")],
                    "tool": [("object_0002", "TRUE")],
                },
                (
                    (
                        "INSERTABLE_IN",
                        "object_0002",
                        "object_0001",
                        "FALSE",
                    ),
                ),
            ),
            task({"receptacle": 1, "tool": 1}, PAIRWISE),
        )
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertIn("REQUIRED_RELATION_FALSE", result["reason_codes"])

    def test_true_assignment_precedes_indeterminate_assignment(self):
        result = evaluate_task_witness(
            graph(
                {
                    "receptacle": [("object_0001", "TRUE")],
                    "tool": [
                        ("object_0002", "UNKNOWN"),
                        ("object_0003", "TRUE"),
                    ],
                },
                (
                    (
                        "INSERTABLE_IN",
                        "object_0002",
                        "object_0001",
                        "UNKNOWN",
                    ),
                    (
                        "INSERTABLE_IN",
                        "object_0003",
                        "object_0001",
                        "TRUE",
                    ),
                ),
            ),
            task({"receptacle": 1, "tool": 1}, PAIRWISE),
        )
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(
            result["selected_witness"]["tool"], ["object_0003"]
        )

    def test_multiple_witnesses_are_resolved_lexicographically(self):
        result = evaluate_task_witness(
            graph(
                {
                    "support": [
                        ("object_0003", "TRUE"),
                        ("object_0001", "TRUE"),
                        ("object_0002", "TRUE"),
                    ]
                }
            ),
            task({"support": 2}),
        )
        self.assertEqual(
            result["selected_witness"]["support"],
            ["object_0001", "object_0002"],
        )

    def test_hidden_object_absent_from_graph_is_not_candidate(self):
        result = evaluate_task_witness(
            graph({"receptacle": []}),
            task({"receptacle": 1}),
        )
        self.assertEqual(result["observed_candidates"]["receptacle"], [])
        self.assertEqual(result["status"], "INCOMPLETE")

    def test_property_thresholds_generate_true_false_and_unknown(self):
        requirements = [
            {
                "property": "opening_width_m",
                "minimum": 0.06,
                "unit": "m",
                "allowed_statuses": ["MEASURED"],
            }
        ]
        measured = {
            "geometric_properties": {
                "opening_width_m": {
                    "value": 0.08,
                    "unit": "m",
                    "status": "MEASURED",
                    "method": "test",
                }
            }
        }
        too_small = {
            "geometric_properties": {
                "opening_width_m": {
                    "value": 0.04,
                    "unit": "m",
                    "status": "MEASURED",
                    "method": "test",
                }
            }
        }
        self.assertEqual(
            evaluate_geometric_requirements(measured, requirements)["status"],
            "TRUE",
        )
        self.assertEqual(
            evaluate_geometric_requirements(too_small, requirements)["status"],
            "FALSE",
        )
        self.assertEqual(
            evaluate_geometric_requirements({}, requirements)["status"],
            "UNKNOWN",
        )

    def test_semantic_category_keys_are_rejected(self):
        invalid = task({"support": 1})
        invalid["roles"]["support"]["candidate_function"] = "plate"
        with self.assertRaisesRegex(ValueError, "semantic keys"):
            load_task_requirements(invalid)


if __name__ == "__main__":
    unittest.main()
