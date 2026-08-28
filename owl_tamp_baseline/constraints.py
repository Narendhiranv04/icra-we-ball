"""Restricted validator for model-generated OWL-TAMP constraints."""

from __future__ import annotations

import ast
from typing import Iterable

from .models import Constraint, ValidationError


HELPERS = {
    "within_distance": 3,
    "inside": 2,
    "supported_by": 2,
    "collision_free": 1,
    "reachable": 1,
    "upright": 1,
}


def validate_constraints(constraints: Iterable[Constraint]) -> tuple[Constraint, ...]:
    result = []
    for constraint in constraints:
        try:
            tree = ast.parse(constraint.expression, mode="eval")
        except SyntaxError as error:
            raise ValidationError("constraint expression is not valid syntax") from error
        call = tree.body
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            raise ValidationError("constraint expression must be one helper call")
        if call.func.id not in HELPERS or len(call.args) != HELPERS[call.func.id]:
            raise ValidationError("constraint uses an unknown helper or wrong arity")
        if call.keywords or any(
            not isinstance(argument, (ast.Name, ast.Constant)) for argument in call.args
        ):
            raise ValidationError("constraint helper arguments must be IDs or literals")
        result.append(constraint)
    return tuple(result)
