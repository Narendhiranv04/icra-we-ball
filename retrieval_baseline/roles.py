"""Function phrases for the retrieval baseline.

Deliberately free of category nouns.  Prompting with "side table" or "cup"
would supply the answer and reduce the baseline to a label lookup, which is
what the proposed method's semantic evidence already does.  These phrases
describe only what the role must *do* or how it must be shaped.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Role:
    key: str
    phrase: str
    count: int
    kind: str  # "object" or "region"


LIVING_ROOM_ROLES = (
    Role(
        key="personal_support",
        phrase="a flat raised surface within arm's reach of a single seated person",
        count=2,
        kind="region",
    ),
    Role(
        key="shared_support",
        phrase="a flat raised surface reachable from two seats at once",
        count=1,
        kind="region",
    ),
    Role(
        key="drink_vessel",
        phrase="a small open vessel for drinking from",
        count=2,
        kind="object",
    ),
    Role(
        key="under_dish",
        phrase="a small shallow flat dish that another object rests on",
        count=2,
        kind="object",
    ),
    Role(
        key="handheld_control",
        phrase="a small handheld device covered in buttons",
        count=1,
        kind="object",
    ),
)

LIVING_ROOM_TASK_TEMPLATE = (
    # (vessel index, dish index, personal support index)
    (0, 0, 0),
    (1, 1, 1),
)
